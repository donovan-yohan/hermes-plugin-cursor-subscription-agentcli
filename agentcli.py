"""Request-scoped Cursor `agent` transport with Hermes-owned tools via a parking MCP bridge.

Each conversation turn runs the official Cursor Agent CLI (`agent -p`) once, in a private
workspace whose `.cursor/mcp.json` advertises exactly the current Hermes tool inventory through
`bridge_mcp.py`. When the agent calls a Hermes tool, the bridge parks the MCP request (no
response) and the transport ends the chat.completions response with `finish_reason: tool_calls`,
killing the agent process on boundary only when the turn is abandoned. Hermes executes the tool
with its own hooks and approvals; the next create() delivers results to the bridge, which
answers the parked requests so the same agent process continues. A final `result` event ends
the turn.

There is no history-replay seam and no raw-response relay in `agent`: canonical Hermes history
is rendered as text on cold starts, and mid-turn results resume the live Cursor chat. Built-in
Cursor tools are denied by workspace permissions so Hermes stays the only executor.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace

try:
    from .agentcli_setup import INSTALL_HINT, LOGIN_HINT, _resolve
except ImportError:
    from agentcli_setup import INSTALL_HINT, LOGIN_HINT, _resolve


class AgentCLIMissing(RuntimeError):
    """The official Cursor agent CLI this transport drives is not installed (or not on PATH)."""


class AgentCLILoggedOut(RuntimeError):
    """The Cursor agent CLI refused before doing work because it has no usable login."""


CARRIER = 'cursor-subscription-agentcli-experimental.native_assistant'
PREFIX = 'hermes-'
BRIDGE_PORT_ENV = 'HERMES_CURSOR_BRIDGE_PORT'
DENY_BUILTINS = ("Shell(*)", "Read(**)", "Write(**)", "Read(/**)", "Write(/**)", "WebFetch(*)", "WebSearch(*)")
ROUTER_MODELS = ('auto',)
_MODEL_EFFORT = re.compile(r'^(.+)-effort-(low|medium|high|xhigh)$')
SETTLE_SECONDS = 0.75


class Object(SimpleNamespace):
    def model_dump(self, **_):
        def unpack(v):
            if isinstance(v, Object):
                return {k: unpack(x) for k, x in vars(v).items()}
            if isinstance(v, list):
                return [unpack(x) for x in v]
            return copy.deepcopy(v)
        return unpack(self)


def obj(value):
    if isinstance(value, dict):
        return Object(**{k: copy.deepcopy(v) if k in ('reasoning_details', 'native_usage') else obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [obj(v) for v in value]
    return value


def projection(message):
    calls = []
    for tc in message.get('tool_calls') or []:
        f = tc['function']
        args = f['arguments']
        calls.append({'id': tc['id'], 'name': f['name'],
                      'input': json.loads(args) if isinstance(args, str) else args})
    return {'content': (message.get('content') or '').strip(), 'tool_calls': calls}


def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get('type') == 'text':
                parts.append(block.get('text', ''))
        return ''.join(parts)
    return ''


def render_prompt(messages):
    """Canonical Hermes history as plain text. There is no native replay seam in `agent`."""
    lines = []
    for message in messages:
        role = message.get('role')
        if role in ('system', 'developer'):
            lines.append(f'[System]\n{_text_of(message.get("content"))}')
        elif role == 'assistant':
            details = message.get('reasoning_details') or []
            carriers = [d for d in details if isinstance(d, dict) and d.get('type') == CARRIER]
            if carriers and (len(carriers) != 1 or carriers[0].get('version') != 1):
                raise ValueError('Unsupported native assistant carrier version')
            text = (message.get('content') or '').strip()
            if text:
                lines.append(f'[Assistant]\n{text}')
            for call in projection(message)['tool_calls']:
                lines.append(f'[Assistant tool call] {call["name"]}({json.dumps(call["input"], ensure_ascii=False)})'
                             f' [id: {call["id"]}]')
        elif role == 'tool':
            lines.append(f'[Tool result for {message.get("tool_call_id")}]\n{_text_of(message.get("content"))}')
        elif role == 'user':
            content = message.get('content')
            if isinstance(content, list):
                parts = []
                for block in content:
                    kind = block.get('type')
                    if kind == 'text':
                        parts.append(block.get('text', ''))
                    elif kind == 'image_url':
                        url = block.get('image_url', {}).get('url', '')
                        media = url[5:].split(';', 1)[0] if url.startswith('data:') else 'unknown'
                        parts.append(f'[image: {media}; not rendered to this transport]')
                    elif kind in ('image', 'document'):
                        parts.append(f'[{kind}: not rendered to this transport]')
                    else:
                        raise ValueError(f'Unsupported content block: {kind}')
                content = '\n'.join(parts)
            lines.append(f'[User]\n{content}')
        else:
            raise ValueError(f'Unsupported message role: {role}')
    return '\n\n'.join(lines)


def bridge_key(name, arguments):
    payload = json.dumps({'name': name, 'arguments': arguments}, sort_keys=True,
                         separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def continuation(messages, turn):
    """Deliverable Hermes results when history is exactly this turn's parked calls plus results.

    Accepts an optional trailing user message (Hermes steering arrives as its own user row after
    the newest tool result). Returns ``(results, steering)`` or ``None`` on any mismatch.
    """
    rows = list(messages)
    steering = ''
    if rows and rows[-1].get('role') == 'user':
        steering = _text_of(rows[-1].get('content'))
        rows = rows[:-1]
    end = len(rows)
    while end > 0 and rows[end - 1].get('role') == 'tool':
        end -= 1
    if end == 0 or rows[end - 1].get('role') != 'assistant':
        return None
    calls = projection(rows[end - 1])['tool_calls']
    if not calls or len(rows) - end != len(calls):
        return None
    by_id = {message.get('tool_call_id'): message for message in rows[end:]}
    results = []
    for call in calls:
        park = turn.parked.get(call['id'])
        if park is None or call['id'] not in by_id or call['id'] in turn.completed:
            return None
        message = by_id[call['id']]
        result = _text_of(message.get('content'))
        if not isinstance(result, str):
            return None
        results.append({'key': park['key'], 'name': park['name'], 'result': result,
                        'is_error': bool(message.get('is_error'))})
    return results, steering


def resume_prompt(results, steering):
    parts = ['[Hermes tool results]']
    for r in results:
        body = r['result'] if r['result'].strip() else '(empty result)'
        parts.append(f'{r["name"]} -> ' + (f'ERROR: {body}' if r['is_error'] else body))
    if steering.strip():
        parts.append(f'[User steering] {steering.strip()}')
    parts.append('Continue the task with these results. Do not call the same tool again with the '
                 'same arguments unless the task needs different ones. Reply to the user when done.')
    return '\n\n'.join(parts)


def normalize_input_schema(schema):
    """Mirror the host's Anthropic-shaped normalization: advisory top-level combinators out,
    nullable unions flattened, object schemas guaranteed properties (handlers re-validate)."""
    try:
        from tools.schema_sanitizer import strip_nullable_unions
        normalized = strip_nullable_unions(schema, keep_nullable_hint=False)
    except ImportError:
        normalized = copy.deepcopy(schema)
    if any(key in normalized for key in ('oneOf', 'allOf', 'anyOf')):
        normalized = {k: v for k, v in normalized.items() if k not in ('oneOf', 'allOf', 'anyOf')}
        normalized.setdefault('type', 'object')
    if normalized.get('type') == 'object' and not isinstance(normalized.get('properties'), dict):
        normalized = {**normalized, 'properties': {}}
    return normalized


def request_body(kwargs):
    allowed = {'model', 'messages', 'tools', 'stream', 'stream_options', 'max_tokens', 'max_completion_tokens',
               'timeout', 'tool_choice', 'parallel_tool_calls', 'n'}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError('Unsupported request parameters: ' + ', '.join(sorted(unknown)))
    if kwargs.get('n', 1) != 1 or kwargs.get('tool_choice', 'auto') not in ('auto', None):
        raise ValueError('Only n=1 and tool_choice=auto are supported')
    if kwargs.get('parallel_tool_calls') is False:
        raise ValueError('parallel_tool_calls=False is unsupported')
    if kwargs.get('stream_options') not in (None, {}, {'include_usage': True}, {'include_usage': False}):
        raise ValueError('Unsupported stream_options')
    manifest, names = [], set()
    for tool in kwargs.get('tools') or []:
        if tool.get('type') != 'function':
            raise ValueError('Only function tools are supported')
        f = tool['function']
        name = f['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', name) or name in names:
            raise ValueError('Tool names must be unique ASCII identifiers of at most 50 characters')
        if f.get('strict'):
            raise ValueError('Strict function schemas are unsupported')
        names.add(name)
        schema, description = f.get('parameters', {'type': 'object'}), f.get('description', '')
        if not isinstance(schema, dict) or not isinstance(description, str):
            raise ValueError('Tool schema must be an object and description a string')
        schema = normalize_input_schema(schema)
        manifest.append({'name': name, 'description': description, 'inputSchema': schema})
    return manifest, names


def _own_process_group():
    if os.name == 'nt':
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'start_new_session': True}


def kill_process_tree(process):
    """Kill `agent` and every descendant so a cancelled request leaves no live Cursor child."""
    if process.poll() is not None:
        return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)  # windows-footgun: ok — the nt branch above never reaches this line
    except (ProcessLookupError, PermissionError):
        pass


class Request:
    def __init__(self, client):
        self.client, self.process = client, None
        self.stream = None
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.protect_park = False  # set while yielding a boundary: the parked agent must survive

    def cancel(self, *, force=False):
        # A parked live turn survives ordinary stream teardown; only real cancellation
        # (timeout, abort, client cancel/close) kills the process tree.
        if self.protect_park and not force:
            return
        self.cancelled.set()
        with self.lock:
            if self.process is not None:
                kill_process_tree(self.process)

    def spawn(self, command, **kwargs):
        with self.lock:
            if self.cancelled.is_set():
                raise RuntimeError('Cursor request cancelled')
            self.process = subprocess.Popen(command, **kwargs, **_own_process_group())
        return self.process


class Stream:
    def __init__(self, iterator, request):
        self.iterator, self.request = iterator, request
        self._advancing = threading.Lock()
        request.stream = weakref.ref(self)

    def __iter__(self):
        return self

    def __next__(self):
        with self._advancing:
            return next(self.iterator)

    def close(self):
        self.request.cancel()
        # An active consumer unwinds itself after cancellation. A paused/unstarted
        # generator has no active owner and can be finalized here.
        if self._advancing.acquire(blocking=False):
            try:
                self.iterator.close()
                with self.request.client._lock:
                    self.request.client._requests.discard(self.request)
            finally:
                self._advancing.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AsyncStream:
    def __init__(self, stream):
        self.stream = stream

    def __aiter__(self):
        return self

    async def __anext__(self):
        def advance():
            try:
                return True, next(self.stream)
            except StopIteration:
                return False, None
        try:
            present, item = await asyncio.to_thread(advance)
        except asyncio.CancelledError:
            self.stream.close()
            raise
        if not present:
            raise StopAsyncIteration
        return item

    async def aclose(self):
        self.stream.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()


def _new_turn(client, request, env, command_argv, manifest, names, model, prompt, timeout):
    """Spawn one `agent -p` process with its bridge listener, private workspace and reader."""
    root = Path(tempfile.mkdtemp(prefix='cursor-agentcli-'))
    try:
        (root / 'tools.json').write_text(json.dumps(manifest), encoding='utf-8')
        cursor_dir = root / '.cursor'
        cursor_dir.mkdir()
        bridge = Path(__file__).with_name('bridge_mcp.py')
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(4)
        # The bridge port travels as a literal argv: Cursor sanitizes MCP child env, so
        # an env var never reaches the bridge process.
        (cursor_dir / 'mcp.json').write_text(json.dumps({'mcpServers': {'hermes': {
            'command': sys.executable, 'args': [str(bridge), str(root / 'tools.json'), str(root / 'bridge.jsonl'),
                                                str(listener.getsockname()[1])]}}}), encoding='utf-8')
        (cursor_dir / 'cli.json').write_text(json.dumps(
            {'permissions': {'allow': ['Mcp(hermes:*)'], 'deny': list(DENY_BUILTINS)}}), encoding='utf-8')
        resolved = _resolve(command_argv, env)
        if resolved is None:
            raise AgentCLIMissing(INSTALL_HINT)
        env = dict(env)
        env[BRIDGE_PORT_ENV] = str(listener.getsockname()[1])
        p = request.spawn(resolved + ['-p', '--output-format', 'stream-json', '--model', model,
                                      '--trust', '--approve-mcps', '--workspace', str(root)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True, encoding='utf-8', cwd=client._workdir(), env=env)
        p.stdin.write(prompt + '\n')
        p.stdin.close()
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise
    events = queue.Queue()

    def read():
        try:
            for line in p.stdout:
                events.put(json.loads(line))
        except Exception as error:
            events.put(error)
        finally:
            # The consumer may close while paused at a yielded chunk. Reaping belongs to this
            # owner thread, never cancel().
            p.wait()
            events.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    turn = SimpleNamespace(p=p, reader=reader, events=events, root=root, listener=listener,
                           session_id=None, parked={}, completed={}, results=[], names=names,
                           emitted='', final_text=None, closed=False, teardown=None, conn=None)

    def teardown():
        if turn.closed:
            return
        turn.closed = True
        kill_process_tree(p)
        for sock in (turn.conn, listener):
            try:
                if sock is not None:
                    sock.close()
            except OSError:
                pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        reader.join(timeout=5)
        for pipe in (p.stdin, p.stdout):
            if pipe and not pipe.closed:
                pipe.close()
        shutil.rmtree(root, ignore_errors=True)
        with client._lock:
            client._turns.pop(id(turn), None)

    turn.teardown = teardown
    with client._lock:
        client._turns[id(turn)] = turn
    return turn


def _dispatch_event(event, turn, emitted):
    """Consume one stream-json event. Returns ('text'|'thinking', delta) to stream, else None."""
    kind = event.get('type')
    if kind == 'system' and event.get('subtype') == 'init':
        turn.session_id = event.get('session_id')
        return None
    if kind == 'assistant':
        if event.get('timestamp_ms') is None:
            # Final echo: duplicate of the streamed text. Some runs (short answers, resumed turns)
            # stream no text deltas at all and only emit this row — keep it as the final text.
            text = ''.join(b.get('text', '') for b in event.get('message', {}).get('content', []) if b.get('type') == 'text')
            if text:
                turn.final_text = text
            return None
        text = ''.join(b.get('text', '') for b in event.get('message', {}).get('content', []) if b.get('type') == 'text')
        if not text:
            return None
        if text.startswith(emitted['text']):
            delta = text[len(emitted['text']):]
            if not delta:
                return None
            emitted['text'] = text
            return ('text', delta)
        # A timestamped assistant row that does not extend the prefix starts a new burst
        # (Cursor resets its running text after a tool round).
        emitted['text'] = text
        return ('text', text)
    if kind == 'thinking' and event.get('subtype') == 'delta' and event.get('text'):
        return ('thinking', event['text'])
    if kind == 'tool_call':
        envelope = event.get('tool_call') or {}
        mcp = envelope.get('mcpToolCall') or {}
        inner = mcp.get('args') if isinstance(mcp.get('args'), dict) else {}
        call_id = event.get('call_id') or mcp.get('toolCallId') or inner.get('toolCallId')
        if event.get('subtype') == 'started':
            name = (mcp.get('toolName') or inner.get('toolName')
                    or ((mcp.get('name') or '')[len(PREFIX):] if mcp.get('name') else ''))
            if call_id and name and name in turn.names and call_id not in turn.parked:
                arguments = inner.get('args') if isinstance(inner.get('args'), dict) else {}
                turn.parked[call_id] = {'name': name, 'arguments': arguments,
                                        'key': bridge_key(name, arguments)}
        else:
            result = mcp.get('result') or envelope.get('result') or {}
            if not isinstance(result, dict) or call_id not in turn.parked:
                return None
            if 'success' in result:
                # The bridge answered this call (or a duplicate raced it); nothing to park.
                success = result['success']
                turn.completed[call_id] = {'result': _text_of(success.get('content')) or '(empty tool result)',
                                              'is_error': bool(success.get('isError'))}
            elif 'permissionDenied' in result or 'error' in result:
                # Denied before the bridge: surface the denial to Hermes as an errored call so
                # the model's view (denial) and Hermes' view stay consistent.
                turn.completed[call_id] = {'result': json.dumps(result), 'is_error': True}
    return None


class Client:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, command=None, args=None, env=None, timeout=600, **_):
        # Hermes snapshots routing metadata from client-shaped objects; this is not a credential.
        self.api_key = 'external-process'
        self.base_url = 'process://cursor-subscription-agentcli-experimental'
        self.env = dict(env) if env is not None else None
        source_env = self.env if self.env is not None else os.environ
        self.command = command or source_env.get('CURSOR_AGENTCLI_COMMAND') or 'agent'
        self.args = list(args or [])
        self.timeout = timeout if isinstance(timeout, (int, float)) else 600
        self._lock, self._requests, self._closed = threading.Lock(), set(), False
        self._turns = {}
        self._live = None  # the parked turn awaiting Hermes tool results
        self._cwd = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def cancel(self):
        """Fast cross-thread cancellation: signal owned groups; never close caller-thread FDs."""
        with self._lock:
            requests = tuple(self._requests)
            turns = tuple(self._turns.values())
        for request in requests:
            request.cancel(force=True)
        for turn in turns:
            turn.teardown()

    def close(self):
        with self._lock:
            self._closed = True
            requests = tuple(self._requests)
            turns = tuple(self._turns.values())
            self._live = None
        for request in requests:
            request.cancel(force=True)
            stream = request.stream() if request.stream else None
            if stream is not None:
                stream.close()
        for turn in turns:
            turn.teardown()
        if self._cwd is not None:
            shutil.rmtree(self._cwd, ignore_errors=True)

    def _workdir(self):
        """One stable cwd per client so Cursor's workspace identity and caches stay warm."""
        with self._lock:
            if self._cwd is None:
                self._cwd = tempfile.mkdtemp(prefix='cursor-agentcli-cwd-')
            return self._cwd

    def create(self, **kwargs):
        # Hermes' auxiliary seam returns this same object and awaits create.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return self._acreate(**kwargs)
        return self._create(**kwargs)

    async def _acreate(self, **kwargs):
        task = asyncio.create_task(asyncio.to_thread(self._create, **kwargs))
        try:
            result = await task
            return AsyncStream(result) if kwargs.get('stream') else result
        except asyncio.CancelledError:
            self.cancel()
            raise

    def _create(self, **kwargs):
        if not isinstance(kwargs.get('model'), str) or not kwargs['model']:
            raise ValueError('model is required')
        request = Request(self)
        with self._lock:
            if self._closed:
                raise RuntimeError('Cursor client is closed')
            self._requests.add(request)
        stream = Stream(self._run(request, kwargs), request)
        if kwargs.get('stream'):
            return stream
        try:
            for chunk in stream:
                if hasattr(chunk, '_response'):
                    return chunk._response
            raise RuntimeError('Native response missing')
        finally:
            stream.close()

    # ── turn plumbing ─────────────────────────────────────────────────────────────────────────

    def _bridge_connect(self, turn, timeout):
        turn.listener.settimeout(timeout)
        try:
            conn, _ = turn.listener.accept()
        except (socket.timeout, OSError) as error:
            raise TimeoutError('Cursor bridge was never established') from error
        conn.settimeout(1.0)
        return conn

    @staticmethod
    def _deliver(conn, results):
        conn.sendall((json.dumps({'type': 'deliver', 'results': results}) + '\n').encode('utf-8'))

    def _boundary_chunk(self, turn, kwargs):
        calls = [{'id': call_id, 'type': 'function',
                  'function': {'name': park['name'],
                               'arguments': json.dumps(park['arguments'], separators=(',', ':'), allow_nan=False)}}
                 for call_id, park in turn.parked.items() if call_id not in turn.completed]
        if not calls:
            raise RuntimeError('Tool boundary without parkable calls')
        message = {'role': 'assistant', 'content': turn.emitted or None, 'tool_calls': calls,
                   'reasoning_content': None}
        carrier = {'type': CARRIER, 'version': 1, 'messages': [], 'projection': projection(message)}
        message['reasoning_details'] = [carrier]
        chunk = self._chunk(kwargs['model'], {'content': None, 'tool_calls': [dict(c, index=i) for i, c in enumerate(calls)]},
                            'tool_calls')
        chunk._response = obj({'id': turn.session_id or 'cursor-native', 'model': kwargs['model'],
                               'object': 'chat.completion',
                               'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': message}],
                               'usage': None})
        return chunk

    def _final_chunk(self, turn, kwargs):
        result = turn.results[-1]
        if result.get('subtype') != 'success' or result.get('is_error'):
            raise RuntimeError('Native request failed: ' + str(result.get('subtype')))
        usage = result.get('usage') or {}
        if not all(isinstance(usage.get(k), (int, float)) for k in ('inputTokens', 'outputTokens')):
            raise RuntimeError('Native result missing complete token usage')
        text = turn.final_text if turn.final_text is not None else turn.emitted
        message = {'role': 'assistant', 'content': text or None, 'tool_calls': None, 'reasoning_content': None}
        carrier = {'type': CARRIER, 'version': 1, 'messages': [], 'projection': projection(message)}
        message['reasoning_details'] = [carrier]
        inp = usage['inputTokens'] + usage.get('cacheReadTokens', 0) + usage.get('cacheWriteTokens', 0)
        normalized_usage = {'prompt_tokens': inp, 'completion_tokens': usage['outputTokens'],
                            'total_tokens': inp + usage['outputTokens'],
                            'prompt_tokens_details': {'cached_tokens': usage.get('cacheReadTokens', 0)},
                            'native_usage': usage}
        response = obj({'id': turn.session_id or 'cursor-native', 'model': kwargs['model'],
                        'object': 'chat.completion',
                        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': message}],
                        'usage': normalized_usage})
        chunk = self._chunk(kwargs['model'], {'content': None}, 'stop', normalized_usage)
        chunk._response = response
        return chunk

    def _pump(self, turn, emitted, request, kwargs, deadline, conn, timeout):
        conn = turn.conn
        """Stream events until a result, a tool boundary, or process exit. True on boundary."""
        while True:
            if request.cancelled.is_set():
                raise RuntimeError('Cursor request cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Cursor request timed out')
            try:
                event = turn.events.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if turn.p.poll() is not None and turn.events.empty():
                    return False
                awaiting = [c for c in turn.parked if c not in turn.completed]
                if awaiting:
                    if turn.conn is None:
                        # First parked call: the bridge client connects now and stays attached
                        # to the turn — it must survive into the follow-up create().
                        turn.conn = self._bridge_connect(turn, min(remaining, 10))
                    # Settle window: parallel tool calls arrive a few hundred ms apart; give the
                    # batch a moment of stream silence before cutting at the boundary.
                    time.sleep(SETTLE_SECONDS)
                    quiet = True
                    while not turn.events.empty():
                        nxt = turn.events.get_nowait()
                        if nxt is None or isinstance(nxt, Exception):
                            continue
                        handled = _dispatch_event(nxt, turn, emitted)
                        if handled:
                            yield self._chunk(kwargs['model'], {'content': handled[1]} if handled[0] == 'text' else {'reasoning_content': handled[1]})
                        quiet = False
                    if quiet and turn.p.poll() is None:
                        return True
                continue
            if event is None:
                return False
            if isinstance(event, Exception):
                raise RuntimeError('Invalid native stream-json output: ' + repr(str(event)[:300])) from event
            deadline = time.monotonic() + timeout  # read-idle deadline: reset on every event
            if event.get('type') == 'result':
                turn.results.append(event)
                return False
            handled = _dispatch_event(event, turn, emitted)
            if handled:
                yield self._chunk(kwargs['model'], {'content': handled[1]} if handled[0] == 'text' else {'reasoning_content': handled[1]})

    def _run(self, request, kwargs):
        turn = None
        conn = None
        try:
            timeout = kwargs.get('timeout', self.timeout)
            timeout = getattr(timeout, 'read', timeout)
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError('timeout must be positive seconds')
            model = kwargs['model']
            if model not in ROUTER_MODELS:
                match = _MODEL_EFFORT.match(model)
                if match:
                    model = match.group(1)  # effort stays Cursor's own choice; the suffix is inert
            manifest, names = request_body(kwargs)
            messages = kwargs.get('messages') or []
            prompt = None

            with self._lock:
                live, self._live = self._live, None
            if live is not None:
                prepared = continuation(messages, live)
                if prepared is None:
                    # History diverged (compaction, retry, new conversation): the live Cursor
                    # chat is no longer canonical. Abandon it and cold-start.
                    live.teardown()
                    live = None
                else:
                    results, steering = prepared
                    if live.conn is None:
                        live.conn = self._bridge_connect(live, min(timeout, 10))
                    self._deliver(live.conn, results)
                    prompt = resume_prompt(results, steering)
                    turn = live
            if turn is None:
                if not messages or messages[-1].get('role') not in ('user', 'tool'):
                    raise ValueError('History must end in a user or tool message; assistant prefill is unsupported')
                prompt = render_prompt(messages)
                env = self.env if self.env is not None else dict(os.environ)
                turn = _new_turn(self, request, env, [self.command] + self.args, manifest, names, model, prompt, timeout)

            emitted = {'text': ''}
            deadline = time.monotonic() + timeout
            boundary = yield from self._pump(turn, emitted, request, kwargs, deadline, conn, timeout)
            if request.cancelled.is_set():
                raise RuntimeError('Cursor request cancelled')
            turn.emitted = emitted['text']
            if boundary:
                # Register the parked turn BEFORE yielding: a consumer that breaks out of the
                # stream never resumes this generator, so post-yield code would not run.
                with self._lock:
                    self._live = turn
                # The agent stays alive parked on the bridge; do not wait for its exit here.
                request.protect_park = True
                yield self._boundary_chunk(turn, kwargs)
                return
            turn.p.wait(timeout=max(0.1, deadline - time.monotonic()))
            turn.reader.join(timeout=5)
            if not turn.results:
                raise RuntimeError('Cursor agent exited without a result')
            final = self._final_chunk(turn, kwargs)
            turn.teardown()  # before yielding: the consumer may never resume this generator
            yield final
        finally:
            request.cancel(force=not request.protect_park or request.cancelled.is_set())
            with self._lock:
                self._requests.discard(request)

    @staticmethod
    def _chunk(model, delta, finish=None, usage=None):
        return obj({'id': 'cursor-native', 'model': model, 'object': 'chat.completion.chunk',
                    'choices': [{'index': 0, 'delta': {'content': None, 'tool_calls': None,
                                                       'reasoning_details': None, **delta},
                                 'finish_reason': finish}],
                    'usage': usage})

"""A fake `agent` CLI: Cursor's stream-json protocol plus its MCP client, in one process.

Stages are selected with the FAKE_STAGE env var. The fake prints system/assistant/tool_call
events on stdout, spawns bridge_mcp.py exactly as Cursor would (env-provided port), parks on
the bridge until Hermes delivers results, then finishes. Every run appends a line to
FAKE_LOG: ``<stage> <pid> <stdin-prompt-first-line>`` so tests can assert which process ran
and what prompt it received.
"""
import json
import os
import socket
import subprocess
import sys
import time

BRIDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bridge_mcp.py')
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fake_tools.json')


def emit(payload):
    print(json.dumps(payload), flush=True)


def assistant(text, stamp=True):
    message = {'role': 'assistant', 'content': [{'type': 'text', 'text': text}]}
    emit({'type': 'assistant', 'message': message, 'session_id': 'fake-sid',
          **({'timestamp_ms': '1'} if stamp else {})})


def tool_call(call_id, name, arguments):
    emit({'type': 'tool_call', 'subtype': 'started', 'call_id': call_id,
          'tool_call': {'toolCallId': call_id,
                        'mcpToolCall': {'args': {'args': arguments, 'toolCallId': call_id, 'name': 'hermes-' + name,
                                                 'toolName': name, 'serverIdentifier': 'hermes'}}},
          'session_id': 'fake-sid'})


def tool_completed(call_id, name, result):
    emit({'type': 'tool_call', 'subtype': 'completed', 'call_id': call_id,
          'tool_call': {'toolCallId': call_id,
                        'mcpToolCall': {'args': {'args': {}, 'toolCallId': call_id, 'name': 'hermes-' + name,
                                                 'toolName': name},
                                        'result': result}},
          'session_id': 'fake-sid'})


def start_bridge():
    """Spawn bridge_mcp.py the way Cursor would: MCP client on our stdin/stdout side."""
    port = os.environ.get('HERMES_CURSOR_BRIDGE_PORT')
    bridge = subprocess.Popen([sys.executable, BRIDGE, TOOLS, os.environ['FAKE_LOG'] + '.bridge'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                              env={**os.environ, 'HERMES_CURSOR_BRIDGE_PORT': port or '0'})
    if not port:
        raise SystemExit('fake: HERMES_CURSOR_BRIDGE_PORT missing')
    bridge.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {}}) + '\n')
    bridge.stdin.write(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/initialized'}) + '\n')
    bridge.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'}) + '\n')
    bridge.stdin.flush()
    return bridge


def call_tool(bridge, call_id, name, arguments):
    bridge.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': call_id, 'method': 'tools/call',
                                   'params': {'name': name, 'arguments': arguments}}) + '\n')
    bridge.stdin.flush()


def receive_results(bridge, count):
    results = []
    while len(results) < count:
        row = json.loads(bridge.stdout.readline())
        if row.get('id') in (1, 2):
            continue
        content = (row.get('result') or {}).get('content') or []
        text = ''.join(block.get('text', '') for block in content if isinstance(block, dict))
        results.append({'result': text.removeprefix('ERROR: '), 'is_error': text.startswith('ERROR: ')})
    return results


def log(stage, prompt):
    with open(os.environ['FAKE_LOG'], 'a', encoding='utf-8') as log_file:
        log_file.write(f"{stage} {os.getpid()} {prompt.splitlines()[0][:120] if prompt else ''}\n")


def result(final_text, usage=None):
    emit({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': final_text,
          'session_id': 'fake-sid', 'usage': usage or {'inputTokens': 10, 'outputTokens': 5,
                                                       'cacheReadTokens': 0, 'cacheWriteTokens': 0}})


def main():
    stage = os.environ.get('FAKE_STAGE', 'plain')
    prompt = sys.stdin.read()
    log(stage, prompt)
    emit({'type': 'system', 'subtype': 'init', 'session_id': 'fake-sid', 'model': 'Auto',
          'permissionMode': 'default'})

    if stage == 'plain':
        emit({'type': 'thinking', 'subtype': 'delta', 'text': 'hmm', 'session_id': 'fake-sid'})
        emit({'type': 'thinking', 'subtype': 'completed', 'session_id': 'fake-sid'})
        for word in ('Hello', 'Hello there'):
            assistant(word)
        assistant('Hello there', stamp=False)
        result('Hello there')

    elif stage == 'tool':
        assistant("I'll check.")
        bridge = start_bridge()
        tool_call('call-1', 'get_weather', {'city': 'Oslo'})
        call_tool(bridge, 'call-1', 'get_weather', {'city': 'Oslo'})
        results = receive_results(bridge, 1)
        tool_completed('call-1', 'get_weather', {'success': {'content': [{'text': {'text': results[0]['result']}}], 'isError': False}})
        for word in ('Oslo:', f"Oslo: {results[0]['result']}"):
            assistant(word)
        assistant(f"Oslo: {results[0]['result']}", stamp=False)
        result(f"Oslo: {results[0]['result']}", {'inputTokens': 100, 'outputTokens': 8, 'cacheReadTokens': 50, 'cacheWriteTokens': 0})

    elif stage == 'parallel':
        assistant('Two calls.')
        bridge = start_bridge()
        tool_call('call-a', 'get_weather', {'city': 'Paris'})
        tool_call('call-b', 'get_weather', {'city': 'Tokyo'})
        call_tool(bridge, 'call-a', 'get_weather', {'city': 'Paris'})
        call_tool(bridge, 'call-b', 'get_weather', {'city': 'Tokyo'})
        results = receive_results(bridge, 2)
        for item in results:
            tool_completed('call-a' if item['result'] == 'Sunny' else 'call-b', 'get_weather',
                           {'success': {'content': [{'text': {'text': item['result']}}], 'isError': False}})
        assistant('Done: ' + '; '.join(item['result'] for item in results))
        assistant('Done: ' + '; '.join(item['result'] for item in results), stamp=False)
        result('Done: ' + '; '.join(item['result'] for item in results))

    elif stage == 'resume_echo':
        # Second process of a cold-start-after-divergence test: normal finish.
        assistant('fresh')
        assistant('fresh', stamp=False)
        result('fresh')

    elif stage == 'denied':
        assistant('Trying.')
        tool_call('call-d', 'get_weather', {'city': 'Oslo'})
        tool_completed('call-d', 'get_weather', {'permissionDenied': {'error': 'MCP tool execution blocked: hermes-get_weather'}})
        assistant('Blocked; answering anyway.')
        assistant('Blocked; answering anyway.', stamp=False)
        result('Blocked; answering anyway.')

    elif stage == 'hang':
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        emit({'type': 'thinking', 'subtype': 'delta', 'text': 'working', 'session_id': 'fake-sid'})
        with open(os.environ['FAKE_LOG'] + '.pids', 'w', encoding='utf-8') as pid_file:
            pid_file.write(f'{os.getpid()} {child.pid}')
        time.sleep(60)
        result('never')

    elif stage == 'logged_out':
        print('Not logged in', file=sys.stderr)
        sys.exit(1)

    elif stage == 'models':
        print('auto - Auto (current, default)')
        print('gpt-5.6-sol-high - GPT-5.6 Sol 1M High')
        print('claude-opus-5-thinking-high - Claude Opus 5 1M Thinking\u200b')
        print('garbage line')
        print('gpt-5.6-sol-high - GPT-5.6 Sol 1M High (duplicate)')

    elif stage == 'status':
        print(json.dumps({'status': 'authenticated', 'isAuthenticated': True}))


if __name__ == '__main__':
    main()

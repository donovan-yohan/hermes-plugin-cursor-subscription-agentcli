"""Transport behavior of the agentcli client against the fake `agent` CLI."""
import json
import os
import time
from pathlib import Path

import pytest

from agentcli import CARRIER, Client, AgentCLIMissing, AgentCLILoggedOut


TOOLS = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Weather',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]


def create(client, **kwargs):
    return client.chat.completions.create(model='auto', messages=[
        {'role': 'system', 'content': 'You are Hermes.'},
        {'role': 'user', 'content': 'Weather in Oslo?'}], tools=TOOLS, **kwargs)


def read_log(path):
    rows = [line.split(' ', 2) for line in Path(path).read_text().splitlines()]
    return [(stage, int(pid), rest) for stage, pid, rest in rows]


def test_plain_text_turn(client, fake_env):
    response = create(client, stream=True)
    final = None
    deltas = []
    for chunk in response:
        if chunk.choices[0].delta.content:
            deltas.append(chunk.choices[0].delta.content)
        if hasattr(chunk, '_response'):
            final = chunk._response
    assert ''.join(deltas) == 'Hello there'
    assert final.choices[0].finish_reason == 'stop'
    assert final.choices[0].message.content == 'Hello there'
    assert final.usage.prompt_tokens == 10 and final.usage.completion_tokens == 5
    assert final.usage.native_usage['cacheReadTokens'] == 0
    assert final.choices[0].message.reasoning_details[0]['type'] == CARRIER
    assert read_log(fake_env / 'fake.log')[0][0] == 'plain'
    # Client stable cwd + private per-turn workspace, both cleaned up.
    assert Path(client._workdir()).is_dir()
    assert not list(Path(fake_env).glob('cursor-agentcli-*'))
    assert not client._turns


def test_tool_roundtrip_streams_boundary_and_final(client, fake_env, stage):
    stage('tool')
    boundary = None
    stream = create(client, stream=True)
    for chunk in stream:
        if chunk.choices[0].finish_reason == 'tool_calls':
            boundary = chunk
    assert boundary is not None
    response = next(c for c in [boundary] if hasattr(c, '_response'))._response
    calls = response.choices[0].message.tool_calls
    assert [(c.function.name, json.loads(c.function.arguments)) for c in calls] == [('get_weather', {'city': 'Oslo'})]
    assert calls[0].id
    # The live turn is parked on the client; the fake process is still alive.
    assert client._live is not None and client._live.p.poll() is None
    assert client._live.session_id == 'fake-sid'

    second = client.chat.completions.create(model='auto', messages=[
        {'role': 'system', 'content': 'You are Hermes.'},
        {'role': 'user', 'content': 'Weather in Oslo?'},
        {'role': 'assistant', 'content': "I'll check.", 'tool_calls': [
            {'id': calls[0].id, 'type': 'function',
             'function': {'name': 'get_weather', 'arguments': '{"city": "Oslo"}'}}]},
        {'role': 'tool', 'tool_call_id': calls[0].id, 'content': 'Rainy, 9C'}], tools=TOOLS)
    assert second.choices[0].finish_reason == 'stop'
    assert second.choices[0].message.content == 'Oslo: Rainy, 9C'
    assert second.usage.prompt_tokens == 100 + 50
    # One process total: the second request resumed the live chat instead of respawning.
    log = read_log(fake_env / 'fake.log')
    assert [row[0] for row in log] == ['tool']
    assert client._live is None and not client._turns


def test_parallel_tool_calls_deliver_as_one_batch(client, fake_env, stage):
    stage('parallel')
    stream = create(client, stream=True)
    response = None
    for chunk in stream:
        if hasattr(chunk, '_response'):
            response = chunk._response
    calls = response.choices[0].message.tool_calls
    assert [c.function.name for c in calls] == ['get_weather', 'get_weather']
    ids = [c.id for c in calls]
    assert ids[0] != ids[1]
    second = client.chat.completions.create(model='auto', messages=[
        {'role': 'user', 'content': 'Weather?'},
        {'role': 'assistant', 'content': 'Two calls.', 'tool_calls': [
            {'id': ids[0], 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}},
            {'id': ids[1], 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Tokyo"}'}}]},
        {'role': 'tool', 'tool_call_id': ids[0], 'content': 'Sunny'},
        {'role': 'tool', 'tool_call_id': ids[1], 'content': 'Rain'}], tools=TOOLS)
    assert second.choices[0].message.content == 'Done: Sunny; Rain'
    assert [row[0] for row in read_log(fake_env / 'fake.log')] == ['parallel']


def test_diverged_history_abandons_live_turn_and_cold_starts(client, fake_env, stage):
    stage('tool')
    stream = create(client, stream=True)
    for chunk in stream:
        pass
    assert client._live is not None
    old_pid = client._live.p.pid
    # History that does not match the parked calls: live turn is torn down; the next request
    # cold-starts a fresh process (resume_echo stage) and answers normally.
    stage('resume_echo')
    response = client.chat.completions.create(model='auto', messages=[
        {'role': 'user', 'content': 'Different conversation entirely.'}], tools=TOOLS)
    assert response.choices[0].message.content == 'fresh'
    stages = [row[0] for row in read_log(fake_env / 'fake.log')]
    assert stages == ['tool', 'resume_echo']
    assert client._live is None and not client._turns


def test_permission_denied_inline_continues_turn(client, fake_env, stage):
    # A locally denied MCP call never reaches the bridge: the model sees the denial in the same
    # run and finishes; Hermes receives the final answer, not a tool boundary.
    stage('denied')
    response = create(client)
    assert response.choices[0].finish_reason == 'stop'
    assert response.choices[0].message.content == 'Blocked; answering anyway.'
    assert client._live is None and not client._turns


def test_cancel_kills_process_tree(client, fake_env, stage):
    stage('hang')
    hanging = Client(timeout=30)
    try:
        stream = hanging.chat.completions.create(model='auto', messages=[
            {'role': 'user', 'content': 'hang'}], tools=TOOLS, stream=True)
        first = next(iter(stream))
        assert first.choices[0].delta.reasoning_content == 'working'
        pids = Path(os.environ['FAKE_LOG'] + '.pids').read_text().split()
        parent, child = int(pids[0]), int(pids[1])
        deadline = time.time() + 5
        while not Path(os.environ['FAKE_LOG'] + '.pids').exists() and time.time() < deadline:
            time.sleep(0.05)
        hanging.cancel()
        time.sleep(1.0)
        assert Path(f'/proc/{parent}') .exists() is False if os.path.isdir('/proc') else True
        if os.path.isdir('/proc'):
            assert not Path(f'/proc/{child}').exists()
        assert not hanging._turns
        stream.close()
    finally:
        hanging.close()


def test_logged_out_maps_to_hint(client, fake_env, stage):
    stage('logged_out')
    with pytest.raises((AgentCLILoggedOut, RuntimeError)) as excinfo:
        create(client)
    assert 'agent login' in str(excinfo.value) or 'without a result' in str(excinfo.value)


def test_missing_binary_raises_install_hint(monkeypatch):
    monkeypatch.setenv('CURSOR_AGENTCLI_COMMAND', str(Path(os.devnull)))
    client = Client(timeout=30)
    try:
        with pytest.raises((AgentCLIMissing, RuntimeError)):
            create(client)
    finally:
        client.close()


def test_client_close_finalizes_parked_turn(client, fake_env, stage):
    stage('tool')
    stream = create(client, stream=True)
    for chunk in stream:
        if chunk.choices[0].finish_reason == 'tool_calls':
            break
    assert client._live is not None
    pid = client._live.p.pid
    client.close()
    assert client._live is None and not client._turns
    import signal
    import subprocess as sp
    with (None or open(os.devnull)) as _:
        pass
    assert sp.run(['ps', '-p', str(pid)], capture_output=True).returncode != 0


def test_async_stream_roundtrip(client, fake_env):
    import asyncio

    async def run():
        stream = await create(client, stream=True)
        text = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                text.append(chunk.choices[0].delta.content)
        await stream.aclose()
        return ''.join(text)

    assert asyncio.run(run()) == 'Hello there'

"""Pure history/parameter translation plus installed-plugin discovery through Hermes core."""
import json
from types import SimpleNamespace
import os
import shutil
import sys
from pathlib import Path

import pytest

from agentcli import (CARRIER, continuation, normalize_input_schema, projection, render_prompt,
                      request_body, resume_prompt, bridge_key)
from agentcli_setup import INSTALL_HINT, parse_models, _resolve


def test_render_prompt_roundtrip_shapes():
    messages = [
        {'role': 'system', 'content': 'SYS'},
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'doing it', 'tool_calls': [
            {'id': 'c1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{"city": "Oslo"}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'Rainy'},
        {'role': 'user', 'content': 'thanks'},
    ]
    text = render_prompt(messages)
    assert '[System]\nSYS' in text and '[User]\nhi' in text and '[User]\nthanks' in text
    assert '[Assistant tool call] get_weather({"city": "Oslo"}) [id: c1]' in text
    assert '[Tool result for c1]\nRainy' in text


def test_render_prompt_images_degrade_to_placeholders():
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'look'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAA'}}]}]
    text = render_prompt(messages)
    assert '[image: image/png; not rendered to this transport]' in text and 'look' in text


def test_request_body_normalizes_schemas_and_names():
    tools = [{'type': 'function', 'function': {'name': 'ok_name', 'description': 'd',
              'parameters': {'type': 'object', 'oneOf': [], 'properties': {'a': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}}}}},
             {'type': 'function', 'function': {'name': 'no_props', 'description': 'd', 'parameters': {'type': 'object'}}}]
    manifest, names = request_body({'tools': tools})
    assert set(names) == {'ok_name', 'no_props'}
    assert manifest[0]['inputSchema'].get('oneOf') is None
    assert manifest[1]['inputSchema']['properties'] == {}


def test_request_body_rejects_unknown_and_strict():
    with pytest.raises(ValueError, match='Unsupported request parameters'):
        request_body({'temperature': 0.5, 'tools': []})
    with pytest.raises(ValueError, match='Strict'):
        request_body({'tools': [{'type': 'function', 'function': {'name': 'x', 'description': 'd',
                     'parameters': {'type': 'object'}, 'strict': True}}]})
    with pytest.raises(ValueError, match='unique'):
        request_body({'tools': [{'type': 'function', 'function': {'name': 'x', 'description': 'd', 'parameters': {}}},
                                {'type': 'function', 'function': {'name': 'x', 'description': 'd', 'parameters': {}}}]})


def test_normalize_input_schema_without_hermes_core():
    schema = {'type': 'object', 'anyOf': [{'type': 'object'}], 'properties': {'x': {'type': ['string', 'null']}}}
    out = normalize_input_schema(schema)
    assert 'anyOf' not in out and 'properties' in out


def test_bridge_key_stable_and_argument_sensitive():
    assert bridge_key('get_weather', {'city': 'Oslo'}) == bridge_key('get_weather', {'city': 'Oslo'})
    assert bridge_key('get_weather', {'city': 'Oslo'}) != bridge_key('get_weather', {'city': 'Paris'})


def test_continuation_matches_parked_calls():
    turn = SimpleNamespace(parked={'call-1': {'name': 'get_weather', 'arguments': {'city': 'Oslo'}, 'key': 'k1'},
                                   'call-2': {'name': 'get_weather', 'arguments': {'city': 'Paris'}, 'key': 'k2'}},
                           completed={}, names={'get_weather'}, parked_keys=set())
    calls = [{'id': 'call-1', 'function': {'name': 'get_weather', 'arguments': '{"city": "Oslo"}'}},
             {'id': 'call-2', 'function': {'name': 'get_weather', 'arguments': '{"city": "Paris"}'}}]
    messages = [{'role': 'assistant', 'content': '', 'tool_calls': calls},
                {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'Sunny'},
                {'role': 'tool', 'tool_call_id': 'call-2', 'content': 'Rainy'},
                {'role': 'user', 'content': 'also check Lima'}]
    results, steering = continuation(messages, turn)
    assert [r['name'] for r in results] == ['get_weather', 'get_weather']
    assert steering == 'also check Lima'
    prompt = resume_prompt(results, steering)
    assert 'get_weather -> Sunny' in prompt and 'get_weather -> ERROR: ' not in prompt.split('Lima')[0]
    assert '[User steering] also check Lima' in prompt


def test_continuation_rejects_mismatch():
    turn = SimpleNamespace(parked={'call-1': {'name': 'get_weather', 'arguments': {}, 'key': 'k'}},
                           completed={}, names={'get_weather'}, parked_keys=set())
    assert continuation([{'role': 'user', 'content': 'new topic'}], turn) is None
    assert continuation([{'role': 'assistant', 'content': 'x', 'tool_calls': [
        {'id': 'other', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'other', 'content': 'r'}], turn) is None
    turn_completed = SimpleNamespace(parked=turn.parked,
                                     completed={'call-1': {'result': 'r', 'is_error': False}},
                                     names={'get_weather'}, parked_keys=set())
    assert continuation([{'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'call-1', 'type': 'function', 'function': {'name': 'get_weather', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'r'}], turn_completed) is None


def test_parse_models_and_resolve():
    rows = parse_models('auto - Auto (current, default)\n'
                        'gpt-5.6-sol-high - GPT-5.6 Sol 1M High\n'
                        'claude-opus-5-thinking-high - Claude Opus 5 1M Thinking\u200b\n'
                        'garbage\n'
                        'gpt-5.6-sol-high - GPT-5.6 Sol 1M High\n')
    assert [r['id'] for r in rows] == ['auto', 'gpt-5.6-sol-high', 'claude-opus-5-thinking-high']
    assert rows[1]['label'] == 'GPT-5.6 Sol 1M High'


def test_resolve_prefers_override_and_falls_back(monkeypatch, tmp_path):
    shim = tmp_path / 'agent-shim'
    shim.write_text('#!/bin/sh\nexit 0\n')
    shim.chmod(shim.stat().st_mode | 0o111)
    monkeypatch.setenv('CURSOR_AGENTCLI_COMMAND', str(shim))
    assert _resolve(None, os.environ)[0] == str(shim)
    monkeypatch.delenv('CURSOR_AGENTCLI_COMMAND')
    resolved = _resolve(None, {})  # PATH-dependent: may find the real CLI or nothing; never raises
    assert resolved is None or Path(resolved[0]).exists()


def test_provider_registers_through_hermes_core(tmp_path, monkeypatch):
    import providers

    home = tmp_path / 'hermes-home'
    installed = home / 'plugins' / 'cursor-subscription-agentcli-experimental'
    shutil.copytree(Path(__file__).resolve().parents[1], installed,
                    ignore=shutil.ignore_patterns('.git', 'tests', 'evals', '__pycache__', '*.log'))
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('plugins:\n  enabled: []\n', encoding='utf-8')
    for name in tuple(sys.modules):
        if name.startswith('_hermes_user_provider_'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(providers, '_REGISTRY', {})
    monkeypatch.setattr(providers, '_ALIASES', {})
    monkeypatch.setattr(providers, '_PROVIDER_LIST_CACHE', None)
    monkeypatch.setattr(providers, '_discovered', False)
    providers._discover_providers()
    profile = providers.get_provider_profile('cursor-subscription-agentcli-experimental')
    assert profile is not None
    assert profile.auth_type == 'external_process'
    assert profile.default_aux_model == 'auto'
    assert 'auto' in profile.fallback_models
    assert profile.setup_status()['available'] is True
    rows = profile.discover_models()
    assert rows and rows[0]['id'] == 'auto'
    # The client factory returns the transport, not an HTTP client. Dual-import layout means
    # the installed copy's Client class is distinct from the flat-imported one: assert the shape.
    monkeypatch.setenv('CURSOR_AGENTCLI_COMMAND', 'definitely-missing-agent-cli')
    client = profile.create_client(timeout=5)
    assert type(client).__name__ == 'Client'
    assert client.chat.completions.create.__func__ is not None
    assert client.base_url.startswith('process://')
    client.close()


def test_setup_status_install_hint(monkeypatch, tmp_path):
    from agentcli_setup import setup_status
    monkeypatch.setenv('CURSOR_AGENTCLI_COMMAND', str(tmp_path / 'no-such-agent'))
    status = setup_status()
    assert status['available'] is False and INSTALL_HINT in status['detail']

"""Shared fixtures: a fake `agent` binary plus the plugin loaded exactly as installed."""
import json
import os
import shutil
import stat
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agentcli import (AgentCLILoggedOut, AgentCLIMissing, Client, render_prompt, resume_prompt,  # noqa: E402
                      bridge_key, continuation)

FAKE = str(REPO_ROOT / 'tests' / 'fake_agent.py')


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Point CURSOR_AGENTCLI_COMMAND at a python3 shim running the fake agent."""
    shim = tmp_path / 'bin'
    shim.mkdir()
    binary = shim / 'fake-agent'
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('CURSOR_AGENTCLI_COMMAND', str(binary))
    monkeypatch.setenv('FAKE_LOG', str(tmp_path / 'fake.log'))
    monkeypatch.setenv('PYTHONPATH', str(REPO_ROOT))
    monkeypatch.delenv('HERMES_CURSOR_BRIDGE_PORT', raising=False)
    monkeypatch.delenv('FAKE_STAGE', raising=False)
    return tmp_path


@pytest.fixture
def stage(monkeypatch):
    def set_stage(name):
        monkeypatch.setenv('FAKE_STAGE', name)
    return set_stage


@pytest.fixture
def client(fake_env):
    client = Client(timeout=30)
    yield client
    client.close()


def read_log(path):
    rows = [line.split(' ', 2) for line in Path(path).read_text().splitlines()]
    return [(stage, int(pid), rest) for stage, pid, rest in rows]


def chunks(stream):
    return list(stream)


def messages_for(calls, results, content='I will check.'):
    """Hermes-shaped turn: assistant tool_calls + tool rows (+ optional steering)."""
    tool_calls = [{'id': call['id'], 'type': 'function', 'function': dict(call['function'])} for call in calls]
    out = [{'role': 'assistant', 'content': content, 'tool_calls': tool_calls}]
    for call, result in zip(calls, results):
        out.append({'role': 'tool', 'tool_call_id': call['id'], 'content': result})
    return out

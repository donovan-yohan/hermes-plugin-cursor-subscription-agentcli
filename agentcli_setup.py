"""Setup-time probes of the Cursor agent CLI: login state and the account's model list. No inference."""
import json
import os
import re
import shutil
import subprocess

INSTALL_HINT = ("Cursor Agent CLI is not installed (no `agent` on PATH). Install it with "
                "`curl https://cursor.com/install -fsS | bash` or set CURSOR_AGENTCLI_COMMAND to the binary.")
LOGIN_HINT = "Cursor Agent CLI is installed but not logged in. Run `agent login`, then select this provider again."
COMMAND_ENV = 'CURSOR_AGENTCLI_COMMAND'


def _resolve(command, env):
    """``[exe, *args]`` for the agent CLI, or ``None`` when it is not installed."""
    path = env.get('PATH') or os.defpath
    if command:
        command = [command] if isinstance(command, str) else list(command)
        heads = [command[0]]
    else:
        override = env.get(COMMAND_ENV)
        command = [override or 'agent']
        heads = [override] if override else ['agent', 'cursor-agent']
    for head in heads:
        exe = head if os.path.isabs(head) and os.access(head, os.X_OK) else shutil.which(head, path=path)
        if exe:
            return [exe] + command[1:]
    return None


def setup_status(command=None, env=None, timeout=20):
    """``{available, logged_in, plan, detail, login_command}`` from ``agent status --format json``."""
    env = dict(env if env is not None else os.environ)
    resolved = _resolve(command, env)
    if resolved is None:
        return {'available': False, 'logged_in': False, 'plan': '', 'detail': INSTALL_HINT, 'login_command': None}
    try:
        run = subprocess.run(resolved + ['status', '--format', 'json'], env=env, stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
        status = json.loads(run.stdout) if run.stdout.strip().startswith('{') else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        status = {}
    logged_in = status.get('isAuthenticated') is True
    return {'available': True, 'logged_in': logged_in, 'plan': 'Cursor' if logged_in else '',
            'detail': '' if logged_in else LOGIN_HINT, 'login_command': resolved + ['login']}


_MODEL_ROW = re.compile(r'^([A-Za-z0-9][\w.\-\[\]=,]*) - (.+?)(?:\s+\((?:current|default)[^)]*\))*$')


def parse_models(text):
    rows = []
    for line in text.splitlines():
        line = re.sub(r'[\u200b-\u200d\ufeff]', '', line).strip()
        match = _MODEL_ROW.match(line)
        if match and match.group(1) not in {r['id'] for r in rows}:
            rows.append({'id': match.group(1), 'label': match.group(2).strip(), 'note': ''})
    rows.sort(key=lambda r: r['id'] != 'auto')
    return rows


def discover_models(command=None, env=None, timeout=40):
    """The account's model list from ``agent models`` as ``[{id, label, note}]``, or ``None``."""
    env = dict(env if env is not None else os.environ)
    resolved = _resolve(command, env)
    if resolved is None:
        return None
    try:
        run = subprocess.run(resolved + ['models'], env=env, stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_models(run.stdout) or None if run.returncode == 0 else None

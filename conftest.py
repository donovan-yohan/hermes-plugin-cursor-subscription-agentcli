"""Test bootstrap and shared fixture.

The standalone plugin imports Hermes core (`providers`, `agent`) from a checkout: set
``HERMES_AGENT_REPO`` to your hermes-agent clone (default ``~/.hermes/hermes-agent``).
"""
import os
import sys
from pathlib import Path

HERMES_AGENT_REPO = Path(os.environ.get('HERMES_AGENT_REPO') or Path.home() / '.hermes' / 'hermes-agent').expanduser()
REPO_ROOT = Path(__file__).resolve().parent
for path in (str(HERMES_AGENT_REPO), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

collect_ignore = ['__init__.py', 'agentcli.py', 'agentcli_setup.py', 'bridge_mcp.py', 'model_catalog.py', 'evals']

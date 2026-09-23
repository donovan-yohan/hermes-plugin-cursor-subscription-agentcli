"""Cursor Subscription AgentCLI (Experimental) — standalone Hermes model-provider registration."""
import logging

from providers import register_provider
from providers.base import ProviderProfile

# Dual import: the Hermes loader imports this directory as a package; the flat test path does not.
try:
    from .agentcli_setup import INSTALL_HINT, _resolve
    from .model_catalog import FALLBACK_MODELS, MODEL_ALIASES
except ImportError:
    from agentcli_setup import INSTALL_HINT, _resolve
    from model_catalog import FALLBACK_MODELS, MODEL_ALIASES

logger = logging.getLogger(__name__)


class CursorAgentCLIProfile(ProviderProfile):
    model_metadata = MODEL_ALIASES

    def get_model_context_length(self, model):
        # Cursor routes models itself and applies its own windows; no id is pinned here.
        return None

    def get_usage_cost(self, model, usage):
        from agent.usage_pricing import CostResult
        return CostResult(amount_usd=None, status='unknown', source='none', label='n/a',
                          notes=('Cursor subscription metering is not exposed by the CLI',))

    def create_client(self, **client_kwargs):
        try:
            from .agentcli import Client
        except ImportError:
            from agentcli import Client
        return Client(**client_kwargs)

    def fetch_models(self, **kwargs):
        rows = self.discover_models(**kwargs)
        return [row['id'] for row in rows] if rows else None

    def setup_status(self, **kwargs):
        try:
            from .agentcli_setup import setup_status
        except ImportError:
            from agentcli_setup import setup_status
        return setup_status(**kwargs)

    def discover_models(self, **kwargs):
        try:
            from .agentcli_setup import discover_models
        except ImportError:
            from agentcli_setup import discover_models
        return discover_models(**kwargs)

    def build_api_kwargs_extras(self, *, reasoning_config=None, **_):
        # Reasoning effort is chosen by Cursor's auto router; nothing to forward.
        return {}, {}


profile = CursorAgentCLIProfile(
    name='cursor-subscription-agentcli-experimental',
    display_name='Cursor Subscription AgentCLI (Experimental)',
    description='Cursor Subscription AgentCLI (Experimental) (your Cursor subscription via the official `agent` CLI; Hermes owns tools)',
    api_mode='chat_completions',
    auth_type='external_process',
    supports_health_check=False,
    env_vars=(),
    base_url='process://cursor-subscription-agentcli-experimental',
    process_command='agent',
    process_args=(),
    process_command_env_vars=('CURSOR_AGENTCLI_COMMAND',),
    default_aux_model='auto',
    fallback_models=FALLBACK_MODELS,
    model_aliases=dict(MODEL_ALIASES),
)
register_provider(profile)

# The provider stays registered when the CLI is missing so `hermes model` can show the
# install hint; the request path (`agentcli.Client`) refuses with the same message.
if _resolve(None, {}) is None:
    logger.warning("%s: %s", profile.display_name, INSTALL_HINT)

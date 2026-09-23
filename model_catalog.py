"""Model metadata for the Cursor subscription provider.

`auto` is Cursor's own router and the only id this plugin pins. Everything else comes live from
`agent models` at setup time; an unpinned id passes through unchanged and carries no context claim.
"""

MODEL_ALIASES = {'auto': {'context_window': None}}

FALLBACK_MODELS = (
    'auto',
    'composer-2.5',
    'gpt-5.6-sol-high',
    'gpt-5.3-codex',
    'claude-sonnet-5-thinking-high',
    'claude-opus-5-thinking-high',
    'gemini-3.7-flash-high',
)

# Cursor Subscription AgentCLI (Experimental), Hermes plugin scope

Status: scoping only. No provider code yet.

The goal is a Hermes model-provider plugin, like
[`hermes-plugin-claude-subscription-directsdk`](https://github.com/NousResearch/hermes-plugin-claude-subscription-directsdk),
that runs requests on a Cursor subscription through the official Cursor `agent` CLI. Hermes would keep its own agent loop, tools, approvals and compaction.

Verdict: feasible, but the Claude plugin's transport design does not carry over. The ownership model does. The provider needs a different request boundary, described below.

## What the Claude plugin does

Each `chat.completions.create` call spawns `claude -p` once:

1. It replays Hermes history as stream-json frames with `shouldQuery:false`.
2. It advertises Hermes tools through an inert MCP server that refuses every call.
3. It runs with `--max-turns 1`, `--tools ''` and `--permission-mode dontAsk`.
4. It points `ANTHROPIC_BASE_URL` at a loopback relay that forwards only the first Messages request and captures the raw Anthropic response, including signed thinking and `tool_use` blocks.
5. It returns those `tool_use` blocks to Hermes as OpenAI-style `tool_calls`. Hermes executes them, and the next `create` replays everything.

That design depends on four Claude Code features.

| Claude Code feature | Cursor `agent` equivalent |
| --- | --- |
| stream-json input with per-frame `shouldQuery:false` history replay | None. Input is a single prompt (argv or stdin). Continuity exists only through `--resume <chatId>` on Cursor's server-side chat. |
| `--max-turns 1` (stop at the tool boundary) | None. The CLI loops until the model answers. |
| `ANTHROPIC_BASE_URL` relay (capture the raw model response) | `--endpoint` exists, but it points at Cursor's private backend protocol, not a model API. Relaying it is not viable. |
| `--system-prompt-file` | None. `.cursor/rules/*.mdc` in the workspace, or a prompt preamble, are the candidates. |

## Proposed design: a parking MCP bridge

Let the Cursor agent call the Hermes tools, then hold the call inside the MCP server until Hermes answers it.

1. `create(messages, tools)` with no pending session: write the tool manifest and `.cursor/mcp.json` into a private workspace, and start `agent -p --output-format stream-json --stream-partial-output --trust --approve-mcps --workspace <dir> --model <m>`. Send the rendered prompt on stdin.
2. Stream `assistant` text deltas to Hermes as content chunks.
3. When the agent calls `hermes-<tool>`, the bridge parks the JSON-RPC `tools/call` and reports it to the plugin over a local socket. The plugin ends the response with `finish_reason: tool_calls` and keeps the process alive.
4. Hermes runs its own hooks, approvals and the tool. The next `create` ends in `tool` messages whose `tool_call_id`s match the parked calls. The plugin releases them to the bridge as MCP results, and the same agent process continues.
5. The final `result` event ends the turn with `usage` (`inputTokens`, `outputTokens`, `cacheReadTokens`, `cacheWriteTokens`).

When no parked session matches, because of a restart, compaction, an edited history or a new conversation, the plugin cold-starts. It renders the full Hermes transcript as text. `--resume` does not help here, because Hermes owns history and may rewrite it.

Cursor's built-in tools stay registered but are denied through workspace `.cursor/cli.json` permissions. Per-workspace permissions are the only denial mechanism. I have not verified they cover every built-in tool yet.

## Spike results

Run on 2026-09-23 with Cursor CLI `2026.09.18-9a7762b`, a Team subscription and `gemini-3.7-flash-high`. The probes are in `spikes/`.

| Probe | Result |
| --- | --- |
| Workspace `.cursor/mcp.json` stdio server, `--approve-mcps --trust` | Loaded. The agent listed and called `get_weather`, and the MCP log shows `initialize`, `tools/list`, `tools/call` with the right arguments. |
| stream-json events | `system/init`, `user`, `tool_call started/completed` (with `mcpToolCall.args` before execution), `assistant`, and `result/success` with token usage and `session_id`. |
| `.cursor/cli.json` deny `Shell(*)`, `Read(**)`, `Write(**)` | Both the built-in read and the shell call returned `Permission denied`. The agent reported the denials and did not work around them. |
| Parked `tools/call` held for about 90 s, then answered | The agent waited, used the late answer ("Rainy, 9C") and exited 0. The idle time did not count as API time. This is the probe the whole design depends on. |
| 120 MCP tools | Cursor discovered them through a namespace lookup and called `tool_117` correctly. The old ~40-tool cap did not apply. |
| Prompt on stdin | Works. That avoids the argv limit for large Hermes transcripts. |

## Known gaps and risks

- The model sees Cursor's own system prompt and tool framing, plus Hermes' prompt. Hermes cannot own the system prompt exactly the way the Claude plugin does.
- There is no signed-thinking replay and no provider-native carrier. Cold starts lose native reasoning state and pay a full uncached prompt.
- There is no per-request admission boundary. A cancelled or timed-out turn can still have spent Cursor usage. Cancellation must kill the process tree, as the Claude plugin does.
- Parallel tool calls: Cursor may send several `tools/call` requests at once. The plugin needs a short settle window to collect a batch before returning to Hermes. This is not measured yet.
- Parked processes use memory and must time out. Hermes subagents and compaction must never reuse another owner's parked session.
- Usage is reported per agent turn, not per model call. Mapping it onto Hermes' per-call accounting is approximate. Cursor does not report cost.
- `stream-json` is a CLI output format, not a stable SDK contract. Pin and qualify CLI versions the way the Claude plugin pins Claude Code versions.
- Cursor's terms of service for driving the CLI from another agent harness have not been reviewed.

## Next steps

1. Measure how Cursor sends parallel tool calls and pick a batch-settle rule.
2. Check whether `.cursor/rules` can carry the Hermes system prompt, and whether deny rules cover every built-in tool (web search, edit, grep).
3. Build `cursor_agent.py` (Client with `chat.completions.create`), `bridge_mcp.py` (parking server) and `__init__.py` (ProviderProfile, `auth_type='external_process'`, discovery from `agent models`, setup from `agent status`).
4. Port the Claude plugin's lifecycle tests: cancel, close, async streams, missing CLI, logged out.

## Spikes

- `spikes/mcp_log.py <tools.json> <log>` is a stdio MCP server that logs every JSON-RPC frame and answers `tools/call` with a fixed result.
- `spikes/mcp_park.py <tools.json> <dir>` writes `call.json` on `tools/call` and blocks until `<dir>/go` exists, then returns its text.

Example:

```sh
mkdir -p /tmp/cs/.cursor && cd /tmp/cs
cp path/to/spikes/* .
echo '{"mcpServers":{"hermes":{"command":"python3","args":["'$PWD'/mcp_park.py","'$PWD'/tools.json","'$PWD'"]}}}' > .cursor/mcp.json
echo '{"permissions":{"allow":["Mcp(hermes:*)"],"deny":["Shell(*)","Read(**)","Write(**)"]}}' > .cursor/cli.json
echo "Use get_weather for Oslo, then answer in one sentence." | \
  agent -p --output-format stream-json --trust --approve-mcps --workspace "$PWD" &
# wait for call.json, then:
echo "Rainy, 9C" > go
```

MIT licensed.

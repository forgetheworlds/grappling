# Harness availability probe — 2026-10-07

Machine: 4-core ARM aarch64 Linux (oracle), network OK. Probe-only: no project work was run
through any harness. The only end-to-end run performed was the single authorized free-model
ping on opencode (it failed; exact error below). No paid/quota/credit usage.

## Driver

| item | result |
|---|---|
| `delegate` driver | **missing** |
| exact path | n/a — `~/bin/delegate` does not exist; `type -a delegate` → not found; not in `~/.local/bin`, `~/.npm-global/bin`, `/usr/local/bin` |
| `delegate --help` / `delegate probe` | cannot run (binary absent) |
| skill doc | `~/.agents/skills/harness-delegation/SKILL.md` exists and describes the driver (sessions, `probe`, `--harness auto`), but the driver itself is not installed. `~/bin` contains only `ultracode*` symlinks → UltraCode-Shim. |

Since the driver is missing, per-harness equivalents (binary presence + version + local, zero-cost
commands) were used instead of `delegate probe`.

## Harness table

| harness | present | probe result | cost model | headless OK | recommendation |
|---|---|---|---|---|---|
| **opencode** | yes — `/home/ubuntu/.opencode/bin/opencode`, v1.3.3 | `opencode models` OK (88 entries). Free set listed: `opencode/mimo-v2.6-flash-free`, `opencode/space-bunny-free`, `exo-free`, `fledge-alpha-free`, `ling-3.0/3.1-flash-free`, `longcat-2.5-preview-free`, `muse-spark-1.3-contributor-free`, `nemotron-3-ultra-free`, `nemotron-3.5-lightning-free`. **`opencode-go/*`: NOT LISTED** (zero entries — no IDs invented). **Connectivity ping FAILED** (below). | free tier gated; fireworks via API key (paid); opencode-go paid (but not listed) | yes, mechanically (`opencode run`) | **NOT usable now.** Free tier blocked (no console auth); opencode-go absent; fireworks excluded. Restore via `opencode auth login` if free tier is wanted. |
| **cursor** | no — `cursor`/`cursor-agent` NOT on PATH; only stale `~/.cursor/projects/...` MCP metadata from past IDE use, no CLI | n/a | Cursor Pro/paid | **not viable** — no binary | skip; install `cursor-agent` only if ever needed |
| **freebuff** | yes — `~/.npm-global/bin/freebuff` → `lib/node_modules/freebuff/index.js` | presence check only (not run — spends credits) | Freebucks credits, metered + hourly window | no native headless (skill doc: TUI via tmux) | do not use (credit cost + TUI) |
| **codex** | yes — `/home/ubuntu/.npm-global/bin/codex`, `codex-cli 0.147.0` | `--version` only (not run — spends quota) | Codex plan quota (5h/weekly) | yes (`codex exec`) | opt-in only; excluded by default here |
| **claude** | yes — `~/.npm-global/bin/claude` (also `~/.local/bin/claude`) | presence only (not run) | Anthropic plan credits | yes (`claude -p`) | not wired; opt-in only |
| **gemini** | yes — `~/.nvm/versions/node/v24.14.0/bin/gemini` | presence only (not run) | Google account quota | yes | not wired; opt-in only |

## opencode connectivity test (the one authorized run)

Command (120s timeout, wall time measured):

```
opencode run -m opencode/mimo-v2.6-flash-free "Reply with exactly: HARNESS_OK"
```

Result: **failed** — no `HARNESS_OK` returned. Exact output (exit code 0, wall 3.6s):

```
> build · mimo-v2.6-flash-free

Error: Error from provider (Console): OpenCode's free tier can only be used from within OpenCode
```

Supporting state (`opencode auth list`, zero-cost): `~/.local/share/opencode/auth.json` holds
**0 credentials**; only `FIREWORKS_API_KEY` is present as an environment variable. [INFERENCE]
The free tier is refused because this machine has no OpenCode console login; the fireworks key
works (it is what lists `fireworks-ai/*` models) but those models are excluded by the operator.

## Operator constraints recorded

- `fireworks-ai/*` (incl. `deepseek-v4p1-flash`, `glm-5p3-flash`, routers like
  `deepseek-flash-latest`): **EXCLUDED-BY-OPERATOR** — listed as available but never run.
- Intended stack `opencode-go/*` (deepseek-flash / mimo-flash variants): **NOT LISTED** in
  `opencode models`; no ping was attempted against invented IDs.

## Bottom line

The `delegate` driver is absent, and **zero harnesses are currently verified usable for free
delegation**: opencode's free tier refuses unauthenticated console use (exact error above),
`opencode-go/*` does not exist in the model list, fireworks is operator-excluded, cursor's CLI
binary is missing, and freebuff/codex/claude/gemini all cost credits/quota. Immediate remedies,
in order: (1) `opencode auth login` to restore the free tier, or (2) install the missing
`~/bin/delegate` driver before any harness delegation is attempted.

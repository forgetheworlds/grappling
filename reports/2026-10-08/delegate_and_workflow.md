# Delegate validation + workflow fix — 2026-10-08 (DelegateTester)

Model running this agent: `opencode-go/glm-5.3-flash` (from system prompt).
Scope: (1) validate omp `delegate` tool, (2) validate/fix opencode headless + freebuff,
(3) diagnose/fix xd://workflow. Driver = NEW file `/home/ubuntu/bin/delegate`
(Python stdlib, exe). No other project files changed (report + notes.md only).

## Stage 1 — omp `delegate` tool

### Before fix
- omp tool spawns the missing driver: `posix_spawn '/home/ubuntu/bin/delegate'` →
  `ENOENT` instantly for harness=opencode AND auto (0.0s). It has **no own auth path** —
  it is a thin `execFile` wrapper over `~/bin/deligate`-equivalent driver
  (`~/.omp/agent/extensions/delegate.ts` builds args: `run [task|--brief F]
  [--session N] [--cwd D] [--harness H] [--model M] [--timeout SEC]`,
  `ask <answer> --session N`, `probe|list|last|template`).
- Fix: implemented the driver at `/home/ubuntu/bin/delegate` per that exact contract.
  Both harness engines implemented (opencode headless CLI; freebuff TUI-in-tmux with
  structured `log.jsonl` verdicts). `auto` = failover probe opencode → cursor
  (cursor not installed); freebuff NEVER auto-selected (skill rule).

### opencode headless (certified)
- `opencode run` WORKS headless on this box as of today (prior 2026-10-07 fact
  "free tier can only be used from within OpenCode" NO LONGER reproduces; no
  credentials in auth.json — free tier is usable anonymous now (binary 1.18.35)).
- Model: `opencode/mimo-v2.6-flash-free` (this is the real v2.6 flash; $0).
- Trivial ping: HEALTHY ~12-15 s/turn (DIRECT calls).
- Multi-turn: `-s <session_id>` works; driver stores the id from `--format json`
  events (`sessionID`); verified_ask returned the remembered literal text.

### freebuff TUI (certified)
- Driven by driver inside tmux (`fb-<session>`); submit via `tmux load-buffer - /
  paste-buffer` + `Enter` (NOT paste via send-keys `'\n'` semantically), with
  submission verified ("Your message is saved" HUD / "messages queued" / busy
  thread view / composer placeholder return); verdict read from
  `~/.config/manicode/projects/<proj>/chats/<iso>/log.jsonl` — last
  `End agent ... shouldEndTurn:true` record (`fullResponse`, `model`, `stepCreditsUsed`),
  completion = ≥1 new log line after send + record line-index AFTER our prompt's
  line + composer not holding draft + quiet ≥9 s.
- Two trivial runs returned verbatim replies (`FREEBUFF_OK`, `TOOL_FB_OK`, …); credits
  burn ~10 Freebucks/hr per turn (metered hourly "NNm left"; balance read from TUI).
- **Freebuff model catalog FACT** (verified in binary 0.2.22 + picker UI + run log):
  Picker offers exactly: Solar Pro 4 (0 FB/hr), Ling 3.1 Flash (stalled), Laguna S 2.1
  (stalled), Glyph Cluster (preview 0 FB/hr), Solar Mini 4 (5), "MiMo 2.6 Flash" (10,
  Recommended), GLM 5.3 Flash (15), DeepSeek V4.1 Flash (15).
  ⚠ The "MiMo 2.6 Flash" card runs `mimo/mimo-v2.5` (binary: `CH=aV.mimoV25`,
  `s6=aV.mimoV26Pro`; run log confirmed `mimo/mimo-v2.5`). **Freebuff has NO mimo
  v2.6 flash** — the operator's requested model exists on opencode only.
  Deepseek flash IS selectable: card "DeepSeek V4.1 Flash" → `deepseek/deepseek-v4-flash`
  (legacy agent id `base2-free-deepseek-flash` in binary; picker exposes it directly).

### Verdict table

| path | reachable | model | returned well | cost (this session) | notes |
|---|---|---|---|---|---|
| omp tool → opencode | YES (after driver fix) | `opencode/mimo-v2.6-flash-free` | `FINAL_OC_OK` 17.2 s; multi-turn ask ok | $0 | driver at ~/bin/delegate required |
| omp tool → freebuff | YES (after driver fix) | card "MiMo 2.6 Flash" → server `mimo/mimo-v2.5` (label drift, see above) | `TOOL_FB_OK` 23.6 s | 10 FB/hr per turn (balance 105→95→85→…) | TUI drive is inherently ~30-70 s/turn |
| omp tool → auto | opencode first (cursor absent) — resolves; probe | = opencode row | verified via direct api; probe JSON lists rows | — | freebuff never auto-picked (by design) |
| driver ask → opencode session t1 | YES | mimo v2.6 free | `DELEGATE_OK` remembered, 13-14 s | $0 | `-s ses_…` continuation |
| driver ask → freebuff TUI chat | YES (same chat continues) | same card model | `FB_GUARD_OK` remembered verbatim from prior turn, 48 s | credits | queue can host multiple pending turns |
| driver run → freebuff with turn anchor | YES | same | `FB_GUARD_OK` 125 s (queued behind earlier turns), correct text despite queue drain | credits | line-anchored guard prevents stale-answer mixups |
| cursor | NO binary present | — | — | — | probe reports clearly |

### Caveats / known limits (documented)
1. **execFile-spawned opencode needs stdin=DEVNULL**: opencode v1.18.35 hangs
   forever when its stdin is an open pipe with no EOF (reproduced: node
   `execFile → opencode` hung 60 s empty; bash pipe held open → hang; FIXED in
   driver by `stdin=DEVNULL`; node-execFile rerun → 12.7 s `NODEFIX_OK`).
2. Free-tier burst throttling: two tool-routed opencode calls that raced against
   each other + parallel test traffic hit 240/180 s empty timeouts (empty stdout,
   no server hint). Direct sequential calls succeed. In status: transient, it is
   server-side; the driver now answers `{ok:false, timed_out, stderr/raw}` so the
   agent fail-over can retry.
3. Freebuff queue gap: if the previous turn is still queued, a fresh send can wait
   longer than the 9 s quiet window between turns; the driver now anchors the
   verdict to the chat-log line of OUR prompt (fix for the "previous turn's text
   as answer" failure the skill warns about).
4. The eval/omp caller can background long tool calls and abort them via the tool
   signal (`aborted or timed out` at ~19 s) — caller-layer artifact, retryable;
   the driver itself survived every direct invocation.

## Stage 2 — xd://workflow

### Repro
- Foreground probe with `export default` → run `minimal-esm-probe-…` status=failed,
  total=0, tokens=0; surfaced as "Workflow was aborted".

### Root cause (file/line references — `@quintinshaw/pi-dynamic-workflows@^3.13.1`)
- `dist/workflow.js` `parseWorkflowScript()` (lines ~1384-1413) strips ONLY the FIRST
  `ExportNamedDeclaration` (the `meta` export) — `body = slice(first.start, first.end)`.
- Body is then executed as a CLASSIC script: `new vm.Script(wrapped)` inside
  `(async () => { … })()` (line ~1174). Any leftover `export` keyword (e.g.
  `export default async () => …` — what the omp-workflow-esm-fallback skill told us
  to write) is a V8 `SyntaxError: Unexpected keyword 'export'` in a classic script;
  the escape seals the run and the tool reports `Workflow was aborted`
  (`workflow-tool.js` ~262 wraps `WORKFLOW_ABORTED`).
- The engine is NOT broken for its documented shape. Verified probe (plain body):
  `export const meta = {[...]} / log(...) / const r = await agent('Reply with exactly:
  WF_AGENT_OK') / return {agentReply: r}` → **completed with 1 agent,
  `{"agentReply":"WF_AGENT_OK"}`, 970 tok** (run minimal-probe2-muytcpra-dxoy8o).

### Fix applied (config/skill-level only — no engine patch)
- Rewrote `~/.omp/agent/managed-skills/omp-workflow-esm-fallback/SKILL.md` to state the
  real one-export rule, the true failure signature, and the verified probe. Future
  agents/agents-in-workflow will no longer write `export default` scripts.
- Engine patch NOT applied (third-party plugin dist; risk to live session; unnecessary
  once scripts match the doc). Adapters possible upstream by making
  `parseWorkflowScript` strip additional exports, filed as the improvement path.

### Versions
- omp = `@oh-my-pi/pi-coding-agent` **18.8.3 == npm latest** (no upgrade available).
  (`omp` on npm is unrelated 1.0.0.)
- Workflow engine plugin: `pi-dynamic-workflows` ^3.13.1 (`~/.omp/plugins/node_modules`).
- `~/.omp/logs` only shows unrelated "Skill description compression" 402 warnings
  (free model role); no workflow engine entries.

## Session bookkeeping
- Freebuff Freebucks: 105 → 85 remaining at last read (10/hr burn pattern; ~4 turns +
  idle hourly spend).
- tmux sessions used: fb-trivial (freebuff chat kept alive for continuation).
- Files: NEW `/home/ubuntu/bin/delegate` (~28 KB, python3). MODIFIED:
  `~/.omp/agent/managed-skills/omp-workflow-esm-fallback/SKILL.md`;
  `~/.omp/logs` (warnings) — read only. notes.md appended (SEE BELOW).

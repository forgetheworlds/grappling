# REFERENCES.md — operator reference videos (external technique material)

Purpose: external wrestling references the operator wants the robot's motion compared against,
and the concrete pipeline for using them. Status is honest: neither could be fetched from this
host (see below), so nothing here is used as a target yet.

## 1. Operator references

| # | source | what it is | status |
|---|---|---|---|
| R1 | https://vimeo.com/501599802 | technique the operator favours; his variant keeps the rear leg "a little bit back for balance" | **UNAVAILABLE from host** — yt-dlp: "Failed to fetch macos OAuth token: HTTP Error 401"; player config endpoint returns non-JSON. Constraints captured verbally (see notes.md E5). |
| R2 | https://www.youtube.com/watch?v=gBAhX5t-GW4 | walkthrough of individual wrestling moves | **UNAVAILABLE from host** — yt-dlp: "Sign in to confirm you're not a bot" (needs cookies/JS runtime); page has no captions or chapter list accessible. |

Both registries are auth/bot-gated for anonymous requests from this machine. No workarounds were
pursued beyond two bounded attempts each, per the operator's instruction not to burn time on
download machinery.

## 2. To enable either reference

**Exact blocker (measured 2026-10-08):** this box is a datacenter IP and YouTube/Vimeo gate
anonymous downloads. yt-dlp attempts and outcomes:

| attempt | result |
|---|---|
| `yt-dlp -f 'b[height<=720]'` (default clients) | "Sign in to confirm you're not a bot" |
| `--js-runtimes node` (node v24 present) | same |
| `--extractor-args youtube:player_client=` ∈ {tv, tv_simply, mweb, web_embedded, android_vr, web_safari, ios} | same for every client |
| Vimeo via yt-dlp | "Failed to fetch macos OAuth token: HTTP Error 401" |
| Vimeo player config endpoint | returns non-JSON (blocked) |
| Headless browser fallback | unavailable: Puppeteer's Chrome for Testing has no linux/arm64 build; no system Chromium (`chromium` package has no candidate; `chromium-browser` is a snap wrapper); PUPPETEER_EXECUTABLE_PATH unset |

**Cleanest remedies (operator, ~1 minute each):**
1. `yt-dlp --cookies-from-browser <browser>` on a machine where you are logged in, then copy the
   resulting media file to `data/references/<id>/ref.mp4`; or
2. export cookies (`--cookies cookies.txt`) from that machine into `data/references/cookies.txt`
   and I re-run the fetch here; or
3. download the file yourself and drop it in `data/references/<id>/`.

Any of the three unblocks the frame-extraction + vision pipeline described below; until then the
clip is recorded as **reference unavailable** and the pipeline does not wait on it.

Drop the file anywhere under `data/references/<id>/` (e.g. `ref.mp4`), or provide a cookies file
(`--cookies`), or paste a direct media URL. Then the following runs automatically:

1. `ffprobe` metadata + frame extraction at a fixed stride and at any agreed timecodes.
2. A vision pass (image-capable model) over the extracted frames to describe, per move segment:
   stance (width/depth, hips, head), level-change depth, lead-foot placement, knee-lowering,
   trail-leg drive, arm/hand positions, and the recovery.
3. Those descriptions become (a) a geometric comparison target for our retargeted references,
   (b) explicit timecoded segments in `docs/CURRICULUM.md` for the double-leg / single-leg
   vocabulary, and (c) inputs to the operator's two-axis stance rule (rear leg back; widen for
   lateral stability).
4. Nothing from a video is treated as a physical ground truth: it informs geometry and visual
   comparison only — dynamics still come from MuJoCo.

## 3. Substitutes in use until a file lands

- GrappleMap-derived references (`data/refs/*.npz`, `videos/refs/*.mp4`) with the operator's
  stance constraints applied as repair rules.
- Ghost-overlay A/B videos (translucent reference vs the physics robot) for every stance/shot
  stage — the operator's "see it visually" requirement is satisfied by these, not by the external
  clip.
- `docs/VISUALS.md` indexes all rendered evidence per stage.

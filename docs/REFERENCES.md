# REFERENCES.md — operator reference material (external technique video)

## 1. Status

| # | source | status |
|---|---|---|
| R1 | Vimeo 501599802 | **DROPPED** (operator, 2026-10-08: "dont need vimeo"). Recorded for history only: Vimeo returns "web client only works when logged-in" — the machine's Firefox profile has no Vimeo session. |
| R2 | YouTube gBAhX5t-GW4 — *"How to stance and motion drills for wrestling"*, Footwork Trainer (Isaac J. Knable), 13:44 | **AVAILABLE** — downloaded 2026-10-08 (h264 640×360, 823.5 s, 33.8 MiB) at `data/references/yt_gBAhX5t-GW4/ref.mp4` (gitignored). |

### Working fetch recipe (documented — this host is a datacenter IP and YouTube blocks anonymous pulls)

```bash
# bot-check is cleared by an existing browser session's cookies; the n-challenge needs a JS runtime
.venv/bin/pip install -U yt-dlp yt-dlp-ejs          # 2026.08.19 + solver distribution
.venv/bin/yt-dlp \
  --cookies-from-browser "firefox:/home/ubuntu/snap/firefox/common/.mozilla/firefox/<profile>.default" \
  --js-runtimes node \
  -f 'b[height<=720]' -o ref.mp4 --no-playlist 'https://www.youtube.com/watch?v=gBAhX5t-GW4'
```
Without cookies: every player client (`tv`, `tv_simply`, `mweb`, `web_embedded`, `android_vr`,
`web_safari`, `ios`) returns "Sign in to confirm you're not a bot". Without `--js-runtimes node`
+ `yt-dlp-ejs`: "n challenge solving failed → the page needs to be reloaded".

## 2. Chapter map (author's own structure — maps onto the solo-drill tasks)

| ch | time | title | our task |
|---|---|---|---|
| 1 | 0:00–0:26 | Intro | — |
| 2 | 0:26–1:23 | **STANCE** | T3 / our STANCE reference + operator stance rules |
| 3 | 1:23–2:13 | **STALKING** | T3 shuffle/approach (T2 locomotion) |
| 4 | 2:13–3:22 | **CIRCLING** | T3 circle/angle change |
| 5 | 3:22–4:21 | **LEVEL CHANGE** | T5 level change (T3 lowered stance) |
| 6 | 4:21–6:40 | **FAKE** | not required this milestone (noted for later selection work) |
| 7 | 6:40–9:09 | **SHOTS** | **T5 penetration step** (primary technique) |
| 8 | 9:09–9:59 | **DOWNBLOCK** | defense — later phase (not this milestone) |
| 9 | 9:59–11:02 | **PEPSI** | operator-specific drill concept; summarise, don't train yet |
| 10 | 11:02–13:43 | **KNEE SPRAWL** | T5/T6 lead-knee toward/onto mat + trail-leg recovery |

## 3. How it is used

**Operator correction (2026-10-08): motion capture, not description.** The video is processed into
*poses*, and the deliverable is full **transitions** (start posture → intermediate → end posture)
usable by training — the same shape of object as our GrappleMap references, but sourced from real
footage:

1. CPU pose estimation over the whole video (2D landmarks + approximate 3D), stored per-frame with
   timestamps and confidences (`data/references/yt_gBAhX5t-GW4/pose/`).
2. Phase boundaries detected from the landmark signals themselves (pelvis/knee height minima, foot
   displacement, velocity spikes) → an explicit transition list per chapter.
3. Each transition retargeted to the G1 with the existing retargeting machinery →
   `data/refs_video/<transition>.npz` in our reference format, relationship-preserving, with the
   operator's two-axis stance rule as tie-breaker where a posture is unholdable.
4. Visual verification per transition: detected-skeleton overlay on the source frames plus the G1
   executing the retargeted motion (`videos/refs_video/`), with honest reliability notes (2D is
   reliable; monocular depth/scale/camera motion are approximate).
5. A compact numeric stance spec is a byproduct, not the goal.

It informs *geometry and visual comparison only* — dynamics and contact still come from MuJoCo.
The raw video and source-frame overlays are never committed (copyright + size: see .gitignore);
in-repo evidence is the transition index, the G1-side videos and a few stills.

## 4. Substitutes already in use

GrappleMap-derived references (`data/refs/*.npz`, `videos/refs/*.mp4`) with the operator's stance
rules as repair policy; ghost-overlay A/B videos per stance/shot stage; `docs/VISUALS.md` indexes
all rendered evidence.

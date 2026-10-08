# Static support-envelope audit of the 7 retargeted references (uncertainty U3) + reference integrity

**Agent:** SupportEnvelope · **Date:** 2026-10-08 · **Repo:** `/home/ubuntu/grappling`
**Deliverables:** `scripts/audit_support_envelope.py`, `data/support_envelope.json`,
`reports/2026-10-08/support_envelope.png`, `reports/2026-10-08/support_envelope.md` (this file).
**Assignment:** `reports/2026-10-08/motor_audit.md` uncertainty **U3** — quantify how much of the
retargeted GrappleMap geometry is statically infeasible for the G1 and how much ankle torque
would be needed; extended (parent request, 2026-10-08) to a **reference-integrity** check that
decides between *reference regeneration* and *stabiliser* work.

**Verification:** `.venv/bin/python scripts/audit_support_envelope.py` → prints the tables
below, writes the JSON and the PNG, and self-checks: class counts sum to frame counts; region
geometry verified against hand-computed cases; the min-max LP's feasibility agrees with the
geometric class on **all 2115 non-airborne robot-frames (0 mismatches)**; integrity counts and
verdict rules internally consistent; all 7 techniques × 2 robots present.
Runtime ≈ 30 s wall.

**Reading rule (stated up front):** *statically infeasible* means **the CoM is outside the foot
contact region at that frame, so no ankle torque can hold that pose** — it does **not** mean the
reference is wrong. Every dynamic motion (a dive, a penetration step, a fall, a clinch leaning on
the opponent) puts the CoM outside the feet on purpose. What the number answers is: *how much of
this reference can be imitated quasi-statically*, and hence why position-servo replay topples
(Phase 2) and how much of M4/M5 needs a dynamic controller instead of a pose tracker.

---

## 0. Answer first

**Per technique: share of frames whose CoM is outside the foot support region** (both robots;
"standing" = frames whose reference pelvis ≥ 0.45 m, the phase-2 stay-up gate; holdable =
contiguous episodes where the CoM is inside the support region):

| technique | frames (a/b) | INFEASIBLE % all | INFEASIBLE % standing | AIRBORNE % | holdable windows (s) | worst CoM deficit (m) a / b |
|---|---|---|---|---|---|---|
| STANCE | 118 / 118 | 100 / 100 | 100 / 100 | 0 / 0 | **none** | 0.002 / 0.010 |
| DOUBLE_LEG | 155 / 155 | 81.9 / 21.9 | 81.7 / 25.4 | 9.0 / 58.1 | 0.28 / 0.62 | 0.774 / 0.255 |
| SINGLE_LEG | 230 / 230 | 71.3 / 73.0 | 72.9 / 90.2 | 7.4 / 19.6 | 0.98 / 0.34 | 0.657 / 0.452 |
| BODY_LOCK | 176 / 176 | 84.1 / 67.0 | 84.1 / 81.9 | 0 / 25.0 | 0.56 / 0.26 | 0.377 / 0.328 |
| SNAPDOWN | 46 / 46 | 100 / 100 | 100 / 100 | 0 / 0 | **none** | 0.388 / 0.092 |
| SPRAWL | 288 / 288 | 70.5 / 63.5 | 65.1 / 56.8 | 29.5 / 29.2 | none / 0.32 | 0.431 / 0.747 |
| STAND_UP | 244 / 244 | 96.7 / 95.1 | 100 / 100 | 3.3 / 4.9 | **none** | 0.588 / 0.575 |

("worst CoM deficit" = most negative margin, sign flipped; the best margin over the whole
technique is still negative for STANCE (−0.1/0.9 cm), SNAPDOWN (−2.9/−5.1 cm), SPRAWL a
(−2.9 cm) and STAND_UP (−2.5/−3.6 cm) — those four have no holdable frame at all.)

Aggregate: **6.9 % of all robot-frames and 9.2 % of standing robot-frames** have the CoM inside
the foot support region. STANCE, SNAPDOWN and STAND_UP have **zero** frames that can be held
statically by either robot; SINGLE_LEG, BODY_LOCK, DOUBLE_LEG, SPRAWL each have ≤ 1.0 s of
holdable window spread over 3.1–5.7 s.

**Ankle torque is not the binding constraint.** Where a static solution exists, the smallest
achievable *worst-ankle* moment (load shared optimally between the feet, CoP placed inside the
contact patches) is at most **23.9 Nm — less than half of the 50 Nm `actuatorfrcrange` limit**
(largest values: SPRAWL b f138 23.9 Nm, DOUBLE_LEG a f13 23.7 Nm, STAND_UP a f137 23.9 Nm).
The limits are geometric: the CoP must lie inside the contact patches, and for most of these
references it cannot. **Implication for M4/M5:** no ankle-strength or gain fix can make a
position replay stand; the required work is (i) a small reference adjustment on the *standing*
postures and (ii) a dynamic balance layer for the transient phases (exactly M1/M4 in
`docs/MOTOR_CURRICULUM.md`).

**3 worst frames (standing phase, deepest support violation):**

| # | technique | robot | frame | t (s) | CoM margin | foot state | CoM→ankle lever | toppling moment |
|---|---|---|---|---|---|---|---|---|
| 1 | SINGLE_LEG | a (shooter) | 31 | 0.62 | **−0.657 m** | one foot down | 0.616 m | 201 Nm |
| 2 | SINGLE_LEG | a | 30 | 0.60 | −0.656 m | one foot down | 0.618 m | 202 Nm |
| 3 | SINGLE_LEG | a | 32 | 0.64 | −0.656 m | one foot down | 0.616 m | 200 Nm |

Frames 28–32 of SINGLE_LEG a are one episode: the shooter's CoM is 0.62 m from the planted
ankle while the body is already low (CoM 0.56–0.57 m high, pelvis 0.64–0.65 m) and diving past
its foot — the pose is a lunge in free fall, not a supportable posture. The deepest violation in
**any** phase is DOUBLE_LEG a f152 (t = 3.04 s, margin **−0.774 m**, pelvis 0.35 m, ground phase)
and the highest toppling moment is DOUBLE_LEG a f152 too (230 Nm). For all of these the min-max
LP is *infeasible*: no static ankle torque exists at all, so the 201–230 Nm numbers are the
gravitational moment that must be opposed *dynamically* (step, fall, or opponent reaction), not
a servo demand.

---

## 1. Method — every choice and every approximation

### 1.1 Kinematic replay (not dynamics)

For each technique in `data/refs/<TECH>.npz` and each frame we set the scene `qpos`
(`qpos[0:36]` = robot a, `qpos[36:72]` = robot b) and call `mujoco.mj_forward` on
`robots/wrestling_scene.xml`. No dynamics, no PD servos, no contact solver: this measures **the
geometry the reference encodes**, not the drift of a particular controller (which is what
`reports/2026-10-07/retarget.md` §6 already measured and `reports/2026-10-08/wrestling_env.md`
§3.1 re-measured for STANCE). `mj_forward` (not `mj_kinematics`) because we need
`subtree_com`.

### 1.2 CoM

CoM = `data.subtree_com[a_pelvis]` / `[b_pelvis]`: the pelvis is the root body of each robot's
kinematic subtree, so this is the whole robot. Cross-checked against a manual mass-weighted sum
over that robot's bodies (exact to 1e-4 m; total mass 33.341 kg/robot). Horizontal position =
xy part; we ignore the CoM height except for reporting.

### 1.3 Foot contact — same sites and same rule as the teacher

Sites `{a,b}_{left,right}_{toe,heel}`; a sole site counts as touching when
**`site_z < FOOT_Z_TOUCH = 0.045 m`** — the teacher's phase-label threshold
(`src/teacher/controller.py:65-66`), so the audit and the teacher agree on when a foot is down.
This is a height threshold, not the collision solver. Measured planted sole heights are
0.000–0.020 m above the mat in the clean techniques, i.e. safely inside the band; a site below
the mat (reference ground penetration, up to 0.16 m in SPRAWL) also counts as touching, and the
penetration is reported separately (§5) and by the verdict rules.
A tight `0.02 m` threshold is also reported (`nofoot<2cm%`) as an independent check.

### 1.4 Support region — three representations, and why the *footprint* hull classifies

* **(a) literal site hull** — convex hull of the touching toe/heel **sites**. The request's
  primary form, but degenerate by construction: both sites sit on the foot **midline** (g1.xml
  local `y = 0`), so a flat single foot yields a 1-D region and single support would be
  "infeasible" for any CoM that is not exactly on the line. Kept for the record (§3.2).
* **(b) toe–heel line union** — the union of the planted feet's heel→toe segments (the request's
  simpler form). A 1-D region; reported as the distance from the CoM to the nearest point of the
  union, with the same 0.02 m band.
* **(c) footprint hull — the PRIMARY CLASSIFIER.** Convex hull of the world-projected **sole
  contact patches** of the touching (foot, site) pairs. A patch is the pair of sole collision
  spheres the scene already defines (`robots/g1/g1.xml:108-115`: r = 5 mm, heel at local
  x = −0.05, y = ±0.025; toe at x = +0.12, y = ±0.03), projected onto the mat: for a horizontal
  floor the contact point is exactly under the sphere centre, so the projection is exact. For
  **point contacts the CoP region *is* the convex hull of the contact points**, so this is the
  physical support region of the scene as it actually collides, 0.175 m long × 0.06 m wide per
  foot. Using (a) or (b) instead would understate the support by the whole foot width and by the
  inter-foot area.

Margin sign convention: distance to the region boundary, **positive inside, negative outside**
(degenerate 1-D regions: margin ≤ 0, magnitude = distance to the segment).

### 1.5 Class per frame, band, and the standing/ground split

`STABLY_FEASIBLE` (margin ≥ band), `MARGINAL` (0 ≤ margin < band), `INFEASIBLE` (margin < 0),
`AIRBORNE` (no touching sole site — a separate class, because a foot-only criterion does not
apply; the task asked for it explicitly). **Band = 0.02 m**, the same order as the retargeter's
accepted ground-penetration tolerance and small against the 0.175 m foot; the JSON carries the
class counts at bands 0.01/0.02/0.05 (the counts are not band-sensitive at the technique level:
in every technique the "feasible" total changes by < 2 % of frames between bands 0.01 and 0.05).

Contact state per frame: `one` / `both` feet planted, or `airborne_ground` (nothing on the feet
but the lowest body site is within 0.05 m of the mat → the body is supported by knees/hands/torso)
/ `airborne_float` (nothing at all within 0.05 m → the reference floats; see §5). Frames whose
**reference pelvis** (qpos[2]) is below 0.45 m are `ground` frames: the body is on the mat and the
foot-only criterion does not govern there, so class counts are also reported for the standing
subset only (that is the subset M4/M5 must imitate as a posture).

### 1.6 Ankle torque — two estimators, both static

Model: the mass above the ankles as an inverted pendulum, ground reaction vertical through the
CoP, foot mass neglected; then the ankle moment is exactly the body's gravitational moment about
the ankle axis.

* **`tau_pendulum` (the requested rough model)** = `m·g·d`, `d` = horizontal distance from the CoM
  to the support foot's **ankle axis** (single support) or to the **midpoint of the two ankle
  axes** (double support). `m = body_subtreemass[pelvis] = 33.34 kg`, `g = 9.81`. Exact for single
  support; for double support it is the rigid-pendulum equivalent, because with two feet the
  load share is statically indeterminate. Split into `tau_pitch` (fore-aft, about the ankle_pitch
  axis) and `tau_roll` (lateral, about the ankle_roll axis) using the foot-body frame.
* **`tau_minmax` (load-sharing-aware)** = the **smallest achievable worst-ankle moment** over
  every static allocation of the body weight to the touching feet and every CoP placement inside
  that foot's patch:
  `min t s.t. ΣNᵢ = mg, ΣNᵢCoPᵢ = mg·CoM_xy, CoPᵢ ∈ patchᵢ, t ≥ |Nᵢ(CoPᵢ − ankleᵢ)|`.
  Solved with `scipy.optimize.linprog` on the cone representation
  `CoPᵢ = Σλⱼvⱼ / Σλⱼ`, `Nᵢ = Σλⱼ`, with the norm discretised over 12 unit directions (≤ 3.4 %
  under-estimate). It is infeasible **exactly** when the CoM is outside the contact hull — this
  is used as an independent check of the geometric class and passes on all 2115 non-airborne
  frames. `tau_minmax` is the number that decides "could *any* static controller hold this?";
  `tau_pendulum` is the number that says how hard gravity is torquing the body.
* A third, deliberately naive bound `tau_naive_one_ankle = mg·maxᵢ|CoM − ankleᵢ|` (one ankle takes
  the whole moment) is stored in the JSON **only to show why the naive metric must not be used**:
  it prices a load path no controller would take, so it over-reports by 2–5×. Examples: the
  staggered STANCE gives 75.4 Nm (a) / 66.9 Nm (b) where the pendulum demand is 30.5 / 24.9 Nm and
  the measured actuator torque at the tipping point was 39 Nm; BODY_LOCK a frame 35 gives 116.8 Nm
  where the load-sharing minimum is 21.6 Nm; DOUBLE_LEG a frame 13 gives 147.4 Nm vs 23.7 Nm.

Limits of both: no dynamics (inertia, CoM velocity, capture point), no external grappling load,
no torso/arm inertia, no compliant-mesh contact, no foot slip, no actuator rate limit; the
min-max LP additionally assumes ideal load sharing. Because the criterion uses the **feet only**,
every number is a *lower bound on support*: in clinch frames the opponent is a second contact
path (pair CoM–CoM distance < 0.6 m in 81–100 % of frames for every technique except STANCE,
which is 0 % — see §5.3).

### 1.7 Reference-integrity checks (parent request, same replay)

1. **Contact availability** — per-foot min sole-site height, frames where neither foot is within
   0.02 m of the mat, and the longest contact-free episode measured on the **lowest body site**
   (`airborne_float`).
2. **Placement** — deepest sole-site penetration below the mat, and how many frames exceed 2 cm
   and 5 cm.
3. **Montage seams** — per frame: max joint delta over the 29 joints, base translation, base spin
   (quaternion angle) and max sole-site jump, compared against **the retimer's own limits**
   (6 rad/s, 3 m/s, 8 rad/s at 50 Hz → 0.12 rad, 0.06 m, 0.16 rad per frame; `retarget.md` §3)
   plus an 0.08 m/frame foot-jump limit. Inside a segment the retimer guarantees those limits, so
   any frame above them is a seam artifact.
4. **Contact-transition sanity** — frames with one foot planted whose CoM is more than 0.05 m
   outside that foot's support.
5. **Verdict per technique** (rules printed by the script, thresholds in the JSON meta):
   `REF_INVALID`/REGENERATE (sustained > 0.40 s contact-free flight with the body > 0.05 m up;
   ≥ 5 cm sole penetration on more than 2 % of frames; or a > 2× retime-limit jump);
   `REF_VALID_ADJUSTABLE`/ADJUST (seam artifacts, or a standing deficit ≤ 0.05 m);
   `REF_VALID_DYNAMIC`/STABILISE (contact-consistent and continuous: either ≥ 40 % of standing
   frames feasible with the demand within 50 Nm, or a large transient CoM excursion). Secondary
   defects are listed in `also` so nothing is hidden.

---

## 2. Data snapshot (what exactly was audited)

| technique | frames | duration s | npz sha256 (16) | mtime | robot a role (b = counterpart) |
|---|---|---|---|---|---|
| STANCE | 118 | 2.34 | `00ef9d08548002ce` | 02:09 | symmetric start pose (a faces +x) |
| DOUBLE_LEG | 155 | 3.08 | `11cfd824bb408d68` | 01:39 | shooter/attacker (b = defender) |
| SINGLE_LEG | 230 | 4.58 | `f0cebe55e4664897` | 01:43 | shooter/attacker (b = defender) |
| BODY_LOCK | 176 | 3.50 | `788c708096c30ebc` | 01:33 | flanker/mover (b = opponent) |
| SNAPDOWN | 46 | 0.90 | `751a4d396bc53ef9` | 01:49 | snapper (b = snapped opponent) |
| SPRAWL | 288 | 5.74 | `3f1ca890d81e4dcc` | 01:51 | attacker, shot chain (b = defender, ends prone) |
| STAND_UP | 244 | 4.86 | `e7a80311e1a65886` | **06:04** | recoverer, bottom (b = opponent) |

Roles are read off `reports/2026-10-07/retarget.md` §4 (the npz `meta.roles` string is generic).

**Two snapshot notes (important for reading the numbers):**

1. **`STAND_UP.npz` was replaced by a concurrent agent at 06:04 while this audit was running**
   (470 → 244 frames; the previous revision is preserved as `data/refs/STAND_UP.airborne.npz`).
   On the pre-fix revision this audit measured **6.30 s (a) / 6.80 s (b) of contact-free flight,
   72 % airborne frames, lowest body site a median 0.23–0.31 m above the mat, 74–82 % of frames
   with no foot within 2 cm** — i.e. the previous STAND_UP was not placeable at all (which also
   explains its stay-up 0.000 in earlier evidence). The 06:04 revision removes that float
   (§5) but keeps the support problem: **100 % of its standing frames are still infeasible** and
   it has no holdable window (§3). All numbers in this report are for the 06:04 revision
   (`e7a80311e1a65886`); the JSON records the hash so the result is anchored.
2. The committed frame counts differ from `retarget.md`'s table (e.g. DOUBLE_LEG 155 vs 178,
   SPRAWL 288 vs 334 frames); the committed npz files are what the env and teacher consume, so
   they are what is audited here.

---

## 3. Results

### 3.1 Classes, support states and torque per technique (script output, verbatim)

```
technique   r frames stand stably%  marg%   inf% inf/stand%  air% float%  1ft%  2ft%  worst lever m@ s ankle moment Nm@ s       inf episodes s
----------------------------------------------------------------------------------------------------------------------------------------------
STANCE      a    118   118     0.0    0.0  100.0      100.0   0.0    0.0   0.0 100.0        0.093@0.00          30.5@0.00         0.00-2.34 x1
STANCE      b    118   118     0.0    0.0  100.0      100.0   0.0    0.0   0.0 100.0        0.076@0.00          24.9@0.00         0.00-2.34 x1
DOUBLE_LEG  a    155   142     2.6    6.5   81.9       81.7   9.0    7.1  16.8  74.2        0.703@3.04         230.0@3.04         0.28-3.08 x2
DOUBLE_LEG  b    155   134    11.0    9.0   21.9       25.4  58.1   54.2  19.4  22.6        0.306@0.70         100.0@0.70         0.62-1.44 x2
SINGLE_LEG  a    230   192     0.0   21.3   71.3       72.9   7.4    0.0  48.3  44.3        0.618@0.58         202.1@0.58         0.00-4.58 x3
SINGLE_LEG  b    230   174     0.0    7.4   73.0       90.2  19.6   19.1  65.7  14.8        0.486@3.68         159.1@3.68         0.00-3.68 x2
BODY_LOCK   a    176   176    10.2    5.7   84.1       84.1   0.0    0.0   5.7  94.3        0.433@2.38         141.6@2.38         0.00-3.50 x2
BODY_LOCK   b    176   138     0.0    8.0   67.0       81.9  25.0   14.2  56.8  18.2        0.342@2.84         111.9@2.84         0.00-2.84 x4
SNAPDOWN    a     46    46     0.0    0.0  100.0      100.0   0.0    0.0   4.3  95.7        0.375@0.88         122.8@0.88         0.00-0.90 x1
SNAPDOWN    b     46    46     0.0    0.0  100.0      100.0   0.0    0.0   0.0 100.0        0.207@0.82          67.8@0.82         0.00-0.90 x1
SPRAWL      a    288   189     0.0    0.0   70.5       65.1  29.5   28.8  10.1  60.4        0.484@4.04         158.4@4.04         0.00-5.74 x2
SPRAWL      b    288   243     2.8    4.5   63.5       56.8  29.2   28.1  24.0  46.9        0.683@5.74         223.5@5.74         0.12-5.74 x4
STAND_UP    a    244    91     0.0    0.0   96.7      100.0   3.3    0.0  29.9  66.8        0.498@3.10         162.8@3.10         0.00-4.86 x2
STAND_UP    b    244    91     0.0    0.0   95.1      100.0   4.9    4.1  17.6  77.5        0.474@1.38         155.0@1.38         0.00-4.86 x2
```

Column notes: `stand` = frames with reference pelvis ≥ 0.45 m; `stably%`/`marg%`/`inf%` are
shares of **all** frames on the footprint hull; `inf/stand%` is the share of **standing** frames
that are infeasible; `air%` = no foot contact; `float%` = nothing (body included) within 0.05 m;
`1ft%`/`2ft%` = one/both feet planted; `ankle moment` = `tau_pendulum` at its worst frame;
`inf episodes` = first and last INFEASIBLE frame time and the number of runs (gaps ≤ 0.04 s
merged). Per-frame class runs and the full INFEASIBLE episode tables are in the JSON
(`infeasible_episodes` has start/end frame indices and times for every episode; e.g. STANCE
0–117 for both robots, SINGLE_LEG a 0–50 / 100–188 / 206–229, BODY_LOCK b 0–18 / 22–63 /
77–108 / 117–142).

### 3.2 Sensitivity to the support-region representation

F/M/I/A = STABLY_FEASIBLE / MARGINAL / INFEASIBLE / AIRBORNE counts:

| technique | r | footprint hull (classifier) | site hull (request a) | toe–heel line union (request b) |
|---|---|---|---|---|
| STANCE | a / b | 0/0/118/0 · 0/0/118/0 | 0/0/118/0 · 0/0/118/0 | 0/0/118/0 · 0/0/118/0 |
| DOUBLE_LEG | a / b | 4/10/127/14 · 17/14/34/90 | 0/0/141/14 · 1/30/34/90 | 0/0/141/14 · 0/4/61/90 |
| SINGLE_LEG | a / b | 0/49/164/17 · 0/17/168/45 | 0/14/199/17 · 0/0/185/45 | 0/0/213/17 · 0/0/185/45 |
| BODY_LOCK | a / b | 18/10/148/0 · 0/14/118/44 | 9/12/155/0 · 0/2/130/44 | 0/0/176/0 · 0/10/122/44 |
| SNAPDOWN | a / b | 0/0/46/0 · 0/0/46/0 | 0/0/46/0 · 0/0/46/0 | 0/0/46/0 · 0/0/46/0 |
| SPRAWL | a / b | 0/0/203/85 · 8/13/183/84 | 0/0/203/85 · 7/3/194/84 | 0/0/203/85 · 0/0/204/84 |
| STAND_UP | a / b | 0/0/236/8 · 0/0/232/12 | 0/0/236/8 · 0/0/232/12 | 0/0/236/8 · 0/0/232/12 |

The literal site hull is **more conservative** than the physical footprint exactly where the
CoM sits off the foot midline or between the feet (SINGLE_LEG a: 14 vs 49 feasible frames;
BODY_LOCK a: 21 vs 28; SPRAWL b: 10 vs 21); the toe–heel line union is the most conservative
(1-D, so it only admits frames with the CoM essentially on a planted foot's midline). The three
representations agree on the qualitative answer everywhere (no technique flips from
"mostly infeasible" to "feasible"), which is the robust part of the finding. The classifier's
choice cannot change any recommendation in §6.

### 3.3 Holdable windows (CoM inside the support region, ≥ 0.10 s)

| technique | robot a | robot b |
|---|---|---|
| STANCE | none | none |
| DOUBLE_LEG | 0.28 s (0.00–0.26) | 0.62 s (0.00–0.60) |
| SINGLE_LEG | 0.98 s (1.02–1.98) | 0.34 s (0.34–0.66) |
| BODY_LOCK | 0.56 s (0.70–1.24) | 0.26 s (1.28–1.52) |
| SNAPDOWN | none | none |
| SPRAWL | none | 0.32 s (0.00–0.10, 2.76–2.94) |
| STAND_UP | none | none |

STANCE (100 % infeasible), SNAPDOWN and STAND_UP never hold statically; the other four have
≤ 1.0 s. The longest window in the whole corpus is SINGLE_LEG a's 0.98 s window at
t = 1.02–1.98 s — that is where a quasi-static pose tracker could plausibly work, and it is the
natural place to anchor a per-phase success criterion.

### 3.4 Ankle-torque detail (`tau_pendulum` vs `tau_minmax`)

| technique | r | worst pendulum lever (m @ s) | worst `tau_pendulum` (Nm) | median / p95 `tau_minmax` (Nm) | worst `tau_minmax` (Nm) | frames min-max > 50 Nm |
|---|---|---|---|---|---|---|
| STANCE | a / b | 0.093@0.00 / 0.076@0.00 | 30.5 / 24.9 | — / — | — | 0 / 0 |
| DOUBLE_LEG | a / b | 0.703@3.04 / 0.306@0.70 | 230.0 / 100.0 | 22.0 / 15.0 | 23.7 / 15.4 | 0 / 0 |
| SINGLE_LEG | a / b | 0.618@0.58 / 0.486@3.68 | 202.1 / 159.1 | 19.1 / 19.3 | 22.5 / 20.8 | 0 / 0 |
| BODY_LOCK | a / b | 0.433@2.38 / 0.342@2.84 | 141.6 / 111.9 | 10.2 / 18.5 | 21.6 / 19.3 | 0 / 0 |
| SNAPDOWN | a / b | 0.375@0.88 / 0.207@0.82 | 122.8 / 67.8 | — / — | — | 0 / 0 |
| SPRAWL | a / b | 0.484@4.04 / 0.683@5.74 | 158.4 / 223.5 | — / 17.8 | — / 23.9 | 0 / 0 |
| STAND_UP | a / b | 0.498@3.10 / 0.474@1.38 | 162.8 / 155.0 | — / — | 23.9 (a) | 0 / 0 |

`—` = no frame of that robot has a static solution at all, so a min-max demand does not exist.
**Zero frames anywhere require more than 50 Nm** in the min-max sense; the worst is 23.9 Nm.
Where the pendulum model exceeds 50 Nm (many frames) the cause is that it prices an
*unachievable* CoP (CoM outside the patches) or a naive single-ankle load path.

**Cross-check against the existing measurement:** STANCE frame 0 is a double-support pose with
`tau_pendulum` = 30.5 Nm (a) / 24.9 Nm (b) — pitch 30.0 Nm, roll 5.5 Nm for a. The measured
actuator torque in the PD replay reached **≈39 Nm of the 50 Nm limit as the body tipped**
(`reports/2026-10-08/wrestling_env.md` §3.1). Static demand at the reference pose 30 Nm plus the
toppling motion = the measured 39 Nm: the two independent methods are consistent, and they now
agree on the *reason* as well (§6, STANCE is 1–10 mm outside the support).

---

## 4. Figure

`reports/2026-10-08/support_envelope.png` — one panel per technique: **CoM margin (m) vs time**
for robot a (solid) and b (dashed), with the 0.02 m marginal band shaded; a coloured strip to the
right of each panel shows the per-frame class (`green` STABLY_FEASIBLE, `amber` MARGINAL,
`red` INFEASIBLE, `grey` AIRBORNE), one row per robot, and **black hatching** marks frames where
nothing at all is within 0.05 m of the mat (reference float).

What to look for: (1) how rarely the line is above the amber band; (2) STANCE/SNAPDOWN are solid
red, i.e. never holdable; (3) the deep red excursions (SINGLE_LEG a ≈ 0.6 s, DOUBLE_LEG a after
2.6 s) are the dive/shot phases where the CoM is far beyond the feet; (4) SPRAWL a/b and
STAND_UP show long grey/hatched spans — the phases where the pair is off the ground or (before
the 06:04 fix) floating; (5) DOUBLE_LEG b's hatching is the defender's 1.6 s of flight (§5).

---

## 5. Reference integrity

### 5.1 Integrity table (script output, verbatim)

```
technique   r nofoot<2cm% longest float med low-z    pen max m@ s  max dq rad@ s max dBase m max dSpin foot jump m                  seam frames
-----------------------------------------------------------------------------------------------------------------------------------------------
STANCE      a         0.0          0.00     0.006               -     0.001@0.04       0.000     0.000       0.000                            -
STANCE      b         0.0          0.00     0.013               -     0.001@2.04       0.000     0.000       0.000                            -
DOUBLE_LEG  a        50.3          0.22     0.020      0.012@0.68     0.119@0.02       0.023     0.081       0.036                            -
DOUBLE_LEG  b        91.0          1.62     0.071               -     0.109@0.68       0.020     0.065       0.047                            -
SINGLE_LEG  a        46.5          0.00     0.018      0.016@4.48     0.070@0.96       0.016     0.050       0.045                            -
SINGLE_LEG  b        84.3          0.88     0.035               -     0.087@1.22       0.014     0.036       0.037                            -
BODY_LOCK   a        15.9          0.00     0.008      0.009@1.80     0.098@0.02       0.014     0.074       0.054                            -
BODY_LOCK   b       100.0          0.26     0.039               -     0.114@3.50       0.024     0.082       0.036                            -
SNAPDOWN    a         6.5          0.00     0.011      0.036@0.14     0.119@0.02       0.019     0.038       0.031                            -
SNAPDOWN    b        65.2          0.00     0.022               -     0.062@0.90       0.014     0.035       0.029                            -
SPRAWL      a        37.2          1.66     0.006      0.100@2.64     0.110@5.74       0.024     0.103       0.053                            -
SPRAWL      b        38.2          1.18     0.009      0.161@2.64     0.109@3.72       0.023     0.159       0.090 126@2.52, 127@2.54, 128@2.56, 129@2.58...
STAND_UP    a        16.8          0.00     0.002      0.014@1.72     0.074@1.72       0.011     0.053       0.029                            -
STAND_UP    b        27.5          0.20     0.006      0.021@2.68     0.074@1.22       0.010     0.039       0.032                            -
```

Findings:

* **Sustained contact-free flight is now confined to DOUBLE_LEG b** (1.62 s, body median
  0.071 m up, max 0.457 m; nofoot<2cm on 91 % of frames). The defender is the lifted partner in
  a double-leg, so part of this is intended; 1.6 s with *nothing* within 5 cm of the mat and no
  ground reference at all is nevertheless not placeable as data. DOUBLE_LEG a's 50 %
  no-foot frames are ≤ 0.22 s hops (a normal shot) and are not flagged.
* **SPRAWL has a real placement defect:** sole sites are 2–16 cm below the mat on 41/288 frames
  (a) and 95/288 (b), spread over t ≈ 2.48–3.36 s and 5.52–5.74 s — i.e. once the pair goes to
  the ground the mat-tracking is not floor-consistent, well beyond a single splice frame. 14 (a)
  and 49 (b) of those frames exceed 5 cm.
* **The SPRAWL splice seam is independently visible** at frames 126–132 (t ≈ 2.52–2.64 s): joint
  delta 0.110 rad (limit 0.12), base spin 0.159 rad (limit 0.16), **sole-site jump 0.090 m
  (limit 0.08)** — the only frames in the corpus that violate the retimer's own limits. The
  penetration maximum (0.161 m) sits 3 frames later, so the seam and the penetration are the same
  event. Everything else is continuous: no other technique has a single frame above the retime
  limits, which says the montage chaining itself (PCHIP + retime) is doing its job.
* **STAND_UP (06:04 revision) is clean on flight** (0–0.20 s float, nofoot<2cm 17–28 %) but has
  2 frames with the sole 1.4–2.1 cm below the mat. The pre-fix revision was not: 6.3–6.8 s of
  flight and 74–82 % no-foot frames.
* **SNAPDOWN a** has 9 frames (4–12, t = 0.08–0.24 s) up to 3.6 cm below the mat — a small,
  localized clamp issue, not a structural one.
* Small penetrations (≤ 2.1 cm, mostly 0.009–0.016 m) appear in almost every technique —
  DOUBLE_LEG a 0.012 m, SINGLE_LEG a 0.016 m, BODY_LOCK a 0.009 m, STAND_UP b 0.021 m — consistent
  with the retarget report's accepted ground-penetration tolerance (≤ 0.011 m reported); the
  frames above 2 cm are the ones listed above.

### 5.2 Contact-transition sanity (request item 3)

Frames with **one** foot planted whose CoM is > 0.05 m outside that foot's support:
DOUBLE_LEG a 26 (worst −0.774 m), b 30 (−0.255), SINGLE_LEG a 111 (−0.657), b 151 (−0.452),
BODY_LOCK a 10 (−0.377), b 95 of 100 (−0.328), SPRAWL a 29 (−0.431), b 69 (−0.747), SNAPDOWN a 2
(−0.388), STAND_UP a 73 (−0.588), b 43 (−0.575). In every case the single-support violation is the
**intended** behaviour of a lunge/dive/shot (the CoM passes the planted foot and the body falls
forward into the technique), not a teleport or a mis-typed foot state: the *feet* stay on the mat
and the CoM leaves. There is no frame anywhere in the corpus where a foot is airborne and the CoM
is comfortably inside the other foot's support — the largest margin over all single-support frames
is **+8 mm** (BODY_LOCK b) — and no frame where the CoM jumps with the feet planted.

### 5.3 Pair contact range (is the opponent a second support path?)

| technique | CoM–CoM median (m) | within 0.4 m | within 0.6 m |
|---|---|---|---|
| STANCE | 0.843 | 0 % | 0 % |
| DOUBLE_LEG | 0.335 | 73 % | 81 % |
| SINGLE_LEG | 0.347 | 71 % | 90 % |
| BODY_LOCK | 0.268 | 100 % | 100 % |
| SNAPDOWN | 0.516 | 0 % | 100 % |
| SPRAWL | 0.309 | 80 % | 99 % |
| STAND_UP | 0.370 | 71 % | 100 % |

STANCE is the only technique where the two robots never interact, so its support deficit must be
solved by the feet alone. In BODY_LOCK (100 % of frames within 0.4 m) and SPRAWL/SINGLE_LEG the
opponent is almost always within grappling range, so the foot-only deficit there is an **upper
bound** on the demand: the pair's contact can legitimately carry part of the load (the phase-2
report already measured 1.3–5.6 cm of inter-robot mesh penetration in these clips, i.e. contact is
happening). This audit does not solve the contact-force distribution; it flags that the
foot-only criterion is conservative for clinch techniques.

### 5.4 Verdicts — data defect, adjustable geometry, or controller limit?

```
STANCE      REF_VALID_ADJUSTABLE   ADJUST      the standing posture is statically infeasible by only 9.5 mm: shift the CoM or plant the feet (the rest of the trajectory is contact-consistent)
DOUBLE_LEG  REF_INVALID            REGENERATE  robot b has 1.62 s of contact-free flight with the body a median 0.071 m above the mat: the reference is not placeable on the mat
SINGLE_LEG  REF_VALID_DYNAMIC      STABILISE   contact-consistent and continuous, but the CoM leaves the support region by up to 0.657 m during the technique (0.98 s of holdable windows): the excursion is a dynamic transient
BODY_LOCK   REF_VALID_DYNAMIC      STABILISE   ... up to 0.377 m (0.56 s of holdable windows)
SNAPDOWN    REF_VALID_ADJUSTABLE   ADJUST      9 frame(s) drive a sole up to 0.036 m below the mat (worst at frame 7): clamp/re-place them; the standing CoM excursion still reaches 0.388 m with 0.00 s of holdable windows
SPRAWL      REF_INVALID            REGENERATE  robot b's sole is 0.161 m below the mat on 49 frames: the vertical placement is not physical
                                          also: 7 frames around [126,127,128,129] violate the retime limits (dq 0.110 rad, spin 0.159 rad, foot jump 0.090 m)
                                          also: 95 frames drive a sole up to 0.161 m below the mat (worst at frame 132)
STAND_UP    REF_VALID_ADJUSTABLE   ADJUST      2 frame(s) drive a sole up to 0.021 m below the mat (worst at frame 134): clamp/re-place them; the standing CoM excursion still reaches 0.588 m with 0.00 s of holdable windows
```

**Which techniques I would rebuild:** **DOUBLE_LEG** (defender's vertical placement: 1.62 s of
flight, 91 % of frames with no foot within 2 cm — re-place the defender track / re-solve the pair
floor) and **SPRAWL** (floor-inconsistent ground phases on a third of the frames plus the splice
seam at 2.52–2.64 s — re-place the ground phases and resample the transition). Both are
*data* problems, not controller problems; the other five techniques are physically consistent
enough to train against, with two local clamps (SNAPDOWN frames 4–12, STAND_UP frame 134) and one
genuine geometry nudge (STANCE, §6).

---

## 6. Interpretation for M4/M5 reference adjustment

1. **A position-tracking imitator cannot stand up on these references, and that is now
   quantified:** only 6.9 % of robot-frames (9.2 % of standing ones) have the CoM inside the foot
   support region, and STANCE/SNAPDOWN/STAND_UP have none. Phase 2's toppling under PD replay is
   therefore not a gain-tuning problem — the *intended* geometry is a falling trajectory in those
   frames. M4/M5 must treat imitation as **dynamic** imitation (velocity/CoM trajectory tracking
   with a balance layer), or explicitly gate the imitation loss to the holdable windows
   (§3.3).
2. **Ankle strength is not the bottleneck.** Everywhere a static solution exists the min-max
   ankle demand is ≤ 23.9 Nm (< 50 % of the actuator limit); the naive "CoM-to-ankle lever"
   metric over-reports by 2–5× because it prices a load path an optimal controller would not use.
   Effort spent on ankle gains/torque limits is misdirected; effort on *CoP placement* (foot
   flatness, stance width, step timing) is the real lever.
3. **STANCE needs a ~1 cm fix, not a redesign.** Its violation is **1.2 mm (a) / 9.5 mm (b)** —
   the CoM projects just outside the thin support ribbon spanned by the two forefoot contact
   patches, because the heels are 6–7 cm off the mat (measured sole heights: toe 0.005–0.034 m,
   heel 0.059–0.072 m). Two equivalent fixes: plant the heels (extending the support region
   0.17 m backwards) or shift the CoM ~2 cm forward. This reconciles the two earlier
   measurements: EnvBuilder's 39 Nm tipping torque is exactly what a pose 1 mm past the support
   boundary does under any perturbation. The `docs/MOTOR_CURRICULUM.md` M4 "stance" task should be
   built on the *heel-planted* variant, not on the current STANCE.npz.
4. **The three single-support violations that matter most are all one dive** (SINGLE_LEG a
   0.56–0.66 s, CoM 0.62 m past the ankle, 200 Nm toppling moment). No reference adjustment will
   make that holdable; it is a ballistic phase and needs either a dynamic controller or a
   different support strategy (step-through). The same is true of DOUBLE_LEG a's −0.774 m dive at
   t ≈ 3.0 s (ground phase; 230 Nm toppling moment).
5. **Reference work order** (from §5.4): (i) rebuild DOUBLE_LEG's defender placement and
   SPRAWL's ground placement + splice seam — both are measured data defects and will silently
   corrupt any imitation/BC training that touches them; (ii) clamp SNAPDOWN frames 7–15 and
   STAND_UP frame 134; (iii) re-place STANCE with the heels down; (iv) then train M1/M4 against
   the remaining references, which are contact-consistent and have 0.26–0.98 s holdable windows.
6. **A per-phase criterion is available for M5 scoring:** the holdable windows in §3.3 are the
   frames where a pose-level success check is meaningful; outside them the metric must be
   dynamic (CoM velocity, step placement, opponent contact), otherwise a correct dynamic
   execution will be scored as failure.

---

## 7. Limits and what this audit does not say

* Kinematic replay describes **reference geometry**, not a physics roll-out; no joint tracking
  error, contact compliance, or actuator saturation enters the classes. It is a *necessary*
  condition test for quasi-static feasibility, not a prediction of the PD replay.
* The contact rule is a teacher-style **height threshold**, not the collision solver; a sole site
  below the mat counts as touching, which is why penetration is reported separately and why the
  SPRAWL seam frames are excluded from the "good news" (their support region is where a foot has
  been pushed through the floor).
* Both torque models are **static**: no dynamics, no external grappling load, no torso/arm
  inertia, no slip, no actuator rates. The min-max LP assumes ideal, freely chosen load sharing;
  a real position-controlled pair will not realise it, so the ≤ 23.9 Nm figure is a *lower bound
  on what an ideal controller needs*, not a guarantee that a PD controller can supply it.
* Ground-phase frames (reference pelvis < 0.45 m) are governed by knees/hands/torso contact that
  this audit does not model, and the opponent is a second support path (§5.3). Both make the
  foot-only deficit an **upper bound on the true demand**; the standing subset (§3, `inf/stand%`)
  is the cleaner number.
* "Infeasible" is a statement about a *single instant held still*; a frame can be perfectly fine
  as a transient. The holdable-window column, not the percentage, is the M4/M5-relevant summary.
* Thresholds (band 0.02 m, contact 0.045 m, float 0.05 m, penetration 0.05 m, seam limits from
  the retimer, verdict rules) are all constants in the script and echoed in the JSON `meta`, so
  any of them can be re-run with different values; the class counts are insensitive to the band
  (§1.5) and the verdicts are identical for a penetration threshold anywhere in 0.03–0.06 m
  (checked at 0.03/0.04/0.05/0.06 m: same verdict and action for all 7 techniques).

## 8. Files and how to re-run

| file | role |
|---|---|
| `scripts/audit_support_envelope.py` | the audit: kinematic replay, three support regions, min-max LP torque, integrity checks, verdicts, tables, JSON, figure |
| `data/support_envelope.json` | 96 KB of per-technique per-robot numbers: class counts (all/standing/ground/band sensitivity/site-hull/line-union), support states, worst lever/torque frames with times, episode tables (frame indices + times), integrity numbers, seam outliers, verdicts with evidence, npz hashes |
| `reports/2026-10-08/support_envelope.png` | the figure (§4) |
| `reports/2026-10-08/support_envelope.md` | this report |

```bash
.venv/bin/python scripts/audit_support_envelope.py    # ~30 s, prints tables, writes JSON + PNG
```

Nothing outside the four files above was modified (`data/refs`, `src/**` and `notes.md`
untouched; the orchestrator merges facts into the ledger). No video is produced: the deliverable
is analytical (tables + a figure) and no motion is rendered, so the VISUAL-evidence rule for
motion deliverables does not apply.

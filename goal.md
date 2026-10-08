# GOAL.md — Project Northstar

Full operator brief preserved verbatim in `docs/MISSION.md` — authoritative for
intent. This file is the working distillation of it; on conflict MISSION.md wins.

Learned humanoid wrestling: two Unitree G1 robots in MuJoCo wrestle a full standing
match. Primary artifact: `final_wrestling_match.mp4`.

## Strong definition of done

Two learned G1 humanoids begin each exchange standing and autonomously wrestle.
They remain upright, manage distance, enter/leave attack range, execute multiple
GrappleMap-derived takedowns and defenses, adapt them under resistance, recover to
standing after nonterminal ground states (knees/hands/sprawl), lose an exchange when
clearly put on their back, reset, and continue until the match clock expires.
Behavior is chosen from match state, never a predetermined sequence.

## Methodology chain (core philosophy)

```
GrappleMap → selected wrestling movements → physically feasible G1 references
→ imitation/behavior cloning → wrestling motor vocabulary → resistance post-training
→ attack vs defense → free technique selection → historical self-play → full match
```

- GrappleMap teaches WHAT useful wrestling movement looks like (relationships, not mocap).
- Imitation/BC teaches the body HOW to perform it (teacher = PD/impedance controller).
- RL/PPO teaches it to SUCCEED under perturbation and resistance.
- Self-play teaches WHEN to attack, defend, retreat, recover, switch, re-engage.

Never ask RL to discover mechanics humans already know. Never literal-replay
schematic keyframes: an edge means start-relationship → key geometry → end-relationship.

## Phase map

| # | Phase | Exit criterion |
|---|-------|----------------|
| 0 | Scaffolding + environment | repo, docs, sim stack verified on ARM CPU |
| 1 | Assets | GrappleMap parser + tests; G1 MJCF stands under PD; vocab selected |
| 2 | Reference motions | valid G1 references (survive MuJoCo) for each vocab technique |
| 3 | Teacher + BC | technique-conditioned policy reproduces refs; perturb-and-correct |
| 4 | Wrestling env | back-to-mat detector, exchange loop, anti-stall rules |
| 5 | Resistance + A/D | PPO stages A–E; techniques work vs resisting opponent |
| 6 | Free wrestling | no technique command; tactical behavior; obs/action ablations |
| 7 | Self-play league | historical pool; match + technique evaluation |
| 8 | Final video + report | final_wrestling_match.mp4 + technical report + deliverables audit |

## Hard rules

- Back-to-mat = exchange loss (persistent dorsal contact + torso orientation; NOT one collision bit).
- Knees/hands/sprawl are NOT losses; recovery to standing must remain possible.
- Game structure itself rewards staying off the back; minimal shaping rewards.
- Prevent run-away-forever: bounded area, exchange clock, light engagement pressure.
- Critic estimates return (may use privileged state). Technique-quality judge is a
  SEPARATE component (technique scorer), never conflated.
- Failed shots must not terminate; attacker recovers or is sprawled on, continues.
- Simplicity rule at every decision: smallest system demonstrating the next missing
  behavior. Complexity requires evidence (ablation), never aesthetics.
- Hardware: ARM AArch64, ~8 CPU cores, 16–24 GB RAM, NO GPU. Lightweight, parallel
  CPU sim, small nets, cached preprocessing, resumable experiments.
- Scope stops at back-to-mat. No continued ground BJJ.

## Deliverables (23)

1 working repo; 2 ARM setup instructions; 3 GrappleMap parser; 4 wrestling curriculum;
5 G1 reference motions; 6 imitation/BC pipeline; 7 technique-validity evaluation;
8 resistance/PPO pipeline; 9 standing recovery behavior; 10 takedown/back-contact
detector; 11 free-wrestling environment; 12 historical self-play; 13 final policy
architecture + justification; 14 exact observation spec; 15 exact action spec;
16 trained checkpoints; 17 experiment ledger; 18 technique evaluation; 19 match
evaluation; 20 failure videos; 21 technique/drilling videos; 22 final technical
report; 23 final_wrestling_match.mp4.

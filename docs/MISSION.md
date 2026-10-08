# MISSION.md — Full Operator Brief (verbatim record, 2026-10-07)

> **STATUS NOTE (2026-10-08, RepoTrim) — the ACTIVE milestone is `docs/SOLO_DRILL.md`.**
> The two-robot wrestling pipeline described below is **superseded for the current milestone**
> and was approved by the operator for deletion. It is *scheduled* for removal, not yet removed:
> the two-robot code is currently an import-level prerequisite of the active path
> (`src/rl/net.py` imports `wrestling.env`; `src/solo/fall.py` imports `wrestling.backdet` for the
> back-to-mat termination of SOLO_DRILL §4), and `src/rl/{vec,trainer,rollout}.py` are held as
> pattern-source until the solo vec backend lands. **When it is removed it will live only in git
> history, recoverable with `git show 72956f3:<path>`** (`72956f3` is the ref immediately before the
> 2026-10-08 trim; the first trim commit is `46fb84f`). The ordered excision plan, the deleted-file
> manifest and the kept-off-mandate list are in `reports/2026-10-08/repo_trim.md`. Read this file as
> the long-term intent record; read `docs/SOLO_DRILL.md` for what is being built now.

Transport artifacts (doubled line-break corruption) repaired; wording otherwise unchanged.
This file is the AUTHORITATIVE record of intent. `goal.md` is the working distillation;
if they ever conflict on detail, this file wins and `goal.md` gets updated.

---

# Mission

Build a complete learned humanoid wrestling system in MuJoCo using two Unitree G1-like robots.

The final goal is an uninterrupted simulated wrestling match in which both robots:

- begin standing in wrestling stance;
- attempt to remain on their feet;
- move, circle, close distance, retreat, and re-engage autonomously;
- recognize when they are at a useful range to attack;
- attempt recognizable wrestling takedowns;
- defend takedowns;
- recover from failed shots;
- recover from knees/hands/other nonterminal ground states back to standing;
- choose between multiple learned attacks and defenses;
- adapt learned techniques to opponent movement and resistance;
- lose an exchange when their back is taken to the mat;
- reset to standing after each completed exchange;
- continue until the match clock expires.

The primary artifact is:

`final_wrestling_match.mp4`

The final behavior must be learned, reactive, and visibly resemble competent wrestling.

Do not solve the entire ground-BJJ problem yet.

This project deliberately stops at the point where a wrestler's back reaching the mat ends the exchange.

The purpose is to prove the complete methodology on standing wrestling before extending the same system into continued ground grappling.

---

# Central training philosophy

Do not ask reinforcement learning to discover wrestling mechanics that humans already know.

Use:

**GrappleMap to teach what useful wrestling movements look like.**

Use:

**imitation learning / behavior cloning / motion tracking to teach the body how to perform them.**

Use:

**reinforcement learning to make those movements work against perturbation and resistance.**

Use:

**self-play to teach when to attack, defend, retreat, recover, switch techniques, and re-engage.**

The conceptual progression is:

```text
GrappleMap
    ↓
selected wrestling movements
    ↓
physically feasible G1 reference motions
    ↓
imitation / behavior learning
    ↓
wrestling motor vocabulary
    ↓
resistance post-training
    ↓
attack vs defense
    ↓
free technique selection
    ↓
historical self-play
    ↓
full wrestling match
```

This is the core methodology.

Keep the implementation as simple as possible.

---

# Hardware

You are running autonomously on an Oracle VPS.

Assume:

- Linux;
- ARM / AArch64;
- CPU-only;
- roughly 8 CPU cores;
- approximately 16 to 24 GB RAM;
- no usable GPU.

Training time is relatively abundant.

Compute is not.

Prefer:

- MuJoCo;
- lightweight ARM-compatible libraries;
- small neural networks;
- parallel CPU simulation;
- cached preprocessing;
- resumable experiments.

Do not spend substantial effort forcing GPU-first frameworks onto this machine.

---

# Autonomous operation

Own the project end to end.

You may spawn subagents and dynamic workflows.

Use strong reasoning models for:

- architectural decisions;
- interpreting experiments;
- debugging difficult failures;
- deciding whether to pivot.

Use fast subagents for:

- repository exploration;
- implementation tasks;
- benchmark scripts;
- testing;
- log analysis;
- candidate technique searches;
- small ablations.

Do not ask the user routine implementation questions.

If something fails:

1. identify the failing layer;
2. design the smallest useful test;
3. collect evidence;
4. change only what the evidence supports;
5. continue.

Avoid both blind persistence and constant architectural redesign.

---

# Source knowledge: GrappleMap

Use:

https://github.com/Eelis/GrappleMap

Inspect the repository directly.

GrappleMap is a public-domain graph of grappling positions and transitions.

It provides:

- two-person poses;
- transition sequences;
- sparse intermediate keyframes;
- position names;
- transition names;
- tags;
- top/bottom mover metadata;
- references;
- mirroring;
- player swaps;
- graph connectivity.

Its pose representation contains 23 3D landmarks per human.

It contains substantial wrestling-relevant material including takedowns, shots, sprawls, standing positions, body locks, single legs, double legs, stand-ups, and related transitions.

Verify exact contents directly.

---

# Important GrappleMap limitation

GrappleMap is schematic.

It is not motion capture.

Its frames do not provide:

- accurate real-world timing;
- torques;
- actions;
- forces;
- detailed contact dynamics;
- reliable hand fighting.

Therefore interpret an edge roughly as:

```text
starting relationship
        ↓
important intermediate geometry
        ↓
ending relationship
```

Then find a physically executable G1 realization of that transition.

Do not attempt literal keyframe replay.

---

# Scope of the initial wrestling vocabulary

Select a small but useful vocabulary.

Possible attack families:

- double leg;
- single leg;
- body lock takedown.

Possible defensive behaviors:

- hips back;
- sprawl;
- widen base;
- angle away;
- recover stance.

Possible general behaviors:

- wrestling stance;
- approach;
- retreat;
- circle;
- close distance;
- recover from failed shot;
- return from knees/hands to standing;
- disengage and re-enter.

Do not maximize the number of techniques.

The goal is enough diversity for meaningful wrestling.

Five reliable movements are better than thirty poor ones.

---

# Phase 1: compile a GrappleMap movement to G1

For each selected technique:

```text
GrappleMap frames
      ↓
map human landmarks to G1 sites
      ↓
IK / optimization
      ↓
physically plausible G1 poses
      ↓
smooth trajectory
      ↓
retime
      ↓
simulate
      ↓
repair
      ↓
valid reference
```

Do this jointly for both grapplers when necessary.

---

# Preserve wrestling geometry

Prioritize functional relationships such as:

- attacker pelvis relative to opponent pelvis;
- torso orientation;
- attack depth;
- base width;
- shoulder-to-hip relationship;
- attacker head position;
- relative COM;
- leg relationship;
- relevant arm/leg proximity.

Exact Cartesian reproduction of every human landmark is not the objective.

The G1 has different morphology.

Preserve the mechanism.

---

# Physics repair

Reference trajectories must survive MuJoCo.

Check:

- joint limits;
- impossible velocities;
- excessive accelerations;
- ground penetration;
- robot-robot penetration;
- self-collision;
- instability;
- impossible support configuration;
- actuator saturation.

Modify timing and geometry when needed.

The reference only needs to remain recognizably the same wrestling movement.

---

# Phase 2: teach the movement

Before adversarial reinforcement learning, teach the controller how the movement should occur.

Preferred progression:

1. simple reference tracking;
2. behavior cloning;
3. perturb-and-correct BC;
4. automated DAgger;
5. imitation RL only if required.

Do not automatically use PPO for this stage.

---

# Teacher generation

GrappleMap contains desired states, not robot actions.

Generate action labels using an algorithmic controller.

For example:

```text
current robot state
+
desired robot reference
      ↓
PD / impedance / trajectory optimizer
      ↓
teacher action
```

Collect:

```text
state + reference + opponent state
→ teacher joint target
```

Train from that dataset.

---

# Do not train only on the perfect trajectory

Perturb training states.

Examples:

- joint noise;
- root orientation error;
- velocity error;
- phase error;
- opponent position variation;
- starting distance variation;
- stance variation.

Train the controller to correct back toward successful execution.

This is the difference between a feedback policy and an animation player.

---

# Technique-conditioned controller

Initially tell the policy explicitly which technique it is performing.

For example:

```text
DOUBLE_LEG
SINGLE_LEG
BODY_LOCK
SPRAWL
RECOVER
STANCE
```

A candidate policy interface during the teaching stage is:

```text
self state
+
opponent state
+
technique / reference
+
phase
      ↓
policy
      ↓
joint position targets
```

Try a single shared network first.

Allow specialists if they clearly work better.

Do not force architectural elegance.

---

# Technique-validity supervision

During imitation and early post-training, maintain a measure of whether a commanded technique still resembles its intended movement family.

This is separate from the PPO critic.

Possible technique-validity inputs:

- relative body-site geometry;
- phase;
- key joint configurations;
- relevant contact relationships;
- root orientation;
- progression through important reference states.

Possible implementations include:

- distance to the reference trajectory;
- distance to a technique manifold;
- a classifier;
- a small discriminator;
- hand-engineered relational geometry for the first version.

Its purpose is to prevent the following failure:

```text
reward says "double leg"
but policy discovers
"run shoulder-first into opponent"
```

and calls that a double leg.

However, do not make technique conformity rigid.

As resistance increases, the policy must be allowed to adapt.

Gradually reduce the importance of exact imitation.

---

# Important distinction: critic versus technique judge

The actor-critic **critic** should estimate expected return:

$$V(s)$$

or an equivalent value function.

It may use privileged simulator state during training.

It should not be confused with a motion-quality classifier.

If you need a component that asks:

> "Does this still look like a legitimate double leg?"

build or calculate that separately.

Call it something such as:

- technique scorer;
- motion prior;
- reference score;
- technique discriminator.

This distinction should remain clear in the implementation.

---

# Phase 3: wrestling-specific game rules

The first complete game should deliberately reward staying off the back.

An exchange begins:

```text
both robots standing
```

and ends when one wrestler's back contacts the mat in a sufficiently clear and persistent way.

That wrestler loses the exchange.

Then:

```text
score
↓
reset both standing
↓
next exchange
```

---

# Back-to-mat terminal rule

Implement a robust dorsal-back contact detector.

Do not use one noisy collision bit.

Use some combination of:

- designated back/torso geom contact with mat;
- torso orientation;
- contact persistence for a short confirmation period;
- possibly pelvis/torso height.

Determine thresholds experimentally.

The intended semantic rule is:

> **Once a wrestler is clearly put onto their back, that wrestler loses the exchange.**

This is a deliberately simplified wrestling game rule.

It does not need to replicate one exact real-world wrestling ruleset.

---

# Ground states that are NOT automatically losses

A wrestler may temporarily be:

- on one or both knees;
- on hands and knees;
- posting with hands;
- partially crouched;
- sprawled;
- recovering from a shot;
- in another non-back-exposed ground configuration.

In these states the policy should be allowed to continue.

It may attempt to:

```text
base
↓
recover posture
↓
stand back up
↓
resume wrestling
```

Do not reset merely because a knee or hand touches the floor.

---

# Natural incentive to remain upright

The game structure itself should provide a strong incentive:

```text
BACK CONTACT
= lose exchange
```

Therefore the policy should naturally learn:

- stance;
- balance;
- takedown defense;
- shot recovery;
- standing recovery;
- avoiding bad rotations.

Avoid drowning this natural objective in dozens of shaping rewards.

---

# Prevent the trivial "run away forever" strategy

If the only objective is:

> don't get put on your back

both robots may learn to avoid engagement.

Monitor for this.

Use the simplest anti-stalling mechanism necessary.

Possible mechanisms:

- bounded competition area;
- penalty for leaving the area;
- limited exchange clock;
- draw/no-positive-reward if neither wrestler attacks;
- light activity/engagement incentive;
- tournament scoring that rewards takedowns rather than merely surviving.

Do not aggressively reward moving toward the opponent every frame.

That can create equally bad behavior.

The desired strategy is:

```text
maintain safe stance
+
manage distance
+
seek useful opportunity
+
commit when appropriate
```

---

# Phase 4: PISTY-style progression

Use dense guidance early and progressively remove it.

---

## Stage A: imitation

Command:

```text
DOUBLE_LEG
```

Question:

> Can you reproduce the movement?

Partner is cooperative.

Strong reference reward is acceptable.

---

## Stage B: randomized drilling

Still command:

```text
DOUBLE_LEG
```

but randomize:

- distance;
- angle;
- stance;
- timing;
- opponent pose.

Question:

> Can you adapt the same movement?

---

## Stage C: dynamic drilling partner

Opponent is a physically simulated controller.

It responds naturally to forces but does not actively defeat the move.

Question:

> Can the learned technique work against an actual moving body?

---

## Stage D: resistance

Opponent now:

- backs up;
- widens stance;
- moves hips;
- changes angle;
- partially sprawls.

Technique command remains fixed.

Question:

> Can the double leg adapt and still succeed?

This is where PPO or another RL method becomes increasingly useful.

---

## Stage E: attack versus learned defense

Example:

```text
attacker:
DOUBLE_LEG

defender:
SPRAWL
```

Both sides possess movement priors.

Now train competitively.

Attacker learns:

- shot timing;
- entry depth;
- angle;
- drive;
- continuation;
- recovery.

Defender learns:

- timing;
- base;
- hip positioning;
- sprawl direction;
- recovery.

---

# Reward evolution

Early:

```text
technique similarity
+
progress
+
outcome
```

Middle:

```text
reduced technique similarity
+
strong outcome
```

Late:

```text
win / loss of exchange
+
minimal anti-exploit shaping
```

Do not preserve exact imitation forever.

The eventual wrestler is allowed to develop effective variations.

---

# Failed shot behavior matters

A bad shot should not automatically terminate the exchange.

The attacker may end up:

- on knees;
- hands posted;
- underneath but not on the back;
- partially sprawled on.

The policy should learn:

```text
failed attack
↓
base
↓
recover
↓
stand
↓
re-engage
```

This behavior is important.

A wrestler who can only attack successfully or collapse is not competent.

---

# Phase 5: free wrestling

Once attacks and defenses work, remove explicit technique commands.

Previously:

```text
goal = DOUBLE_LEG
```

Now:

```text
goal = WIN_EXCHANGE
```

The controller must decide:

- whether to approach;
- whether to maintain range;
- whether to retreat;
- whether to circle;
- when to shoot;
- which attack to use;
- when to sprawl;
- when to recover;
- when to disengage;
- when to re-enter.

This is where the project becomes actual wrestling.

---

# END-STAGE POLICY DESIGN IS IMPORTANT

Do not prematurely lock the final neural-network interface.

Once the individual skills work, explicitly research the best observation and action representation for free wrestling.

The final policy must solve more than technique execution.

It must solve:

```text
distance management
stance management
opportunity recognition
attack timing
attack selection
defense
recovery
re-engagement
```

Treat this as a serious experimental question.

---

# Candidate final observations

A good baseline should contain enough information to reason about wrestling while remaining compact.

## Self proprioception

Consider:

- root orientation relative to gravity;
- root height;
- root linear velocity;
- root angular velocity;
- joint positions;
- joint velocities;
- foot contacts;
- knee/hand contacts;
- important body-site positions;
- whether own back is approaching the mat.

## Opponent state

Express primarily in self-relative coordinates.

Consider:

- opponent pelvis position;
- opponent pelvis velocity;
- torso orientation;
- torso velocity;
- shoulder locations;
- hips;
- knees;
- feet;
- elbows/hands where useful;
- opponent support/contact state;
- approximate stance width;
- relative heading;
- distance.

The policy should be able to infer things such as:

```text
too far to shoot
good entry range
opponent moving backward
opponent overcommitted forward
opponent beginning a shot
opponent hips available
danger of being taken down
```

---

# Do not over-observe

Do not automatically feed every MuJoCo field to the actor.

Run ablations.

Important information should remain.

Redundant privileged information should not.

A privileged critic may still receive richer simulator state.

---

# Temporal information

Some wrestling decisions depend on motion, not pose alone.

Examples:

- opponent is stepping toward me;
- opponent is retreating;
- opponent just changed level;
- opponent is beginning a shot.

Joint/root velocities may already provide enough information.

If not, test:

1. small frame stack;
2. small GRU.

Do not jump directly to large recurrent or transformer architectures.

---

# Final action-space question

Explicitly test what the final policy should output.

There are several plausible designs.

## Option 1: direct motor control

```text
wrestling observation
      ↓
one policy
      ↓
joint position targets
```

This is the simplest conceptual architecture.

It may allow completely fluid behavior.

Test it first if the imitation-trained controller provides a strong initialization.

## Option 2: residual motor control

```text
current learned/reference behavior
+
policy residuals
```

Useful if the learned techniques remain strong but require adaptation.

## Option 3: skill selection plus motor execution

High-level outputs something like:

```text
STANCE
APPROACH
RETREAT
CIRCLE_LEFT
CIRCLE_RIGHT
DOUBLE_LEG
SINGLE_LEG
BODY_LOCK
SPRAWL
RECOVER
```

A learned motor controller executes the selected skill.

This may make tactical learning easier.

Do not adopt it automatically.

Use it if direct control struggles with long-horizon decision-making.

## Option 4: one network with multiple heads

A compromise worth testing:

```text
shared observation encoder
       ↓
    ┌──┴───────────┐
    ↓              ↓
intention head   motor head
```

The intention head may represent:

- skill;
- motion phase;
- tactical objective.

The motor head outputs joint targets conditioned on that intention.

This keeps most learning inside one model while providing useful structure.

Again, use evidence.

---

# End-stage behavior we want to see

The final controller should learn states such as:

```text
too far
→ close distance
```

then:

```text
good wrestling range
→ maintain stance / circle
```

then:

```text
opponent presents opportunity
→ shoot
```

or:

```text
opponent posture unfavorable
→ back off / change angle
```

then:

```text
opponent attacks
→ defend / sprawl
```

then:

```text
shot fails
→ recover to feet
```

This is much more important than simply having a menu of techniques.

---

# Technique confines during late training

A technique-validity scorer can remain useful during post-training.

For example, while explicitly drilling:

```text
DOUBLE_LEG
```

it can detect whether the policy has drifted completely outside double-leg-like movement.

However, during free wrestling, do not overconstrain the policy to textbook technique boundaries.

Real useful behavior may involve:

- aborted shots;
- blended attacks;
- partial entries;
- transitions between single and double;
- unusual robot-specific finishes.

The final criterion should become:

```text
valid wrestling
+
competitive effectiveness
+
no simulator exploits
```

rather than:

```text
perfect textbook imitation
```

---

# PPO critic

Use the critic conventionally to estimate future return.

The critic may use privileged state such as:

- complete robot states;
- precise contact information;
- exact opponent velocities;
- distance to back-exposure terminal state.

This can improve training.

The actor need not receive every privileged signal.

---

# Phase 6: historical self-play

Once free wrestling works at all, maintain opponent diversity.

Use:

- current checkpoint;
- recent checkpoints;
- older checkpoints;
- specialist attackers;
- specialist defenders;
- possibly independent seeds.

Do not let both agents co-adapt exclusively to each other.

---

# Full match format

A simple match can use:

```text
3 simulated minutes
```

or another sensible duration.

Each exchange:

```text
standing start
↓
free wrestling
↓
one wrestler's back reaches mat
↓
other wrestler scores
↓
standing reset
↓
continue
```

Possible score:

```text
1 point per won exchange
```

or:

```text
2 points per takedown
```

The exact number is less important than consistent incentives.

---

# Simultaneous/ambiguous falls

Handle edge cases.

If both backs contact the mat almost simultaneously:

- examine which occurred first;
- consider who caused/control-led the transition;
- or declare no score and reset if ambiguity is too high.

Prefer a robust rule over a complicated one.

Record ambiguous events for inspection.

---

# Evaluation

Do not evaluate only final score.

Track:

## Movement

- time spent standing;
- stance stability;
- recovery-to-standing success;
- accidental fall rate.

## Attacking

- attack attempts;
- attacks by type;
- successful takedowns;
- failed-shot recovery rate.

## Defense

- sprawl/defense attempts;
- successful defenses;
- recoveries after partial takedowns.

## Tactics

- average engagement distance;
- excessive retreat frequency;
- technique diversity;
- attack timing distribution;
- re-engagement after failed exchange.

## Competition

- exchange win rate;
- match win rate;
- historical-opponent performance.

---

# Technique-specific evaluation

Continue testing:

```text
double leg
single leg
body lock
sprawl
recovery
```

independently.

Free self-play must not silently destroy them.

---

# Ablations

At minimum consider:

## Imitation value

Compare:

```text
random-init PPO
```

versus:

```text
GrappleMap imitation
→ PPO
```

Measure:

- samples to first successful takedown;
- final success;
- visual quality.

## Technique prior value

Compare late resistance training:

```text
no technique constraint
```

versus:

```text
light technique-validity guidance
```

Determine whether it prevents degenerate attacks.

## Final action representation

If practical compare:

```text
direct motor policy
```

against:

```text
explicit skill-selection architecture
```

This is especially relevant to the final-stage question of learning good wrestling behavior.

---

# Simplicity rule

At every decision ask:

**What is the simplest system that demonstrates the next missing behavior?**

If BC works:

do not add imitation PPO.

If direct control works:

do not add hierarchy.

If current observations allow good range management:

do not add recurrence.

If a simple technique-distance score prevents drift:

do not train a discriminator.

If self-play works with a small opponent pool:

do not build a complex league.

Complexity requires evidence.

---

# Outcome engineering

At all times identify the visible blocker.

Examples:

```text
robots cannot reproduce double leg
→ fix imitation/reference
```

```text
double leg works only from exact position
→ perturb-and-correct
```

```text
double leg works on passive opponent but not moving one
→ resistance training
```

```text
attacker falls onto own back
→ recovery/control objective
```

```text
robots simply retreat forever
→ fix game incentives
```

```text
techniques work but attacks happen from absurd distance
→ fix observations/tactical learning
```

```text
robots attack constantly without setting up
→ improve outcome incentives/self-play
```

```text
robots know attacks but choose poorly
→ investigate final policy/action representation
```

Always work on the current bottleneck.

---

# Expected emergent end-stage behavior

The ideal final policy is not simply an animation selector.

It should begin to discover wrestling concepts such as:

- maintain base;
- manage distance;
- enter and leave attack range;
- circle;
- wait for an opportunity;
- pressure forward;
- retreat when vulnerable;
- shoot when geometry is favorable;
- defend immediately when opponent changes level;
- recover after a stuffed shot;
- re-engage instead of remaining on knees;
- vary attacks against different defensive reactions.

Do not hard-code those sequences.

Create the state, skills, objective, and curriculum from which they can be learned.

---

# Final video

`final_wrestling_match.mp4` should visibly contain things like:

```text
stance
→ movement
→ range management
→ attack attempt
→ defense
→ failed-shot recovery
→ both regain feet
→ re-engagement
→ different attack
→ one wrestler put onto back
→ score
→ reset
```

The exact sequence should emerge from the policies.

Do not prearrange it.

---

# Required deliverables

Produce:

1. working repository;
2. ARM setup instructions;
3. GrappleMap parser;
4. selected wrestling curriculum;
5. G1 reference motions;
6. imitation/BC pipeline;
7. technique-validity evaluation;
8. resistance/PPO pipeline;
9. standing recovery behavior;
10. takedown/back-contact detector;
11. free-wrestling training environment;
12. historical self-play;
13. final policy architecture and justification;
14. exact observation specification;
15. exact action specification;
16. trained checkpoints;
17. experiment ledger;
18. technique evaluation;
19. match evaluation;
20. failure videos;
21. technique/drilling videos;
22. final technical report;
23. `final_wrestling_match.mp4`.

---

# Strong definition of done

The strongest success condition is:

**Two learned Unitree G1 humanoids begin each exchange standing and autonomously wrestle. They actively attempt to remain upright, manage distance, enter and leave attack range, execute multiple GrappleMap-derived takedowns and defenses, adapt those movements against resistance, recover to standing after unsuccessful ground exchanges when their backs have not touched the mat, and lose an exchange when clearly put onto their backs. They choose their behavior from the evolving state of the match rather than following a predetermined sequence.**

The progression should be experimentally visible:

```text
GrappleMap
↓
imitation
↓
recognizable wrestling movements
↓
resistance
↓
robust techniques
↓
range management and technique choice
↓
self-play
↓
wrestling
```

Use imitation to teach **what the movement is**.

Use RL to teach **how to make it succeed or survive resistance**.

Use self-play to teach **when to attack, defend, retreat, recover, and re-engage**.

Keep everything else subordinate to that objective.

---

# Operator addendum: orchestration directives

- "use deepseek mimo and zai models only from zai provider using go and command code for
  deepseek and mimo, if space bunny is free on command code you can use it but default to
  those. 5.3 flash is good. use subagents liberally. keep a notes.md as orchestrator
  knowledge, do not write code yourself."
- Refined: "use task, and workflows for the most part … don't use delegate."

Operating meaning: orchestrator (main assistant) writes docs/briefs/ledgers/workflow
scripts only; ALL project code is produced by `task` subagents and `workflow` agent()
fan-outs (zai-family models). `delegate` (external harnesses) is NOT used.

---

# Engineering and Research Operating Standard

These instructions govern **how you approach the mission**, not what the mission is.

Treat this project as an engineering research program whose outputs must survive skeptical review.

The objective is not to produce a large amount of code or to report apparent progress.

The objective is to build a system whose behavior is real, reproducible, measurable, understandable, and progressively closer to the stated outcome.

---

## 1. Think from the outcome backward

At every point, maintain a clear chain:

```text
FINAL OUTCOME
    ↓
required observable capabilities
    ↓
current missing capability
    ↓
hypothesis about why it is missing
    ↓
smallest experiment that tests that hypothesis
    ↓
evidence
    ↓
next decision
```

Do not work merely because a task sounds useful.

Before substantial work, ask:

> If this succeeds, what specific uncertainty or bottleneck does it remove?

If there is no clear answer, reconsider the task.

---

## 2. Separate facts, hypotheses, and decisions

Do not blur them together.

Maintain the distinction:

```text
FACT:
Measured or directly verified.

HYPOTHESIS:
Our current explanation.

DECISION:
What we will do next based on current evidence.
```

Example:

```text
FACT:
The double-leg reference falls in 78% of rollouts.

FACT:
Most failures begin when the attacker's rear foot loses contact.

HYPOTHESIS:
The retargeted human timing shifts the COM too quickly for the G1.

TEST:
Slow phases 0.30–0.55 by 20% without changing pose geometry.

RESULT:
Failure drops from 78% to 31%.

CONCLUSION:
Timing is an important contributor, though not necessarily the only one.
```

This is the expected reasoning standard.

Do not write:

> "The issue was balance."

unless you actually established that.

---

## 3. Never confuse plausibility with verification

Something making intuitive sense is not evidence that it works.

For every important assumption, determine whether it is:

```text
known from source/code
measured experimentally
strongly inferred
weakly inferred
unknown
```

Prefer direct inspection and experiments.

Examples:

Do not assume:

- a library supports ARM because it probably should;
- a controller is stable because the loss decreases;
- a technique is robust because one video looked good;
- a reward reflects wrestling quality because return increased;
- a pretrained checkpoint matches the exact G1 model;
- two coordinate frames use the same convention;
- an implementation matches a paper because the README claims it does.

Verify.

---

## 4. Use a hypothesis-driven loop

For substantial problems, use:

```text
OBSERVATION
↓
HYPOTHESIS
↓
PREDICTION
↓
EXPERIMENT
↓
MEASUREMENT
↓
ACCEPT / REJECT / REFINE
```

The **prediction** step matters.

Before running an experiment, state what should happen if the hypothesis is correct.

Example:

```
Observation:
BC performs the motion perfectly from the nominal state
but fails after 5–10 cm root displacement.

Hypothesis:
The dataset has insufficient off-reference states.

Prediction:
Adding perturbed states with teacher corrections should improve
perturbed success substantially without requiring more model capacity.

Experiment:
Train the same architecture on perturb-and-correct data.

Measure:
Nominal success
5 cm displacement success
10 cm displacement success
angle perturbation success
```

This prevents random tweaking.

---

## 5. Prefer the simplest explanation first

When something fails, investigate likely causes in an efficient order.

For example:

```text
bad data
↓
wrong coordinate transform
↓
incorrect reset/state initialization
↓
invalid physics/contact
↓
observation bug
↓
action scaling bug
↓
reward bug
↓
optimizer/training issue
↓
insufficient model capacity
↓
fundamental algorithm limitation
```

Do not respond to every failure by increasing network size or introducing a new learning algorithm.

Many apparent "AI problems" are ordinary engineering bugs.

---

## 6. Verify interfaces before optimizing internals

Most complex-system failures happen at interfaces.

Explicitly verify:

```text
GrappleMap coordinates
        ↓
retargeting

retargeted motion
        ↓
MuJoCo model

reference trajectory
        ↓
controller

controller
        ↓
actuator interpretation

simulation state
        ↓
policy observation

policy output
        ↓
action scaling

environment
        ↓
reward

reward
        ↓
training algorithm
```

For every interface, know:

- coordinate frame;
- units;
- ordering;
- dimensions;
- sign convention;
- normalization;
- sampling rate;
- expected valid range;
- ownership;
- failure behavior.

Whenever possible, encode these in types, assertions, tests, or explicit metadata.

---

## 7. Make units explicit

Never rely on implicit units.

Use and document:

```text
meters
radians
seconds
radians/sec
meters/sec
Newton-meters
Hz
```

Be particularly suspicious of:

```text
degrees vs radians
world vs local coordinates
XYZ vs Z-up/Y-up conventions
left-handed vs right-handed frames
absolute vs relative positions
simulation step vs policy step
physics steps vs environment transitions
```

Unit and frame errors can produce believable but completely incorrect behavior.

---

## 8. Build executable invariants

Important assumptions should become automated checks.

Examples:

```text
all joint targets within valid limits
reference timestamps strictly increasing
no NaN/Inf in observations
expected observation dimension fixed
action dimension matches actuator dimension
quaternions normalized
mirroring twice returns original motion approximately
player swap twice returns original approximately
simulation reset produces valid state
reference frame zero matches intended initial configuration
```

Fail loudly.

Do not silently clip or repair serious errors unless the repair is intentional and logged.

---

## 9. Test transformations independently

Any transformation that changes representation should have dedicated tests.

Examples:

```text
GrappleMap decode
mirror
player swap
canonicalization
world → pair frame
pair frame → world
human landmark → G1 [brief ends here]
```

---

*End of operator brief.*

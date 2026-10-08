export const meta = {
  name: 'wave1_assets',
  description: 'Phase 0/1 bring-up: sim environment install, GrappleMap parser, Unitree G1 model',
};

const COMMON = [
  'You are a subagent working in the repo /home/ubuntu/grappling (this is your cwd).',
  'FIRST read AGENTS.md, goal.md, and notes.md in the repo root.',
  'Hardware FACTS: 4-core ARM aarch64 (Neoverse-N1), 23GB RAM, NO GPU, passwordless sudo,',
  'ffmpeg + git + cmake installed, python3.12 (system numpy/scipy present, but use the project venv),',
  'network access OK. Repo is otherwise empty besides docs and workflows.',
  'Rules: do not modify goal.md, notes.md (except appending FACTS at the very end is allowed),',
  'or anything outside your listed deliverables. Keep everything lightweight; no long-running jobs.',
  'Create directories as needed. Your final message: concise list of FACTS + deliverable paths +',
  'anything blocked. Also write the full report to reports/2026-10-07/<your-name>.md.',
].join('\n');

export default async ({ agent, parallel, log }) => {
  log('Stage 1: EnvSetup — blocking prerequisite (installs mujoco/torch into .venv)');

  await agent(
    COMMON + '\n\n' + [
      'TASK: Bring up the CPU-only ARM Python simulation stack for this project.',
      '',
      'Steps:',
      '1. python3 -m venv .venv; source .venv/bin/activate; upgrade pip.',
      '2. Install: mujoco, numpy, scipy, pytest, imageio, imageio-ffmpeg, matplotlib.',
      '3. Install CPU torch for aarch64: pip install torch --index-url https://download.pytorch.org/whl/cpu',
      '   (verify: python -c "import torch; print(torch.__version__, torch.cpu.is_available())").',
      '   If that index fails on this platform, try the default PyPI wheel (also CPU on aarch64) and',
      '   record exactly what worked.',
      '4. Headless MuJoCo rendering: try MUJOCO_GL=egl first; if that fails install apt libs',
      '(sudo apt-get install -y libegl1 libosmesa6-dev libgl1) and try MUJOCO_GL=osmesa.',
      '5. Write scripts/smoke_env.py: prints mujoco/numpy/torch versions, builds a trivial MJCF',
      '(plane + small sphere free joint), steps 100 steps, and renders one offscreen PNG to',
      'reports/2026-10-07/smoke_env.png using the working GL backend. Must run green from repo root',
      'with the venv active.',
      '6. Write docs/SETUP.md: exact ARM setup commands from bare Ubuntu to running smoke_env.py,',
      'including which MUJOCO_GL backend to export.',
      '7. Run smoke_env.py yourself; paste its stdout into your report.',
      '',
      'Acceptance: smoke_env.py runs clean, PNG exists and is non-trivial (>5KB), torch imports CPU-only,',
      'SETUP.md is complete enough that a fresh machine reproduces it.',
      'If anything fails, record the exact error + what you tried in the report; do not fake success.',
    ].join('\n'),
    { name: 'EnvSetup' },
  );

  log('Stage 2 (parallel): GrappleMapParser + G1Model');

  const [gmap, g1] = await parallel([
    () => agent(
      COMMON + '\n\n' + [
        'TASK: vendor GrappleMap and build a verified parser for it.',
        '',
        'Steps:',
        '1. git clone --depth 1 https://github.com/Eelis/GrappleMap third_party/GrappleMap',
        '2. Inspect the repo (README, source) and determine EXACTLY: where the position/transition',
        'graph data lives, the pose representation (verify the claim "23 3D landmarks per player" —',
        'count them yourself), coordinate frame + units + which plane is the ground, how transitions',
        'store sparse keyframes, what tags/metadata exist (mover, player designations), and whether',
        'mirror / player-swap operations are defined in the source (find their exact semantics).',
        '3. Write src/grapplemap/__init__.py and src/grapplemap/parser.py (stdlib + numpy ONLY) that',
        'loads the full graph into Python: every node (position) with its two-player pose, every edge',
        '(transition) with frames, tags, mover metadata. Expose clean dataclasses or dicts + numpy',
        'arrays with documented shapes.',
        '4. Write tests/test_grapplemap.py (pytest): counts match the raw files, pose shapes are as',
        'documented, timestamps/frames monotonic, landmark count is exactly what you verified,',
        'mirror and player-swap roundtrips (transform T(T(x)) == x) if the repo defines them;',
        'skip-with-explanation any op the repo does not define. Tests must pass: run',
        '`python -m pytest tests/ -x -q` from repo root with .venv active.',
        '5. Export data/grapplemap/vocab_candidates.json: an inventory of wrestling-relevant edges —',
        'double leg, single leg, body lock, sprawl, stand-up from ground, wrestling stance/clinch',
        'entries, and anything adjacent (reshots, snaps, reattacks). For each: exact edge name,',
        'start node, end node, tags, frame count, mover. Curation criterion: standing→standing or',
        'standing→ground transitions relevant to takedown offense/defense/recovery.',
        '6. Write reports/2026-10-07/grapplemap.md with: data-format FACTS (frame, units, ground',
        'plane, landmark semantics, counts of nodes/edges), your proposed 5-8 technique vocabulary',
        'for this project mapped to specific edges (attack families: double leg / single leg /',
        'body lock; defense: sprawl / hips back; recovery: stand-up), and caveats about schematic timing.',
      ].join('\n'),
      { name: 'GrappleMapParser' },
    ),
    () => agent(
      COMMON + '\n\n' + [
        'TASK: obtain and characterize a standing Unitree G1 MJCF model.',
        '',
        'The venv now has mujoco installed (EnvSetup finished).',
        '',
        'Steps:',
        '1. git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie third_party/menagerie',
        '2. Copy the unitree_g1 robot (prefer the 23-dof variant if multiple exist; document what',
        'variants exist) into robots/g1/ keeping mesh paths working.',
        '3. Write scripts/smoke_g1.py (repo root, venv active) that:',
        '   - loads robots/g1 model, prints: nq, nv, nu, na, timestep, total mass, gravity compensation',
        '     torque estimate if cheap, full joint list (name, type, range in radians), actuator list',
        '     (name, joint, ctrlrange, gain/bias params), and whether keyframes are defined;',
        '   - picks a standing qpos (keyframe if present, else engineering a sensible one and',
        '     documenting it), places the robot on the ground;',
        '   - holds that pose for 5 s of sim time using the model actuators if they are position-',
        '     type (ctrl = initial qpos), else a simple joint-space PD computing torques from',
        '     (q0 - q, -dq) — either way clamp to ctrlrange/force limits;',
        '   - reports pelvis height and torso up-axis tilt at t=0/1/2.5/5 s so standing stability is',
        '     a measured FACT, not an assumption;',
        '   - renders front + side PNGs (reports/2026-10-07/g1_front.png, g1_side.png) using the GL',
        '     backend documented in docs/SETUP.md.',
        '4. Write reports/2026-10-07/g1_model.md: the full spec table above + stability result +',
        '   which variant chosen and why + any model edits you made (e.g. contact tweaks) with reasons.',
        '',
        'Acceptance: smoke_g1.py runs clean from repo root, PNGs exist, report contains the complete',
        'joint/actuator spec (this feeds Phase 2 retargeting), and a clear verdict: does G1 stand',
        'under plain PD for 5 s (yes/no + drift numbers).',
      ].join('\n'),
      { name: 'G1Model' },
    ),
  ]);

  log('Wave 1 complete');
  return { gmap_summary: String(gmap).slice(0, 2000), g1_summary: String(g1).slice(0, 2000) };
};

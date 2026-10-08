"""GrappleMap -> G1 reference retargeting (Phase 2, deliverable 5).

Pipeline (see reports/2026-10-07/retarget.md):
1. ``landmarks``   GrappleMap joint -> G1 landmark <site> mapping (attached at
                    load time via MjSpec; robots/g1/g1.xml is never edited).
2. ``gmframe``     edge-chain assembly into one Y-up world + rigid alignment.
3. ``world``       per-player scale, facing-+-x rotation, centroid at origin,
                    Y-up -> Z-up conversion.
4. ``solve``       per-frame joint solve (both players jointly) + 50 Hz
                    resampling with velocity/acceleration limits.
5. ``techniques``  the 7-technique recipes from docs/CURRICULUM.md.
6. ``scene``       two-G1 wrestling scene (MjSpec-composed).

Entry points: scripts/build_refs.py (npz), scripts/validate_refs.py
(validation + repair + videos).
"""

from .landmarks import SITE_SPECS, GM_JOINT_TO_SITE, WEIGHT_PRESETS, load_g1_spec
from .scene import build_scene_spec, load_scene_model, write_scene_xml
from .techniques import TECHNIQUES, build_technique_targets

__all__ = [
    "SITE_SPECS", "GM_JOINT_TO_SITE", "WEIGHT_PRESETS", "load_g1_spec",
    "build_scene_spec", "load_scene_model", "write_scene_xml",
    "TECHNIQUES", "build_technique_targets",
]

# ============================================
# HIPPOCAMPUS + AMYGDALA CROPS (input for gpu_hippo_cnn.py)
# ============================================
# The clearest TLE signal is in the medial temporal lobe (hippocampal
# volumes alone reach AUC 0.86 for left vs right), but Step 5's patches are
# a blind 32 mm grid. This cuts one box around each hippocampus + amygdala
# at full 1 mm detail instead.
#
# Uses Step 5's own alignment (rigid to MNI152 1 mm + symmetry midline), so
# the boxes sit at fixed MNI coordinates:
#   centre (+-26, -20, -14) mm, size 48 (x) x 64 (y) x 48 (z) mm
#   = hippocampus head to tail and the amygdala, with a margin for
#     individual anatomy (alignment is rigid, not warped).
# The RIGHT box is mirrored along x, so both boxes run lateral -> medial and
# one shared CNN can compare them.
#
# Output: RUN_DIR/hippo_crops/<sub>.npy  float16 (2 sides [L, R-mirrored],
#         2 channels [T1, FLAIR], 48, 64, 48)   + a few QC pictures
# Run:    python hippo_crops.py [n_parallel]          (all subjects in subject_list.txt)
#         TLE_BOX=temporal python hippo_crops.py ...   (whole temporal lobe -> temporal_crops/)

import os
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")    # parallel subjects, 1 thread each

from pipeline_config import RUN_DIR, load_subject_list
import step5_patch_extraction as s5

# Two box types. "hippo": hippocampus + amygdala. "temporal": the whole
# temporal lobe (hippocampus, amygdala, temporal pole, lateral temporal
# cortex) for lesions outside the hippocampus (DNT, cavernoma, FCD).
BOXES = {
    "hippo": ({"L": (-26.0, -20.0, -14.0), "R": (26.0, -20.0, -14.0)}, (48, 64, 48), "hippo_crops"),
    "temporal": ({"L": (-38.0, -16.0, -22.0), "R": (38.0, -16.0, -22.0)}, (56, 80, 48), "temporal_crops"),
}
BOX = os.environ.get("TLE_BOX", "hippo")
CENTRE_MM, SIZE, _folder = BOXES[BOX]
OUT_DIR = Path(RUN_DIR) / _folder


def crop_subject(sub_id):
    out = OUT_DIR / f"{sub_id}.npy"
    if out.exists():
        return sub_id, "exists"
    paths = s5.get_subject_paths(s5.NORMALIZED_DIR / sub_id)
    # Step 5 writes the MNI template files into its cache folder on first use;
    # parallel workers sharing one folder read each other's half-written
    # files, so every worker process gets its own folder.
    cache = OUT_DIR / "_mni_cache" / str(os.getpid())
    t1, flair, affine, _ = s5.align_to_mni(paths.t1, paths.flair, cache)
    inv = np.linalg.inv(affine)
    sides = []
    for side in ("L", "R"):
        c = np.round(inv @ np.array([*CENTRE_MM[side], 1.0]))[:3].astype(int)
        sl = tuple(slice(ci - s // 2, ci + s // 2) for ci, s in zip(c, SIZE))
        box = np.stack([t1[sl], flair[sl]])                                # (2, 48, 64, 48)
        if side == "R":
            box = box[:, ::-1]                                             # mirror x: lateral -> medial
        sides.append(box)
    np.save(out, np.stack(sides).astype(np.float16))
    return sub_id, "ok"


def qc_picture(sub_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = np.load(OUT_DIR / f"{sub_id}.npy").astype(np.float32)
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), dpi=90)
    for i, side in enumerate(("left", "right (mirrored)")):
        for j, ch in enumerate(("T1", "FLAIR")):
            vol = a[i, j]
            ax[i, 2 * j].imshow(np.rot90(vol[:, :, SIZE[2] // 2]), cmap="gray", vmin=-6, vmax=3)
            ax[i, 2 * j].set_title(f"{side} {ch}: axial", fontsize=9)
            ax[i, 2 * j + 1].imshow(np.rot90(vol[:, SIZE[1] // 2 + 6, :]), cmap="gray", vmin=-6, vmax=3)
            ax[i, 2 * j + 1].set_title(f"{side} {ch}: coronal", fontsize=9)
    for x in ax.ravel():
        x.axis("off")
    fig.suptitle(f"{sub_id}: {BOX} boxes", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "qc" / f"{sub_id}.png")
    plt.close(fig)


if __name__ == "__main__":
    n_par = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    only = sys.argv[2:]                                                    # optional subject IDs (testing)
    (OUT_DIR / "qc").mkdir(parents=True, exist_ok=True)
    subjects = only or sorted(load_subject_list())
    done = Parallel(n_jobs=n_par, verbose=5)(delayed(crop_subject)(s) for s in subjects)
    print(f"{sum(st == 'ok' for _, st in done)} new, {sum(st == 'exists' for _, st in done)} already there")
    for s in subjects[:6]:
        qc_picture(s)
    print(f"crops -> {OUT_DIR}  (QC pictures for the first {min(6, len(subjects))} subjects in {OUT_DIR / 'qc'})")

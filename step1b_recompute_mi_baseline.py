# ============================================
# STEP 1b: RECOMPUTE THE "MI BEFORE" BASELINE (one-off repair)
# ============================================
# The first balanced run of Step 1 recorded mi_before = 0.0 for every
# subject (mattes_mi() never actually evaluated the metric - fixed in
# step1_registration.py). mi_after and the registered volumes were
# correct, so instead of re-running ~77 min of registration this fills in
# the real baseline and re-runs the group-level Wilcoxon test.
#
# Baseline = MI at the GEOMETRY-centred starting transform, i.e. the
# images as they are before any optimisation. (The full Step 1 records the
# baseline of whichever initialiser won; geometry won for the large
# majority of subjects, so the two agree closely.)

import os

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

import step1_registration as s1

if __name__ == "__main__":
    report_path = os.path.join(s1.OUT_DIR, "step1_registration_report.csv")
    report = pd.read_csv(report_path)
    t1_map, flair_map = s1.find_subject_files(s1.DATA_ROOT)

    for sub_id in tqdm(report["sub_id"].unique(), desc="Recomputing MI baseline"):
        t1 = sitk.ReadImage(t1_map[sub_id], sitk.sitkFloat32)
        flair = sitk.ReadImage(flair_map[sub_id], sitk.sitkFloat32)
        for direction, fixed, moving in [("FLAIR_to_T1", t1, flair), ("T1_to_FLAIR", flair, t1)]:
            init = sitk.CenteredTransformInitializer(
                fixed, moving, sitk.AffineTransform(3),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            row = (report["sub_id"] == sub_id) & (report["direction"] == direction)
            report.loc[row, "mi_before"] = s1.mattes_mi(fixed, moving, init)

    report.to_csv(report_path, index=False)
    print(f"Updated mi_before in {report_path}")

    group_results = [r for d in ["FLAIR_to_T1", "T1_to_FLAIR"]
                     if (r := s1.group_level_test(report, d))]
    for r, d in zip(group_results, ["FLAIR_to_T1", "T1_to_FLAIR"]):
        sub = report[report["direction"] == d]
        r["mean_mi_before"] = sub["mi_before"].mean()
        r["mean_mi_after"] = sub["mi_after"].mean()
        r["n_improved"] = int((sub["mi_after"] < sub["mi_before"]).sum())
    group_df = pd.DataFrame(group_results)
    group_df.to_csv(os.path.join(s1.OUT_DIR, "step1_group_statistical_test.csv"), index=False)
    print(group_df.to_string(index=False))

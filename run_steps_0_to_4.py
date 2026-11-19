# ============================================
# DRIVER: bring the whole cohort through Steps 0-4
# ============================================
# Safe to run any time, and to re-run after stopping it - every step skips
# subjects whose outputs already exist.
#
#   1. Step 0 selects the cohort (excluding anyone who failed registration),
#      downloads anything missing, and Step 1 registers new subjects.
#      Repeated until every selected subject is registered, because a
#      failed subject is replaced by a newly matched one on the next Step 0.
#   2. Steps 2, 3 and 4 then run once for everyone still missing them.
#
# Run:  python run_steps_0_to_4.py

import os
import subprocess
import sys

from pipeline_config import CODE_DIR, RUN_DIR, load_subject_list

MAX_ROUNDS = 5


def run(script, *args):
    print(f"\n########## {script} {' '.join(args)} ##########", flush=True)
    result = subprocess.run([sys.executable, "-u", os.path.join(CODE_DIR, script), *args])
    if result.returncode != 0:
        sys.exit(f"{script} exited with code {result.returncode} - stopping here.")


def unregistered_subjects():
    reg_dir = os.path.join(RUN_DIR, "registered_outputs")
    return sorted(s for s in load_subject_list()
                  if not os.path.exists(os.path.join(reg_dir, s, f"{s}_FLAIR_in_T1space.nii.gz")))


if __name__ == "__main__":
    from step0_select_subjects import registration_failures

    for round_no in range(1, MAX_ROUNDS + 1):
        # Re-select only if a selected subject has FAILED registration.
        # Re-running Step 0 otherwise could reshuffle pairs that are fine.
        failed = {f"sub-{n}" for n in registration_failures()}
        if not os.path.exists(os.path.join(CODE_DIR, "subject_list.txt")) or failed & set(unregistered_subjects()):
            run("step0_select_subjects.py")
        run("download_subjects.py")
        run("step1_registration.py")
        missing = unregistered_subjects()
        print(f"\nRound {round_no}: {len(missing)} selected subject(s) not registered {missing}", flush=True)
        if not missing:
            break
    else:
        sys.exit(f"Still {len(missing)} unregistered after {MAX_ROUNDS} rounds - check the Step 1 log.")

    for script in ("step2_skull_stripping.py", "step3_bias_correction.py", "step4_intensity_normalization.py"):
        run(script)
    print("\nSteps 0-4 complete for the whole cohort. Next: step5_patch_extraction.py")

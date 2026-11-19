# HANDOVER — TLE Detection & Lateralization from Brain MRI (capstone)

Written 2026-11-13 by the previous Claude (Claude Code, desktop app on the user's Windows laptop) for a fresh Claude with zero context.
Everything below is from the working session, the project files, earlier session transcripts and a persistent memory file.
Anything not known is marked **UNKNOWN**; all gaps are collected in section 8.

**Package contents (this zip)**

| Path in zip | What |
|---|---|
| `HANDOVER.md` | this document |
| `code/` | every current script (Steps 0–13, server tools, GPU models, experiments, final test), config, requirements, README, .gitignore/.gitattributes — exact copies of the laptop's `TLE\` folder as of 2026-11-13 |
| `cohort/` | `labels.tsv` (cohort + held-out split), `subject_list.txt`, `heldout_test.txt`, `lateralization_discovery_patients.txt`, `scanner_by_subject.csv`, `subjects_to_download.txt` |
| `results/` | final metric CSVs: held-out test metrics + per-subject predictions, CV ensemble tables, classifier-experiment tables |
| `docs/` | project Word documents (pipeline steps, dataset guide, older summaries), `PROJECT_CONTEXT_PROMPT.md` (user's older tutor briefing, run-1 numbers) + `project_pipeline_thumbnail.png` (pipeline diagram) |

No MRI scans or images of subjects are included. The cohort files contain only the **pseudonymous public IDs of the IDEAS dataset** (e.g. `sub-61`, released under CC0 with anonymised metadata) plus group, side, pathology, scanner, sex and age band; they are required to reproduce the locked held-out split. If the user prefers, delete `cohort/` and re-create it (see 7.3).

---

## 0. READ THIS FIRST (corrections to the request text)

1. **The data are structural MRI only (sMRI: T1-weighted + FLAIR). There is NO fMRI.** The IDEAS dataset contains no functional MRI (no `func/` folders). The user's handover request mentions "fMRI and sMRI" — that is inaccurate; do not look for fMRI.
2. **The pipeline has 14 numbered stages: Step 0 to Step 13** (Step 0 = subject selection). People often call it "the 13-step pipeline" (Steps 1–13). After Step 13, several **additional models** were added (FreeSurfer volumetry, 3D CNNs on hippocampus / temporal-lobe crops, ensembles, held-out final test). These additional models give the final reported results.
3. **The held-out test set has already been used once.** Never re-run `final_test.py` to "improve" numbers and never tune anything on the 44 test subjects. Any new experiment must be reported as development cross-validation only.
4. **The user is a learner as well as the project owner.** At the time of handover the user was being taught the pipeline step by step (currently Step 8, piece 3). See section 7.5 for how they like to be taught and how they like to work.

---

## 1. PROJECT OVERVIEW

### 1.1 Identity
- **Title (chosen by the user):** "Spectral Hemisphere Graph Learning for Temporal Lobe Epilepsy Detection and Lateralization from Brain MRI"
- **Type:** B.Tech CS final-year capstone (user is 2027 batch, in placement season). Team: the user + group members ("my group", "teammates", "my friend" who studied Steps 7–8). Team member names: **UNKNOWN**.
- **Mentor:** a faculty mentor gave feedback (see 4.6). Name **UNKNOWN**.
- **User's GitHub:** account `pavneetgrewal021-hash` (git user.name). Git email set to `pavneetgrewal021@gmail.com` (fixed from a typo on 2026-11-01). Repository for this project: **not created yet** (see 4.4).
- **Project portal description:** a ~120-word and a 50-word description were written for the capstone portal (text **UNKNOWN** verbatim; regenerate from section 4 if needed).

### 1.2 Research questions
1. **Detection:** can we classify a person as TLE patient vs healthy control from T1 + FLAIR MRI?
2. **Lateralization:** for TLE patients, can we tell whether the epileptogenic temporal lobe is **left or right** (= side that was surgically resected, `Op_Side`)?

### 1.3 Dataset
| Item | Value |
|---|---|
| Name | IDEAS — Imaging Database for Epilepsy And Surgery |
| Source | OpenNeuro **ds005602 v1.0.0** (https://openneuro.org/datasets/ds005602), raw files also on S3 `https://s3.amazonaws.com/openneuro.org/ds005602` |
| Paper | Taylor P.N. et al., "The Imaging Database for Epilepsy And Surgery (IDEAS)", arXiv:2406.06731 (cite it) |
| Licence | CC0 |
| Site | UCLH / National Hospital for Neurology and Neurosurgery, London; all GE 3 T scanners |
| Full dataset | 542 subject folders: 442 patients (`sub-1`…`sub-463`, non-continuous) + 100 controls (`sub-4001`…`sub-4100`; ID ≥ 4001 = control) |
| Scanners | GE DISCOVERY MR750, GE SIGNA HDx, GE SIGNA EXCITE (EXCITE has 0 controls → excluded) |
| Modalities used | T1w and FLAIR (`anat/`), BIDS, `.nii.gz` + `.json` sidecars |
| Clinical tables (NOT on OpenNeuro) | figshare: patient table `Metadata_Release_Anon.csv` (figshare item id 25612080; on the laptop at `C:\Users\pavne\Downloads\25612080 (1)\Metadata_Release_Anon.csv`), control table `Metadata_Controls_Release.csv` (laptop: `C:\Users\pavne\Downloads\Metadata_Controls_Release.csv`), `readme_metadata.pdf` defines the codes |
| FreeSurfer tables | figshare `25612065.zip` (laptop: `C:\Users\pavne\Downloads\25612065.zip`), files used: `aseg_vol.txt`, `controls_aseg_vol.txt` (rows = measures, columns = subject IDs; IDs prefixed `sub-` by our loader). Extracted copies: laptop scratch folder (temporary) and server `/workspace/tle_code/fs_tables/` (server access ended) |
| Wrong file to avoid | `Capstone\code\participants.tsv` belongs to a DIFFERENT dataset (UNAM ds004469). It caused the very first run to match 0 subjects. Never use it. |

Patient eligibility: only patients whose `Op_Type` starts with "T" (temporal-lobe surgery; 356 of 443). 17 controls have no FLAIR (404 on download) → excluded.

Metadata columns. Patient table (443 rows / 442 unique IDs): ID, Sex, Binned_Onset_Age, FUS, fqFUS, FBTCS, fqFBTCS, SE, **Op_Side** (L/R resected = lateralization label), **Op_Type** (T/F/P/O lobe; Lx lobe resection, Lesx lesion resection, Hx hemispherectomy), **Pathology** (HS, DNT, FCD, CAV, GL, EFG, DUAL, TREBLE, OTHER, TBC), OP MEMO, Number_ASMs, Binned_Age_at_Scan, Binned_Age_at_Surgery, ILAE_Year1–5. Control table: ID, Sex, Binned_Age_at_Scan. The two tables use different end age bands; Step 0 harmonises them to <20, 20–24 … 50–54, 55+.

Scanner counts among usable subjects (controls / temporal patients with FLAIR): MR750 68 / 92, HDx 15 / 163, EXCITE 0 / 78 → strong scanner confound, hence within-scanner matching. 30 MR750 patients use non-standard protocols (e.g. 0.42 mm in-plane); no controls do.

Note: the user once said the **official project title may mention functional data** — IDEAS has none; the chosen title (1.1) does not mention it.

### 1.4 Final cohort (219 subjects)
- **59 matched pairs** (patient + control): same scanner, same standard protocol, same sex, age band at most 1 band apart. 48 MR750 pairs + 11 HDx pairs. Sex 35 F / 24 M in both groups; 30 left / 29 right TLE; 46 pairs exact age band, 13 one band apart.
- **101 lateralization-only patients** (no control partner), used only for left vs right.
- **All patients: 160** (85 left / 75 right). Pathology: HS (hippocampal sclerosis) 96, OTHER 21, DNT 19, CAV (cavernoma) 12, DUAL 8, TREBLE 2, TBC 1, FCD 1.
- Standard protocols required: MR750 1 mm isotropic T1 + FLAIR; HDx T1 0.94 × 1.1 × 0.94 mm, HDx FLAIR 0.94 × 0.94 × 5 mm (thick 2D slices).
- `labels.tsv` columns: `participant_id, group (patient/control), cohort (matched / lateralization_only), pair, scanner, sex, age_band, tle_side (L/R/n/a), pathology, match_quality, split (dev/test)`.

### 1.5 Held-out split (locked 2026-10-25, `make_heldout_split.py`, seed 20261025)
| Split | TLE vs control | Left vs right |
|---|---|---|
| Development (all tuning and model choice) | 47 pairs = 94 subjects | 128 patients (47 + 81 lat-only) |
| Held-out test (used ONCE, 2026-10-27) | 12 pairs = 24 subjects (10 MR750 + 2 HDx, sides balanced) | 32 patients (12 from pairs + 20 lat-only, stratified) |

### 1.6 Preprocessing state of the data
- Steps 1–4 (registration → normalisation) are **complete for all 219** on the laptop (`Capstone\run_balanced\normalized_outputs\`). The server used a copy of these.
- Steps 5–13 were run twice: on the laptop (run 2, 9 Oct, all 219 + 4 leftover old controls) and on the server (26 Oct, exactly 219, development-only for Steps 12/13).
- All server results were copied back to `C:\Users\pavne\OneDrive\Documents\Capstone\server_results_final\` (300 files, 39 MB) before access ended.

### 1.7 Targets, deadlines, deliverables
- **Mentor's target:** ~85% accuracy. Reached on held-out test (95.8% / 93.8%); CV 82.3% / 87.5%.
- **College GPU server:** available until **2026-11-01**. The user applied (or was applying) for an extension; outcome **UNKNOWN**.
- **Extension application deliverables (prepared):** project thumbnail (`docs/project_pipeline_thumbnail.png`), video plan + narration script (section 4.5), 300-word project summary (section 4.5). Whether submitted: **UNKNOWN**.
- **Final capstone deadline / thesis / report format: UNKNOWN.** A paper was not discussed.
- **Resume:** user's resume lists this project; updated bullet block given (section 4.5).

---

## 2. THE STAGES (Step 0 – Step 13) + ADDITIONAL MODELS

General facts for all steps:
- Code folder (laptop): `C:\Users\pavne\OneDrive\Documents\TLE\` (also in `code/` of this zip). Each step is one script `stepN_*.py`.
- Every step reads paths from `pipeline_config.py`: `DATA_ROOT` (env `TLE_DATA_ROOT`, default `C:\Users\pavne\OneDrive\Documents\Capstone`) and `RUN_DIR` (env `TLE_RUN_DIR`, default `<DATA_ROOT>\run_balanced`). Outputs land in sub-folders of `RUN_DIR`; figures in `RUN_DIR/figures/stepN`, metrics in `RUN_DIR/metrics/stepN`, logs in `RUN_DIR/logs`.
- Steps 1–5 process only IDs in `subject_list.txt`; Steps 1–4 are resumable (skip finished subjects); old outputs were never deleted but renamed `*_OLD_*`.
- Libraries: Python 3.11 (laptop) / 3.10.12 (server); numpy, pandas, scipy, scikit-learn, SimpleITK, nibabel, nilearn, PyWavelets, torch, networkx, umap-learn, matplotlib, joblib, tqdm, hd-bet (versions in `code/requirements.txt`).
- Status: **all stages done** (working). Known weaknesses are listed per stage and in 4.3.

### Step 0 — Subject selection (`step0_select_subjects.py`, plus `download_subjects.py`, `run_steps_0_to_4.py`)
- Purpose: fair cohort — matched patients/controls that differ only in disease (not scanner, protocol, sex, age), plus extra patients for lateralization.
- Inputs: IDEAS patient + control CSVs (hard-coded paths at top of script: `PATIENT_METADATA`, `CONTROL_METADATA`, `DATA_ROOT`), JSON sidecars from OpenNeuro (cached in `scanner_by_subject.csv`).
- Outputs (in code folder): `subject_list.txt`, `labels.tsv`, `subjects_to_download.txt`, `scanner_by_subject.csv`.
- Key parameters: `PAIRS_PER_SCANNER = {"DISCOVERY MR750": 60, "SIGNA HDx": 11}`, `MAX_AGE_BAND_GAP = 1`, `STANDARD_PROTOCOL` (voxel sizes above), `ADD_LATERALIZATION_PATIENTS = True`, `REGISTRATION_FAILURES = {7, 122, 127, 399, 408, 431, 4053}`, `RANDOM_SEED = 42`. Scarcer group matched first; L/R balanced.
- `download_subjects.py`: 8 parallel connections, stall timeout 30 s, 6 attempts, size checks (campus network is slow per connection).
- `run_steps_0_to_4.py`: driver that repeats Step 0 → download → Step 1 until every selected subject registers (failures replaced), then Steps 2–4.
- Result: 59 pairs + 101 lat-only = 219. Runtime ~1 min (+downloads ~9 MB/subject).

### Step 1 — Registration FLAIR ↔ T1 (`step1_registration.py`)
- Purpose: align each FLAIR to the same person's T1 (voxel-to-voxel correspondence).
- In: raw `sub-X_T1w.nii.gz`, `sub-X_FLAIR.nii.gz`. Out: `RUN_DIR/registered_outputs/sub-X/` (T1 + `FLAIR_in_T1space`), report `step1_registration_outputs/step1_registration_report.csv`, checkerboard QC figures.
- Method: SimpleITK **affine** (12 DOF). Mattes MI, 50 bins, 20% random sampling with fixed seed 42, linear interpolation, gradient descent (lr 1.0, 200 iterations, convergence 1e-6 over window 10), scales from physical shift, 3-level pyramid (shrink 4-2-1, smoothing 2-1-0 mm). Two initialisations (GEOMETRY and MOMENTS), better final MI kept. Both directions registered (FLAIR→T1 used).
- Quality gate: MI must not get worse AND reach ≤ −0.35 in both directions; if optimisation worsens MI, the initial transform is kept. Dice of rough masks reported for information only.
- Result: all 219 pass. FLAIR→T1 MI −0.430 → −0.495, improved in 102/128 listed, Wilcoxon p = 1.8e-18. 7 failures replaced (listed above).
- Runtime ~46 s/subject.

### Step 2 — Skull stripping (`step2_skull_stripping.py`)
- Purpose: keep only brain.
- Method: **HD-BET** (pip `hd-bet`, v2.0.1 on laptop) on T1, no test-time augmentation, device from env `TLE_HDBET_DEVICE` (default `cpu`). Largest connected component; same mask applied to FLAIR. Per-subject temp folder `OUT_DIR/_hdbet_tmp/<sub>` (parallel-safe).
- Checks: brain volume 900–2200 cm³; mask stability under erode/dilate Dice ≥ 0.95; group check — mask volume by sex and group (Mann-Whitney) on matched pairs.
- Out: `RUN_DIR/skullstripped_outputs/sub-X/` (T1 brain, FLAIR brain, `_brain_mask`).
- Result: 219/219 pass, volumes 1052–2143 cm³; men > women p < 0.0001; patients = controls p = 0.55.
- Runtime ~40–80 s/subject CPU.

### Step 3 — Bias field correction (`step3_bias_correction.py`)
- N4ITK (SimpleITK), T1 and FLAIR separately, inside brain mask. Gate: blocked only if CoV (std/mean in brain) worsens > 2%; 3% improvement is a soft check.
- Out: `RUN_DIR/biascorrected_outputs/sub-X/`.
- Result: 219/219; T1 CoV 0.296 → 0.259 (−13%), FLAIR 0.325 → 0.247 (−22%), Wilcoxon p = 7e-37. Slowest step (~3 min/subject).

### Step 4 — Intensity normalisation, WhiteStripe (`step4_intensity_normalization.py`)
- Find the white-matter histogram peak (T1: **right-most / brightest prominent peak**; FLAIR: tallest peak), take ±25% band (`whitestripe_width_pct 0.25`), rescale band to mean 0, robust spread 1 (MAD-based). Only brain voxels (mask and value ≠ 0) are normalised; background is written as exactly 0 (`norm_arr = zeros; norm_arr[valid] = ...`). Ignores out-of-FOV FLAIR zeros and a 2-voxel blended edge.
- Group checks: pairwise KS tests between subjects' histograms; outlier flags on raw peaks.
- Out: `RUN_DIR/normalized_outputs/sub-X/` (`sub-X_T1w_normalized.nii.gz`, `sub-X_FLAIR_normalized.nii.gz`, `sub-X_brain_mask.nii.gz`).
- Result: 438/438 images pass; FLAIR mean KS 0.064; T1 KS 0.209 (scanner difference, not group). 3 FLAIRs with unusual raw peaks (sub-24, sub-358, sub-65) never checked visually.
- Verified 2026-11 on sub-61: of 1.3 M brain voxels only 600 (T1) / 30 (FLAIR) have |value| < 1e-6, so "exactly 0" reliably means background.

### Step 5 — Patch extraction (`step5_patch_extraction.py`)
- Rigid (6 DOF, `sitk.Euler3DTransform`) registration of the skull-stripped T1 to nilearn's skull-stripped **MNI152 1 mm template** (grid 197 × 233 × 189; affine = identity scale with offsets (−98, −134, −72), so voxel x 98 = world x 0). Init: `CenteredTransformInitializer` MOMENTS on **brain masks** (T1 ≠ 0). Mattes MI 50 bins, **10%** sampling seed 42, gradient descent lr 1.0, 200 its, pyramid 4-2-1 / 2-1-0 mm. T1 and FLAIR resampled (linear) with the same transform. No pass/fail gate; final MI saved as `mni_rigid_mi`.
- Midline correction: mask = |T1| > 1e-6; try sagittal planes c within ±20 mm of the bounding-box centre; overlap = count(mask AND mirror(mask) about c); best c → whole-voxel shift `98 − c` (no interpolation) for T1 and FLAIR.
- Grid: 32 × 32 × 32 mm patches, stride 16 (50% overlap). x-starts anchored on the midline (`offset = 98 % 16 = 2` → 2, 18, …, 162 = 11), y 13, z 10 → 1,430 candidates. Reject if > 70% background (|v| < 1e-6) in T1 OR FLAIR, or if the patch spans world x = 0 (only the start-82 column). Hemisphere = sign of world x of centre (RAS: < 0 left).
- `Atlas_Region` = "NA" (no atlas file exists).
- Out: `RUN_DIR/patch_outputs/sub-X/patches.npy` (N × 2 × 32 × 32 × 32 float32, ~92 MB/subject) + `metadata.csv` (Subject_ID, Patch_ID, Center_Coordinate, Atlas_Region, Hemisphere, Background_Percentage); metrics `metrics/step5/subject_metrics.csv` etc.
- Result: 77,264 patches (≈350/subject; sub-61: 343 = 170 L / 173 R); L/R ratio 0.90–1.05 for all; patches per subject patient = control (p = 0.68). ~5 s/subject.
- `align_to_mni()` is re-used by `hippo_crops.py`.

### Step 6 — 3D wavelet decomposition (`step6_localized_spectral_decomposition.py`)
- `pywt.wavedecn(patch, "db4", level=3, mode="symmetric")` per patch and channel → 22 sub-bands (approx 10³, level-3 details 7 × 10³, level-2 7 × 13³, level-1 7 × 19³ = 71,392 coefficients).
- Gate: inverse transform reconstruction MSE ≤ 1e-6 (actual ~1e-30). Energy ratio consistency (CV ≤ 0.10) reported only; it is always "False" (ratio ≈ 2.4 because of boundary extension) — harmless.
- 7 statistics computed immediately (moved here from Step 7 to save disk): Energy (Σv²), Entropy (64-bin histogram, bits), Mean, Variance, Skewness, Kurtosis (excess), MAD.
- **Known weakness (found 2026-11):** statistics are computed on `pywt.coeffs_to_array` packed 52³ array (140,608 slots) of which **69,216 (49%) are padding zeros**, and all bands are pooled. Effects: MAD ≈ 0 for every patch (useless), entropy ~0.65 bits and kurtosis ~300+ distorted, approximation band dominates. Inherited from the original design. Fix idea: per-sub-band statistics on real coefficients only (dev-CV only). pywt also warns "Level 3 is too high" (max clean level for 32 voxels with 8-tap db4 is 2).
- Out: `RUN_DIR/spectral_outputs/sub-X/spectral_features_T1.npy`, `_FLAIR.npy` (N × 7) + `metadata.csv` with recon MSE / energy ratio.

### Step 7 — Tokens (`step7_spectral_token_generation.py`)
- Pure assembly: per patch join metadata + 7 T1 + 7 FLAIR statistics → 14-number token. Checks: row counts match, NaN, duplicates. Saves raw (unscaled) values.
- Out: `RUN_DIR/token_outputs/tokens.csv`, `tokens.npy` (N × 14 float32), `feature_names.json`; PCA/UMAP/correlation/histogram plots.
- Result: 77,264 tokens for 219 subjects (laptop file: 78,551 / 223 subjects including 4 leftover run-1 controls sub-4001/4067/4073/4077).

### Step 8 — Cross-scale embedding (`step8_cross_scale_spectral_consistency.py`)
- Self-supervised (no labels). Three "scale views" per token built from the SAME 32 mm tokens by Gaussian-weighted averaging with same-subject neighbours (centre-to-centre distance): `SCALES = (24, 32, 48)`, radius = scale/2 (12/16/24 mm), bandwidth = radius/2 (6/8/12 mm), weight = exp(−d²/(2·bw²)), normalised. On the 16 mm grid: view 24 = token itself (no neighbour within 12 mm), view 32 = self 55% + 6 face neighbours 7.5% each, view 48 = self 18% + 6 face (7.5%) + 12 edge neighbours (3.1%). Names are misleading: really "original / lightly / heavily smoothed".
- Standardisation: one mean/std per feature computed over all three views pooled (std 0 → 1).
- Model: projection head 14 → 64 (ReLU) → 32, L2-normalised. Loss: multi-positive InfoNCE / NT-Xent, temperature 0.07; positives = other views of the same token, negatives = all other tokens' views in the batch. Adam lr 1e-3, 50 epochs, batch 256 tokens (×3 views), seed 42. Inputs are ONLY the 14 features (position/hemisphere deliberately not fed — would let the model cheat / leak L-R).
- Validation (primary): cross-scale top-1 retrieval on 3,000 sampled tokens before vs after, paired exact McNemar: 24↔32 85.3% → 98.1% (p 1e-96), 32↔48 50.6% → 90.8%, 24↔48 18.3% → 54.4%. Secondary (kept for reference only, wrong metric): raw cosine Wilcoxon.
- Out: `RUN_DIR/embedding_outputs/embeddings_scale24/32/48.npy` (N × 32), `token_index.csv`, `projection_head.pt`.

### Step 9 — Hemisphere graphs (`step9_hemisphere_graph_construction.py`)
- Node feature = 32-dim embedding averaged over the 3 scales; each subject has an L and an R graph; edges = 5 nearest patches in space (`TOP_K = 5`), weight = cosine similarity. Spatial edges only (no fMRI).
- Out: `RUN_DIR/graph_outputs/sub-X/sub-X_L_graph.npz`, `_R_graph.npz`. ~176 nodes, ~509 edges per graph.

### Step 10 — Graph representation (`step10_graph_representation_learning.py`)
- Path A GAT: 2-layer graph attention (4 heads, hidden 16, out 16, dropout 0.1) + attention pooling, self-supervised GraphCL (20% edge drop, 10% feature mask), temperature 0.07, Adam lr 1e-3, 40 epochs, seed 42.
- Path B PCA: mean + max of node features + graph statistics → PCA to 16 dims.
- Stability check: leave-one-subject-out retraining (15 epochs) — slow (~2.6 h); env `TLE_SKIP_STABILITY=1` skips it (used on server).
- Out: `RUN_DIR/embedding_outputs/graph_embeddings_gat.npy`, `graph_embeddings_pca.npy` (2 × subjects × 16), `graph_embedding_index.csv` (Subject_ID, Hemisphere).
- Finding: PCA vectors beat GAT vectors in every downstream comparison.

### Step 11 — Adaptive hemisphere fusion (`step11_adaptive_hemisphere_fusion.py`)
- Gate network (hidden 16) outputs α: fused = α·L + (1−α)·R, concatenated with L − R → 32 numbers; trained label-free by reconstruction (Adam lr 1e-3, 100 epochs). Baselines: average and concatenation.
- Out: `RUN_DIR/fusion_outputs/fusion_embeddings_adaptive|average|concat.npy`, `alpha_weights.csv`.
- Result: α ≈ 0.40 ± 0.006 for everyone — does not adapt or lateralise (honest negative result). Not used in final models.

### Step 12 — Original classifier, TLE vs control (`step12_lightweight_classifier.py`)
- Matched cohort only, development subjects only (`dev_only`). Logistic regression, linear SVM, MLP (16). Nested LOOCV (inner 3-fold), 100 permutations, 1000 bootstrap, 10 seeds, repeated pair-grouped 5-fold × 10; fusion ablation; scanner-only / sex+age / dummy baselines.
- Out: `RUN_DIR/classifier_outputs/`, `metrics/final/`, `figures/final/`.
- Dev results (server, 47 pairs): best repeated CV average-fusion + LR 69.7% ± 2.3, AUC 0.724; MLP 69.5%, AUC 0.726; LOOCV LR 64.9% / 0.684. Baselines at chance.

### Step 13 — Original lateralization (`step13_lateralization.py`)
- All (dev) patients. Pre-registered primary config: PCA vectors, control-normalised asymmetry z-scores ((L − R − control mean)/control std), mirror augmentation (z → −z, label flipped, training only), logistic regression. LOOCV, repeated 5-fold × 10, 1000 permutations, bootstrap, HS vs non-HS, ablation, confirmation test (train on discovery patients in `lateralization_discovery_patients.txt`, test new patients, 10,000 permutations). Ignores subjects missing from `labels.tsv`.
- Dev results (server, 128 patients): LOOCV 56.2%, AUC 0.587, permutation p = 0.085; confirmation (39 discovery → 89 new) 55.1%, AUC 0.557, p = 0.19 → **not significant on dev**.
- Out: `RUN_DIR/lateralization_outputs/`, `metrics/step13/`, `figures/step13/`.

### Additional modules (after Step 13) — these produce the final results
| Script | Purpose | Key settings | Output (server `RUN_DIR/...`) | Status |
|---|---|---|---|---|
| `make_heldout_split.py` | lock 44 test subjects, add `split` column, write `heldout_test.txt`; refuses to run twice | seed 20261025, `TEST_PAIRS = {MR750: 10, HDx: 2}`, `TEST_LAT_ONLY = 20` | `labels.tsv` (backup `labels_before_split_backup.tsv`, not in repo) | done |
| `experiment_classifiers.py` | 9 classifiers × ~20 feature sets (gat/pca × LR, avg, diff, absdiff, avg_absdiff, asymz; fusion_adaptive; tok_all/T1/FLAIR; with `TLE_VOL_DIR`: vol and vol+partner) | 5-fold × 10 repeats (`TLE_EXP_REPEATS`), `TLE_EXP_JOBS` parallel; exports `load_features`, `metrics`, `splits` | `experiments/classifiers_<task>.csv` | done |
| `experiment_volumetry.py` | FreeSurfer hippocampus/amygdala volumes ÷ eTIV × 1e3 + asymmetry index (L − R)/(L + R); LR CV; no-training rule "smaller hippocampus = epileptic side" | `load_volumes(folder)` | printed (log `exp_volumes.log`) | done |
| `gpu_attention_mil.py` | 3D CNN patch encoder (2→16→32→64→128 + position MLP) + gated attention MIL (Ilse 2018) over all Step 5 patches | 5-fold × 3, 40 epochs, AdamW lr 3e-4 wd 1e-2 one-cycle, 256 random patches per bag per step, 8 bags/batch, bags kept on GPU | `gpu_mil/<task>/` | done; **not kept** |
| `hippo_crops.py` | cut one box per side at fixed MNI coords using Step 5 alignment; right box mirrored on x; float16 (2 sides, 2 channels, D, H, W) + QC PNGs | `TLE_BOX=hippo`: centres (±26, −20, −14) mm, 48 × 64 × 48; `TLE_BOX=temporal`: (±38, −16, −22), 56 × 80 × 48; joblib, per-PID MNI cache dir | `hippo_crops/` or `temporal_crops/<sub>.npy` | done (219 on server; laptop has ~35 for smoke tests) |
| `gpu_hippo_cnn.py` | shared encoder (4 × [conv3 → InstanceNorm → LeakyReLU 0.1 → maxpool], 2→16→32→64→64, avgpool, dropout 0.3); TLE head on [(fL+fR)/2, \|fL−fR\|] → 32 → 1; side head = Linear(fL − fR, bias=False) (antisymmetric). Also fits the tabular model on the same folds | AdamW lr 1e-3 (env `TLE_CNN_LR`), wd 5e-2, one-cycle, 80 epochs, batch 16, bf16 autocast, augmentation: roll ±3 voxels, intensity ×(1±0.1), noise 0.05, random L/R swap (label flipped for side); 5-fold × 5 repeats; `TLE_CROP_DIR` selects crops | `gpu_hippo/<crop>/<task>[_lr<x>]/` per_repeat_metrics, oof_predictions, summary | done |
| `combine_oof.py` | equal-weight averages of all combinations of hippo_cnn, temporal_cnn, tabular out-of-fold predictions; rank by mean AUC | — | `gpu_hippo/ensembles_<task>.csv` | done |
| `final_test.py` | train locked models on ALL dev subjects (each CNN 5 seeds averaged), predict 44 test subjects once; bootstrap CIs (pairs resampled for TLE), binomial p, HS/non-HS; refuses to run again | seeds 1042–1046 for CNNs, 2000 bootstrap | `final_test/<task>/final_metrics.csv`, `test_predictions.csv` | **done once — do not rerun** |
| `lr_sweep_summary.py` | learning-rate sensitivity of the CNNs (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) | requires runs with `TLE_CNN_LR` | `gpu_hippo/lr_sweep_<task>.csv` | **ran on the server** (`/workspace/tle_data/lr_sweep.log` exists, 2026-11-19); results not yet read/interpreted |
| `run_parallel.py` | run Steps 1–4 as N shards (`TLE_SHARD=i/n`), merge per-shard reports | `THREADS_PER_SHARD = 4`, Step 2 max 48 shards | merged report CSVs | done (server) |
| `step1b_recompute_mi_baseline.py` | recompute "MI before" for an already-registered run | — | `step1_final_cohort_FLAIR_to_T1_mi.csv` | done |

"Tabular" model: TLE vs control = LDA (lsqr, shrinkage auto) on [hipp_L, hipp_R, amyg_L, amyg_R, |hipp_AI|, |amyg_AI|] + PCA avg & |L − R| (32) with StandardScaler; side = logistic regression on [hipp_AI, amyg_AI].

---

## 3. CHANGE LOG WITH REASONS (chronological)

Dates are 2026. "Laptop" = user's Windows 11 laptop; "server" = college GPU server.

### Before 27 Sep — first run (original code, in `Capstone\code\step*.py`)
| Before | Problem | Now / what happened |
|---|---|---|
| Patients only, 75 subjects | nothing to compare against | balanced cohort with controls |
| Labels from `participants.tsv` | file is from UNAM ds004469; 0 of 75 IDs matched → no classification at all | labels from IDEAS metadata; controls = ID ≥ 4001 |
| Skull stripping with `nilearn.compute_brain_mask` | it pastes an MNI template mask onto native-space images: cut brains, included skull, volumes ~1640 cm³ for everyone, no male/female difference, controls 64 cm³ bigger than patients (p = 0.002) = scanner confound | replaced by HD-BET (28 Sep). **Results of that first run are invalid.** |

### 27 Sep
- Decision: 50 temporal patients + 50 age/sex-matched controls; rerun from Step 1. Created `step0_select_subjects.py` (writes `subject_list.txt`, `labels.tsv`) and `pipeline_config.py` (shared paths, fixed cohort, resumable steps).

### 28 Sep
- **Step 1 "MI before" was always 0.0** (0-iteration registration read a value never computed) → Wilcoxon against a fake baseline. Fixed with `MetricEvaluate`; `step1b_recompute_mi_baseline.py` for the saved run.
- Random voxel sampling without seed → borderline subjects flipped pass/fail between runs → fixed seed 42.
- Step 2 → HD-BET (CPU, ~40 s/subject). Validated: M > F p = 1e-4, patient vs control p = 0.55. Upper volume limit raised 2000 → 2200 cm³ after visual check of sub-132.
- **Scanner confound:** controls 86% MR750 vs patients 36%; a classifier could reach ~75% on scanner alone → pairs matched within scanner; EXCITE excluded. Cohort v2 backed up in `TLE\cohort_v2_age_sex_only\`.
- 17 controls lack FLAIR → excluded. `download_controls.py` renamed `download_subjects.py`.

### 29 Sep
- Step 1: keep initial transform when optimisation worsens MI (well-aligned subjects like sub-237, sub-88 were being rejected).
- Step 0 reads registration failures automatically; 7 failures replaced (sub-7, 122, 127, 399, 408, 431, control 4053).
- Step 4: T1 used the **tallest** histogram peak; on MR750 T1 that is grey matter → half the cohort normalised to GM, split by scanner (71.5% of pairs differed). Fixed to brightest prominent peak (WhiteStripe paper). Old output `normalized_outputs_OLD_tallest_peak`.
- **Protocol confound:** 11 MR750 patients had non-1 mm voxels (0.42–1.5 mm), all controls 1 mm → patches in voxels had different physical size (385 vs 276 patches). Step 0 now requires standard protocol (cohort v3 backup `cohort_v3_scanner_matched\`); Step 5 resamples to 1 mm.
- Step 5 rebuilt: MNI rigid alignment + symmetry midline (native x = 0 was not midline, e.g. sub-4001 629k vs 801k voxels), midline-anchored grid (edge-anchored grid gave unequal L/R counts, e.g. 194 vs 135), FLAIR background check, midline-crossing patches rejected. Old outputs `patch_outputs_OLD_native_grid`, `spectral_outputs_OLD_native_grid`.
- Step 6: statistics computed inside Step 6, full coefficients no longer saved (30 GB → 8.5 MB; identical values to 1e-15).
- Step 8 validation: raw cosine before/after said "worse" (0.925 → 0.896, p = 1.0) because it compared different spaces; replaced by cross-scale retrieval + McNemar.
- Step 12: sklearn `StratifiedGroupKFold` ignored the seed when every group is 1 patient + 1 control → identical folds every repeat → own pair-grouped shuffling. Baselines scored 0% in LOOCV (balanced classes artefact) → evaluated with repeated 80/20 instead.
- Steps 0–6 done for 100 subjects (run 1), user asked to pause after Step 6; later Steps 7–12.
- **Run 1 results (100 subjects, no held-out set):** nested LOOCV linear SVM 77% / AUC 0.83, permutation p = 0.01; repeated CV average-fusion LR 76.7 ± 2.5%; scanner-only 50%, demographics 54.5%. Step 11 α constant (0.40 ± 0.01). Run-1 outputs kept in `run_balanced\results_v1_100subjects\`, docs `*_v1_100subjects.docx`.

### 8 Oct
- Added **Step 13** (lateralization). v1 on 50 patients: GAT AUC 0.59, p = 0.18 (not significant); PCA variants 0.65–0.72 → PCA configuration **pre-registered** (fixed in code before new patients processed).
- Cohort v2 (219): `cohort` column (matched = 59 pairs; lateralization_only = 101). Step 0 rule: same sex + age gap ≤ 1 band (old "nearest age band" produced 5 opposite-sex pairs). Discovery patients saved for a confirmation test. `download_subjects.py` made 8-parallel.

### 9 Oct — Run 2 (219 subjects, laptop, before held-out split)
- Step 12 (59 pairs): MLP nested LOOCV 74.6% / AUC 0.75, p = 0.01; linear SVM 63.6% / 0.71; repeated CV best 72.9% / 0.78; baselines at chance.
- Step 13 (160 patients): LOOCV 63.1% / AUC 0.68, p = 0.001; confirmation 50 → 110 new: 58.2% / 0.63, p = 0.008; HS 66.7% vs non-HS 57.8%.
- Issue: 4 run-1 controls (sub-4001, 4067, 4073, 4077) had leftover `patch_outputs` and leaked into the label-free Steps 6–11 (not into labels). Step 13 crashed on them → now ignores subjects not in `labels.tsv`. Resolved on the server run (exactly 219).
- The user's resume (sent before results) used estimates 75% / 0.80 / 65%.

### ~22–24 Oct (laptop)
- User studied Steps 1–4 in depth; documents for a friend (Steps 7–8 study doc); resume LaTeX rewrites (TLE replaced the "Resume Screening" project; one page; volunteer section kept; estimated numbers tagged `% ESTIMATE`).
- Dataset/pipeline docx updated to run 2; mentor document created (no "mentor" wording, simple style).
- Literature review: TLE vs control ~85–87% with CNNs on large cohorts (Kaestner et al. 2024, n = 1178; drops to ~72% at n = 300); ENIGMA non-lesional 53–62%; lateralization by hippocampal volumetry 88–94%; no published classification on IDEAS found. (From memory of the chat; re-verify before citing.)
- ds004199 evaluated (FCD type II, Bonn, Siemens) → **not usable** for TLE; would add a site confound. Rejected.

### 25–26 Oct — GPU server
- Mentor feedback received (4.6). College GPU server granted until 1 Nov.
- Server engineering: env vars `TLE_DATA_ROOT` / `TLE_RUN_DIR`; `TLE_SHARD` + `report_file()` for parallel shards; `run_parallel.py`; per-subject HD-BET temp dirs (shared `_hdbet_tmp` broke when other processes deleted `dataset.json`); empty-report guard in Steps 1–4 (`raise SystemExit(0)`) because shards whose subjects were all done crashed on an empty DataFrame; `merge_reports` skips `EmptyDataError` files (first size-based fix failed the laptop test).
- **CRLF bug:** `subject_list.txt` uploaded with Windows line endings → every download URL ended in `\r` → fixed with `sed`, removed bad dirs, re-downloaded. (Now prevented by `.gitattributes eol=lf`.)
- The same launch command was accidentally pasted twice → two runners → killed both, cleaned up, restarted.
- 3 borderline controls failed Step 1 on the server (MI −0.3494 / −0.3476 / −0.3495) → used the laptop's registrations (`registered_3_controls.zip`).
- **Extreme slowness:** container CPU quota is **15 CPUs** (`/sys/fs/cgroup/cpu.max` = 1500000/100000) although `nproc` shows 224. Server Steps 2–4 were abandoned; the laptop's `normalized_outputs` were uploaded in 5 zips (`upload_normalized/normalized_part1–5of5.zip`); thread limits OMP/MKL/OPENBLAS/ITK = 14 added to env on the server.
- HD-BET on GPU: user wanted it; it was not faster and parallel GPU processes crashed → `TLE_HDBET_DEVICE=cpu` on the server (note: the laptop copy of `env_server.sh` still says `cuda`; which value the server finally used is UNKNOWN — irrelevant now since Steps 2–4 came from the laptop).
- Server ran Steps 5–11 (chain, `TLE_SKIP_STABILITY=1`), then **held-out split** created (`make_heldout_split.py`), `dev_only()` added to Steps 12/13; Steps 12/13 re-run on dev only (results in section 2).
- `gpu_attention_mil.py`: first version slow because bags were prepared on CPU in float16 → bags moved to GPU. Result TLE 71.6% ± 1.6, AUC 0.767, MCC 0.44 (31 min). Side: ~chance (e.g. repeat 2 acc 0.461, AUC 0.469) → abandoned.
- `experiment_classifiers.py`: best TLE pca_avg_absdiff + LDA 73.8% / 0.814; PCA > GAT; FLAIR tokens (AUC 0.77) ≫ T1 tokens (0.59); side best only AUC 0.65 (tok_T1 LDA).

### 27 Oct
- FreeSurfer volumetry: TLE 71.1% / AUC 0.735; side 73.6% / 0.855; no-training rule 70.3% (HS 94.6%, non-HS 37.0%). With volumes: TLE vol + pca_avg_absdiff LDA 79.0% ± 3.3, AUC 0.877 (sens 75.5, spec 82.6, MCC 0.58); side vol linear SVM 73.8% / 0.859.
- `hippo_crops.py` (first attempt failed: parallel workers raced on Step 5's shared MNI template cache → per-PID cache dir; laptop ran out of memory with 6 jobs → 2 on laptop, 12 on server).
- `gpu_hippo_cnn.py` hippocampus: TLE CNN 79.6% / 0.839 (HS 76%, non-HS 76%); CNN + tabular 79.1% / 0.907. **Side CNN 86.9% / 0.945** (HS 94.9%, non-HS 75.9%).
- Improvement round: temporal-lobe box + `combine_oof.py`. Locked by the fixed rule (highest mean AUC): TLE = temporal CNN + tabular (82.3% / 0.922); side = hippo + temporal + tabular (87.5% / 0.949).
- `final_test.py` written, laptop smoke-tested (laptop lacked test crops → cropped 2 test controls sub-4061, sub-4070 for the smoke test only), run ONCE on the server: TLE 95.8% / 0.986; side 93.8% / 0.984.
- All results archived (`tle_results_final.tar.gz`, 39 MB) → laptop `Capstone\server_results_final\`.

### 28 Oct – 1 Nov
- Group doc "TLE Pipeline: All Steps and What Changed" (Claude Docs artifact on the OLD account).
- Atlas question answered (why no atlas: none provided; rigid-only alignment; FreeSurfer gives subject-specific segmentations; region boxes worked) — atlas-region graph proposed as optional extra, not done.
- Learning-rate check (mentor request): `TLE_CNN_LR` env added to `gpu_hippo_cnn.py` (output folder suffix `_lr<value>`), `lr_sweep_summary.py`; laptop smoke-tested; bundle `tle_update12.zip`. Server run status UNKNOWN (user only pasted the check command, never its output).
- GPU-extension application material: thumbnail PNG, video shot list + narration, 300-word summary.
- GitHub preparation (1 Nov): local repo initialised in `TLE\` (`git init -b main`), `.gitignore`, `.gitattributes`, `README.md`, `requirements.txt`, `results/` added, 54 files staged (~1.2 MB). **No commit yet, no remote** (waiting for the user to create an empty public repo and give the URL). User chose: **public** repo, **include** cohort/label files. IDEAS licence CC0 confirmed.
- Global git email typo fixed (`gmial` → `gmail`).

### 1–3 Nov — teaching (no code changes)
- Taught Step 5 (technical), Step 6 (slowly, piece by piece; discovered the padding-zero issue), Step 7, Step 8 pieces 1–2 and started piece 3.

### Abandoned approaches — do not retry without a new reason
- nilearn `compute_brain_mask` skull stripping (template mask, wrong).
- Tallest-peak WhiteStripe for T1.
- Native-space hemisphere assignment / edge-anchored patch grid / voxel-sized patches.
- Dice-based registration gate (unreliable for oblique FLAIR).
- Raw cosine Wilcoxon as Step 8 validation.
- Adaptive fusion gate as a lateralising mechanism (α constant).
- GAT hemisphere vectors as the main graph features (PCA consistently better).
- Attention MIL over all patches for lateralization (chance).
- Server Steps 2–4 under the 15-CPU quota; HD-BET on the GPU.
- ds004199 (FCD, Siemens) as extra data.
- Picking the best of many Step 13 variants post hoc (replaced by pre-registration + confirmation).

---

## 4. CURRENT STATE

### 4.1 Final results (the numbers to report)
**Development CV** = 5-fold × 5 repeats (TLE: matched pairs kept in the same fold; side: stratified), mean over repeats. **Held-out** = 44 locked subjects, evaluated once.

| Task | Locked model | Dev CV | Held-out test |
|---|---|---|---|
| TLE vs control | temporal-lobe CNN + tabular (LDA on volumes + PCA avg/\|L−R\|) | 82.3% ± 3.6, AUC 0.922 ± 0.017, sens 76.6, spec 88.1, F1 0.81, MCC 0.65 (n = 94) | **95.8%** (23/24) [95% CI 88–100], AUC 0.986 [0.94–1.00], sens 100, spec 91.7, F1 0.96, MCC 0.92, binomial p < 1e-4 |
| Left vs right | hippocampus CNN + temporal CNN + tabular (LR on asymmetry) | 87.5% ± 1.9, AUC 0.949 ± 0.007, sens (left) 90.3, spec (right) 84.3, F1 0.885, MCC 0.75 (n = 128) | **93.8%** (30/32) [84–100], AUC 0.984 [0.93–1.00], sens 94.1, spec 93.3, MCC 0.875; HS 22/22, non-HS 8/10 |

Held-out errors: TLE — control **sub-4041** called patient (p = 0.66). Side — **sub-221** (left, DNT) and **sub-66** (right, OTHER, p = 0.52). Each member alone also scored 87.5–93.8% on the test set.

**How to present honestly:** test > CV is mostly chance on a small test set (1 subject ≈ 4 points) plus training on more subjects; CV is the stable estimate; single site, no external validation.

Ensemble table (dev CV, mean AUC; from `results/cv_ensembles_*.csv`):

| Combination | TLE acc / AUC | Side acc / AUC |
|---|---|---|
| hippo CNN | 79.6 / 0.839 | 86.9 / 0.945 |
| temporal CNN | 78.7 / 0.865 | 86.6 / 0.942 |
| tabular | 78.1 / 0.872 | 73.4 / 0.854 |
| hippo + temporal | 81.3 / 0.865 | 87.2 / 0.948 |
| hippo + tabular | 79.1 / 0.907 | 86.1 / 0.940 |
| temporal + tabular | **82.3 / 0.922** | 87.0 / 0.940 |
| all three | 81.5 / 0.918 | **87.5 / 0.949** |

Ablation (TLE, dev CV): original Step 12 69.7% / 0.72 → volumes only 71.1 / 0.74 → PCA graph features only 73.8 / 0.81 → volumes + PCA 79.0 / 0.88 → + temporal CNN 82.3 / 0.92. Graph features add ~8 points over volumes; for side they reach only AUC 0.65.

Other results: see section 2 (Step 12/13 dev, MIL, classifiers, volumetry).

### 4.2 What was being worked on last
1. **Teaching:** Step 8, piece 3 (standardisation of the 14 features to z-scores, pooled over the 3 views). Pending check question to the user: "after standardisation, if a patch has T1 Energy z = 0, what does that tell you?" (answer: exactly average energy). Next pieces planned: 4 = projection head network (14 → 64 → 32, L2 norm), 5 = InfoNCE loss (positives/negatives, temperature), 6 = training loop, 7 = validation (retrieval + McNemar), 8 = outputs. Then Steps 9–13 and the new models.
2. **GitHub:** waiting for repo URL (see 4.4).
3. **GPU extension application:** materials delivered; submission and approval status UNKNOWN.

### 4.3 Known bugs / limitations
- Step 6 statistics include 49% padding zeros and pool all 22 sub-bands (MAD useless). See Step 6.
- Steps 8–11 and the PCA were fitted on all 219 subjects without labels, held-out images included (transductive). No label leakage.
- Choosing the best of 7 ensemble combinations on CV slightly inflates the CV numbers (held-out test guards against this).
- Step 1–3 report CSVs list only 128 / 198 / 214 of 219 subjects (bookkeeping when subjects left/re-entered the cohort); images are fine.
- Laptop `token_outputs` etc. include 4 leftover run-1 controls (server run does not).
- `step0_select_subjects.py` has hard-coded metadata paths (not env vars).
- `step13_lateralization.py` comment says best Step 12 model is linear SVM (77% / 0.83) — outdated run-1 comment.
- `run_parallel.py` / `env_server.sh` comments disagree on HD-BET device.
- 3 FLAIRs with unusual raw intensity (sub-24, sub-358, sub-65) not visually checked.
- HDx FLAIR has 5 mm slices vs 1 mm MR750 (scanner-matched, but mixed).
- No external validation.

### 4.4 Next steps (priority order)
1. **Push to GitHub.** User creates an empty PUBLIC repo on github.com (no README/licence), gives the URL; then in `C:\Users\pavne\OneDrive\Documents\TLE\`: `git commit` (message ending with the attribution line the harness requires), `git remote add origin <url>`, `git push -u origin main` (Git Credential Manager opens a browser login — the user logs in; never handle passwords). `gh` CLI is NOT installed. Repo link is needed for the GPU-extension form ("Project Link") and resume.
2. Read the LR sweep results (it ran): on the server `grep -c "^saved" /workspace/tle_data/lr_sweep.log; grep FAILED /workspace/tle_data/lr_sweep.log; sed -n '/learning-rate sensitivity/,$p' /workspace/tle_data/lr_sweep.log` (16 runs expected). Interpret as a sensitivity check only — locked models stay at lr 1e-3. Download `/workspace/last_results.tar.gz` (contains `lr_sweep.log`, `gpu_hippo`, `token_outputs`) to the laptop as backup.
3. Continue teaching from Step 8 piece 3 (user's explicit current activity).
4. Export the Claude-hosted documents (group doc, mentor doc, Steps 7–8 study doc) from the old account to Word/PDF — they will not be accessible from a new account (URLs in 7.6).
5. Update mentor doc + pipeline/dataset docx with the final results (pending since 27 Oct).
6. Optional, dev-CV only: per-sub-band wavelet statistics (+ mother-wavelet comparison db4 vs haar/sym4/coif — mentor asked), atlas-region graph (Harvard-Oxford via nilearn on MNI-aligned patch centres), answer "normalized vector" question once its meaning is clarified.

### 4.5 Texts already given to the user (reuse, don't regenerate differently)
**Resume block (LaTeX, Jake-Gutierrez template; replaces the estimate version):**
```latex
  \resumeProjectHeading
    {\textbf{Temporal Lobe Epilepsy Detection from Brain MRI} $|$ \emph{Python, PyTorch, Scikit-learn, SimpleITK}}{}
  \resumeItemListStart
    \resumeItem{Built an end-to-end pipeline to detect Temporal Lobe Epilepsy (TLE) and its affected side from 3D brain MRI (T1w + FLAIR) of \textbf{219 subjects}, automating registration, HD-BET skull stripping, N4 bias correction, and MNI alignment.}
    \resumeItem{Modelled each hemisphere as a graph of wavelet (db4) patch features with contrastive embeddings and a \textbf{Graph Attention Network}, then added 3D CNNs on hippocampal and temporal-lobe crops trained on an \textbf{NVIDIA H100}, ensembled with brain volumetry.}
    \resumeItem{Achieved \textbf{95.8\% accuracy (AUC 0.99)} for TLE vs. healthy controls and \textbf{93.8\% (AUC 0.98)} for left/right lateralization on a locked held-out test set (82\% / 88\% in repeated cross-validation).}
    \resumeItem{Prevented leakage and bias by matching patients to controls on scanner, resolution, sex, and age, locking a held-out test set before modelling, and selecting models by a pre-fixed rule; scanner-only baselines scored at chance.}
  \resumeItemListEnd
```
Resume file on laptop: `C:\Users\pavne\OneDrive\Documents\Claude code\Pavneet_Resume.tex` is an OLDER version (still "Resume Screening"); where the user's current resume source lives is UNKNOWN (they likely use Overleaf).

**Project summary (GPU-extension form, ~300 words):** "Temporal lobe epilepsy (TLE) is the most common form of drug-resistant epilepsy, and surgery can make many patients seizure-free — but only if the epileptic side of the brain is identified correctly. … [full text was given in chat; key content: problem, motivation, IDEAS 219 subjects, matching, 44 locked test subjects, preprocessing, spectral hemisphere graph learning, region-focused 3D CNNs on H100, equal-weight ensemble, CV 82.3%/0.92 and 87.5%/0.95, held-out 95.8%/0.99 and 93.8%/0.98, all HS lateralised correctly]". Exact wording UNKNOWN beyond this outline.

**Video plan (~2.5 min screen recording; form requires voice):** 1) thumbnail 0:00–0:15; 2) NiiVue raw `sub-61_T1w.nii.gz` → `normalized_outputs/sub-61/sub-61_T1w_normalized.nii.gz`; 3) `figures/step5/patch_overlay.png`, `hippo_crops/qc/sub-1.png`; 4) VS Code `gpu_hippo_cnn.py` (HippoNet, train_cnn); 5) `server_results_final/final_test.log` + results tables; 6) thumbnail again. Narration lines were provided per shot. User initially wanted no audio; told them the form explicitly asks for voice; their final choice UNKNOWN.

### 4.6 Mentor's requests (25 Oct) and status
| Request | Status |
|---|---|
| Justify patch size 32 / stride 16 | answered verbally (hippocampal cross-section + context; power of 2 for 3-level wavelet; 16 mm too small, 64 mm too few nodes; 50% overlap → any point within 8 mm of a centre). Not written into a doc. |
| "Normalized vector" | meaning never clarified — **UNKNOWN** |
| Accuracy ~85% | met on held-out; CV 82/88 |
| Summary chart of previous papers | literature review done in chat (see 3, 22–24 Oct); a formal chart/table document **not made** |
| More attention models | attention MIL done (71.6% TLE; side chance) |
| Check PCA | done — PCA beats GAT |
| Try other classifiers | done — 9 classifiers |
| Different feature-vector combinations | done — ~20 sets |
| Mother wavelet from previous studies | not done (user deprioritised: "not caring about mother wavelet") |
| Full metrics (F1, AUC, accuracy …) | done everywhere |
| Change learning rate | code ready; run status UNKNOWN |

---

## 5. CODE

All current code is in `code/` of this zip (exact copies). Directory structure on the laptop:

```
C:\Users\pavne\OneDrive\Documents\
├── TLE\                                 ← code + cohort files (git repo, not yet committed/pushed)
│   ├── pipeline_config.py               shared paths, dev_only(), report_file(), shard support
│   ├── step0_select_subjects.py         Step 0
│   ├── download_subjects.py             Step 0 helper (OpenNeuro S3, 8 parallel)
│   ├── run_steps_0_to_4.py              driver Steps 0-4
│   ├── step1_registration.py            Step 1
│   ├── step1b_recompute_mi_baseline.py  Step 1 helper
│   ├── step2_skull_stripping.py         Step 2
│   ├── step3_bias_correction.py         Step 3
│   ├── step4_intensity_normalization.py Step 4
│   ├── step5_patch_extraction.py        Step 5
│   ├── step6_localized_spectral_decomposition.py  Step 6
│   ├── step7_spectral_token_generation.py         Step 7
│   ├── step8_cross_scale_spectral_consistency.py  Step 8
│   ├── step9_hemisphere_graph_construction.py     Step 9
│   ├── step10_graph_representation_learning.py    Step 10
│   ├── step11_adaptive_hemisphere_fusion.py       Step 11
│   ├── step12_lightweight_classifier.py           Step 12
│   ├── step13_lateralization.py                   Step 13
│   ├── make_heldout_split.py            held-out split (done, refuses rerun)
│   ├── experiment_classifiers.py        classifier × feature experiments
│   ├── experiment_volumetry.py          FreeSurfer volume features
│   ├── gpu_attention_mil.py             attention MIL (not kept)
│   ├── hippo_crops.py                   hippocampus / temporal crops
│   ├── gpu_hippo_cnn.py                 crop CNN + tabular, OOF predictions
│   ├── combine_oof.py                   ensembles
│   ├── final_test.py                    one-time held-out evaluation (done)
│   ├── lr_sweep_summary.py              LR sensitivity summary
│   ├── run_parallel.py                  sharded Steps 1-4 (server)
│   ├── setup_server.sh, env_server.sh   server setup / environment
│   ├── requirements.txt, README.md, .gitignore, .gitattributes
│   ├── labels.tsv, subject_list.txt, heldout_test.txt, lateralization_discovery_patients.txt,
│   │   scanner_by_subject.csv, subjects_to_download.txt
│   ├── results\                         final CSVs (copied from server results)
│   ├── *.docx, project_pipeline_thumbnail.png
│   ├── cohort_v2_age_sex_only\, cohort_v3_scanner_matched\, cohort_v4_final_100\   old cohort backups (git-ignored)
│   ├── labels_before_split_backup.tsv   (git-ignored)
│   ├── tle_code_server.zip, tle_update1..12.zip, registered_3_controls.zip, upload_normalized\   server upload bundles (git-ignored)
│   └── PROJECT_CONTEXT_PROMPT.md        user's older tutor briefing (git-ignored; copy in docs/)
└── Capstone\                            ← DATA_ROOT
    ├── dataset_description.json, README, CHANGES
    ├── sub-*\anat\                      raw BIDS T1w/FLAIR (+ JSON)
    ├── code\                            ORIGINAL pre-project code + wrong participants.tsv (do not use)
    ├── run_balanced\                    laptop RUN_DIR (all step outputs, figures, metrics, logs,
    │                                    *_OLD_* superseded outputs, results_v1_100subjects\,
    │                                    hippo_crops\ / temporal_crops\ (~35 subjects), gpu_hippo\*_smoke)
    └── server_results_final\            everything copied back from the server (logs + run_server\…)
```

Environment / requirements (laptop tested versions): numpy 2.2.2, pandas 2.2.3, scipy 1.15.2, scikit-learn 1.6.1, SimpleITK 2.5.5, nibabel 5.4.2, nilearn 0.13.1, PyWavelets 1.8.0, torch (laptop reports `2.14.0+cpu`), networkx 3.6.1, matplotlib 3.10.1, umap-learn 0.5.12, joblib 1.4.2, tqdm 4.67.1, hd-bet 2.0.1, Python 3.11 (`C:\Python311`). Server: Python 3.10.12, torch 2.4.0a0+f70bd71a48.nv24.06 (NVIDIA container build; must not be replaced — see 6.3). `requirements.txt` uses `>=` pins.

Environment variables used by the code: `TLE_DATA_ROOT`, `TLE_RUN_DIR`, `TLE_SHARD` (i/n), `TLE_HDBET_DEVICE` (cpu/cuda), `TLE_SKIP_STABILITY=1`, `TLE_EXP_REPEATS`, `TLE_EXP_JOBS`, `TLE_VOL_DIR` (FreeSurfer tables folder), `TLE_MIL_SMOKE=1` (tiny CPU smoke test for MIL / CNN / final test), `TLE_MIL_REPEATS`, `TLE_MIL_EPOCHS`, `TLE_BOX` (hippo/temporal), `TLE_CROP_DIR` (hippo_crops/temporal_crops), `TLE_CNN_REPEATS`, `TLE_CNN_EPOCHS`, `TLE_CNN_LR`, `TLE_FINAL_SEEDS`.

---

## 6. UNIVERSITY GPU WORKFLOW

### 6.1 System
| Item | Value |
|---|---|
| Type | **JupyterLab web portal** running inside a container (no SLURM, no SSH used) |
| Address | `172.16.224.125:1003` — campus network only (protocol http/https **UNKNOWN**; VPN option **UNKNOWN**) |
| Login | through the JupyterLab page; credentials **<user provides>** |
| User inside container | `root` (prompt `root@dgxhnode5:/workspace/...`) — host is a DGX node `dgxhnode5` |
| GPU | NVIDIA H100 80GB HBM3, **MIG slice 3g.40gb** (~40 GB), driver 535.161.08, CUDA 12.5 |
| CPU | container quota **15 CPUs** (`/sys/fs/cgroup/cpu.max` = `1500000 100000`); `nproc` misleadingly shows 224 |
| Storage | `/workspace` = 100 GB persistent NFS share (survives restarts); root overlay 3.5 T (not persistent) |
| Python / torch | Python 3.10.12; torch 2.4.0a0 NVIDIA build (`nv24.06`) |
| Access period | originally until 2026-11-01; **still accessible on 2026-11-19** (extension details UNKNOWN). On that date: `/workspace` 101 G, 30 G used; `tle_data` 34 G with all 219 subjects and every `run_server` output folder; `tle_code` had 27 of 28 scripts (missing `run_steps_0_to_4.py`); `lr_sweep.log` and `/workspace/last_results.tar.gz` exist. A full sync bundle `tle_sync_full.zip` (all code + cohort + docs + results + this handover + IDEAS metadata tables in `tle_code/metadata/`) was prepared to make the server copy complete. |

### 6.2 Layout on the server
- Code: `/workspace/tle_code/` (also `/workspace/tle_code/fs_tables/` FreeSurfer tables)
- Data: `/workspace/tle_data/` (BIDS `sub-*`), run outputs `/workspace/tle_data/run_server/`
- Logs of background jobs: `/workspace/tle_data/*.log` (steps1to4.log, steps5to11.log, steps12to13.log, mil_tle.log, mil_side.log, exp_classifiers.log, exp_volumes.log, hippo.log, temporal.log, final_test.log, lr_sweep.log)
- Upload bundles: `/workspace/*.zip`; constraints file `/workspace/constraints.txt`

### 6.3 Environment setup
- One-time: `bash setup_server.sh` (in the unzipped code folder). It writes `pip freeze | grep torch…` to `/workspace/constraints.txt` and installs `SimpleITK nibabel nilearn PyWavelets networkx umap-learn matplotlib tqdm hd-bet` **with `-c constraints.txt`** so pip does not replace NVIDIA's PyTorch; prints torch/CUDA; downloads the 219 subjects.
- Every new terminal: `source /workspace/tle_code/env_server.sh` (sets `TLE_DATA_ROOT`, `TLE_RUN_DIR`, `TLE_HDBET_DEVICE`, cds to code). On the server the file also had thread limits appended (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS` = 14).
- No modules / conda used.

### 6.4 How code gets there and how jobs run
1. Claude edits code on the laptop and **tests it on the laptop first** (smoke mode / small subsets). The user insisted on this after frustration with server back-and-forth.
2. Claude bundles changed files into `C:\Users\pavne\OneDrive\Documents\TLE\tle_updateN.zip` with paths `tle_code/<file>` (Python `zipfile`). Last bundle: `tle_update12.zip`.
3. User uploads the zip via the JupyterLab file browser into `/workspace`.
4. User pastes ONE command line Claude gives, typically:
   `cd /workspace && python -c "import zipfile; zipfile.ZipFile('tle_updateN.zip').extractall()" && cd tle_code && source env_server.sh && nohup bash -c '<commands>' > /workspace/tle_data/<name>.log 2>&1 &`
5. Monitoring: user pastes outputs of `tail`/`grep` commands Claude provides, e.g. `grep -A12 -E "^(tle|side): " /workspace/tle_data/temporal.log`, `ps aux | grep -c ...`, `nvidia-smi | grep MiB`.
6. Results back: `tar czf /workspace/<name>.tar.gz ...` then JupyterLab file browser → right-click → Download; Claude extracts on the laptop.

### 6.5 Commands used per stage (server)
- Steps 1–4 (abandoned on server for speed): `python run_parallel.py [n_shards]`
- Steps 5–11 chain: `nohup bash -c 'for s in step5_patch_extraction step6_localized_spectral_decomposition step7_spectral_token_generation step8_cross_scale_spectral_consistency step9_hemisphere_graph_construction step10_graph_representation_learning step11_adaptive_hemisphere_fusion; do echo "########## $s"; TLE_SKIP_STABILITY=1 python -u $s.py || { echo "FAILED $s"; break; }; done; echo "CHAIN DONE"' > /workspace/tle_data/steps5to11.log 2>&1 &` (exact text approximate; structure as used for Steps 12–13 below)
- Steps 12–13: `nohup bash -c 'for s in step12_lightweight_classifier step13_lateralization; do echo "########## $s"; OMP_NUM_THREADS=8 python -u $s.py || { echo "FAILED $s"; break; }; done; echo "CHAIN DONE"' > /workspace/tle_data/steps12to13.log 2>&1 &`
- MIL: `OMP_NUM_THREADS=4 nohup python -u gpu_attention_mil.py tle > /workspace/tle_data/mil_tle.log 2>&1 &` (and `side`)
- Classifiers: `TLE_EXP_JOBS=8 python -u experiment_classifiers.py tle; ... side` (> exp_classifiers.log)
- Volumes: `python -u experiment_volumetry.py fs_tables; export TLE_VOL_DIR=fs_tables; TLE_EXP_JOBS=8 python -u experiment_classifiers.py tle; ... side` (> exp_volumes.log)
- Hippo CNN: `python -u hippo_crops.py 12 && TLE_VOL_DIR=fs_tables python -u gpu_hippo_cnn.py tle && TLE_VOL_DIR=fs_tables python -u gpu_hippo_cnn.py side` (> hippo.log; ~6 s per fold)
- Temporal + ensembles: `TLE_BOX=temporal python -u hippo_crops.py 12 && export TLE_VOL_DIR=fs_tables TLE_CROP_DIR=temporal_crops && python -u gpu_hippo_cnn.py tle && python -u gpu_hippo_cnn.py side && python -u combine_oof.py tle && python -u combine_oof.py side` (> temporal.log)
- Final test (done once): `export TLE_VOL_DIR=fs_tables && nohup bash -c 'python -u final_test.py tle && python -u final_test.py side' > /workspace/tle_data/final_test.log 2>&1 &`
- LR sweep (status UNKNOWN): `export TLE_VOL_DIR=fs_tables && nohup bash -c 'for lr in 1e-4 3e-4 3e-3 1e-2; do for c in hippo_crops temporal_crops; do for t in tle side; do TLE_CNN_LR=$lr TLE_CROP_DIR=$c python -u gpu_hippo_cnn.py $t || echo "FAILED $lr $c $t"; done; done; done; python lr_sweep_summary.py tle; python lr_sweep_summary.py side' > /workspace/tle_data/lr_sweep.log 2>&1 &`
- Archive: `cd /workspace/tle_data && tar czf /workspace/tle_results_final.tar.gz --ignore-failed-read *.log run_server/{final_test,gpu_hippo,gpu_mil,experiments,classifier_outputs,lateralization_outputs,metrics,figures,logs,embedding_outputs,fusion_outputs}`

### 6.6 Problems hit on the server and fixes
| Problem | Fix |
|---|---|
| CRLF line endings in `subject_list.txt` → URLs ending in `\r` | `sed -i 's/\r$//'`, deleted bad dirs, re-downloaded; now `.gitattributes` forces LF |
| Same command pasted twice → duplicate runners | kill both, clean, restart; give one-line commands |
| 3 controls fail Step 1 MI gate by ~0.001 | used laptop registrations (`registered_3_controls.zip`) |
| HD-BET shared temp dir deleted by other processes | per-subject temp dir |
| Concurrent HD-BET weight downloads | resolved once weights existed |
| `merge_reports` EmptyDataError | try/except |
| All shards crash on empty report | empty-report guard in Steps 1–4 |
| Very slow everything | 15-CPU quota discovered; skip server Steps 2–4, upload laptop outputs, thread limit 14 |
| MIL slow | bags kept on GPU, batches built on device |
| `hippo_crops.py` crash: `Unable to determine ImageIO reader for .../_mni152_brain_mask_1mm.nii.gz` | per-process MNI cache dir |
| Grep returned nothing because job still running | check `tail` + process count before assuming failure |
| Stray `]` at prompt broke a pasted command | press Ctrl+C before pasting |
| pip could replace NVIDIA torch | install with constraints file |

---

## 7. CONVENTIONS AND DECISIONS

### 7.1 Naming
- Subjects `sub-<ID>`; controls ≥ 4001. Step scripts `stepN_<name>.py`; outputs `<thing>_outputs/sub-X/`; superseded outputs renamed `*_OLD_<reason>`; server update bundles `tle_updateN.zip`.
- Hemisphere labels "L"/"R" (RAS, world x < 0 = left). Task names `tle` (patient vs control) and `side` (left vs right; positive class = left).

### 7.2 Seeds
- 42 everywhere (registration sampling, numpy RNGs, torch, CV splits base seed; repeats use SEED + r).
- Held-out split seed 20261025. CNN fold seeds `SEED + 100*r + k`; final-test CNN seeds 1042–1046; bootstrap rng 42.

### 7.3 Splits and leakage control
- 44-subject held-out set locked before any new modelling; `dev_only()` in every experiment; `make_heldout_split.py` and `final_test.py` refuse to run twice. Held-out test run exactly once (27 Oct). **Do not touch it again.**
- TLE CV: matched pairs kept together (own pair-grouped shuffling; sklearn StratifiedGroupKFold ignores seed here). Side CV: StratifiedKFold.
- Scalers, LDA/LR, AsymZ (control reference from training fold for TLE) fitted inside folds; CNNs retrained per fold.
- Labels never used in Steps 6–11 (self-supervised); known transductive caveat.
- Model selection rule fixed in advance: highest mean AUC; equal-weight ensembles only (no fitted weights).
- Step 13 primary analysis pre-registered; confirmation on unseen patients.
- Matching on scanner, protocol, sex, age; scanner-only and demographics-only baselines must stay at chance.
- To recreate the cohort without the shipped files: run Step 0 with the IDEAS tables, then `make_heldout_split.py` (deterministic with seed 20261025 given the same `labels.tsv`) — but prefer the shipped `cohort/` files.

### 7.4 Reporting conventions
- Always report accuracy, balanced accuracy, sensitivity, specificity, PPV, NPV, F1, AUC, MCC (function `metrics()` in `experiment_classifiers.py`), mean ± std over repeats, HS vs non-HS subgroups, and both CV and held-out numbers together.

### 7.5 How the user likes to work (stated preferences)
- **Test on the laptop before giving server commands**; give ONE pasteable command at a time; tell exactly what to paste back. The user got very frustrated by server back-and-forth ("its so frustrating").
- User wants to **leverage the GPU** whenever sensible.
- Priorities set by the user: "just work on improving accuracy"; mother-wavelet work deprioritised.
- **Teaching style:** move slowly, one small "piece" at a time, plain language first, then technical "how"; worked numeric examples; visual diagrams when asked ("show me visually"); end each piece with a check question and wait. User says "next" to continue, asks for recaps ("revise piece 1, 2"), and dislikes extra documents when they asked for chat ("give me a recap here only").
- Ask before making extra deliverables (e.g. said "ask me yes or no before making a video").
- Documents for others: simple, not AI-sounding, no "mentor report" wording; diagrams simple.
- Resume: one page, keep Volunteer section, compact Education; user insisted on estimated numbers earlier (tagged `% ESTIMATE`), now replaced with real ones. The user makes final calls.
- Writes casually (sometimes Hindi/Hinglish, e.g. mentor notes "krke dekho"); short messages.
- Use they/them for the user and teammates unless told otherwise.
- From the user's own tutor briefing (`docs/PROJECT_CONTEXT_PROMPT.md`): when asked to teach, *teach, don't build* (no code changes unless asked); intuition and analogy first, then technical detail, then why it was chosen; always connect to the project's real numbers; explain the "why" behind every fix; check understanding with 2–3 short questions; on request role-play a critical mentor/reviewer; well-structured answers, define every new term; say clearly when something isn't known instead of guessing.

### 7.6 External artefacts on the OLD Claude account (export before switching)
- Group doc "TLE Pipeline: All Steps and What Changed": https://claude.ai/artifact/G9tF4DWAFuZfAjqFfE4Vyu
- Mentor doc "TLE Detection from Brain MRI": https://claude.ai/code/artifact/4dbc1d52-c91c-4c50-81f4-53d36453fb20
- Steps 7–8 study doc for a friend: https://claude.ai/artifact/VX5dyw6XVuqDYT1r41jxJT
These are private to the old account; the new account will not be able to open them.

---

## 8. GAPS (UNKNOWN items) and first questions to ask

### 8.1 UNKNOWN list
1. Capstone final deadline, submission format (report/thesis/viva), and grading rubric.
2. Team member names and who does what; mentor's name.
3. Meaning of the mentor's "normalized vector" request.
4. GPU-server extension terms and end date (server was still accessible on 2026-11-19).
5. Learning-rate sweep results (the run exists on the server; numbers not yet read).
6. Whether the video (with or without voice) and thumbnail/summary were submitted.
7. GitHub repo URL (not created yet when this handover was written).
8. Server URL protocol (http/https), VPN availability off campus, exact login method — **<user provides>** for credentials.
9. Final value of `TLE_HDBET_DEVICE` on the server copy of `env_server.sh` (irrelevant to results).
10. Exact text of the capstone portal descriptions and of the 300-word summary (outline in 4.5).
11. Location of the user's current resume source (Overleaf?) — laptop `.tex` is outdated.
12. (Resolved) `TLE\PROJECT_CONTEXT_PROMPT.md` is an older (run-1 era, 100 subjects) tutor briefing the user wrote for another chat; included in `docs/` for its teaching instructions; its numbers are superseded by this handover.
13. Exact Step 5–11 server launch command text (structure known, wording approximate).
14. Literature numbers (Kaestner 2024 etc.) are from memory of the chat — verify before citing.
15. Whether the user wants the remaining mentor items (prior-work chart, mother-wavelet comparison) done at all.

### 8.2 Questions the new Claude should ask first
1. "Do you still have access to the college GPU server? If yes, did the learning-rate sweep finish — can you paste `grep -c "^saved" /workspace/tle_data/lr_sweep.log`?"
2. "Have you created the GitHub repo? Send me the URL and I'll commit and push the prepared files from `C:\Users\pavne\OneDrive\Documents\TLE\`."
3. "What is the capstone deadline and what must be submitted (report, presentation, viva, code)?"
4. "Do you want to continue learning from Step 8 piece 3 (standardisation), where we stopped?"
5. "Did you export the three Claude-hosted documents from your old account?"
6. "What did your mentor mean by 'normalized vector'?"
7. "Were the GPU-extension materials submitted, and with voice-over or silent?"


## UPDATE 2026-11-19 — learning-rate sweep results (dev CV, 5-fold x 5)
Final ensemble AUC, TLE vs control: 1e-4 0.873 | 3e-4 0.891 | 1e-3 0.922 (82.3%) | 3e-3 0.922 | 1e-2 0.908
Final ensemble AUC, left vs right:  1e-4 0.925 | 3e-4 0.944 | 1e-3 0.949 (87.5%) | 3e-3 0.949 | 1e-2 0.950
Conclusion: 1e-3 lies on a stable plateau (1e-3 to 3e-3); 1e-4 under-trains the CNNs (~55% alone). Locked models unchanged.
Files: /workspace/tle_data/run_server/gpu_hippo/lr_sweep_tle.csv, lr_sweep_side.csv. Mentor's learning-rate request: answered.
Server status 2026-11-19: /workspace/tle_code complete (28 scripts, docs/, results/, metadata/, HANDOVER.md); data + all outputs in /workspace/tle_data.

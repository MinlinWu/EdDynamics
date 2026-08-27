# Minimal mechanism

Run after the empirical output has been generated. Execute in order:

```bash
REPO_ROOT=/path/to/EdDynamics
OUTPUT_ROOT=/path/to/outputs
MECHANISM_MAIN=${REPO_ROOT}/s2_minimal_mechanism/main

python ${MECHANISM_MAIN}/run_minimal_mechanism_family_ablation.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --output-root ${OUTPUT_ROOT}/stage2_phase1_unified_minimality \
  --bootstrap-reps 300 --numba-threads 32 --overwrite

MECH_PHASE1_STAGE1_ROOT=${OUTPUT_ROOT}/stage1 \
MECH_PHASE1_HANDOFF_PATH=${OUTPUT_ROOT}/stage2_phase1_unified_minimality/metadata/phase1_minimal_mechanism_handoff.json \
MECH_PHASE1_OUTPUT_ROOT=${OUTPUT_ROOT}/stage2_phase1 \
python ${MECHANISM_MAIN}/tune_offset_dual_channel_phase1.py

python ${MECHANISM_MAIN}/freeze_offset_dual_channel_phase2.py \
  --phase1-script ${MECHANISM_MAIN}/tune_offset_dual_channel_phase1.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --handoff ${OUTPUT_ROOT}/stage2_phase1_unified_minimality/metadata/phase1_minimal_mechanism_handoff.json \
  --phase1-output-root ${OUTPUT_ROOT}/stage2_phase1 \
  --output-root ${OUTPUT_ROOT}/stage2_phase2_freeze

python ${MECHANISM_MAIN}/confirm_offset_dual_channel_phase3.py \
  --frozen-manifest ${OUTPUT_ROOT}/stage2_phase2_freeze/metadata/phase2_frozen_model_manifest.json \
  --phase1-script ${MECHANISM_MAIN}/tune_offset_dual_channel_phase1.py \
  --stage1-root ${OUTPUT_ROOT}/stage1 \
  --output-root ${OUTPUT_ROOT}/stage2_phase3_confirm \
  --write-full-predictions
```

Then run `robustness/README.md` and `scripts/README.md` when their prerequisite outputs exist.

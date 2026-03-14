#!/usr/bin/env bash
# run_all_oracles.sh
#
# Run the ProteinAE autoencode pipeline on ALL example PDBs with:
#   1) No guidance  (plain autoencoding baseline)
#   2) Rg guidance  (target = 1.5 nm)
#   3) Contact density guidance (target = 6.0)
#   4) H-bond guidance (target = 0.3)
#   5) Clash guidance (target = 0.0)
#
# Then run analyze_outputs.py to collate results into a single table.
#
# Usage:
#   bash run_all_oracles.sh          # run everything
#   bash run_all_oracles.sh --skip-run  # skip inference, only analyze

set -euo pipefail

PYTHON="/home/vishak/miniforge3/envs/proteinae/bin/python"
SCRIPT="proteinfoundation/autoencode.py"
EXAMPLES_DIR="examples"
ROOT_DIR="/home/vishak/IIITH-III-II/mlns/ProteinAE_v1"

cd "$ROOT_DIR"

# All PDB files in examples/
PDB_IDS=(
  "7v11"
  "1crn"
  "1ubq"
  "2gb1"
  "1bdd"
  "1csp"
  "1hrc"
  "1fkb"
  "2ci2"
  "1aho"
  "1ptq"
)

# Oracle configurations: "name|output_dir|scale|target"
CONFIGS=(
  "none|output_all_baseline|0|"
  "rg|output_all_rg|1.0|1.5"
  "contacts|output_all_contacts|1.0|6.0"
  "hbond|output_all_hbond|1.0|0.3"
  "clash|output_all_clash|1.0|0.0"
)

SKIP_RUN=false
if [[ "${1:-}" == "--skip-run" ]]; then
  SKIP_RUN=true
fi

TOTAL=0
PASS=0
FAIL=0
FAIL_LIST=()

if [[ "$SKIP_RUN" == false ]]; then
  echo "============================================================"
  echo "  ProteinAE — Full Oracle Batch Test"
  echo "  Structures : ${#PDB_IDS[@]}"
  echo "  Configs    : ${#CONFIGS[@]} (baseline + 4 oracles)"
  echo "  Total runs : $(( ${#PDB_IDS[@]} * ${#CONFIGS[@]} ))"
  echo "============================================================"
  echo ""

  for config_line in "${CONFIGS[@]}"; do
    IFS='|' read -r ORACLE OUT_DIR SCALE TARGET <<< "$config_line"

    echo ""
    echo "============================================================"
    if [[ "$ORACLE" == "none" ]]; then
      echo "  Config: BASELINE (no guidance)"
    else
      echo "  Config: oracle=$ORACLE  scale=$SCALE  target=$TARGET"
    fi
    echo "  Output: $OUT_DIR/"
    echo "============================================================"

    for pdb_id in "${PDB_IDS[@]}"; do
      PDB_PATH="${EXAMPLES_DIR}/${pdb_id}.pdb"
      TOTAL=$((TOTAL + 1))

      if [[ ! -f "$PDB_PATH" ]]; then
        echo "  [SKIP] $pdb_id — not found"
        continue
      fi

      # Skip if already completed
      if [[ -f "${OUT_DIR}/${pdb_id}/sample.pdb" ]]; then
        echo "  [CACHED] $pdb_id — already done"
        PASS=$((PASS + 1))
        continue
      fi

      echo -n "  [$pdb_id] running ... "

      # Build command
      CMD="$PYTHON $SCRIPT --input_pdb $PDB_PATH --output_dir $OUT_DIR --mode autoencode"
      if [[ "$ORACLE" != "none" ]]; then
        CMD="$CMD --guidance_oracle $ORACLE --guidance_scale $SCALE --guidance_target $TARGET"
      fi

      if $CMD > "${OUT_DIR}_${pdb_id}.log" 2>&1; then
        echo "OK"
        PASS=$((PASS + 1))
      else
        echo "FAIL (see ${OUT_DIR}_${pdb_id}.log)"
        FAIL=$((FAIL + 1))
        FAIL_LIST+=("${ORACLE}:${pdb_id}")
      fi
    done
  done

  echo ""
  echo "============================================================"
  echo "  Inference complete: $PASS passed, $FAIL failed out of $TOTAL"
  if [[ ${#FAIL_LIST[@]} -gt 0 ]]; then
    echo "  Failed: ${FAIL_LIST[*]}"
  fi
  echo "============================================================"
fi

# Run the analysis / collation script
echo ""
echo ">>> Running analysis and collation ..."
$PYTHON scripts/analyze_outputs.py \
  --output_dirs output_all_baseline output_all_rg output_all_contacts output_all_hbond output_all_clash \
  --labels baseline rg contacts hbond clash \
  --save_csv results_summary.csv

echo ""
echo ">>> Done. Summary saved to results_summary.csv"

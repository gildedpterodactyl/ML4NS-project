#!/usr/bin/env bash
# run_batch_test.sh
# Run the ProteinAE autoencoding pipeline on 10 diverse PDB structures.
# Usage: bash run_batch_test.sh [oracle] [scale] [target]
#   oracle : rg | contacts | hbond | clash  (optional, default: no guidance)
#   scale  : guidance scale eta             (default: 1.0)
#   target : target value for the oracle    (default: oracle's built-in default)
#
# Examples:
#   bash run_batch_test.sh                      # plain autoencode, no guidance
#   bash run_batch_test.sh rg 1.0 1.5           # radius-of-gyration guidance
#   bash run_batch_test.sh contacts 1.0 6.0     # contact-density guidance

set -euo pipefail

PYTHON="/home/vishak/miniforge3/envs/proteinae/bin/python"
SCRIPT="proteinfoundation/autoencode.py"
EXAMPLES_DIR="examples"
ORACLE="${1:-}"
SCALE="${2:-1.0}"
TARGET="${3:-}"

# 10 diverse test structures (downloaded to examples/)
# Fold types: all-alpha, all-beta, alpha+beta, varied lengths 20–107 res
PDB_IDS=(
  "1crn"   # Crambin          46 res  alpha+beta
  "1ubq"   # Ubiquitin        76 res  alpha+beta
  "2gb1"   # Protein G B1     56 res  alpha+beta
  "1bdd"   # Protein A B-dom  60 res  all-alpha
  "1csp"   # Cold-shock prot  67 res  all-beta
  "1hrc"   # Cytochrome c    104 res  all-alpha
  "1fkb"   # FKBP12          107 res  alpha+beta
  "2ci2"   # CI2               83 res  alpha+beta
  "1aho"   # Scorpion toxin   66 res  alpha+beta
  "1ptq"   # 1ptq              44 res  all-beta
)

# Build output tag and guidance args
if [[ -n "$ORACLE" ]]; then
  OUT_TAG="output_batch_${ORACLE}"
  GUID_ARGS="--guidance_oracle ${ORACLE} --guidance_scale ${SCALE}"
  if [[ -n "$TARGET" ]]; then
    GUID_ARGS="${GUID_ARGS} --guidance_target ${TARGET}"
  fi
else
  OUT_TAG="output_batch_autoencode"
  GUID_ARGS=""
fi

echo "============================================"
echo "  ProteinAE Batch Test"
echo "  Structures : ${#PDB_IDS[@]}"
echo "  Oracle     : ${ORACLE:-none (plain autoencode)}"
[[ -n "$ORACLE" ]] && echo "  Scale      : $SCALE"
[[ -n "$TARGET" ]] && echo "  Target     : $TARGET"
echo "  Output dir : ${OUT_TAG}/"
echo "============================================"

PASS=0
FAIL=0
FAIL_LIST=()

for pdb_id in "${PDB_IDS[@]}"; do
  PDB_PATH="${EXAMPLES_DIR}/${pdb_id}.pdb"
  OUT_DIR="${OUT_TAG}"

  if [[ ! -f "$PDB_PATH" ]]; then
    echo "[SKIP] $pdb_id — file not found at $PDB_PATH"
    continue
  fi

  echo ""
  echo ">>> [$pdb_id] Starting ..."
  # shellcheck disable=SC2086
  if $PYTHON $SCRIPT \
      --input_pdb "$PDB_PATH" \
      --output_dir "$OUT_DIR" \
      --mode autoencode \
      $GUID_ARGS \
      2>&1 | tail -5; then
    echo "    [OK] $pdb_id"
    PASS=$((PASS + 1))
  else
    echo "    [FAIL] $pdb_id"
    FAIL=$((FAIL + 1))
    FAIL_LIST+=("$pdb_id")
  fi
done

echo ""
echo "============================================"
echo "  Results: ${PASS} passed, ${FAIL} failed"
[[ ${#FAIL_LIST[@]} -gt 0 ]] && echo "  Failed: ${FAIL_LIST[*]}"
echo "  Outputs saved to: ${OUT_TAG}/"
echo "============================================"

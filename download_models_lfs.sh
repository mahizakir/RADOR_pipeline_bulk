#!/usr/bin/env bash
set -euo pipefail

# Unified download script
# - creates project-local model cache under ./models
# - can git-clone or add Hugging Face model repos as git submodules
# - optionally runs git-lfs fetch/pull
#
# Usage examples:
#   download_models_lfs.sh                 # default: perform git clone for listed models
#   download_models_lfs.sh --submodules    # add repos as git submodules instead
#   download_models_lfs.sh --submodules --lfs
#   download_models_lfs.sh --lfs          # clone then lfs pull inside each cloned repo


MODELS_DIR="$(pwd)/models"
HF_HOME="$MODELS_DIR/huggingface"
mkdir -p "$MODELS_DIR"
mkdir -p "$HF_HOME"

# Export environment variables so other tools/libraries pick them up
export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

USE_SUBMODULES=0
DO_LFS=0

print_usage(){
  sed -n '1,160p' <<'USAGE'
Usage: download_models_unified.sh [--submodules] [--lfs] [--help]

Options:
  --submodules   Add the models as git submodules under ./models/huggingface/OWNER/REPO
  --lfs          Attempt git-lfs fetch/pull inside each model repo (useful with either clone or submodule modes)
  --help         Show this message

This script downloads two models used by the pipeline:
  - sentence-transformers/paraphrase-MiniLM-L6-v2
  - unsloth/gemma-3-27b-it-unsloth-bnb-4bit

It stores files under ./models/ by default and exports HF cache env vars.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --submodules)
      USE_SUBMODULES=1; shift ;;
    --lfs)
      DO_LFS=1; shift ;;
    -h|--help)
      print_usage; exit 0 ;;
    *)
      echo "Unknown arg: $1"; print_usage; exit 2 ;;
  esac
done

echo "Models directory: $MODELS_DIR"
echo "HF cache (HF_HOME): $HF_HOME"
echo "Mode: $( [ $USE_SUBMODULES -eq 1 ] && echo "submodules" || echo "git clone" )"
echo "git-lfs: $( [ $DO_LFS -eq 1 ] && echo "enabled" || echo "disabled" )"
echo ""

SENT_MODEL="sentence-transformers/paraphrase-MiniLM-L6-v2"
VLLM_MODEL="unsloth/gemma-3-27b-it-unsloth-bnb-4bit"

HUB_URL_BASE="https://huggingface.co"

add_submodule(){
  local repo_id="$1"
  owner=${repo_id%%/*}
  repo=${repo_id#*/}
  target="$HF_HOME/$owner/$repo"
  url="$HUB_URL_BASE/$repo_id"

  mkdir -p "$(dirname "$target")"
  if [ -e "$target" ]; then
    echo "Submodule path already exists: $target"
  else
    echo "Adding submodule $repo_id -> $target"
    git submodule add "$url" "$target" || { echo "submodule add failed for $repo_id"; return 1; }
    # init shallow if possible
    if git help submodule | grep -q -- '--depth'; then
      git submodule update --init --depth 1 -- "$target" || true
    else
      git submodule update --init -- "$target" || true
    fi
  fi
  if [ $DO_LFS -eq 1 ]; then
    if command -v git-lfs >/dev/null 2>&1; then
      echo "Running git-lfs fetch/pull in $target"
      (cd "$target" && git lfs fetch --all || true)
      (cd "$target" && git lfs pull --include="*" --exclude="" || true)
    else
      echo "git-lfs not installed; skipping LFS for $repo_id"
    fi
  fi
}

clone_repo(){
  local repo_id="$1"
  owner=${repo_id%%/*}
  repo=${repo_id#*/}
  target="$HF_HOME/$owner/$repo"
  url="$HUB_URL_BASE/$repo_id"

  mkdir -p "$(dirname "$target")"
  if [ -d "$target" ] && [ "$(ls -A "$target")" ]; then
    echo "Target already exists and is non-empty: $target"
    return 0
  fi
  echo "Cloning $repo_id -> $target"
  if command -v git >/dev/null 2>&1; then
    if git clone --depth 1 --jobs 1 "$url" "$target"; then
      echo "Cloned $repo_id"
    else
      echo "git clone failed for $repo_id"; return 1
    fi
  else
    echo "git not found on PATH"; return 2
  fi

  if [ $DO_LFS -eq 1 ]; then
    if command -v git-lfs >/dev/null 2>&1; then
      echo "Running git-lfs fetch/pull in $target"
      (cd "$target" && git lfs fetch --all || true)
      (cd "$target" && git lfs pull --include="*" --exclude="" || true)
    else
      echo "git-lfs not installed; skipping LFS for $repo_id"
    fi
  fi
}

# Download sentence-transformer
echo "[1/2] Preparing sentence-transformer: $SENT_MODEL"
if [ $USE_SUBMODULES -eq 1 ]; then
  add_submodule "$SENT_MODEL"
else
  clone_repo "$SENT_MODEL"
fi

echo ""
# Download vLLM model
echo "[2/2] Preparing vLLM model: $VLLM_MODEL"
if [ $USE_SUBMODULES -eq 1 ]; then
  add_submodule "$VLLM_MODEL"
else
  clone_repo "$VLLM_MODEL"
fi

echo ""
echo "Summary:"
echo "  HF cache: $HF_HOME"
echo "Operations completed. If any repositories use Git LFS and you didn't enable --lfs, run 'git lfs pull' inside the model directory or re-run with --lfs."

echo "Done. You may now 'git status' and commit submodule entries if you used --submodules."
s
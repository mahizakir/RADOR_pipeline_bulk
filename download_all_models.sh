#!/bin/bash

# ============================================================================
# RADOR Pipeline - Complete Download Script
# Downloads all required models and data files
# ============================================================================

set -e  # Exit on error

# Directory to store downloaded models locally (instead of default cache)
MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"

# Configure HuggingFace / Transformers cache locations so downloads go into MODELS_DIR
export HF_HOME="$MODELS_DIR/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================"
echo -e "${BLUE}RADOR Pipeline - Complete Setup${NC}"
echo "================================================================"
echo ""
echo "This script will download:"
echo "  1. NLTK punkt tokenizer"
echo "  2. Sentence Transformer model (paraphrase-MiniLM-L6-v2)"
echo "  3. vLLM model (unsloth/gemma-3-27b-it-unsloth-bnb-4bit) ~19GB"
echo ""
echo "================================================================"
echo ""

# ============================================================================
# Step 1: Download NLTK Data
# ============================================================================
echo -e "${YELLOW}[1/3] Downloading NLTK punkt tokenizer to project-local ./nltk_data...${NC}"
# Use project-local NLTK data directory under models so the repo keeps all
# downloads in the same `models/` area managed by the script.
NLTK_ROOT="$MODELS_DIR/nltk_data"
export NLTK_DATA="$NLTK_ROOT"
NLTK_DIR="$NLTK_ROOT/tokenizers"
mkdir -p "$NLTK_DIR"

if [ -d "$NLTK_DIR/punkt" ]; then
    echo -e "${GREEN}✓ NLTK punkt already exists, skipping...${NC}"
else
    echo "Downloading punkt tokenizer..."
    wget -q --show-progress https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip -O /tmp/punkt.zip
    unzip -q /tmp/punkt.zip -d "$NLTK_DIR"
    rm /tmp/punkt.zip
    echo -e "${GREEN}✓ NLTK punkt downloaded successfully${NC}"
fi

# Download punkt_tab (optional, for newer NLTK versions)
if [ -d "$NLTK_DIR/punkt_tab" ]; then
    echo -e "${GREEN}✓ NLTK punkt_tab already exists, skipping...${NC}"
else
    echo "Downloading punkt_tab tokenizer (optional)..."
    wget -q --show-progress https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip -O /tmp/punkt_tab.zip 2>/dev/null || echo "punkt_tab not available (optional, skipping)"
    if [ -f /tmp/punkt_tab.zip ]; then
        unzip -q /tmp/punkt_tab.zip -d "$NLTK_DIR"
        rm /tmp/punkt_tab.zip
        echo -e "${GREEN}✓ NLTK punkt_tab downloaded${NC}"
    fi
fi
echo ""

# ============================================================================
# Step 2: Download Sentence Transformer Model
# ============================================================================
echo -e "${YELLOW}[2/3] Downloading Sentence Transformer model...${NC}"
SENTENCE_MODEL="sentence-transformers/paraphrase-MiniLM-L6-v2"

# Use git clone exclusively for the sentence-transformer model so the project
# stores a local copy under the HF_HOME cache. If git is not available or
# cloning fails, exit with an error so the user can install/configure git.
if command -v git &> /dev/null; then
    echo "Using git to clone the sentence-transformers model (shallow clone)..."
    SENT_TARGET="$HF_HOME/sentence-transformers/paraphrase-MiniLM-L6-v2"
    mkdir -p "$(dirname "$SENT_TARGET")"
    if [ -d "$SENT_TARGET" ] && [ "$(ls -A "$SENT_TARGET")" ]; then
        echo -e "${GREEN}✓ Sentence Transformer directory already exists at ${SENT_TARGET}, skipping clone${NC}"
    else
        echo "Cloning https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2 -> $SENT_TARGET"
        if git clone --depth 1 --jobs 1 "https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2" "$SENT_TARGET"; then
            echo -e "${GREEN}✓ Sentence Transformer model cloned to ${SENT_TARGET}${NC}"
        else
            echo -e "${RED}✗ git clone failed for sentence-transformers model. Please ensure 'git' is installed and you have network access.${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}✗ 'git' not found on PATH. Install git and try again.${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Step 3: Download vLLM Model
# ============================================================================
echo -e "${YELLOW}[3/3] Downloading vLLM model (this will take time, ~50GB)...${NC}"
VLLM_MODEL="unsloth/gemma-3-27b-it-unsloth-bnb-4bit"

echo "Model: $VLLM_MODEL"
echo "This is a large download and may take significant time depending on your connection..."
echo ""

# Use git clone exclusively for the vLLM repo. If git is not available or
# cloning fails, exit with an error so the user can install/configure git and
# git-lfs as appropriate.
if command -v git &> /dev/null; then
    echo "Using git to clone the model repository (shallow clone)..."
    TARGET_DIR="$HF_HOME/unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
    mkdir -p "$(dirname "$TARGET_DIR")"
    if [ -d "$TARGET_DIR" ] && [ "$(ls -A "$TARGET_DIR")" ]; then
        echo -e "${GREEN}✓ Model directory already exists at ${TARGET_DIR}, skipping clone${NC}"
    else
        echo "Cloning https://huggingface.co/$VLLM_MODEL -> $TARGET_DIR"
        if git clone --depth 1 --jobs 1 "https://huggingface.co/$VLLM_MODEL" "$TARGET_DIR"; then
            echo -e "${GREEN}✓ vLLM model cloned to ${TARGET_DIR}${NC}"
            echo "Note: If the repository uses Git LFS for large files you may need to run 'git lfs pull' in the model directory or install git-lfs: https://github.com/git-lfs/git-lfs"
        else
            echo -e "${RED}✗ git clone failed. Please ensure 'git' is installed and you have network access. If the repo uses LFS, install 'git-lfs' and run 'git lfs pull' in ${TARGET_DIR}${NC}"
            exit 1
        fi
    fi
else
    echo -e "${RED}✗ 'git' not found on PATH. Install git and try again. For large model files also install git-lfs: https://github.com/git-lfs/git-lfs${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "================================================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================================================"
echo ""
echo "Downloaded files are cached at:"
echo "  • NLTK data:            $NLTK_ROOT/"
echo "  • HuggingFace models:   $HF_HOME (under project ./models/)"
echo ""
echo "Models downloaded:"
echo "  ✓ NLTK punkt tokenizer"
echo "  ✓ Sentence Transformer: paraphrase-MiniLM-L6-v2"
echo "  ✓ vLLM model: unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
echo ""
echo "You can now run the pipeline with:"
echo -e "${BLUE}  python pipeline_code_with_vllm.py <disease_name>${NC}"
echo ""
echo "Example:"
echo -e "${BLUE}  python pipeline_code_with_vllm.py GNE_myopathy${NC}"
echo "================================================================"

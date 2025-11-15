echo 'make sure run this at top level project dir'
echo 'this will require huggingface_hub, python3.11 and stdenv/2023 module loaded'

# Ensure HuggingFace hub cache goes under ./models (project-local)
MODELS_DIR="$(pwd)/models"
mkdir -p "$MODELS_DIR"
export HF_HOME="$MODELS_DIR"
export TRANSFORMERS_CACHE="$HF_HOME"

if [ ! -d "huggingface_hub_venv" ]; then
    echo "Virtual environment huggingface_hub_venv does not exists"
    echo "Creating virtual environment..."
    virtualenv huggingface_hub_venv
    source huggingface_hub_venv/bin/activate
    pip install --no-index --upgrade pip
    pip install --no-index huggingface_hub

else
    echo "Virtual environment huggingface_hub_venv already exists"
    source huggingface_hub_venv/bin/activate
    # https://github.com/huggingface_hub/huggingface_hub/issues/2423
fi
# HF_HUB_DISABLE_XET=1 hf download --max-workers=1 unsloth/gemma-3-27b-it-unsloth-bnb-4bit
HF_HUB_DISABLE_XET=1 hf download --max-workers=1 sentence-transformers/paraphrase-MiniLM-L6-v2
HF_HUB_DISABLE_XET=1 hf download --max-workers=1 unsloth/gemma-3-27b-it-bnb-4bit
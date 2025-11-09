echo 'make sure run this at top level project dir'
echo 'this will require huggingface_hub, python3.11 and stdenv/2023 module loaded'
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
HF_HUB_DISABLE_XET=1 hf download --max-workers=1  https://huggingface.co/unsloth/gemma-3-27b-it-unsloth-bnb-4bit
HF_HUB_DISABLE_XET=1 hf download --max-workers=1  https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2

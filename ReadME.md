# Rare Disease Oracle (RaDOr) Triple Extraction Pipeline

`run_driver.py` is the driver script to automate the execution of the pipeline_code_with_vllm.py pipeline, which generates simplified sentences and structured triples from rare disease literature. The outputs will be used later for building knowledge graphs.



# Overview

The pipeline processes disease-specific text data stored in the folders:

`data/<disease_name>/`

For each disease, it generates two output CSV files and stores them inside the same disease specific folder inside the data folder:

- `generated_simple_sentences_<disease_name>.csv` — contains simplified sentences extracted from the text.
- `triples_<disease_name>.csv` — contains structured triples derived from the simplified sentences.

The goal is to process all 5000+ rare diseases found inside the `data` folder.




  

# Indexing the number of disease folders for run: 
To manage large-scale processing, diseases should be run in ranges because we cannot process 5000+ disease data at once (GPU Time for 1 diseases is between 6 to 200+ minutes):

**Example for partial batch run**
Inside the run_driver.py file:

- `start_index = 1000`  # starts at the 1000th disease (eg. start_index = 11 processes 12th row of csv)
- `end_index = 1100`    # (eg. end_index = 15 stop processing after processing 15th row of csv)

- Full dataset inside data folder has 5094 diseases.


# Rare Disease Oracle (RaDOr) Triple Extraction Pipeline

`run_driver.py` is the driver that orchestrates the pipeline in `pipeline_code_with_vllm.py`.
The pipeline processes disease-specific text found under `data/<disease_name>/`, producing simplified
sentences and structured triples used for downstream knowledge-graph work.

Important artifacts per disease (written to `data/<disease_name>/`):

- `generated_simple_sentences_<disease_name>.csv` — simplified sentences extracted from text chunks.
- `triples_<disease_name>.csv` — triples generated from the simplified sentences.

The repository contains helper scripts to download models and prepare a container environment.

---

## Quickstart — install Python deps

Create a Python environment and install required packages:

```bash
pip install -r requirements.txt
```

## Download required models and tokenizers (recommended)

This project provides `download_all_models.sh` which organizes downloads under `./models/`:

- HuggingFace models: `./models/huggingface/...`
- NLTK data (punkt tokenizer): `./models/nltk_data/tokenizers/punkt`

Run the script from the repository root:

```bash
bash download_all_models.sh
```

Notes:
- The script uses `git clone --depth 1` to fetch HuggingFace repos into `models/huggingface/...`.
  Many model repositories use Git LFS for large weight files. After cloning a repo you may
  need to run `git lfs pull` inside the model directory (and have `git-lfs` installed).
- The script places NLTK data under `models/nltk_data` and exports `NLTK_DATA` when run.

## How the pipeline finds models/tokenizers

- Sentence Transformer: the code prefers the local clone at
  `models/huggingface/sentence-transformers/paraphrase-MiniLM-L6-v2` if present; otherwise it falls back
  to the HF model id and will download into your HF cache on first use.
- vLLM model (Gemma): the pipeline prefers a local clone at
  `models/huggingface/unsloth/gemma-3-27b-it-unsloth-bnb-4bit` if present. You can also specify a custom
  local path using the environment variable `LLM_LOCAL_PATH` or by creating the file
  `models/llm_local_path.txt` containing the path.

## Run the pipeline (host)

Run the driver which iterates over a range of disease folders. Edit `run_driver.py` to set
`start_index` / `end_index`, or pass arguments as implemented by the driver.

```bash
python run_driver.py
```

Log output is appended to `DRAC_output_log_from_pipeline.log`.

---

## Using Apptainer (recommended for reproducible environments)

This repository includes Apptainer artifacts in the `apptainer/` directory:

- `vllm_apptainer.def` — definition file to build an image from scratch
- `vllm_apptainer_no_docker.def` — alternate definition
- `vllm_apptainer.sif` — pre-built Apptainer image (if present)

Build the SIF from the definition (requires root or fakeroot support):

```bash
# As root (or with sudo) on systems that support it
apptainer build vllm_apptainer.sif apptainer/vllm_apptainer.def

# Or, if your system supports fakeroot builds with apptainer:
apptainer build --fakeroot vllm_apptainer.sif apptainer/vllm_apptainer.def
```

If a pre-built `vllm_apptainer.sif` is already present in `apptainer/`, you can use it directly.

Run the pipeline inside the container (GPU-enabled):

```bash
# from repo root, bind the repo so the container can access code and the models folder
apptainer exec --nv --bind $(pwd):/workspace apptainer/vllm_apptainer.sif \
    bash -lc "cd /workspace && bash download_all_models.sh && python pipeline_code_with_vllm.py <disease_name>"
```

Or if you already downloaded models on the host and just want to run the pipeline inside the container:

```bash
apptainer exec --nv --bind $(pwd):/workspace apptainer/vllm_apptainer.sif \
    bash -lc "cd /workspace && python pipeline_code_with_vllm.py <disease_name>"
```

Notes for Apptainer:
- Use `--nv` to enable NVIDIA GPU support (requires NVIDIA drivers and apptainer compiled with GPU support).
- The `--bind $(pwd):/workspace` makes your project available inside the container at `/workspace`.
- If your model clones used Git LFS on the host, ensure the model directory contains actual weight files before running inside the container, or run `git lfs pull` inside the container in the model directory.

## Recommendations & troubleshooting

- If `git clone` fails due to Git LFS, install `git-lfs` and run `git lfs pull` inside the cloned repo:

```bash
sudo apt-get install git-lfs   # or use your distro's package manager
git lfs install
cd models/huggingface/<repo>
git lfs pull
```

If you run this repository on Compute Canada or other HPC systems that use Environment Modules, load the git-lfs module before pulling LFS objects:

```bash
module load git-lfs/3.4.0
git lfs install
git lfs pull
```

- If the pipeline cannot find NLTK data, ensure you ran `download_all_models.sh` or set `NLTK_DATA` to
  point at `models/nltk_data`.
- To explicitly point the pipeline at a different vLLM model directory, set `LLM_LOCAL_PATH`:

```bash
export LLM_LOCAL_PATH=/absolute/path/to/local/gemma
python pipeline_code_with_vllm.py <disease_name>
```

## Additional resources

Helpful external documentation related to model downloads and container setup:

- HuggingFace usage and best practices: https://docs.alliancecan.ca/wiki/Huggingface
- Apptainer (Singularity) documentation: https://docs.alliancecan.ca/wiki/Apptainer


## Contact / next steps

If you'd like, I can:

- add a small self-test mode to `run_driver.py` that runs a single file end-to-end with dummy inputs,
- add `git-lfs` detection to the download script and automatically run `git lfs pull` where available,
- or add flags to `download_all_models.sh` to select which models to download.

---

Happy running — tell me if you want any of the optional improvements added.


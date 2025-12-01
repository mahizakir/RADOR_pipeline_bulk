# Rare Disease Oracle (RaDOr) Triple Extraction Pipeline

`run_driver.py` is the driver script to automate the execution of the paper_processor.py pipeline, which generates simplified sentences and structured triples from rare disease literature. The outputs will be used later for building knowledge graphs.



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

`run_driver.py` is the driver that orchestrates the pipeline in `paper_processor.py`.
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
pip install vllm==0.10.1.1 \
    pandas \
    sentence_transformers \
    bitsandbytes==0.48.2 \
    flashinfer_python==0.4.1 
```
Note `flashinfer_python` is not require, without this vllm will fallback to the PyTorch-native implementation of top-p & top-k sampling. If you are running into trouble with `flashinfer_python`, leave without it

## Run the pipeline (host)
Config Pipeline

## Update configs
Change path to model snapshot in `model_config.json`
A example model config looks like following
```json
{
  "llm_model_name": "unsloth/gemma-3-27b-it-bnb-4bit",
  "llm_model_path": "models/hub/models--unsloth--gemma-3-27b-it-bnb-4bit/snapshots/c08b2ba63738aa8cfc60f06741d6356ef4e60b3f",
  "llm_max_model_len": null,
  "quantization": "bitsandbytes",
  "gpu_memory_utilization": 0.9,
  "tensor_parallel_size": 1,
  "batch_size": 45,
  "similarity_threshold": 0.95,
  "sentence_transformer_model": "sentence-transformers/paraphrase-MiniLM-L6-v2",
  "sentence_transformer_model_path": "models/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2/snapshots/c9a2bfebc254878aee8c3aca9e6844d5bbb102d1",
  "keywords_csv": "static_data/overall_keywords.csv",
  "log_file": null
}
```

Run  `paper_processor.py` directly 
```bash
python paper_processor.py -d <disease_name>
```

Use `run_driver.py`
Run the driver which iterates over a range of disease folders. Edit `run_driver.py` to set
`start_index` / `end_index`, or pass arguments as implemented by the driver.

```bash
python run_driver.py
```

---
## Using Apptainer

This repository includes Apptainer artifacts in the `apptainer/` directory:

- `vllm_apptainer.def` — definition file to build an image from scratch
- `vllm_apptainer.sif` — pre-built Apptainer image (if present)

Build the SIF from the definition (requires root or fakeroot support):

```bash
# As root (or with sudo) on systems that support it
apptainer build apptainer/vllm_apptainer.sif apptainer/vllm_apptainer.def

# Or, if your system supports fakeroot builds with apptainer:
apptainer build --fakeroot vllm_apptainer.sif apptainer/vllm_apptainer.def
```

In case tmp folder size is limited and you incounter  `no space left on device` error,  use following command
```bash
apptainer build --tmpdir $(pwd)/apptainer/tmp  apptainer/vllm_apptainer.sif apptainer/vllm_apptainer.def 
```

If a pre-built `vllm_apptainer.sif` is already present in `apptainer/`, you can use it directly.

## Run the pipeline inside the container (GPU-enabled):
Make sure your models are already downloaded to model diretory, and config file is up-to-data.

```bash
apptainer exec -C --nv -B ${SLURM_TMPDIR}:/workspace \
    -B ${SLURM_TMPDIR}/working_dir/cache:/root/.cache \
    -B ${SLURM_TMPDIR}/working_dir/tmp:/tmp \
    apptainer/vllm_apptainer.sif \
    bash -lc "cd /workspace && python paper_processor.py -d <disease_name>"
```

Notes for Apptainer:
- Use `--nv` to enable NVIDIA GPU support (requires NVIDIA drivers and apptainer compiled with GPU support).
- The `--bind $(pwd):/workspace` makes your project available inside the container at `/workspace`.

# Run on DRAC

## Download Model Files
As most of HPCs doess not allow internet acccess on compute node, it is requried to pre-download model weight on the login node

```bash
bash download_all_models_cc.sh
```

## Update configs
Change path to model snapshot in `model_config.json`
A example model config looks like following

other tried model
```
{
  "llm_model_name": "unsloth/gemma-3-27b-it-bnb-4bit",
  "llm_model_path": "models/hub/models--unsloth--gemma-3-27b-it-bnb-4bit/snapshots/c08b2ba63738aa8cfc60f06741d6356ef4e60b3f",
}
```
## create csv for input
```bash 
python collect_result.py -u
```

## check pogress
```bash 
python collect_result.py
```

## collect data
```bash 
python collect_result.py -c
```

## use slrum
use `slurm_scripts/example_slurm_fir.sh`
## Notes
As for 11/16/2025 - appatiner image build based on `nvcr.io/nvidia/vllm:25.09-py3` use `apptainer/vllm_apptainer_nv_vllm.def` on works on A6000 and L40s, not H100

## Additional resources

Helpful external documentation related to model downloads and container setup:

- HuggingFace usage and best practices: https://docs.alliancecan.ca/wiki/Huggingface
- Apptainer (Singularity) documentation: https://docs.alliancecan.ca/wiki/Apptainer

# TODO 
we need better data handling here, now we are assume data will be in the `data` folder. we should have data as a path in each slurm script and also has collect_results script to create tracking file for each task in a working dir.
#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=vllm_rador_example
#SBATCH --mem=144G
# https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles
#SBATCH --cpus-per-task=6 
# https://docs.alliancecan.ca/wiki/Firs
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-0:15
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/RADOR_pipeline_bulk/jobs/rador_vllm_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/RADOR_pipeline_bulk/jobs/rador_vllm_%A_%a.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

#load modules
module load StdEnv/2023
module load opencv/4.12
module load cudacore/.12.6.3

# set variablesss
RADOR_DP=/home/feiw/project/feiw/RADOR_pipeline_bulk
WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}

# get disease list
N=1
DISEASE_LIST_FILE=${RADOR_DP}/static_data/5094_disease_folders_names.csv
DISEASE_LIST=$(sed -n "1,${N}p" ${DISEASE_LIST_FILE} | tr '\n' ' ')
echo "DISEASE_LIST:"
for d in ${DISEASE_LIST}; do
    echo " ${d}"
done

# prepare working directory on the node
echo create working dir: ${WORKING_DIR}
mkdir -p ${WORKING_DIR}
mkdir -p ${WORKING_DIR}/cache

# Use a node-local cache directory for vLLM to avoid shared-FS compile/cache issues
# Point vLLM's cache root to the job-local cache directory we just created.
export VLLM_CACHE_ROOT=${WORKING_DIR}/cache
echo "VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT}"
# Point flashinfer's cache root to the job-local cache directory we just created.
export FLASHINFER_WORKSPACE_BASE=${WORKING_DIR}/cache
echo "FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE}"

# copy model ckpts to slurm local dir
echo copy model ckpts to slurm local dir
rsync -av --exclude=".*" ${RADOR_DP}/models ${WORKING_DIR}/
# copy pipeline code to slurm local dir
echo copy pipeline code to slurm local dir
rsync -av ${RADOR_DP}/*.py ${WORKING_DIR}/
rsync -av ${RADOR_DP}/static_data ${WORKING_DIR}/
rsync -av ${RADOR_DP}/model_config.json ${WORKING_DIR}/
# link data to slurm local dir
ln -s ${RADOR_DP}/data ${WORKING_DIR}/data

# check GPU visibility
echo "Checking GPU visibility..."
echo cuda_visible_devices=$CUDA_VISIBLE_DEVICES
nvidia-smi             # check memory of the MIG slice you were assigned
nvidia-smi -L          # should show 1 device

# create local env
cd  ${WORKING_DIR}
virtualenv RADOR_env
source RADOR_env/bin/activate
pip install --no-index vllm==0.10.1.1 \
    pandas \
    sentence_transformers \
    bitsandbytes==0.48.2 \
    flashinfer_python==0.4.1 # this is not required but good to have, without this vllm will fallback to the PyTorch-native implementation of top-p & top-k sampling.

# run paper_processor
echo running paper_processor

#export CUDA_LAUNCH_BLOCKING=1
cd ${WORKING_DIR}
for d in ${DISEASE_LIST}; do
    echo processing disease: ${d}
    python paper_processor.py -d ${d}
done
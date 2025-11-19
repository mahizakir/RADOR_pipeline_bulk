#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=vllm_rador_example
#SBATCH --mem=144G
# https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles
#SBATCH --cpus-per-task=6 
# https://docs.alliancecan.ca/wiki/Fir
#SBATCH --gpus=h100:1
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=/home/feiw/project/feiw/RADOR_pipeline_bulk/jobs/rador_vllm_%A_%a.out
#SBATCH --error=/home/feiw/project/feiw/RADOR_pipeline_bulk/jobs/rador_vllm_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=1-2000:12%30 # 12 tasks at a time, each task process 12 diseases

# record start
echo `date`: Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} is allocated resources
# record epoch start for elapsed time calculation
START_TIME=$(date +%s)

#load modules
module load StdEnv/2023
module load apptainer/1.3.5

# set variables
RADOR_DP=/home/feiw/project/feiw/RADOR_pipeline_bulk
WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}

# get disease list
N=12
DISEASE_LIST_FILE=${RADOR_DP}/need_compute_disease.csv
DISEASE_LIST=$(sed -n "${SLURM_ARRAY_TASK_ID},$(($SLURM_ARRAY_TASK_ID+$N-1))p" ${DISEASE_LIST_FILE} | tr '\n' ' ')
echo "DISEASE_LIST:"
for d in ${DISEASE_LIST}; do
    echo " ${d}"
done

# prepare working directory on the node
echo create working dir: ${WORKING_DIR}
mkdir -p ${WORKING_DIR}
mkdir -p ${WORKING_DIR}/working_dir/cache
mkdir -p ${WORKING_DIR}/working_dir/tmp

# copy apptainer container (use vLLM image)
cp ${RADOR_DP}/apptainer/vllm_apptainer.sif ${WORKING_DIR}/

# copy model ckpts to slurm local dir
echo copy model ckpts to slurm local dir
rsync -av --exclude=".*" ${RADOR_DP}/models ${WORKING_DIR}/
# copy pipeline code to slurm local dir
echo copy pipeline code to slurm local dir
rsync -av ${RADOR_DP}/*.py ${WORKING_DIR}/
rsync -av ${RADOR_DP}/static_data ${WORKING_DIR}/
rsync -av ${RADOR_DP}/utils ${WORKING_DIR}/
rsync -av ${RADOR_DP}/model_config.json ${WORKING_DIR}/

# check GPU visibility
echo "Checking GPU visibility..."
echo cuda_visible_devices=$CUDA_VISIBLE_DEVICES
nvidia-smi             # check memory of the MIG slice you were assigned

# export PYTORCH_SDP_BACKEND=math
# export CUDA_LAUNCH_BLOCKING=1
#export CUDA_LAUNCH_BLOCKING=1
# export DISABLE_TQDM=1
# echo "DISABLE_TQDM=${DISABLE_TQDM}"
# export VLLM_DISABLE_FLASHINFER=1
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# run paper_processor
echo running paper_processor
cd ${WORKING_DIR}
for d in ${DISEASE_LIST}; do
    echo processing disease: ${d}
    apptainer exec -C --nv --bind ${WORKING_DIR}:/workspace \
    --bind ${RADOR_DP}/data:/workspace/data \
    --bind ${WORKING_DIR}/working_dir/cache:/root/.cache \
    --bind ${WORKING_DIR}/working_dir/tmp:/tmp \
    ${WORKING_DIR}/vllm_apptainer.sif \
   sh -c "cd /workspace && python paper_processor.py -d ${d}"
done

# record end time and compute elapsed duration
END_TIME=$(date +%s)
ELAPSED=$((END_TIME-START_TIME))
hours=$((ELAPSED/3600))
mins=$(((ELAPSED%3600)/60))
secs=$((ELAPSED%60))
printf "%s: Job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} has completed (elapsed %02d:%02d:%02d)\n" "$(date)" "$hours" "$mins" "$secs"
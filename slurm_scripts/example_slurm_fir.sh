#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=vllm_rador_example
#SBATCH --mem=144G
# https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles
#SBATCH --cpus-per-task=6 
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb
#SBATCH --time=0-3:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=./jobs/rador_vllm_%A_%a.out
#SBATCH --error=./jobs/rador_vllm_%A_%a.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

#load modules
module load StdEnv/2023

RADOR_DP=/home/feiw/project/feiw/frag-gnn
WORKING_DIR=${SLURM_TMPDIR}/${SLURM_JOB_ID}

# set variables
N=10
DISEASE_LIST_FILE=static_data/5094_disease_folders_names.csv
DISEASE_LIST=$(sed -n "1,${N}p" ${DISEASE_LIST_FILE} | tr '\n' ' ')
echo "DISEASE_LIST:"
for d in ${DISEASE_LIST}; do
    echo " ${d}"
done

# set save dir and name
# we can get away with this for 5094 files I hopes
# DEST_DP=/scratch/feiw/rador_data/
# echo DEST_DP: ${DEST_DP}

# prepare working directory on the node
echo create working dir: ${WORKING_DIR}
mkdir -p ${WORKING_DIR}/working_dir/cache
mkdir -p ${WORKING_DIR}/working_dir/tmp

# copy apptainer container (use vLLM image)
module load apptainer/1.2.4
cp apptainer/vllm_apptainer.sif ${WORKING_DIR}/vllm_apptainer.sif

# copy model ckpts to slurm local dir
echo copy model ckpts to slurm local dir
rsync -av --exclude=".*" ${RADOR_DP}/models ${WORKING_DIR}/
# copy pipeline code to slurm local dir
echo copy pipeline code to slurm local dir
rsync -av ${RADOR_DP}/*.py ${WORKING_DIR}/
rsync -av ${RADOR_DP}/static_data ${WORKING_DIR}/
rsync -av ${RADOR_DP}/model_config.json ${WORKING_DIR}/
# copy data to slurm local dir
# echo copy data to slurm local dir
# rsync -av ${RADOR_DP}/data ${WORKING_DIR}/ 
# soft link data dir
ln -s ${RADOR_DP}/data ${WORKING_DIR}/data

# run dag gen
echo running dag gen
cd ${WORKING_DIR}
for d in ${DISEASE_LIST}; do
    apptainer exec -C --nv -B ${WORKING_DIR}:/workspace \
    -B ${WORKING_DIR}/working_dir/cache:/root/.cache \
    -B ${WORKING_DIR}/working_dir/tmp:/tmp \
    apptainer/vllm_apptainer.sif \
    python paper_processor.py ${d}
done
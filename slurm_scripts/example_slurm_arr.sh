#!/bin/bash
#SBATCH --account=rrg-dwishart
#SBATCH --nodes=1
#SBATCH --job-name=vllm_rador_example
#SBATCH --mem=90G
#SBATCH --cpus-per-task=8s
#SBATCH --time=0-1:00
#SBATCH --mail-user=fw4@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --output=./jobs/rador_vllm_%j.out
#SBATCH --error=./jobs/rador_vllm_%j.err
#SBATCH --open-mode=append

# record start
echo `date`: Job ${SLURM_JOB_ID} is allocated resources

#load modules
module load StdEnv/2023

DEST_DP=/scratch/feiw/rador_data/

echo DEST_DP: ${DEST_DP}

# create local dir
echo create local data dir: ${SLURM_TMPDIR}/data
mkdir -p ${SLURM_TMPDIR}/data

mkdir -p ${SLURM_TMPDIR}/working_dir/cache
mkdir -p ${SLURM_TMPDIR}/working_dir/tmp

# copy apptainer container (use vLLM image)
module load apptainer/1.2.4
cp apptainer/vllm_apptainer.sif ${SLURM_TMPDIR}/vllm_apptainer.sif

# copy model ckpts to slurm local dir
echo copy model ckpts to slurm local dir
rsync -av --exclude=".*" models ${SLURM_TMPDIR}
# copy pipeline code to slurm local dir
echo copy pipeline code to slurm local dir
rsync -av *.py ${SLURM_TMPDIR}/
rsync -av static_data ${SLURM_TMPDIR}/
rsync -av model_config.json ${SLURM_TMPDIR}/

# run dag gen
echo running dag gen

apptainer exec -C --nv -B ${SLURM_TMPDIR}:/workspace \
    -B ${SLURM_TMPDIR}/working_dir/cache:/root/.cache \
    -B ${SLURM_TMPDIR}/working_dir/tmp:/tmp \
    apptainer/vllm_apptainer.sif \
    python paper_processor.py yellow_fever
        
# go to frag directory
cd ${SLURM_TMPDIR}/frag/
tar -cf ${SAVE_NAME}.tar ${SAVE_NAME}

# copy tar to scratch
echo create dir at ${DEST_DP}
mkdir -p ${DEST_DP}
echo copy tar: ${SAVE_NAME}.tar to ${DEST_DP}
cp ${SAVE_NAME}.tar ${DEST_DP}

# record ending
echo `date`: Job ${SLURM_JOB_ID} finished running
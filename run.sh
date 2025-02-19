#!/bin/bash
#SBATCH --job-name=ml-job          # Job name
#SBATCH --time=12:00:00            # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu            # Request GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4          # Request 4 CPU cores
#SBATCH --mem-per-cpu=16G          # Request 16GB memory per CPU core

# Load necessary modules
module load cuda
module load miniforge

# Activate conda environment
conda activate ML2

# Add conda's lib directory to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Run GPU job using local data
python src/fine_tune_with_gen.py --general_data_dir "$SCRATCH/Datasets" --data $TMP_SHARED/data --save_dir "$SCRATCH/SavedModels"

# Copy results back to permanent storage
cp -r $TMP_SHARED/results /path/to/permanent/storage/

# Flash storage ($TMP_SHARED) is automatically cleaned after job ends
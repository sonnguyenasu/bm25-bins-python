#!/bin/bash
#SBATCH --job-name=beir_eval
#SBATCH --partition=bigmem
#SBATCH --output=beir.out
#SBATCH --error=beir.err
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=240G
#SBATCH --time=2-00:30:00

set -e

source /home/dblackle/miniconda3/etc/profile.d/conda.sh
conda activate main

# OPTIONAL: sanity check
which python
python --version

pip install beir
pip install elasticsearch



./elasticsearch-8.14.0/bin/elasticsearch \
  -E xpack.security.enabled=false \
  -E thread_pool.search.size=1 \
  -E thread_pool.write.size=1 \
  > /dev/null 2>&1 &


python bier_retrieval.py

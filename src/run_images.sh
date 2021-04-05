#!/bin/sh
#SBATCH -J Shopee 
#SBATCH --cpus-per-task=8         
#SBATCH --nodes=1

#SBATCH --mail-user=xwang423@fordham.edu
#SBATCH --mail-type=ALL
#SBATCH --output=Shopeeout.txt
#SBATCH --error=Shopeestderr.txt

module load cuda10.1/toolkit/10.1.243
module load shared
module load python36
module load ml-pythondeps-py36-cuda10.1-gcc/3.3.0
# source ../../venv/bin/activate

# which python
python3.6 main.py
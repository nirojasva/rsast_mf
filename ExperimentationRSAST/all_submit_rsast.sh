#!/bin/bash
#SBATCH --job-name=mfrsast

#SBATCH --partition=normal    # choix de la partition où soumettre le job
#SBATCH --ntasks=1            # nb de tasks total pour le job
#SBATCH --cpus-per-task=4     # 1 seul CPU pour une task
#SBATCH --mem=32000            # mémoire nécessaire (par noeud) en Mo
 

# Activate the Python virtual environment
source rsast_env/bin/activate

# Execute the Python script
cd ~/mf_rsast/ExperimentationRSAST
python accuracy_rsast.py 

wait

echo "All jobs complete"

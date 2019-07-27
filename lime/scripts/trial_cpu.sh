#BSUB -q cpuqueue
#BSUB -J cpu_hyperparam_gn_fc
#BSUB -n 512
#BSUB -R "rusage[mem=4] span[ptile=48]"
#BSUB -W 160:00
#BSUB -o %J.stdout
#BSUB -eo %J.stderr

module add cuda/10.0
python hyperparameter_tuning_gn_fc.py 

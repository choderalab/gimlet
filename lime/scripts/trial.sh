#BSUB -q gpuqueue
#BSUB -J hyperparam_gn_fc
#BSUB -m "ld-gpu ls-gpu lt-gpu lp-gpu lg-gpu lv-gpu lu-gpu"
#####BSUB -m "lu-gpu"
#BSUB -q gpuqueue -n 48 -gpu "num=4:j_exclusive=yes"
#BSUB -R "rusage[mem=4] span[ptile=48]"
######BSUB -R V100
#BSUB -W 160:00
#BSUB -o %J.stdout
#BSUB -eo %J.stderr

module add cuda/10.0
python hyperparameter_tuning_gn_fc.py 

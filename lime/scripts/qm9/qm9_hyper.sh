#BSUB -q gpuqueue
#BSUB -J qm9_hyper
#BSUB -m "ld-gpu ls-gpu lt-gpu lp-gpu lg-gpu lv-gpu lu-gpu"
#####BSUB -m "lu-gpu"
#BSUB -q gpuqueue -n 24 -gpu "num=2:j_exclusive=yes"
#BSUB -R "rusage[mem=8] span[hosts=1]"
######BSUB -R V100
#BSUB -W 24:00
#BSUB -o %J.stdout
#BSUB -eo %J.stderr

python ht_gn_hyper_qm9.py

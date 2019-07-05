#BSUB -q gpuqueue
#BSUB -J lipo_vanilla
#BSUB -m "ld-gpu ls-gpu lt-gpu lp-gpu lg-gpu lv-gpu lu-gpu"
#####BSUB -m "lu-gpu"
#BSUB -q gpuqueue -n 12 -gpu "num=4:j_exclusive=yes"
#BSUB -R "rusage[mem=4] span[ptile=12]"
######BSUB -R V100
#BSUB -W 24:00
#BSUB -o %J.stdout
#BSUB -eo %J.stderr

module add cuda/10.0
python ht_lipo_vanilla.py

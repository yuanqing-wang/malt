for depth in 2 4 6; do
for width in 16 32 64; do
for learning_rate in 1e-3 1e-4; do

bsub -q gpuqueue -o %J.stdout -gpu "num=1:j_exclusive=yes" -R "rusage[mem=5] span[ptile=1]" -W 0:59 -n 1\
    python run.py \
      --depth $depth \
      --width $width \
      --learning_rate $learning_rate \

done; done; done

!/bin/bash

export PACKAGES=/packages
. $PACKAGES/anaconda3/etc/profile.d/conda.sh

conda activate softlearning

python morphology/plot.py \
  --patterns 'data/gym/MorphingAnt/v0/*-forward_model/*/events.out.tfevents*' \
             '/home/btrabucco/design-baselines/design-bench/data/MorphingAnt/v0/*/*/events.out.tfevents*' \
  --names 'Forward Model' \
          'Dataset' \
  --title 'Ant Morphology Evaluation' \
  --csv-file 'ant.csv' \
  --plt-file 'ant.png'

python morphology/plot.py \
  --patterns 'data/gym/MorphingDog/v0/*-forward_model/*/events.out.tfevents*' \
             '/home/btrabucco/design-baselines/design-bench/data/MorphingDog/v0/*/*/events.out.tfevents*' \
  --names 'Forward Model' \
          'Dataset' \
  --title 'Dog Morphology Evaluation' \
  --csv-file 'dog.csv' \
  --plt-file 'dog.png'

python morphology/plot.py \
  --patterns 'data/gym/MorphingDKitty/v0/*-forward_model/*/events.out.tfevents*' \
             '/home/btrabucco/design-baselines/design-bench/data/MorphingDKitty/v0/*/*/events.out.tfevents*' \
  --names 'Forward Model' \
          'Dataset' \
  --title 'DKitty Morphology Evaluation' \
  --csv-file 'dkitty.csv' \
  --plt-file 'dkitty.png'
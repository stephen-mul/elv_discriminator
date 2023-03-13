#! /bin/bash

for ((c=1; c<=$1; c++))
do
    echo "Run $c starting"
    python gen_dataset.py --raw_dir_path ~/google_earth_data/mergedBilinear --out_dir_name ~/notgan_workdir/elv_vae/data/random_tile_200/ --run_num $c
done
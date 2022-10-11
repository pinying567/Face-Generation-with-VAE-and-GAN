#!/bin/bash
# setting training config here
batch_size=64
epochs=90
lr=0.0002
step=30
z_dim=1024
seed=10
save_dir="checkpoint/GAN_seed10"
checkpoint=""

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --step ${step} --z_dim ${z_dim} --seed ${seed} --save_dir ${save_dir}"
#config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 GAN_main.py ${config}"

echo "${run}"
${run}

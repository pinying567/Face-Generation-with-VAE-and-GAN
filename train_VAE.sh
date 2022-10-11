#!/bin/bash
# setting training config here
batch_size=128
epochs=90
lr=0.0002 # 0.001
step=90
gamma=0.1
z_dim=1024
lamda=1
save_dir="checkpoint/VAE_lamda1_progress"
checkpoint=""

config="--batch_size ${batch_size} --epochs ${epochs} --lr ${lr} --step ${step} --gamma ${gamma} --save_dir ${save_dir} --z_dim ${z_dim} --lamda ${lamda}"
#config="${config} --checkpoint ${checkpoint}" # uncomment this line to load checkpoint

run="python3 VAE_main.py ${config}"

echo "${run}"
${run}

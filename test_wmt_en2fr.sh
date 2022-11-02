#! /usr/bin/bash
set -e

dataroot=/mnt/data2/danqingwang/Dataset/MT
workroot=/mnt/data2/danqingwang/Workspace/ODETransformer

model_dir=$workroot/checkpoints/wmt-en2fr/RK2-learnbale-layer6-Big-RPR-2
# model_dir=$workroot/checkpoints/wmt-en2fr-baseline/transformer-big
checkpoint=checkpoint_best.pt

python3 generate.py \
$dataroot/data-bin/wmt14_en_fr_joint_bpe \
--path $model_dir/$checkpoint \
--gen-subset test \
--batch-size 64 \
--beam 4 \
--lenpen 0.6 \
--output hypo_best2.txt \
--quiet \
--remove-bpe 
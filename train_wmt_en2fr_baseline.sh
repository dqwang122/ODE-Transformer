#! /usr/bin/bash
set -e

# must set this tag
# tag=RK2-learnbale-layer12-Big-RPR
tag=$1

# task=wmt-en2de
task=wmt-en2fr-baseline

# device=0,1,2,3,4,5,6,7
device=0
# device=$2

dataroot=/mnt/data2/danqingwang/Dataset/MT/data-bin
workroot=/mnt/data2/danqingwang/Workspace/ODETransformer

if [ $task == "wmt-en2de" ]; then
        arch=ode_relative_transformer_t2t_wmt_en_de_big
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.002
        warmup=16000
        max_tokens=4096
        update_freq=2
        weight_decay=0.0
        keep_last_epochs=10
        max_epoch=40
        max_update=
        reset_optimizer=0
        data_dir=google
        src_lang=en
        tgt_lang=de
elif [ $task == "wmt-en2ro" ]; then
        arch=ode_relative_transformer_t2t_wmt_en_de_big
        share_embedding=0
        share_decoder_input_output_embed=1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.002
        warmup=8000
        max_tokens=4096
        update_freq=1
        weight_decay=0.0
        keep_last_epochs=20
        max_epoch=20
        max_update=
        reset_optimizer=0
        data_dir=wmt-en2ro
        src_lang=en
        tgt_lang=ro
elif [ $task == "wmt-en2fr" ]; then
        arch=ode_transformer_t2t_wmt_en_de_big
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        reset_optimizer=0
        fp16=1
        lr=0.002
        warmup=16000
        max_tokens=16384
        update_freq=8
        weight_decay=0.0
        keep_last_epochs=10
        max_epoch=20
        max_update=
        data_dir=wmt14_en_fr_joint_bpe
        src_lang=en
        tgt_lang=fr
elif [ $task == "wmt-en2fr-baseline" ]; then
        arch=transformer_vaswani_wmt_en_de_big
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        reset_optimizer=0
        fp16=1
        lr=0.002
        warmup=16000
        max_tokens=16384 
        update_freq=8
        weight_decay=0.0
        keep_last_epochs=10
        max_epoch=20
        max_update=
        data_dir=wmt14_en_fr_joint_bpe
        src_lang=en
        tgt_lang=fr
else
        echo "unknown task=$task"
        exit
fi

save_dir=$workroot/checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="python3 -u train.py $dataroot/$data_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --weight-decay $weight_decay
  --criterion $criterion --label-smoothing 0.1
  --max-tokens $max_tokens
  --update-freq $update_freq
  --encoder-normalize-before 
  --decoder-normalize-before
  --encoder-layers 6
  --dropout 0.1
  --no-progress-bar
  --log-format simple
  --log-interval 100
  --ddp-backend no_c10d 
  --seed 1
  --save-dir $save_dir
  --keep-last-epochs $keep_last_epochs
  --tensorboard-logdir $save_dir" 

adam_betas="'(0.9, 0.997)'"
cmd=${cmd}" --adam-betas "${adam_betas}
if [ $share_embedding -eq 1 ]; then
cmd=${cmd}" --share-all-embeddings "
fi
if [ $share_decoder_input_output_embed -eq 1 ]; then
cmd=${cmd}" --share-decoder-input-output-embed "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi
if [ -n "$dropout" ]; then
cmd=${cmd}" --dropout "${dropout}
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ $reset_optimizer -eq 1 ]; then
cmd=${cmd}" --reset-optimizer "
fi


# export CUDA_VISIBLE_DEVICES=$device
cmd=" "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log

# src=en
# tgt=de
# TEXT=../wmt-en2de
# tag=wmt-en2de
# output=data-bin/$tag

src=en
tgt=fr
TEXT=/mnt/data2/danqingwang/Dataset/MT/wmt14_en_fr
tag=wmt14_en_fr_joint_bpe
output=/mnt/data2/danqingwang/Dataset/MT/data-bin/$tag

# python3 preprocess.py --source-lang $src --target-lang $tgt --trainpref $TEXT/train  --validpref $TEXT/valid --testpref $TEXT/test --destdir $output --workers 32

python3 preprocess.py --source-lang $src --target-lang $tgt \
                      --trainpref $TEXT/train  --validpref $TEXT/valid --testpref $TEXT/test \
                      --destdir $output --workers 32 \
                      --joined-dictionary

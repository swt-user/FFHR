#! /bin/bash 
python  run.py \
            --dataset WN18RR \
            --model RescalH\
            --rank 256\
            --distance dot \
            --xvaier \
            --weight \
            --sparse  \
            --regularizer DURA_RESCAL\
            --reg 0.1 \
            --optimizer Adagrad \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 1000\
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias none\
            --dtype single \
            --curvature \
            --no_act \
            --no_gcn \
            --num_head 1



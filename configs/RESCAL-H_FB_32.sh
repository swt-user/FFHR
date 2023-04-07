#! /bin/bash 
python  run.py \
            --dataset FB237\
            --model RescalH\
            --rank 32\
            --distance dot \
            --regularizer DURA_RESCAL\
            --reg 0.05 \
            --optimizer Adagrad \
            --max_epochs 500 \
            --patience 15 \
            --valid 5 \
            --batch_size 2000\
            --neg_sample_size -1 \
            --init_size 0.001 \
            --learning_rate 0.1 \
            --gamma 0.0 \
            --bias none\
            --dtype double \
            --curvature \
            --no_act \
            --no_gcn \
            --num_head 1




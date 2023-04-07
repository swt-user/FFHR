#! /bin/bash 
python  run.py \
            --dataset FB237 \
            --model RescalH\
            --rank 512\
            --distance dot \
            --xvaier \
            --sparse  \
            --regularizer DURA_RESCAL\
            --reg 0.05 \
            --optimizer Adagrad \
            --max_epochs 300 \
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
            --num_head 1



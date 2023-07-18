#!/bin/bash
pip install -e . || exit 1
python ./bin/main.py --distributed_init_method tcp://127.0.0.1:3456 \
                        --world_size 1 \
                        --config_file "config_local.ini" \
                        --dist_backend "nccl" \
                        --num_workers 0 \
                        --test \
                        --fast

#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068

export WANDB_API_KEY='your WANDB API key'
export TORCH_HOME='/mnt/Work/Dataset/Affwild2_ABAW3/pretrained/'
export WANDB_CONFIG_DIR='/mnt/Work/Dataset/Affwild2_ABAW3/'
export WANDB_CACHE_DIR='/mnt/Work/Dataset/Affwild2_ABAW3/'


# Submission 01
test_config='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_12-55-48/config.yaml'   # Path to config file
test_ckpt='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_12-55-48/version_None/checkpoints/epoch=17-step=6155.ckpt'

python -W ignore main.py --cfg $test_config \
                          TEST_ONLY $test_ckpt

echo "Finished generate results for submission 01"

# Submission 02
test_config='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_14-45-39/config.yaml'   # Path to config file
test_ckpt='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_14-45-39/version_None/checkpoints/epoch=19-step=6839.ckpt'

python -W ignore main.py --cfg $test_config \
                          TEST_ONLY $test_ckpt

echo "Finished generate results for submission 02"

# Submission 03
test_config='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-23_12-47-38/config.yaml'   # Path to config file
test_ckpt='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-23_12-47-38/version_None/checkpoints/epoch=19-step=6839.ckpt'

python -W ignore main.py --cfg $test_config \
                          TEST_ONLY $test_ckpt

echo "Finished generate results for submission 03"

# Submission 04
test_config='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_16-44-13/config.yaml'   # Path to config file
test_ckpt='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/AU_v2/AU_2022-03-22_16-44-13/version_None/checkpoints/epoch=17-step=6155.ckpt'

python -W ignore main.py --cfg $test_config \
                          TEST_ONLY $test_ckpt

echo "Finished generate results for submission 04"

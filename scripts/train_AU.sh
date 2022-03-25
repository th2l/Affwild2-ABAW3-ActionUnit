#!/bin/bash

trap "exit" INT
# shellcheck disable=SC2068

export WANDB_API_KEY='your WANDB API key'
export TORCH_HOME='/mnt/Work/Dataset/Affwild2_ABAW3/pretrained/'
export WANDB_CONFIG_DIR='/mnt/Work/Dataset/Affwild2_ABAW3/'
export WANDB_CACHE_DIR='/mnt/Work/Dataset/Affwild2_ABAW3/'

run_ver='v2'
task='AU'
logger='wandb' #'wandb' none

# Temporal configs
num_enc_dnc=2
tranf_nhead=10
tranf_dim_fc=512
seq_len=256 # old 256

# Backbone configs
model_backbone='regnet-400mf'
freeze_bn=True
model_aux=1.
fusion_strategy=1

# Optimizer configs
optim_name='sgd'
lr_policy='cos-restart'  # 'cos-restart'
warmup_epoch=4
warmup_factor=0.1
wd=5e-5
max_epoch=20 # 25 # old 20
train_bsz=16 # old 16
test_bsz=16

# Focal loss config
focal_gamma=2.
focal_alpha=0.75
base_lr=0.9  # 0.9 # 0.0001 0.005

img_size=112
pretrained_model='none'  # 'none' #
train_dir='/mnt/Work/Dataset/Affwild2_ABAW3/train_logs/'$task'_'$run_ver'/'

test_only='none'
# Run command
python -W ignore main.py --cfg conf/${task}_baseline.yaml \
        TASK $task \
        LOGGER $logger \
        OUT_DIR $train_dir \
        OPTIM.MAX_EPOCH $max_epoch \
        OPTIM.WARMUP_FACTOR $warmup_factor \
        OPTIM.BASE_LR $base_lr \
        OPTIM.NAME $optim_name \
        OPTIM.LR_POLICY $lr_policy \
        OPTIM.WEIGHT_DECAY $wd \
        OPTIM.FOCAL_ALPHA $focal_alpha \
        OPTIM.FOCAL_GAMMA $focal_gamma \
        OPTIM.WARMUP_EPOCHS $warmup_epoch \
        TRAIN.BATCH_SIZE $train_bsz \
        TEST.BATCH_SIZE $test_bsz \
        TRANF.NUM_ENC_DEC $num_enc_dnc \
        TRANF.NHEAD $tranf_nhead \
        TRANF.DIM_FC $tranf_dim_fc \
        MODEL.BACKBONE $model_backbone \
        MODEL.BACKBONE_FREEZE "'block4', 'block3', 'block2'" \
        MODEL.FREEZE_BATCHNORM $freeze_bn \
        MODEL.BACKBONE_PRETRAINED $pretrained_model \
        MODEL.USE_AUX $model_aux \
        MODEL.FUSION_STRATEGY $fusion_strategy \
        DATA_LOADER.SEQ_LEN $seq_len \
        DATA_LOADER.IMG_SIZE $img_size \
        TEST_ONLY $test_only \
        OPTIM.TUNE_LR False

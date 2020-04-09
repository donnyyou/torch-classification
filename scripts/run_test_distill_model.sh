#!/bin/bash

# check the enviroment info
nvidia-smi

PYTHON="python"
#---------------prepare data-------------#
PROJECT_DIR=$(cd $(dirname $0)/../; pwd)
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
cd ${PROJECT_DIR}

MAIN_BACKBONE="resnet152"
PEER_BACKBONE="resnet50"
MODEL_NAME="distill_model"
CHECKPOINTS_NAME="distill_model_"$1

MAIN_PRETRAINED_MODEL="./pretrained_models/7x7resnet152-imagenet.pth"
PEER_PRETRAINED_MODEL="./pretrained_models/7x7resnet50-imagenet.pth"

CONFIGS_FILE='configs/base_image_classifier.conf'
MAX_ITERS=100000
TRAIN_BATCH_SIZE=64
BASE_LR=0.05
LOSS_TYPE="ce_loss"

# SHUFFLE_TRANS_SEQ="random_contrast random_hue random_saturation random_brightness random_perm"
SHUFFLE_TRANS_SEQ="random_contrast random_hue random_saturation random_brightness"
TRANS_SEQ="random_flip resize random_rotate random_resize random_crop random_pad random_erase"

LOG_DIR="${PROJECT_DIR}/log/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"


if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi


${PYTHON} -u main.py --config_file ${CONFIGS_FILE} --gpu 0 1 2 3 --train_batch_size ${TRAIN_BATCH_SIZE} --train_loader default --phase test \
                     --shuffle_trans_seq ${SHUFFLE_TRANS_SEQ} --trans_seq ${TRANS_SEQ} --linear_type nobias --is_warm y --warm_iters 10000 \
                     --main_backbone ${MAIN_BACKBONE} --peer_backbone ${PEER_BACKBONE} --base_lr ${BASE_LR} \
                     --model_name ${MODEL_NAME} --min_count 10 --num_classes 100 \
                     --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                     --main_pretrained ${MAIN_PRETRAINED_MODEL} --peer_pretrained ${PEER_PRETRAINED_MODEL} \
                     --checkpoints_name ${CHECKPOINTS_NAME}  2>&1 | tee ${LOG_FILE}


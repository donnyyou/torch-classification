#!/bin/bash

# check the enviroment info
nvidia-smi

PYTHON="python"
#---------------prepare data-------------#
PROJECT_DIR=$(cd $(dirname $0)/../; pwd)
export PYTHONPATH=${PROJECT_DIR}:${PYTHONPATH}
cd ${PROJECT_DIR}

BACKBONE="resnet50"
MODEL_NAME="cls_model"
CHECKPOINTS_NAME="cls_model_"$1

PRETRAINED_MODEL="./pretrained_models/7x7resnet50-imagenet.pth"

CONFIGS_FILE='configs/base_image_classifier.conf'
MAX_ITERS=20000
TRAIN_BATCH_SIZE=64
BASE_LR=0.05
LOSS_TYPE="ce_loss"

# SHUFFLE_TRANS_SEQ="random_contrast random_hue random_saturation random_brightness random_perm"
SHUFFLE_TRANS_SEQ="random_contrast random_hue random_saturation random_brightness"
TRANS_SEQ="random_flip resize random_rotate random_resize random_crop random_pad random_erase"

TRAIN_LABEL_PATH="/data/donny/train.txt"
VAL_LABEL_PATH="/data/donny/val.txt"
LOG_DIR="${PROJECT_DIR}/log/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"


if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi


${PYTHON} -u main.py --config_file ${CONFIGS_FILE} --gpu 0 1 2 3 --train_batch_size ${TRAIN_BATCH_SIZE} --train_loader default \
                     --shuffle_trans_seq ${SHUFFLE_TRANS_SEQ} --trans_seq ${TRANS_SEQ}  --is_warm y --warm_iters 1000 \
                     --backbone ${BACKBONE} --base_lr ${BASE_LR} --train_label_path ${TRAIN_LABEL_PATH} --val_label_path ${VAL_LABEL_PATH} \
                     --model_name ${MODEL_NAME} --min_count 10 --max_count 10000 --num_classes 2 \
                     --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                     --pretrained ${PRETRAINED_MODEL} --resume checkpoints/cls/cls_model_neg_posv2_latest.pth --resume_continue n \
                     --checkpoints_name ${CHECKPOINTS_NAME}  2>&1 | tee ${LOG_FILE}


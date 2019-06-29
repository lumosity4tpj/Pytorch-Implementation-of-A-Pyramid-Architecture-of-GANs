. ./CONFIG

python3 train.py \
    --train_age ${TRAIN_AGE} \
    --elder_dataroot ${ELDER_DATAROOT} \
    --model_dir ${MODEL_DIR} \
    --train_img_dir ${TRAIN_IMG_DIR} \
    --netD_pth ${NETD_PTH} \
    --netG_pth ${NETG_PTH} \
    --is_training False

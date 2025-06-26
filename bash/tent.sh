GPU_ID=2
DATASET=nct #cifar10
DATA_DIR=/mnt/data1/histology
BATCH_SIZE=128
LR=1e-3
METHOD=tent
MODEL=Quilt



BACKBONE=ViT-B/32
SAVE_DIR=./save/${DATASET}/${DOMAIN}/${BACKBONE}/watt-${WATT_TYPE}-l${WATT_L}-m${WATT_M}
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --data_dir $DATA_DIR --dataset $DATASET --adapt --method $METHOD  --save_dir $SAVE_DIR --backbone $BACKBONE --batch-size $BATCH_SIZE --lr $LR  --model $MODEL

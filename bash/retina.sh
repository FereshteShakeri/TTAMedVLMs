GPU_ID=3
DATASET=messidor #cifar10
DATA_DIR=/export/datasets/public/medical/Retina
BATCH_SIZE=128
LR=1e-3
METHOD=limo
MODEL=ViLRef #CONCH



BACKBONE=ViT-B/32
SAVE_DIR=./save/${DATASET}/${DOMAIN}/${BACKBONE}/watt-${WATT_TYPE}-l${WATT_L}-m${WATT_M}
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --data_dir $DATA_DIR --dataset $DATASET --method $METHOD  --save_dir $SAVE_DIR --backbone $BACKBONE --batch-size $BATCH_SIZE --lr $LR  --model $MODEL --adapt

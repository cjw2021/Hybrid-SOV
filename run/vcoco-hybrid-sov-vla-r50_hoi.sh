set -x
PROJECT_NAME=vcoco
EXP_NAME=007_hybrid-sov-s-vla-topk-hoi_bs2x4_lr1e-4
EXP_DIR=exp_logs/${PROJECT_NAME}_${EXP_NAME}

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=9086 main.py \
    --dataset_file vcoco_self_object \
    --config configs/hybrid-sov/vcoco-hybrid-sov-vla-r50_self-object.yml \
    --config_file slconfig/vcoco-hybrid-sov-vla-r50_self-object.py \
    --hoi_path data/v-coco \
    --batch_size 4 \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --output_dir ${EXP_DIR} \
    --wandb_project ${PROJECT_NAME} \
    --wandb_name ${EXP_NAME} \
    --pretrain_model_path params/rtdetr_r50vd_6x_coco_from_paddle_converted_vcoco.pth \
    --use_wandb

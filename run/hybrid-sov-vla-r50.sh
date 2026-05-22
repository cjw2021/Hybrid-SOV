PROJECT_NAME=hico_det
EXP_NAME=hybrid-sov-vla-r50
EXP_DIR=exp_logs/${PROJECT_NAME}_${EXP_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=8908 main.py \
  --dataset_file hico \
  --hoi_path data/hico_det \
  --config configs/hybrid-sov/hybrid-sov-vla-r50.yml \
  --config_file slconfig/hybrid-sov-vla-r50_hoi.py \
  --batch_size 4 \
  --num_obj_classes 80 \
  --num_verb_classes 117 \
  --output_dir ${EXP_DIR} \
  --wandb_project ${PROJECT_NAME} \
  --wandb_name ${EXP_NAME} \
  --pretrain_model_path params/rtdetr_r50vd_6x_coco_from_paddle_converted_hico.pth \
  --use_wandb

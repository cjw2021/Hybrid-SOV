PROJECT_NAME=hico_det
EXP_NAME=hybrid-sov-vla-dinov3-cnx-l
EXP_DIR=exp_logs/${PROJECT_NAME}_${EXP_NAME}

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=19605 main.py \
  --dataset_file hico \
  --config configs/hybrid-sov/hybrid-sov-vla-dinov3-cnx-l.yml \
  --config_file slconfig/hybrid-sov-vla-dinov3-cnx-l_hoi.py \
  --hoi_path data/hico_det \
  --batch_size 4 \
  --num_obj_classes 80 \
  --num_verb_classes 117 \
  --output_dir ${EXP_DIR} \
  --wandb_project ${PROJECT_NAME} \
  --wandb_name ${EXP_NAME} \
  --use_wandb

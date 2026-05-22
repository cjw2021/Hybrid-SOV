set -x
PROJECT_NAME=hico_rt
EXP_NAME=exp036_hybrid-sov-r50_coco-pretrain_enc-obj-class-topk-topq300_bs4x4-8
EXP_DIR=exp_logs/${PROJECT_NAME}_${EXP_NAME}

export NCCL_TIMEOUT=36000

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_addr="127.0.0.1" --master_port=8908 main.py \
  --dataset_file hico \
  --config configs/sov_dn_rt/sov-dn-rt.yml \
  --config_file slconfig/sov-dn-rt-pure.py \
  --hoi_path data/hico_det \
  --batch_size 4 \
  --num_obj_classes 80 \
  --num_verb_classes 117 \
  --output_dir ${EXP_DIR} \
  --pretrain_model_path params/rtdetr_r50vd_6x_coco_from_paddle_converted_hico.pth \
  --wandb_project ${PROJECT_NAME} \
  --wandb_name ${EXP_NAME} \
  --use_wandb
  # --debug \

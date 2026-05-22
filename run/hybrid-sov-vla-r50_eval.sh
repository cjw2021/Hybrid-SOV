PROJECT_NAME=hico_det
EXP_NAME=hybrid-sov-vla-r50
EXP_DIR=exp_logs/${PROJECT_NAME}_${EXP_NAME}
OUTPUT_DIR=${EXP_DIR}_eval

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=19089 main.py \
    --dataset_file hico \
    --config configs/hybrid-sov/hybrid-sov-vla-r50.yml \
    --config_file slconfig/hybrid-sov-vla-r50_hoi.py \
    --hoi_path data/hico_det \
    --batch_size 2 \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --output_dir "${OUTPUT_DIR}" \
    --wandb_project "${PROJECT_NAME}" \
    --wandb_name "${EXP_NAME}" \
    --resume "params/hico_det_hybrid-sov-vla-r50.pth" \
    --eval

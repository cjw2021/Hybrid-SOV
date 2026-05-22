PROJECT_NAME=vcoco
EXP_NAME=vcoco-hybrid-sov-vla-r50
EXP_DIR=exp_logs/${PROJECT_NAME}_${EXP_NAME}
OUTPUT_DIR=${EXP_DIR}_eval

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

python generate_vcoco_official.py \
    --config configs/hybrid-sov/vcoco-hybrid-sov-vla-r50_self-object.yml \
    -c slconfig/vcoco-hybrid-sov-vla-r50_self-object.py \
    --resume params/vcoco-hybrid-sov-vla-r50.pth \
    --dataset_file vcoco_self_object \
    --hoi_path data/v-coco \
    --num_obj_classes 81 \
    --num_verb_classes 29 \
    --with_clip_label \
    --fix_clip \
    --zero_shot_type default \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 2 \
    --eval

python vsrl_eval_accelerate.py --vcoco_path data/v-coco --detections ${EXP_DIR}_eval

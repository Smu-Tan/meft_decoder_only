## Environment setting
export TRANSFORMERS_CACHE= # TODO: add the path for downloading from tansformers
export HF_DATASETS_CACHE= # TODO: add the path for downloading from tansformers
export HF_METRICS_CACHE= # TODO: add the path for downloading from tansformers
export TOKENIZERS_PARALLELISM=false

# Hyper-parameters
MODEL_NAME="facebook/opt-1.3b"
DIM=64
FACTOR1=0.1
FACTOR2=1
MAX_NORM=1
GROUP=true
SUM=true
FP16=true
SUM_SCALE=0.1
FARCH="layer"
LAYER=16
FREEZE=true

## Task setting  #TODO: adjust to your own path
TASKS=(openbookqa) #TODO: adjust to your task
TRAIN_TOOL=run_revopt_qa.py
EVAL_TOOL=eval_revopt_qa.py
SAVE_DIR=checkpoints
MODEL_VARIANT=farch${FARCH}_x1${FACTOR1}_x2${FACTOR2}_revlayer${LAYER}_fp16${FP16}_maxnorm${MAX_NORM}_group${GROUP}_sum${SUM}_scale${SUM_SCALE}_dim${DIM}_freeze${FREEZE}

mkdir -p $SAVE_DIR/${MODEL_VARIANT}

SEEDS=(3407)
LRS=(1e-4 3e-4 5e-4 7e-4)
BSS=(8 16 32)
EPS=(3 5 10)

for (( t=0; t<${#TASKS[@]}; t++ ))
do
for (( s=0; s<${#SEEDS[@]}; s++ ))
do
first_id=$((t*${#SEEDS[@]}+s))
for (( l=0; l<${#LRS[@]}; l++ ))
do
second_id=$((first_id*${#LRS[@]}+l))
for (( b=0; b<${#BSS[@]}; b++ ))
do
third_id=$((second_id*${#BSS[@]}+b))
for (( e=0; e<${#EPS[@]}; e++ ))
do

task_id=$((third_id*${#EPS[@]}+e))
if [ "$task_id" -eq "$SLURM_ARRAY_TASK_ID" ]
then

TASK_VARIANT=${TASKS[$t]}_seed${SEEDS[$s]}_lr${LRS[$l]}_bs${BSS[$b]}_ep${EPS[$e]}
mkdir -p $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT}

python -u $TRAIN_TOOL \
    --model_name_or_path ${MODEL_NAME} \
    --f_arch ${FARCH} \
    --freeze_irreversible_layers ${FREEZE} \
    --adapter_bottleneck_dim $DIM \
    --x1_factor ${FACTOR1} --x2_factor ${FACTOR2} \
    --num_rev_layers ${LAYER} \
    --sum ${SUM} --sum_scale ${SUM_SCALE} \
    --dataset_name ${TASKS[$t]} \
    --output_dir $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT} \
    --do_train true \
    --do_eval true \
    --save_total_limit ${EPS[$e]} \
    --load_best_model_at_end true \
    --metric_for_best_model "loss" \
    --greater_is_better false \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --max_seq_length 128 \
    --pad_to_max_length false \
    --per_device_train_batch_size ${BSS[$b]} \
    --learning_rate ${LRS[$l]} \
    --num_train_epochs ${EPS[$e]} \
    --overwrite_output_dir true \
    --fp16 ${FP16} \
    --seed ${SEEDS[$s]} \
    --group_by_length ${GROUP} \
    --report_to none \
    --warmup_ratio 0.06 \
    --warmup_steps 0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --weight_decay 0.1 \
    --max_grad_norm ${MAX_NORM} \
    --logging_steps 100 --disable_tqdm true 2>&1 | tee $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT}/out

python -u $EVAL_TOOL \
    --model_name_or_path ${MODEL_NAME} \
    --f_arch ${FARCH} \
    --freeze_irreversible_layers ${FREEZE} \
    --adapter_bottleneck_dim $DIM \
    --x1_factor ${FACTOR1} --x2_factor ${FACTOR2} \
    --num_rev_layers ${LAYER} \
    --sum ${SUM} --sum_scale ${SUM_SCALE} \
    --dataset_name ${TASKS[$t]} \
    --output_dir $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT}/ \
    --do_train true --do_eval true \
    --max_seq_length 128 \
    --fp16 ${FP16} \
    --group_by_length true

cd $SAVE_DIR/${MODEL_VARIANT}/${TASK_VARIANT}
rm -rf checkpoint*
cp eval_results.json eval_results
cp all_results.json all_results
rm *.json
rm *.bin
rm merges.txt

fi
done
done
done
done
done


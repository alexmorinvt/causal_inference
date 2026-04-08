#!/bin/bash

set -e # abandon script on error

model=''
model_path=''
exp_id='exp_id_not_set'
verbose='false'
dev_mode='false'

BASEDIR="$(dirname "$(readlink -f "$0")")"

print_usage() {
  printf "Usage: run_model.sh -m <model_name> [-p <model_path>] [-e <exp_id>] [-v] [-d]\n"
  printf "  -d  Dev/fast mode: skips false negatives, false omission rate, and transitive paths\n"
}

while getopts 'm:p:e:vd' flag; do
  case "${flag}" in
    m) model="${OPTARG}" ;;
    p) model_path="--inference_function_file_path $BASEDIR/${OPTARG}" ;;
    e) exp_id="${OPTARG}" ;;
    v) verbose='true' ;;
    d) dev_mode='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done

mkdir -p $BASEDIR/output/ $BASEDIR/storage/

dev_args=''
if [ "$dev_mode" = 'true' ]; then
  dev_args='--max_path_length 1 --omission_estimation_size 0 --do_not_eval_false_negatives'
fi

conda run --live-stream -n causal_inference python $BASEDIR/main.py \
    --dataset_name weissmann_k562 \
    --output_directory ${BASEDIR}/output/ \
    --data_directory ${BASEDIR}/storage/ \
    --training_regime interventional \
    --model_name ${model} \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter \
    --exp_id ${exp_id} \
    ${model_path} \
    ${dev_args}

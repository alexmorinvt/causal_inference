#!/bin/bash

set -e # abandon script on error

model=''
verbose='false'

print_usage() {
  printf "Usage: ..."
}

while getopts 'm:v' flag; do
  case "${flag}" in
    m) model="${OPTARG}" ;;
    v) verbose='true' ;;
    *) print_usage
       exit 1 ;;
  esac
done

BASEDIR="$(dirname "$(readlink -f "$0")")"

mkdir -p $BASEDIR/output/ $BASEDIR/storage/

conda run --live-stream -n causal_inference python main.py \
    --dataset_name weissmann_k562 \
    --output_directory $BASEDIR/output/ \
    --data_directory $BASEDIR/storage/ \
    --training_regime observational \
    --model_name $model \
    --subset_data 1.0 \
    --model_seed 0 \
    --do_filter \
    --max_path_length -1 \
    --omission_estimation_size 500 \
    &> $BASEDIR/output/last_run.txt
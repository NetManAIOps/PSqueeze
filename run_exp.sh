#!/usr/bin/env bash
DATA_ROOT=Psqueeze-dataset
DATASET=${1:-B5}
LAYER=${2:-1}
N_ELE=${3:-1}
ALGORITHM=${4:psqueeze}
if [[ $DATASET == A* ]]; then
  NAME=A_week_12_cuboid_layer_${LAYER}_n_ele_${N_ELE}
else
  NAME=B_cuboid_layer_${LAYER}_n_ele_${N_ELE}
fi
python run_algorithm.py \
  --name ${NAME} \
  --input-path ${DATA_ROOT}/${DATASET} \
  --output-path output/${DATASET} \
  --algorithm ${ALGORITHM} \
  -j 8
python run_evaluation.py \
  -i ${DATA_ROOT}/${DATASET}/${NAME}/injection_info.csv \
  -p output/${DATASET}/${NAME}.json \
  -c ${DATA_ROOT}/${DATASET}/config.json \
  -g ${DATA_ROOT}/${DATASET}/${NAME} \
  -v

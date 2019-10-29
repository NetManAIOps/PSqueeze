#!/usr/bin/env bash
# **********************************************************************
# * Description   : run experiment script
# * Last change   : 20:46:33 2019-10-29
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : none
# **********************************************************************

echo -e "PSqueeze script: run experiment..."

SCRIPT_DIR=`dirname "$0"`
SCRIPT_DIR=`cd $SCRIPT_DIR; pwd`
MAIN_DIR=`cd ${SCRIPT_DIR}/../; pwd`
DATA_DIR=`cd ${MAIN_DIR}/data/; pwd`
RESULT_DIR=${MAIN_DIR}/result/

[ ! -d "$RESULT_DIR" ] && mkdir "$RESULT_DIR"
RESULT_DIR=`cd ${RESULT_DIR}; pwd`
cd $MAIN_DIR

run_algorithm()
{
    DATASET=$1
    SETTING=$2
    NUM_WORKER=$3
    [ ! -d "${RESULT_DIR}/${DATASET}" ] && mkdir "${RESULT_DIR}/${DATASET}"
    ./run_algorithm.py \
        --name ${SETTING} \
        --input-path ${DATA_DIR}/${DATASET} \
        --output-path ${RESULT_DIR}/${DATASET}/${SETTING}/ \
        --num-workers ${NUM_WORKER}
}

run_evaluation()
{
    DATASET=$1
    SETTING=$2
    ./run_evaluation.py \
        --injection-info ${DATA_DIR}/${DATASET}/${SETTING}/injection_info.csv \
        --predict ${RESULT_DIR}/${DATASET}/${SETTING}/${SETTING}.json \
        --config ${DATA_DIR}/${DATASET}/config.json
}

help_info()
{
    echo -e "USAGE: run.sh {TASK} {DATASET} {SETTING} [NUM_WORKER]"
}

run_B_all()
{
    NUM_WORKER=$1
    DATASET_LIST="B3\nB4\nB5\nB6\nB7"
    CUBOID_X_LIST="1\n2\n3"
    CUBOID_Y_LIST="1\n2\n3"
    while read -r dataset; do
        while read -r cuboid_x; do
            while read -r cuboid_y; do
                setting=B_cuboid_layer_${cuboid_x}_n_ele_${cuboid_y}
                [ ! -d "${RESULT_DIR}/${dataset}" ] && mkdir "${RESULT_DIR}/${dataset}"
                [ ! -d "${RESULT_DIR}/${dataset}/${setting}" ] && mkdir "${RESULT_DIR}/${dataset}/${setting}"
                echo -e "\trun for $dataset $setting now..."
                run_algorithm "$dataset" "$setting" "$NUM_WORKER" \
                    >${RESULT_DIR}/${dataset}/${setting}/runtime.log 2>&1
            done < <(echo -e $CUBOID_Y_LIST)
        done < <(echo -e $CUBOID_X_LIST)
    done < <(echo -e $DATASET_LIST)
}

run_A_all()
{
    NUM_WORKER=$1
    CUBOID_X_LIST="12\n34\n56\n78"
    CUBOID_Y_LIST="1\n2\n3\n4\n5"
    CUBOID_Z_LIST="1\n2\n3\n4"
    while read -r x; do
        while read -r y; do
            while read -r z; do
                setting=new_dataset_A_week_${x}_n_elements_${y}_layers_${z}
                [ ! -d "${DATA_DIR}/A/${setting}" ] && continue
                [ ! -d "${RESULT_DIR}/A" ] && mkdir "${RESULT_DIR}/A"
                [ ! -d "${RESULT_DIR}/A/${setting}" ] && mkdir "${RESULT_DIR}/A/${setting}"
                echo -e "\trun for $setting now..."
                run_algorithm A "$setting" "$NUM_WORKER" \
                    >${RESULT_DIR}/A/$setting/runtime.log 2>&1
            done < <(echo -e $CUBOID_Z_LIST)
        done < <(echo -e $CUBOID_Y_LIST)
    done < <(echo -e $CUBOID_X_LIST)
}

TASK=$1
DATASET=$2
SETTING=$3
NUM_WORKER=${4:-20}

case "$TASK" in
    run)
        run_algorithm "$DATASET" "$SETTING" "$NUM_WORKER"
        ;;
    eval)
        run_evaluation "$DATASET" "$SETTING"
        ;;
    test)
        run_algorithm B3 B_cuboid_layer_1_n_ele_1 "$NUM_WORKER"
        ;;
    B)
        run_B_all "$NUM_WORKER"
        ;;
    A)
        run_A_all "$NUM_WORKER"
        ;;
    *)
        help_info
        ;;
esac

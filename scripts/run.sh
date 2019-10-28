#!/usr/bin/env bash
# **********************************************************************
# * Description   : run experiment script
# * Last change   : 21:17:50 2019-10-28
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
    ./run_algorithm.py \
        --name ${SETTING} \
        --input-path ${DATA_DIR}/${DATASET} \
        --output-path ${RESULT_DIR}/${DATASET}/ \
        --num-workers ${NUM_WORKER}
}

run_evaluation()
{
    DATASET=$1
    SETTING=$2
    ./run_evaluation.py \ 
        -i ${DATA_DIR}/${DATASET}/${SETTING}/injection_info.csv \ 
        -p ${RESULT_DIR}/${DATASET}/${SETTING}.json \
        -c ${DATA_DIR}/${DATASET}/config.json
}

help_info()
{
    echo -e "USAGE: run.sh {TASK} {DATASET} {SETTING} [NUM_WORKER]"
}

run_B_all()
{
    NUM_WORKER=$1
    DATASET_LIST="B3\nB4\nB5\nB6\nB7\nD"
    CUBOID_X_LIST="1\n2\n3"
    CUBOID_Y_LIST="1\n2\n3"
    while read -r dataset; do
        while read -r cuboid_x; do
            while read -r cuboid_y; do
                echo -e "\trun for $dataset B_cuboid_layer_${cuboid_x}_n_ele_${cuboid_y} now..."
                run_algorithm "$dataset" "B_cuboid_layer_${cuboid_x}_n_ele_${cuboid_y}" "$NUM_WORKER" \
                    > /dev/null
            done < <(echo -e $CUBOID_Y_LIST)
        done < <(echo -e $CUBOID_X_LIST)
    done < <(echo -e $DATASET_LIST)
}

TASK=$1
DATASET=$2
SETTING=$3
NUM_WORKER=${4:-1}

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
    *)
        help_info
        ;;
esac

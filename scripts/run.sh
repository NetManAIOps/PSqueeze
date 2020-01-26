#!/usr/bin/env bash
# **********************************************************************
# * Description   : run experiment script
# * Last change   : 13:13:49 2020-01-26
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : none
# **********************************************************************

echo -e "PSqueeze script: run experiment..."

WORKING_DIR=`pwd`
SCRIPT_DIR=`dirname "$0"`
SCRIPT_DIR=`cd $SCRIPT_DIR; pwd`
MAIN_DIR=`cd ${SCRIPT_DIR}/../; pwd`
DATA_DIR=`cd ${MAIN_DIR}/data/; pwd`
RESULT_DIR=${MAIN_DIR}/debug/

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
    DATASET_LIST="B0\nB1\nB2\nB3\nB4"
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
                echo -en "\t\t"
                run_evaluation "$dataset" "$setting" 2>/dev/null | tail -1
            done < <(echo -e $CUBOID_Y_LIST)
        done < <(echo -e $CUBOID_X_LIST)
    done < <(echo -e $DATASET_LIST)
}

eval_B_line()
{
    DATASET=$1
    CUBOID_X=$2
    CUBOID_Y=$3
    SETTING=B_cuboid_layer_${CUBOID_X}_n_ele_${CUBOID_Y}

    LINE="$DATASET""\t""$SETTING"
    VALUE=`run_evaluation $DATASET $SETTING | tail -1`
    [ ! "$?" -eq "0" ] && exit "$?"
    LINE="$LINE""\t""${VALUE// /\\t}"
    echo -e "$LINE"
}

eval_B_all()
{
    TO_PATH="$1"
    DATASET_LIST="B0\nB1\nB2\nB3\nB4"
    CUBOID_X_LIST="1\n2\n3"
    CUBOID_Y_LIST="1\n2\n3"
    while read -r dataset; do
        while read -r cuboid_x; do
            while read -r cuboid_y; do
                eval_B_line "$dataset" "$cuboid_x" "$cuboid_y" \
                    >> "$TO_PATH"
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
                echo -en "\t\t"
                run_evaluation A "$setting" 2>/dev/null | tail -1
            done < <(echo -e $CUBOID_Z_LIST)
        done < <(echo -e $CUBOID_Y_LIST)
    done < <(echo -e $CUBOID_X_LIST)
}

eval_A_line()
{
    X=$1
    Y=$2
    Z=$3
    SETTING=new_dataset_A_week_${X}_n_elements_${Y}_layers_${Z}
    [ ! -d "${DATA_DIR}/A/${SETTING}" ] && exit 0

    LINE="A""\t""$SETTING"
    VALUE=`run_evaluation A $SETTING 2>/dev/null | tail -1`
    [ ! "$?" -eq "0" ] && exit "$?"
    LINE="$LINE""\t""${VALUE// /\\t}"
    echo -e "$LINE"
}

eval_A_all()
{
    TO_PATH="$1"
    X_LIST="12\n34\n56\n78"
    Y_LIST="1\n2\n3\n4\n5"
    Z_LIST="1\n2\n3\n4"
    while read -r X; do
        while read -r Y; do
            while read -r Z; do
                eval_A_line "$X" "$Y" "$Z" \
                    >> "$TO_PATH"
            done < <(echo -e $Z_LIST)
        done < <(echo -e $Y_LIST)
    done < <(echo -e $X_LIST)
}

export_csv()
{
    GREEN='\033[32m'
    RED='\033[0;31m'
    NC='\033[0m'

    SAVE_PATH=${1:-result_summary.csv}
    if [[ ! "$SAVE_PATH" = /* ]]; then
        SAVE_PATH=${WORKING_DIR}/${SAVE_PATH}
    fi 

    if [ -d "$SAVE_PATH" ]; then
        echo -e "Path to a directory already exists, script aborted."
        exit 1
    fi

    echo -e "Which part do you want to export?\n"
    echo -e "${GREEN}0${NC} A evaluation"
    echo -e "${GREEN}1${NC} B evaluation"
    echo -e ""
    echo -en "Enter the index to continue:"
    read INDEX

    if [ "$INDEX" !=  "0" ] &&
        [ "$INDEX" != "1" ]; then
        echo -e "Aborted."
        exit 1
    fi

    if [ -f "$SAVE_PATH" ]; then
        echo -e "${RED}Caution: file ${SAVE_PATH} already exists\n${NC}"
        echo -en "Enter (y/n) to continue:"
        read INPUT
        if [ "$INPUT" = "y" ]; then
            rm "$SAVE_PATH"
        else
            echo -e "Aborted."
            exit 1
        fi
    fi

    touch "$SAVE_PATH" > /dev/null 2>&1
    if [ ! "$?" -eq "0" ]; then
        echo -e "Failed to create file."
        exit 1
    fi

    case "$INDEX" in
        0)
            echo -e "Export A evaluation..."
            eval_A_all "$SAVE_PATH"
            ;;
        1)
            echo -e "Export B evaluation..."
            eval_B_all "$SAVE_PATH"
            ;;
        *)
            echo -e "Should have been aborted earlier."
            exit 2
            ;;
    esac

    echo -e "Export successfully."
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
    test_run)
        run_algorithm A new_dataset_A_week_12_n_elements_1_layers_1 "$NUM_WORKER"
        ;;
    test_eval)
        run_evaluation A new_dataset_A_week_12_n_elements_1_layers_1
        ;;
    B)
        run_B_all "$NUM_WORKER"
        ;;
    A)
        run_A_all "$NUM_WORKER"
        ;;
    export)
        export_csv "$2"
        ;;
    *)
        help_info
        ;;
esac


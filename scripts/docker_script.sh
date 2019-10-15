#!/usr/bin/env bash
# **********************************************************************
# * Description   : docker relevant script
# * Last change   : 21:31:36 2019-10-10
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : none
# **********************************************************************

echo -e "PSqueeze script: docker manager..."

SCRIPT_DIR=`dirname "$0"`
SCRIPT_DIR=`cd $SCRIPT_DIR; pwd`
MAIN_DIR=`cd ${SCRIPT_DIR}/../; pwd`
DATA_DIR=`cd ${MAIN_DIR}/data/; pwd`
RESULT_DIR=${MAIN_DIR}/result/

[ ! -d "$RESULT_DIR" ] && mkdir "$RESULT_DIR"
RESULT_DIR=`cd ${RESULT_DIR}; pwd`
cd $MAIN_DIR

build_image()
{
    IMAGE=$1
    echo -e "\t*build image ${IMAGE}..."
    sudo docker build -t "$IMAGE" "$MAIN_DIR"
}

run_image()
{
    IMAGE=$1
    RESULT_DIR_CONTAINER=/PSqueeze/result
    DATA_DIR_CONTAINER=/PSqueeze/data
    echo -e "\t*run image ${IMAGE}..."

    sudo docker run --rm -d -P \
        --name "$IMAGE" \
        --mount type=bind,source=$RESULT_DIR,target=$RESULT_DIR_CONTAINER \
        --mount type=bind,source=$DATA_DIR,target=$DATA_DIR_CONTAINER \
    #TODO
        
}

help_info()
{
    #TODO
}

TASK=$1
IMAGE=${2:-psqueeze:1.0}
echo -e $IMAGE

case "$TASK" in
    build)
        build_image "$IMAGE"
        ;;
    run)
        run_image "$IMAGE"
        ;;
    *)
        help_info
        ;;
esac

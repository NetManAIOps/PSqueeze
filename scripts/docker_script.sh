#!/usr/bin/env bash
# **********************************************************************
# * Description   : docker relevant script
# * Last change   : 19:19:32 2019-12-19
# * Author        : Yihao Chen
# * Email         : chenyiha17@mails.tsinghua.edu.cn
# * License       : none
# **********************************************************************

echo -e "PSqueeze script: docker manager..."

SCRIPT_DIR=`dirname "$0"`
SCRIPT_DIR=`cd $SCRIPT_DIR; pwd`
MAIN_DIR=`cd ${SCRIPT_DIR}/../; pwd`

run_image()
{
    name=$1
    sudo docker run --rm -it \
        --name "$name" \
        --mount type=bind,source=$MAIN_DIR,target=/psqueeze \
        "psqueeze-env:0.1" \
        bash
}

run_image ${1:-psqueeze-tmp}

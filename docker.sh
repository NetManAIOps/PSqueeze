#!/usr/bin/env bash
#####################################
PORT=10670  # Jupyter Server
SSH_PORT=10623  # OpenSSH
PROJECT_NAME="psqueeze"
#####################################
CMD=${1:-start}
CONTAINER_NAME=${2:-$(basename "$(pwd)")}  # Container Name
echo CMD=${CMD}
echo CONTAINER_NAME=${CONTAINER_NAME}
#####################################
IMAGE=${3:-docker.peidan.me/lizytalk/nichijou:ubuntu20.04-cp3.9.4}
echo IMAGE=${IMAGE}
GPU_OPT=""
echo GPU_OPT=${GPU_OPT}
MEMORY_LIMIT="$(($(free -g | grep -oP "Mem:\s+\K\d+")-1))g"
MEMORY_SWAP_LIMIT="-1"
echo MEMORY_LIMIT=${MEMORY_LIMIT}
echo MEMORY_SWAP_LIMIT=${MEMORY_SWAP_LIMIT}


build () {
  sudo docker pull ${IMAGE}
  sudo docker build . -t ${IMAGE}
  sudo docker push ${IMAGE}
}


start () {
    sudo docker pull ${IMAGE}
    # shellcheck disable=SC2046
    sudo docker run -d --name ${CONTAINER_NAME} --restart=unless-stopped --ipc="host" \
    -v "$(realpath .)":/${PROJECT_NAME} \
    -v "$(realpath ~/data)":/data \
    -v "$(realpath .cache)":/root/.cache \
    -v "$(realpath .jupyter)":/root/.jupyter \
    -v "$(realpath .ssh)":/root/.ssh \
    --hostname "${PROJECT_NAME}-$(hostname)" \
    ${GPU_OPT} \
    --memory ${MEMORY_LIMIT} \
    -v /mnt/mfs/mlstorage-experiments/v-zyl14:/mnt/mfs/mlstorage-experiments/v-zyl14 \
    -v /tmp:/tmp \
    -w /${PROJECT_NAME} \
    --env http_proxy="" --env https_proxy="" \
    -p ${PORT}:8888 -p ${SSH_PORT}:22 -p ${TB_PORT}:6006 \
    --user lizytalk \
    ${IMAGE} \
    zsh -c "source ~/.zshrc && cd /${PROJECT_NAME} && jupyter lab --ip=0.0.0.0 --allow-root";
    sleep 5s;

    sudo docker logs --tail 100 ${CONTAINER_NAME};
    sudo docker exec ${CONTAINER_NAME} zsh -c "sudo killall sshd; sudo mkdir /run/sshd || echo exists; sudo /usr/sbin/sshd";
    sudo docker exec ${CONTAINER_NAME} sudo zsh -c "echo 'lizytalk:lizytalk'| chpasswd";
    sudo docker exec ${CONTAINER_NAME} zsh -c "source ~/.zshrc && cd /${PROJECT_NAME} && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple";
    sudo docker exec ${CONTAINER_NAME} zsh -c "source ~/.zshrc && cd /${PROJECT_NAME} && direnv allow .";
    sudo docker push ${IMAGE}
}

stop () {
    sudo docker stop ${CONTAINER_NAME};
    sudo docker rm ${CONTAINER_NAME};
}

shell () {
  sudo docker exec -it ${CONTAINER_NAME} bash
}


if [ ${CMD} == "start" ]; then
  start
elif [ ${CMD} == "stop" ]; then
  stop
elif [ ${CMD} == "restart" ]; then
  stop;
  start;
elif [ ${CMD} == "build" ]; then
  build;
elif [ ${CMD} == "shell" ]; then
  shell;
fi


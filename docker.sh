#!/usr/bin/env bash
PORT=10670
SSH_PORT=10623
NAME=PSQ
IMAGE=${2:-docker.peidan.me/lizytalk/nichijou:ubuntu20.04-cp3.9.4}
CMD=${1:-start}
MEMORY_LIMIT="$(($(free -g | grep -oP "Mem:\s+\K\d+")-1))g"
MEMORY_SWAP_LIMIT="-1"
echo MEMORY_LIMIT=${MEMORY_LIMIT}
echo MEMORY_SWAP_LIMIT=${MEMORY_SWAP_LIMIT}
echo CMD=${CMD}
if [ ${CMD} == "start" ]; then
    sudo docker run -d --name ${NAME} --restart=unless-stopped --shm-size="4g" \
    -v "$(realpath .)":/${NAME} --user lizytalk \
    -v "$(realpath ~/.cache)":/home/${USER}/.cache \
    -v "$(realpath ~/.jupyter)":/home/${USER}/.jupyter \
    -v "$(realpath ~/.ssh)":/home/${USER}/.ssh \
    --memory ${MEMORY_LIMIT} \
    -v /mnt/mfs/mlstorage-experiments/v-zyl14:/mnt/mfs/mlstorage-experiments/v-zyl14 \
    -v /tmp:/tmp \
    --env http_proxy="" --env https_proxy="" \
    -p ${PORT}:8888 -p ${SSH_PORT}:22 ${IMAGE} \
    zsh -c "source ~/.zshrc && cd /${NAME} && jupyter lab --ip=0.0.0.0";
    sleep 5s;
    sudo docker logs --tail 100 ${NAME};
    sudo docker exec ${NAME} zsh -c "sudo mkdir /run/sshd && sudo /usr/sbin/sshd";
    sudo docker exec ${NAME} sudo bash -c "echo 'lizytalk:lizytalk'| chpasswd";
    sudo docker exec ${NAME} zsh -c "source ~/.zshrc && cd /${NAME} && pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple";
    sudo docker exec ${NAME} zsh -c "source ~/.zshrc && cd /${NAME} && direnv allow .";
elif [ ${CMD} == "stop" ]; then
    sudo docker stop ${NAME};
    sudo docker rm ${NAME};
fi

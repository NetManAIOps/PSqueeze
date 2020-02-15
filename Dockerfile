FROM psqueeze-env:0.1

ARG USER_ID
ARG GROUP_ID

RUN buildDeps='graphviz' \
    && apt update \
    && apt install -y $buildDeps \
    && pip3 install graphviz  \
    && addgroup --gid $GROUP_ID user \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user \
    && rm -rf /var/lib/apt/lists/* \
    && apt purge -y --auto-remove $buildDeps

USER user

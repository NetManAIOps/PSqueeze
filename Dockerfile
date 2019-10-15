FROM python:3

COPY . /PSqueeze
WORKDIR /PSqueeze

RUN apt-get update -y \
    && apt-get install python3-pip -y \
    && pip3 install -r requirements.txt \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove

ENTRYPOINT ["./scripts/run.sh"]

CMD [""]

FROM allennlp/allennlp:v0.8.1

WORKDIR /local
COPY pretrained_bert/ /local/pretrained_bert/

ENTRYPOINT []
CMD ["python", \
        "-m", "allennlp.service.server_simple", \
        "--archive-path", "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/bert-large/model.tar.gz", \
        "--predictor", "bert-for-qa", \
        "--include-package", "pretrained_bert", \
        "--field-name", "passage", \
        "--field-name", "question"]

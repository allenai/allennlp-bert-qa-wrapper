# allennlp-bert-qa-wrapper
This is a simple wrapper on top of pretrained BERT based QA models from [pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-BERT) to make AllenNLP model archives, so that you can serve demos from AllenNLP.

## Docker

To run this with Docker, first build an image.

```
docker build -t allennlp-bert-qa .
```

Then you can run the image.  The cache directory is mounted so you can re-use the cache across
multiple Docker commands.

```
mkdir -p $HOME/.allennlp/
docker run -p 8000:8000 -v $HOME/.allennlp:/root/.allennlp allennlp-bert-qa
```

{
  "dataset_reader": {
    "type": "squad_for_pretrained_bert",
    // "bert-base-uncased" or "bert-large-uncased"
    "pretrained_bert_model_file": "bert-base-uncased"
  },
  // Some small data files in the right format just to have AllenNLP produce a model archive after "training".
  // Training will not change the weights.
  "train_data_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/sample_data/sample-v2.0.json",
  "validation_data_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/sample_data/sample-v2.0.json",
  "model": {
    "type": "bert_for_qa",
    "bert_model_type": "bert_base",
    "model_is_for_squad1": true,
    // Path to a tarball containing bert_config.json and pytorch_model.bin that are outputs from HuggingFace code
    // for BERT base
    "pretrained_archive_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/bert-base/squad1.1/bert_base_archive.tar.gz",
    // for BERT large
    // "pretrained_archive_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/bert-large/bert_large_archive.tar.gz",
  },
  "iterator": {
    "type": "basic",
    "batch_size": 40
  },

  "trainer": {
    "num_epochs": 1,
    "cuda_device": -1,
    "optimizer": {
      "type": "adam",
      "betas": [0.9, 0.9]
    }
  }
}

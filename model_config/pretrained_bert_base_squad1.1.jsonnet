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
    // Path to a tarball containing bert_config.json and pytorch_model.bin that are outputs from HuggingFace code
    // for BERT base
    "pretrained_archive_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/bert-base/bert_base_archive.tar.gz",
    // for BERT large
    // "pretrained_archive_path": "https://s3-us-west-2.amazonaws.com/pradeepd-bert-qa-models/bert-large/bert_large_archive.tar.gz",
    // BERT base threshold; dev tuned number of -1.7497849464416504
    "null_score_difference_threshold": 0.0 
    // BERT large threshold, dev tuned number is -1.9847722053527832
    // "null_score_difference_threshold": 0.0 
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

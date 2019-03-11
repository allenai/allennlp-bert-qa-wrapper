{
  "dataset_reader": {
    "type": "squad_for_pretrained_bert",
    // "bert-base-uncased" or "bert-large-uncased"
    "pretrained_bert_model_file": "bert-large-uncased"
  },
  // Some small data files in the right format just to have AllenNLP produce a model archive after "training".
  // Training will not change the weights.
  "train_data_path": "/Users/pradeepd/data/squad2.0/sample-v2.0.json",
  "validation_data_path": "/Users/pradeepd/data/squad2.0/sample-v2.0.json",
  "model": {
    "type": "bert_for_qa",
    // Path to a tarball containing bert_config.json and pytorch_model.bin that are outputs from HuggingFace code
    "pretrained_archive_path": "/Users/pradeepd/workspace/squad_bert/squad2_bert_large/bert_large_archive.tar.gz",
    // BERT base threshold
    //"null_score_difference_threshold": -1.75 
    // BERT large threshold
    "null_score_difference_threshold": -1.98477 
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

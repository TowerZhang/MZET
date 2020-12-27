# MZET
This repository is for the accepted paper: [MZET: Memory Augmented Zero-Shot Fine-grained Named Entity Typing](https://arxiv.org/pdf/2004.01267.pdf)

## Usage
### Prerequisites
* Strict requirement: Minimum 200G available disk space to store the preprocessed tfrecord data and 16G memory to train the model.
* Python 3.X 
* [bert-as-services](https://github.com/hanxiao/bert-as-service), to load BERT label embedding in label_extract.py.


### File Directory
* First, deploy the Glove embedding file and BERT model in advance, then replace your file directory with the 'BERT_PRETRAINED_DIR' and 'glove_file' in config_train.py.
* Second, create a folder "intermediate" under the path Data/BBN/ to collect the intermediate processed data.


### Model training
```bash
pyrhon build_data.py
python config_train.py
```



### Citation Format

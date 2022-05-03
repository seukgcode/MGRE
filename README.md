# MGRE
Code for paper "Multi-Granularity Neural Networks for Document-Level Relation Extraction".

## Requirements
- python=3.7
- CUDA=10.0
- torch=1.7.0
- stanford-parser=4.2.0
- pytorch_transformers=1.2.0
- numpy=1.16.0
- nltk=3.6.1
- tqdm=4.59.0
- matplotlib=3.0.2
- scikit_learn=0.21.2

## Preprocessing Data
- Download [DocRED](https://github.com/thunlp/DocRED) dataset
- Put the `train_annotated.json`, `dev.json`, `test.json` into the directory `data/`
- Put the `vec.npy`,`word2id.json`,`rel2id.json`,`ner2id.json`, `char2id.json` into the directory `prepro_data/`
```
>> python gen_data_tree.py
>> python gen_data.py         # for Glove
>> python gen_data_bert.py    # for BERT
```

## Training
For Golve:
```
>> python train.py --model_name MRGE --save_name checkpoint_MRGE --train_prefix dev_train --test_prefix dev_dev
```
For BERT:
```
>> python train_bert.py --model_name MRGE_bert --save_name checkpoint_MRGE_bert --train_prefix dev_train --test_prefix dev_dev
```
## Testing
For Golve:
```
>> python test.py --model_name MRGE --save_name checkpoint_MRGE --train_prefix dev_train --test_prefix dev_test
```
For BERT:
```
>> python test_bert.py --model_name MRGE_bert --save_name checkpoint_MRGE_bert --train_prefix dev_train --test_prefix dev_test
```
You will get json file named `result.json`, and then you can submit it to [CodaLab](https://competitions.codalab.org/competitions/20717#learn_the_details).

## Acknowledgement
We refer to the code of [DocRED](https://github.com/thunlp/DocRED). Thanks for their contributions.
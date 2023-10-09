import torch
import numpy as np

from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
# for data preprocessing
from datasets import load_dataset
import json

def preprocess_function(examples:dict):
    # examples = train_set[0]
    # examples.keys() is dict_keys(['paragraphs', 'answer', 'question', 'relevant', 'id'])
    first_sentences = [[context] * 4 for context in examples['question']]

    second_sentences = [
        [context_json[idx] for idx in related_p_idx]
        for related_p_idx in examples['paragraphs']
    ]

    # Flatten out
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True
        # max_length=max_seq_length,
        # padding="max_length" if data_args.pad_to_max_length else False,
    )

    # Unflatten and group by four
    tokenized_inputs = {
        k: [v[i:i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }

    # train or valid set
    if 'relevant' in examples.keys():
        tokenized_inputs['label'] = [
            para.index(p_uid)
            for para, p_uid in zip(examples['paragraphs'], examples['relevant'])
        ]
    else:
        tokenized_inputs['label'] = [0] * len(examples['paragraphs'])

    return tokenized_inputs

datafiles = {"train": "/home/loijilai/CS-hub/DL/adl/paragraph-selection-QA/dataset/train.json",
             "valid": "/home/loijilai/CS-hub/DL/adl/paragraph-selection-QA/dataset/valid.json"}
raw_dataset = load_dataset("json", data_files=datafiles)
# raw_dataset['train][0]['question']  = '舍本和誰的數據能推算出連星的恆星的質量？'

with open("./dataset/context.json", 'r') as f:
    context_json = json.load(f)
train_set = raw_dataset['train']

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# tokenizer.model_max_length = 512
# tokenizer.model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']

tmp = train_set[0]
preprocess_function(tmp)
## train_set = train_set.map(preprocess_function, batched=True)
# More keys: input_ids, token_type_ids, attention_mask, label
# Dataset({
#     features: ['paragraphs', 'answer', 'question', 'relevant', 'id', 'input_ids', 'token_type_ids', 'attention_mask', 'label'],
#     num_rows: 21714
# })
# type(train_set[0]) is <class 'dict>
# train_set[0].keys() is dict_keys(['paragraphs', 'answer', 'question', 'relevant', 'id', 'input_ids', 'token_type_ids', 'attention_mask', 'label'])
# train_set[0]['paragraphs'] = [2018, 6952, 8264, 836]
# train_set[0]['answer'] = {'start': 108, 'text': '斯特魯維'}
# train_set[0]['question'] = '舍本和誰的數據能推算出連星的恆星的質量？'
# train_set[0]['relevant'] = 836
# train_set[0]['id'] = '593f14f960d971e294af884f0194b3a7'
# train_set[0]['input_ids/token_type_ids/attention_mask'] =
#     [[101, 5650, 3315, 1469, 6306, 4638, 3149, 3087, 5543, 2972, 5050, 1139, 6865, 3215, ...],
#     [101, 5650, 3315, 1469, 6306, 4638, 3149, 3087, 5543, 2972, 5050, 1139, 6865, 3215, ...],
#     [101, 5650, 3315, 1469, 6306, 4638, 3149, 3087, 5543, 2972, 5050, 1139, 6865, 3215, ...],
#     [101, 5650, 3315, 1469, 6306, 4638, 3149, 3087, 5543, 2972, 5050, 1139, 6865, 3215, ...]]
# (a list of four elements)
# train_set[0]['label'] = 3


"----------------------------------------------------------------------------------------"
model = AutoModelForMultipleChoice.from_pretrained("bert-base-chinese")


""
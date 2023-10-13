import torch
import numpy as np

from transformers import AutoConfig, AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
# for data preprocessing
from datasets import load_dataset
import json
# for data collator
from dataclasses import dataclass
from typing import Optional, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

def preprocess_function(examples):
    pass


# datafiles = {"train": "/home/loijilai/CS-hub/DL/adl/paragraph-selection-QA/dataset/train.json",
#              "valid": "/home/loijilai/CS-hub/DL/adl/paragraph-selection-QA/dataset/valid.json"}
datafiles = {"train": "/project/dsp/loijilai/adl/dataset1/train.json",
             "valid": "/project/dsp/loijilai/adl/dataset1/valid.json"}
raw_dataset = load_dataset("json", data_files=datafiles)
# raw_dataset['train][0]['question']  = '舍本和誰的數據能推算出連星的恆星的質量？'

with open("/project/dsp/loijilai/adl/dataset1/context.json", 'r') as f:
    context_json = json.load(f)


print("Context: ", context_json[raw_dataset['train']['relevant'][0]])
print("Question: ", raw_dataset['train']['question'][0])
print("Answer: ", raw_dataset['train']['answer'][0])


train_set = raw_dataset['train']
valid_set = raw_dataset['valid']
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# tokenizer.model_max_length = 512
# tokenizer.model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']

train_set = train_set.map(preprocess_function, batched=True)
valid_set = valid_set.map(preprocess_function, batched=True)

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

training_args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
)

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = train_set,
    eval_dataset = valid_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

# Training
trainer.train()
""
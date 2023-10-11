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
    # [[1, 1, 1, 1], [2, 2, 2, 2], ...]
    """
    first_sentences[0] = 
    ['舍本和誰的數據能推算出連星的恆星的質量？',
    '舍本和誰的數據能推算出連星的恆星的質量？',
    '舍本和誰的數據能推算出連星的恆星的質量？',
    '舍本和誰的數據能推算出連星的恆星的質量？']
    """
    first_sentences = [[context] * 4 for context in examples['question']]

    # [[p1, p2, p3, p4], [p1, p2, p3, p4], ...]
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
    tokenized_result = {}
    for k, v in tokenized_examples.items():
        # tokenized_examples.keys() = input_ids, token_type_ids, attention_mask
        tokenized_result[k] = [v[i:i + 4] for i in range(0, len(v), 4)]

    # train and valid set
    if 'relevant' in examples.keys():
        tokenized_result['label'] = [
            para.index(p_uid)
            for para, p_uid in zip(examples['paragraphs'], examples['relevant'])
        ]
    else: # test set, initialize with 0
        tokenized_result['label'] = [0] * len(examples['paragraphs'])

    return tokenized_result

# Data collator transforms dataset into a batch of tensors
@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

# Metric
def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


datafiles = {"train": "/home/loijilai/CS-hub/DL/adl/paragraph-selection-QA/dataset/train.json",
             "valid": "/home/loijilai/CS-hub/DL/adl/paragraph-selection-QA/dataset/valid.json"}
# datafiles = {"train": "/tmp2/loijilai/adl/hw1/dataset/train.json",
#              "valid": "/tmp2/loijilai/adl/hw1/dataset/valid.json"}
raw_dataset = load_dataset("json", data_files=datafiles)
# raw_dataset['train][0]['question']  = '舍本和誰的數據能推算出連星的恆星的質量？'

with open("./dataset/context.json", 'r') as f:
    context_json = json.load(f)
train_set = raw_dataset['train']

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# tokenizer.model_max_length = 512
# tokenizer.model_input_names = ['input_ids', 'token_type_ids', 'attention_mask']

train_set = train_set.map(preprocess_function, batched=True)
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

valid_set = raw_dataset['valid']
valid_set = valid_set.map(preprocess_function, batched=True)


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
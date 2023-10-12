import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from datasets import load_dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

import evaluate


def preprocess_function(examples):
    first_sentences = [[context] * 4 for context in examples["sent1"]]
    """
    first_sentences[0] = 
    ['Members of the procession walk down the street holding small horn brass instruments.',
    'Members of the procession walk down the street holding small horn brass instruments.',
    'Members of the procession walk down the street holding small horn brass instruments.',
    'Members of the procession walk down the street holding small horn brass instruments.']
    """
    question_headers = examples["sent2"]
    second_sentences = [
        [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
    ]
    """
    second_sentences[0] =
    ['A drum line passes by walking down the street playing their instruments.',
    'A drum line has heard approaching them.',
    "A drum line arrives and they're outside dancing and asleep.",
    'A drum line turns the lead singer watches the performance.']
    """

    first_sentences = sum(first_sentences, []) # Flatten out a list of list
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    tokenized_result = {}
    for k, v in tokenized_examples.items():
        # k is input_ids, token_type_ids, attention_mask
        # so result.keys() = input_ids, token_type_ids, attention_mask
        # v is a list of 4000 list inside of it
        # v[0:4] takes the first 4 list inside of v -> [[], [], [], []] Note that result of list slicing is also a list
        tokenized_result[k] = [v[i : i + 4] for i in range(0, len(v), 4)] # key: [ [[], [], [], []], [[], [], [], []] ]
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

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Loading and preprocessing dataset
swag = load_dataset("swag", "regular")
ending_names = ["ending0", "ending1", "ending2", "ending3"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# single_example = swag["train"][0] This is an error
# tk_single_example = preprocess_function(single_example) This is an error
# batch_example = swag["train"][:5]
# tk_batch_example = preprocess_function(batch_example)
tokenized_swag = swag.map(preprocess_function, batched=True)

# Evaluating model
accuracy = evaluate.load("accuracy")

# Training
model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    output_dir="my_awesome_swag_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_swag["train"],
    eval_dataset=tokenized_swag["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
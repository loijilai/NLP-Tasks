import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMultipleChoice, TrainingArguments, Trainer
from datasets import load_dataset
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union

import evaluate

# Inference
prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law does not apply to croissants and brioche."
candidate2 = "The law applies to baguettes."

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # mymodel
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
"""
inputs.items()
inputs_ids = [[prompt, c1], [prompt, c2]]
('input_ids', tensor([[  101,  2605,  2038,  1037,  7852,  2375,  1010,  3393, 11703, 13465,
          3255,  1010,  2007,  9384,  3513,  2006,  2054,  2003,  3039,  1999,
          1037,  3151,  4524, 23361,  2618,  1012,   102,  1996,  2375,  2515,
          2025,  6611,  2000, 13675, 10054, 22341,  2015,  1998,  7987,  3695,
          5403,  1012,   102],
        [  101,  2605,  2038,  1037,  7852,  2375,  1010,  3393, 11703, 13465,
          3255,  1010,  2007,  9384,  3513,  2006,  2054,  2003,  3039,  1999,
          1037,  3151,  4524, 23361,  2618,  1012,   102,  1996,  2375, 12033,
          2000,  4524, 23361,  4570,  1012,   102,     0,     0,     0,     0,
             0,     0,     0]]))
('token_type_ids', tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]))
('attention_mask', tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]))
"""
labels = torch.tensor(0).unsqueeze(0)

model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased") # mymodel
outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
"""
outputs.items()
('loss', tensor(0.6927, grad_fn=<NllLossBackward0>))
('logits', tensor([[-0.8642, -0.8652]], grad_fn=<ViewBackward0>))
"""
logits = outputs.logits

predicted_class = logits.argmax().item()
predicted_class
from transformers import AutoTokenizer, AutoModelForMultipleChoice, AutoConfig
from datasets import load_dataset

datafiles = {"train": "/tmp2/loijilai/adl/paragraph-selection-QA/dataset/train.json",
             "valid": "/tmp2/loijilai/adl/paragraph-selection-QA/dataset/valid.json"}
raw_dataset = load_dataset("json", data_files=datafiles)
config = AutoConfig.from_pretrained('bert-base-chinese')
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModelForMultipleChoice.from_pretrained('bert-base-chinese')
""
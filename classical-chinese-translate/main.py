from curses import raw
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from peft import LoraConfig, get_peft_model
from utils import get_prompt, get_bnb_config, print_trainable_parameters
from ppl import perplexity
import argparse

def parse_args():
    base_model_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/model/Taiwan-LLM-7B-v2.0-chat"
    train_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/train.json"
    eval_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/public_test.json"
    per_device_train_batch_size = 1
    learning_rate = 5e-5
    num_train_epochs = 1
    gradient_accumulation_steps = 2
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=base_model_path)
    parser.add_argument("--train_data_path", type=str, default=train_data_path)
    parser.add_argument("--eval_data_path", type=str, default=eval_data_path)
    parser.add_argument("--debug", action="store_true", default=debug)

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=per_device_train_batch_size,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, default=num_train_epochs, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=gradient_accumulation_steps,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # parser.add_argument("--peft_path", type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config
            load_in_4bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    # model = PeftModel.from_pretrained(model, args.peft_path)
    config = LoraConfig(
        r=64, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    print_trainable_parameters(model)
    # trainable params: 6738415616 || all params: 6738415616 || trainable%: 100.00
    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    # trainable params: 33554432 || all params: 6771970048 || trainable%: 0.50

    # with load_in_4bit=True
    # trainable params: 262410240 || all params: 3500412928 || trainable%: 7.50
    # trainable params: 33554432 || all params: 3533967360 || trainable%: 0.95

    # Load dataset
    data_files = {}
    if args.train_data_path is not None:
        data_files["train"] = args.train_data_path
    if args.eval_data_path is not None:
        data_files["eval"] = args.eval_data_path
    raw_datasets = load_dataset("json", data_files=data_files)

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    # Preprocess dataset

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import json
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import get_prompt, get_bnb_config, print_trainable_parameters
import os
import argparse
from accelerate import Accelerator
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
    get_linear_schedule_with_warmup,
    BitsAndBytesConfig
)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def preprocess_function(examples):
    prompts = [get_prompt(x) for x in examples["instruction"]]  
    outputs = examples["output"]
    # debug
    # tk_p = tokenizer(prompts, add_special_tokens=False)
    # tk_o = tokenizer(outputs, add_special_tokens=False)
    tokenized_inputs = tokenizer(prompts, outputs, 
                                 add_special_tokens=False, 
                                 return_token_type_ids=True,
                                 truncation="only_second",
                                 max_length=2048,
                                 )

    tokenized_inputs["input_ids"] = [[tokenizer.bos_token_id] + x + [tokenizer.eos_token_id] for x in tokenized_inputs["input_ids"]]
    tokenized_inputs["attention_mask"] = [[1] * len(x) for x in tokenized_inputs["input_ids"]]
    tokenized_inputs["token_type_ids"] = [[0] + x + [1] for x in tokenized_inputs["token_type_ids"]]
    return tokenized_inputs

def parse_args():
    base_model_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/model/Taiwan-LLM-7B-v2.0-chat"
    train_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/train.json"
    eval_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/public_test.json"
    output_dir = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/peft_output"
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2
    learning_rate = 5e-5
    num_train_epochs = 1
    gradient_accumulation_steps = 2
    debug = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=base_model_path)
    parser.add_argument("--train_data_path", type=str, default=train_data_path)
    parser.add_argument("--eval_data_path", type=str, default=eval_data_path)
    parser.add_argument("--output_dir", type=str, default=output_dir)
    parser.add_argument("--debug", action="store_true", default=debug)

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=per_device_train_batch_size,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=per_device_eval_batch_size,
        help="Batch size (per device) for the evaluation dataloader.",
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config
    )  
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    # Load LoRA
    lora_config = LoraConfig(
        r=4, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = prepare_model_for_kbit_training(model)
    print_trainable_parameters(model)
    lora_model = get_peft_model(model, lora_config)
    print_trainable_parameters(lora_model)
    # save model
    # model.save_pretrained(os.path.join(args.output_dir, "checkpoint/"))
    # tokenizer.save_pretrained(os.path.join(args.output_dir, "checkpoint/"))

    trainer = transformers.Trainer(
        model=lora_model,
        # train_dataset=data["train"],
        # args=transformers.TrainingArguments(
        #     per_device_train_batch_size=2,
        #     num_train_epochs=5,
        #     gradient_accumulation_steps=4,
        #     warmup_steps=2,
        #     max_steps=10,
        #     learning_rate=2e-4,
        #     fp16=True,
        #     logging_steps=1,
        #     output_dir=os.path.join(args.output_dir, "checkpoint/"),
        #     optim="paged_adamw_8bit"
        # ),
        # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.save_model(os.path.join(args.output_dir, "checkpoint/"))
    print(os.path.exists(os.path.join(args.output_dir, "checkpoint/")))
    # lora_model.save_pretrained(os.path.join(args.output_dir, "checkpoint/"))
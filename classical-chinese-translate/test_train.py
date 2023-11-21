from curses import raw
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
    get_linear_schedule_with_warmup
)

from torch import device, le, nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, batch, return_outputs=False):
        output_mask = batch.pop("token_type_ids") # [batch_size, seq_len]
        label = batch["input_ids"] # [batch_size, seq_len]
        outputs = model(**batch)
        out_logits = outputs.logits # [batch_size, seq_len, vocab_size]

        # calculate loss
        loss_fnc = nn.CrossEntropyLoss(reduction="none")
        shift_logits = out_logits[..., :-1, :].contiguous() # [batch_size, seq_len-1, vocab_size]
        shift_label = label[..., 1:].contiguous() # [batch_size, seq_len-1]
        shift_output_mask = output_mask[..., 1:].contiguous() # [batch_size, seq_len-1]
        ce_loss = loss_fnc(shift_logits.transpose(1, 2), shift_label) # [batch_size, seq_len-1]
        ce_loss_per_batch = ((ce_loss * shift_output_mask).sum(1) / shift_output_mask.sum(1)).sum() # [batch_size].sum()
        return (ce_loss_per_batch, outputs) if return_outputs else ce_loss_per_batch


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
    num_train_epochs = 3
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
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int, default=gradient_accumulation_steps, 
        help="gradient accumulation steps."
    )
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
            # TODO: Does not use bnb_config
            load_in_4bit=True
        )  
        model = prepare_model_for_kbit_training(model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    lora_config = LoraConfig(
        r=4, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )

    print_trainable_parameters(model)
    lora_model = get_peft_model(model, lora_config)
    print_trainable_parameters(lora_model)

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
    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        # load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on training dataset",
    )

    eval_dataset = raw_datasets["eval"].map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["eval"].column_names,
        # load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on evaluation dataset",
    )

    print(f"Size of the train set: {len(train_dataset)}")
    print(f"Size of the eval set: {len(eval_dataset)}")

    data_collator = DataCollatorWithPadding(tokenizer, 
                                            padding = 'longest',
                                            # pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
                                            )

    # prepare optimizer and schedule (linear warmup and decay)
    # optimizer = torch.optim.AdamW(lora_model.parameters(), lr=args.learning_rate)
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    # )

    # training
    trainer = CustomTrainer(
        model=lora_model,
        args=transformers.TrainingArguments(
            output_dir=os.path.join(args.output_dir, "checkpoint/"),
            do_train=True,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            remove_unused_columns=False,
            logging_strategy="steps",
            logging_steps=0.2,
        ),
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        # optimizers=(optimizer, ),
    )
    trainer.train()
    trainer.save_model()
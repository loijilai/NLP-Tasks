import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from utils import get_prompt, get_bnb_config, print_trainable_parameters
from ppl import perplexity
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

def preprocess_function(examples):
    prompts = [get_prompt(x) for x in examples["instruction"]]  
    tokenized_inputs = tokenizer(prompts,
                                 add_special_tokens=False, 
                                 max_length=2048,
                                 )

    return tokenized_inputs

def parse_args():
    base_model_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/model/Taiwan-LLM-7B-v2.0-chat"
    test_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/public_test.json"
    per_device_test_batch_size = 5
    debug = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=base_model_path)
    parser.add_argument("--test_data_path", type=str, default=test_data_path)
    parser.add_argument("--debug", action="store_true", default=debug)
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=per_device_test_batch_size,
        help="Batch size (per device) for the training dataloader.",
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
            load_in_4bit=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    # config = LoraConfig(
    #     r=64, 
    #     lora_alpha=32, 
    #     lora_dropout=0.05, 
    #     bias="none", 
    #     task_type="CAUSAL_LM"
    # )

    # print_trainable_parameters(model)
    # model = get_peft_model(model, config)
    # print_trainable_parameters(model)

    # Load dataset
    data_files = {}
    if args.test_data_path is not None:
        data_files["test"] = args.test_data_path
    raw_datasets = load_dataset("json", data_files=data_files)

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    # Preprocess dataset
    test_dataset = raw_datasets["test"].map(
        preprocess_function,
        batched=True,
        # num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets["test"].column_names,
        # load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )


    data_collator = DataCollatorWithPadding(tokenizer, 
                                            padding = 'longest',
                                            # pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
                                            )
    test_dataloader = DataLoader(
        test_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_test_batch_size
    )

    # Generate sequence
    predictions = []
    gen_kwargs = {
        "num_beams": 1,
    }
    model.eval()
    for i, tokenized_instructions in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            input_ids = torch.tensor(tokenized_instructions["input_ids"]).cuda()
            attn_mask = torch.tensor(tokenized_instructions["attention_mask"]).cuda()
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attn_mask,
                **gen_kwargs,
            )
            # debug
            # um_decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            # decoded_prompt = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            #
            generated_tokens = [generated_token[len(input_id):] for generated_token, input_id in zip(generated_tokens, input_ids)]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            predictions.extend(decoded_preds)
    
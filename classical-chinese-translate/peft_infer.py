import json
import peft
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from utils import get_few_shot_prompt, get_prompt, get_bnb_config, print_trainable_parameters
from ppl import perplexity
import argparse
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)

def preprocess_function(examples):
    if args.few_shot:
        prompts = [get_few_shot_prompt(x) for x in examples["instruction"]]  
    prompts = [get_prompt(x) for x in examples["instruction"]]

    tokenized_inputs = tokenizer(prompts,
                                 add_special_tokens=False, 
                                 max_length=2048,
                                 )
    # tokenized_inputs["input_ids"] = [[tokenizer.bos_token_id] + x + [tokenizer.eos_token_id] for x in tokenized_inputs["input_ids"]]
    # tokenized_inputs["attention_mask"] = [[1] * len(x) for x in tokenized_inputs["input_ids"]]

    return tokenized_inputs

def parse_args():
    base_model_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/model/Taiwan-LLM-7B-v2.0-chat"
    peft_model_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/peft_output/checkpoint/checkpoint-675"
    test_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/private_test.json"
    output_file = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/peft_output/prediction.json"
    per_device_test_batch_size = 8
    debug = False
    zero_shot = False
    few_shot = False
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=base_model_path)
    parser.add_argument("--peft_model_path", type=str, default=peft_model_path)
    parser.add_argument("--test_data_path", type=str, default=test_data_path)
    parser.add_argument("--output_file", type=str, default=output_file)
    parser.add_argument("--debug", action="store_true", default=debug)
    parser.add_argument("--zero_shot", action="store_true", default=zero_shot)
    parser.add_argument("--few_shot", action="store_true", default=few_shot)
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
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)


    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print_trainable_parameters(model)
    if not (args.zero_shot or args.few_shot):
        # Load LoRA
        model = PeftModel.from_pretrained(model, args.peft_model_path)
        print_trainable_parameters(model)


    # Load dataset
    data_files = {}
    if args.test_data_path is not None:
        data_files["test"] = args.test_data_path
    raw_datasets = load_dataset("json", data_files=data_files)

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(20))

    # Preprocess dataset
    test_dataset = raw_datasets["test"].map(
        preprocess_function,
        batched=True,
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
            input_ids = tokenized_instructions["input_ids"].cuda()
            attn_mask = tokenized_instructions["attention_mask"].cuda()
            generated_tokens = model.generate(
                input_ids=input_ids,
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

    # format prediction
    data = []
    print(f"Saving prediction to {args.output_file}...")
    for i in range(len(predictions)):
        data.append({"id":raw_datasets['test']['id'][i], "output":predictions[i]})
    # write to json file
    with open(args.output_file, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    print("Done!")        
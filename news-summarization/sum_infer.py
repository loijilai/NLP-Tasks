#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import json
import os

import nltk
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

def parse_args():
    model_name_or_path = "/tmp2/loijilai/adl/NLP-Tasks/news-summarization/outputs"
    source_prefix = "summarize: "
    output_file = "/tmp2/loijilai/adl/NLP-Tasks/news-summarization/outputs/inference/output.jsonl"
    test_file = "/project/dsp/loijilai/adl/dataset2/public.jsonl"
    per_device_test_batch_size = 64
    max_source_length = 256
    max_target_length = 64
    num_beams = 3
    top_k = None
    top_p = None
    temperature = None
    do_sample = False
    debug = False
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--debug", 
        action="store_true", 
        default=debug,
        help="Trim a dataset to the first 100 examples for debugging."
    )
    parser.add_argument(
        "--test_file", type=str, default=test_file, help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=max_source_length,
        help=(
            "The maximum total input sequence length after tokenization"
            "Sequences longer than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=max_target_length,
        help=(
            "The maximum total sequence length for target text after tokenization."
            "Sequences longer than this will be truncated, sequences shorter will be padded. "
            "during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=source_prefix,
        help="A prefix to add before every source text (useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=num_beams,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--do_sample",
        default=do_sample,
        action="store_true",
        help=("Whether do sampling or not."),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=top_k,
        help=(
            "Top k sampling, value of k."
        ),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=top_p,
        help=(
            "Top p sampling, value of p."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=temperature,
        help=(
            "Value of temperature."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=model_name_or_path,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=per_device_test_batch_size,
        help="Batch size (per device) for the testing dataloader.",
    )
    parser.add_argument("--output_file", type=str, default=output_file, help="Where to store the final model.")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

    # Load dataset
    data_files = {}
    if args.test_file is not None:
        data_files["test"] = args.test_file
    raw_datasets = load_dataset("json", data_files=data_files)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    prefix = "summarize: "

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["test"].column_names # ['date_publish', 'title', 'source_domain', 'maintext', 'split', 'id']

    # Get the column names for input/target.
    text_column = "maintext"

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        inputs = examples[text_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        return model_inputs

    ## DEBUG ##
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))

    test_dataset = raw_datasets["test"].map(
        preprocess_function,
        batched=True,
        # num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]

        return preds

    test_dataloader = DataLoader(
        test_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_test_batch_size
    )

    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # Inference
    model.eval()
    gen_kwargs = {
        "max_length": args.max_target_length,
        "num_beams": args.num_beams,
        # "do_sample": args.do_sample,
        # "top_k": args.top_k,
        # "top_p": args.top_p,
        # "temperature": args.temperature
    }
    predictions = []
    print("Inference...")
    for step, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **gen_kwargs,
            )

            # pad the generated tokens
            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
            )

            generated_tokens = generated_tokens.cpu().numpy()

            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            decoded_preds = postprocess_text(decoded_preds)
            predictions.extend(decoded_preds)
    
    # Write to file
    print("Writing to file...")
    # if args.temperature:
    #     tag = "t"+str(int(args.temperature*10))
    # elif args.top_k:
    #     tag = "k"+str(args.top_k)
    # elif args.top_p:
    #     tag = "p"+str(int(args.top_p*10))
    with open(os.path.join(args.output_file), "w") as f:
        for i in range(len(predictions)):
            json.dump({"title":predictions[i],"id":raw_datasets['test']['id'][i]}, f, ensure_ascii=False)
            f.write("\n")
    print("Done!")


if __name__ == "__main__":
    main()
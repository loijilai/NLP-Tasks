import os
import argparse
import json
from itertools import chain

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    default_data_collator,
)

def parse_args():
    ### Arguments ###
    # CUDA_VISIBLE_DEVICES=2
    model_name_or_path = "/tmp2/loijilai/adl/paragraph-selection-QA/outputs/mc/01-bert-base-chinese" # change this
    test_file = "/project/dsp/loijilai/adl/dataset1/test.json"
    context_file = "/project/dsp/loijilai/adl/dataset1/context.json"
    output_dir = "/tmp2/loijilai/adl/paragraph-selection-QA/outputs/result" # do not change
    max_seq_length = 512 # for tokenizer to do static padding
    test_batch_size = 8
    debug = False
    #################

    parser = argparse.ArgumentParser(description="Inference using a transformers model on a multiple choice task")
    parser.add_argument("--experiment", action="store_true", help="Whether or not to create a folder for experiment result")
    # Dataset
    parser.add_argument(
        "--test_file", type=str, default=test_file, help="A csv or a json file containing the testing data."
    )
    parser.add_argument(
        "--context_file", type=str, default=context_file, help="A csv or a json file containing the context data."
    )
    # Padding & Sequence length
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=max_seq_length,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    # Model, config, tokenizer 
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=model_name_or_path,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    # Inference: batch_size
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=test_batch_size,
        help="Batch size (per device) for the test dataloader.",
    )
    # Environment setting
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Where to store the final model.")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=debug,
        help="Activate debug mode and run inferencing only with a subset of data.",
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Handling output directory
    # find the lastest directory in the output_dir
    if args.experiment:
        latest = 0
        for dir_name in os.listdir(args.output_dir):
            num = int(dir_name.split("_")[0])
            if num > latest:
                latest = num
        args.output_dir = os.path.join(args.output_dir, f"{latest+1:02d}")
        print("output_dir is set to " + args.output_dir)
        os.mkdir(args.output_dir)
        with open(os.path.join(args.output_dir, "model.txt"), "w") as f:
            f.write("MC: " + args.model_name_or_path)
        print("writing model.txt is done!")


    # Loading pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path, config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Data loading
    data_files = {}
    data_files["test"] = args.test_file
    raw_datasets = load_dataset("json", data_files=data_files)
    with open(args.context_file, 'r') as f:
        context_json = json.load(f)

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(10))

    # Data preprocessing and create dataloader
    def preprocess_function(batch_examples):
        # Think batch_exmaples as raw_datasets['train'][:5]
        paragraphs = []
        # paras are four pargraphs index of an example: [p1, p2, p3, p4]
        for paras in batch_examples['paragraphs']:
            for p_index in paras:
                paragraphs.append(context_json[p_index])
            
        questions = []
        for question in batch_examples['question']:
            questions.append([question]*4)

        # labels = []
        # for paras, ans in zip(batch_examples['paragraphs'], batch_examples['relevant']):
            # labels.append(paras.index(ans))

        # Flatten out
        # paragraphs = list(chain(*paragraphs)) paragraphs does not need to be flatten
        questions = list(chain(*questions))

        # Tokenize
        tokenized_examples = tokenizer(
            paragraphs,
            questions,
            max_length=args.max_seq_length,
            # padding = "max_length" to use static padding so we can use default_data_collator
            padding="max_length", 
            truncation=True,
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
        # tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_datasets = raw_datasets.map(
                        preprocess_function,
                        batched=True, 
                        remove_columns=raw_datasets["test"].column_names
                        )
    test_dataset = processed_datasets["test"]
    test_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.test_batch_size)

    # Inference
    labels = []
    model.eval()
    for batch in tqdm(test_dataloader, desc="Inferencing"):
        # move batch to device
        for k, v in batch.items():
            batch[k] = v.to(device)
        with torch.no_grad(): # batch.size() = (batch_size, 4, max_seq_length)
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) # predictions is a tensor of size (batch_size) with predicted labels in it
        labels.extend(predictions.tolist())

    # Writing output_data to json file
    raw_datasets["test"] = raw_datasets["test"].add_column("labels", labels)
    output_data = []
    for dic in raw_datasets["test"]:
        output_data.append(dic)
    print("Saving output_data to {}...".format(args.output_dir))
    with open(args.output_dir + '/' + 'mc_result.json', 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
    print("Done!")


if __name__ == "__main__":
    main()
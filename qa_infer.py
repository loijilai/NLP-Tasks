import os
import argparse
import json
import csv
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator,
)

def parse_args():
    ### Arguments ###
    # CUDA_VISIBLE_DEVICES=1
    model_name_or_path = "/tmp2/loijilai/adl/paragraph-selection-QA/outputs/qa/03-chinese-macbert-base" # change this
    test_file = "/tmp2/loijilai/adl/paragraph-selection-QA/dataset" # do not change
    context_file = "/project/dsp/loijilai/adl/dataset1/context.json"
    output_dir = "/tmp2/loijilai/adl/paragraph-selection-QA/outputs/result" # do not change
    max_seq_length = 512
    test_batch_size = 8
    doc_stride = 128 # for prepare_validation_features
    max_answer_length = 30 # for postprocess_qa_predictions
    max_test_samples = None # for debugging
    #################
    parser = argparse.ArgumentParser(description="Inference using a transformers model on a Question Answering task")
    parser.add_argument(
        "--test_file", type=str, default=test_file, help="A csv or a json file containing the Prediction data."
    )
    parser.add_argument(
        "--context_file", type=str, default=context_file, help="A csv or a json file containing the context data."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=max_seq_length,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=model_name_or_path,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=test_batch_size,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--output_dir", type=str, default=output_dir, help="Where to store the final prediction.")
    parser.add_argument( #add
        "--doc_stride",
        type=int,
        default=doc_stride,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_answer_length", #add
        type=int,
        default=max_answer_length,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    # Debug
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=max_test_samples,
        help=(
            "For debugging purposes or quicker testing, truncate the number of testing examples to this "
            "value if set."
        ),
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # Handling output directory
    # find the lastest directory in the output_dir
    latest = 0
    for dir_name in os.listdir(args.output_dir):
        num = int(dir_name.split("_")[0])
        if num > latest:
            latest = num
    args.output_dir = os.path.join(args.output_dir, f"{latest:02d}")
    print("output_dir is set to: " + args.output_dir)
    args.test_file = os.path.join(args.output_dir, "mc_result.json")
    print("test_file is set to: " + args.test_file)
    with open(os.path.join(args.output_dir, "model.txt"), "a") as f:
        f.write("\nQA: " + args.model_name_or_path)
    print("writing model.txt is done!")
        
    # Loading pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path, config=config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Data loading
    data_files = {}
    data_files["test"] = args.test_file
    raw_datasets = load_dataset("json", data_files=data_files)
    with open(args.context_file, 'r') as f:
        context_json = json.load(f)

    context_list = []
    for label, paras in zip(raw_datasets["test"]["labels"], raw_datasets["test"]["paragraphs"]):
        relevant_paragraph = context_json[paras[label]]
        context_list.append(relevant_paragraph)

    raw_datasets["test"] = raw_datasets["test"].add_column("context", context_list)

    # Data preprocessing and create dataloader
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" # to use static padding
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples


    column_names = raw_datasets["test"].column_names
    question_column_name = "question"
    context_column_name = "context"
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)
    test_examples = raw_datasets["test"]

    # for debugging
    if args.max_test_samples is not None:
        # We will select sample from whole data
        test_examples = test_examples.select(range(args.max_test_samples))

    test_dataset = test_examples.map(
        prepare_validation_features,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on test dataset",
    )

    # for debugging
    if args.max_test_samples is not None:
        # During Feature creation dataset samples might increase, we will select required samples again
        test_dataset = test_dataset.select(range(args.max_test_samples))
    
    test_dataset_for_model = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_dataloader = DataLoader(
        test_dataset_for_model, collate_fn=default_data_collator, batch_size=args.test_batch_size
    )

    # Post-processing:         # eval_examples, eval_dataset, outputs_numpy
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=args.max_answer_length,
            prefix=stage,
        )
        return predictions


    # Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor
    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        """
        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

        Args:
            start_or_end_logits(:obj:`tensor`):
                This is the output predictions of the model. We can only enter either start or end logits.
            eval_dataset: Evaluation dataset
            max_len(:obj:`int`):
                The maximum length of the output tensor. ( See the model.eval() part for more details )
        """

        step = 0
        # create a numpy array and fill it with -100.
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
        # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather_for_metrics
        for i, output_logit in enumerate(start_or_end_logits):  # populate columns
            # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
            # And after every iteration we have to change the step

            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]

            if step + batch_size < len(dataset):
                logits_concat[step : step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

            step += batch_size

        return logits_concat

    # Inference
    print("***** Running Inference *****")
    print(f"  Num examples after tokenized = {len(test_dataset)}")

    all_start_logits = []
    all_end_logits = []

    model.eval()
    for batch in tqdm(test_dataloader, desc="Inferencing"):
        # Move to gpu
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            all_start_logits.append(start_logits.cpu().numpy())
            all_end_logits.append(end_logits.cpu().numpy())

    max_len = max([x.shape[1] for x in all_start_logits])  # Get the max_length of the tensor

    # concatenate the numpy array
    start_logits_concat = create_and_fill_np_array(all_start_logits, test_dataset, max_len)
    end_logits_concat = create_and_fill_np_array(all_end_logits, test_dataset, max_len)

    # delete the list of numpy arrays
    del all_start_logits
    del all_end_logits

    outputs_numpy = (start_logits_concat, end_logits_concat)
    prediction = post_processing_function(test_examples, test_dataset, outputs_numpy)

    # Write prediction to csv file
    print("Saving prediction to {}...".format(args.output_dir))
    with(open(args.output_dir + '/' + 'qa_result.csv', 'w')) as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(["id","answer"])
        for id, answer in prediction.items():
            writer.writerow([id, answer])
    print("Done!")        

if __name__ == "__main__":
    main()
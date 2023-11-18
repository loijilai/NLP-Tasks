from cgi import test
from sklearn import base
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


def perplexity(
    model, tokenizer, data, max_length=2048,
):
    data_size = len(data)
    instructions = [get_prompt(x["instruction"]) for x in data]
    outputs = [x["output"] for x in data]

    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    tokenized_outputs = tokenizer(outputs, add_special_tokens=False)
    output_masks = []

    # Format data
    for i in range(data_size):
        # add bos in front of instruction
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
        # add eos at the end of output
        output_input_ids = tokenized_outputs["input_ids"][i] + [tokenizer.eos_token_id]
        # concatenate instruction and output and update tokenized_instructions
        tokenized_instructions["input_ids"][i] = instruction_input_ids + output_input_ids
        # set attention mask to 1 for all tokens and update tokenized_instructions
        tokenized_instructions["attention_mask"][i] = [1] * len(tokenized_instructions["input_ids"][i])
        # set output mask to 0 for instruction and 1 for output
        output_mask = [0] * len(instruction_input_ids) + [1] * len(output_input_ids)

        # slice to max_length
        tokenized_instructions["input_ids"][i] = torch.tensor(tokenized_instructions["input_ids"][i][:max_length])
        tokenized_instructions["attention_mask"][i] = torch.tensor(tokenized_instructions["attention_mask"][i][:max_length])
        output_mask = torch.tensor(output_mask[:max_length])
        output_masks.append(output_mask)

    # Calculate ppl
    ppls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    for i in tqdm(range(data_size)):
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0) # [1, 211]
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0) # [1, 211]
        output_mask = output_masks[i].unsqueeze(0) # [1, 211]
        label = input_ids

        with torch.no_grad():
            out_logits = model(input_ids, attention_mask=attn_mask).logits # [1, 211, 32000]

        # Remove the last token from the logits. This is because you're predicting the next token 
        # given the previous ones, so the last token doesn't need to be predicted.
        shift_logits = out_logits[..., :-1, :].contiguous() # [1, 210, 32000]
        # when you perform language modeling, you typically predict the next token based on the 
        # preceding tokens, so you don't need the label for the first token in the sequence. 
        # Removing the first label ensures that the ground truth labels correspond correctly 
        # to the predictions made by the model for the subsequent tokens in the sequence.
        shift_label = label[..., 1:].contiguous() # [1, 210]
        shift_output_mask = output_mask[..., 1:].contiguous() # [1, 210]
        ce_loss = loss_fct(shift_logits.transpose(1, 2), shift_label) # [1, 210]
        perplexity_batch = torch.exp(
            (ce_loss * shift_output_mask).sum(1) / shift_output_mask.sum(1)
        )
        ppls += perplexity_batch.tolist()
    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


if __name__ == "__main__":
    base_model_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/model/Taiwan-LLM-7B-v2.0-chat"
    test_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/public_test.json"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=base_model_path,
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    # parser.add_argument(
    #     "--peft_path",
    #     type=str,
    #     required=True,
    #     help="Path to the saved PEFT checkpoint."
    # )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=test_data_path,
        help="Path to test data."
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    model.eval()
    ppl = perplexity(model, tokenizer, data)
    print("Mean perplexity:", ppl["mean_perplexity"])

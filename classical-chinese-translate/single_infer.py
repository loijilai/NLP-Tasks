import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse


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

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            # quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.test_data_path, "r") as f:
        data = json.load(f)

    data_size = len(data)
    instructions = [x["instruction"] for x in data]
    # Tokenize data
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)

    # Generate sequence
    predictions = []
    gen_kwargs = {
        "num_beams": 1,
    }
    model.eval()
    for i in tqdm(range(data_size)):
        with torch.no_grad():
            input_ids = torch.tensor(tokenized_instructions["input_ids"][i]).unsqueeze(0).cuda()
            attn_mask = torch.tensor(tokenized_instructions["attention_mask"][i]).unsqueeze(0).cuda()
            generated_tokens = model.generate(
                input_ids,
                attention_mask=attn_mask,
                **gen_kwargs,
            )
            um_decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            generated_tokens = generated_tokens[:, len(input_ids[0]):]
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            predictions.extend(decoded_preds)
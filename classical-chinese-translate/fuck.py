import argparse
import os

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed

from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

from utils import get_prompt, get_bnb_config, print_trainable_parameters


"""
Fine-Tune Llama-7b on SE paired dataset
"""


def get_args():
    model_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/model/Taiwan-LLM-7B-v2.0-chat"
    train_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/train.json"
    eval_data_path = "/tmp2/loijilai/adl/NLP-Tasks/classical-chinese-translate/dataset/public_test.json"
    debug = True
    num_train_epochs = 1
    batch_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=model_path)
    parser.add_argument("--train_data_path", type=str, default=train_data_path)
    parser.add_argument("--eval_data_path", type=str, default=eval_data_path)
    parser.add_argument("--debug", action="store_true", default=debug)

    parser.add_argument("--seq_length", type=int, default=1024) # for tokenizer
    parser.add_argument("--num_train_epochs", type=int, default=num_train_epochs)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--no_fp16", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")

    return parser.parse_args()


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text =  f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {example['instruction']} ASSISTANT: {example['output']}"
    return text


def create_datasets(tokenizer, args):
    data_files = {}
    if args.train_data_path is not None:
        data_files["train"] = args.train_data_path
    if args.eval_data_path is not None:
        data_files["eval"] = args.eval_data_path
    raw_datasets = load_dataset("json", data_files=data_files)

    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    train_data = raw_datasets["train"]
    valid_data = raw_datasets["eval"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        seq_length=args.seq_length,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        seq_length=args.seq_length,
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Loading the model")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # todo: bnb_config = get_bnb_config()

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_4bit=True)
    model = prepare_model_for_kbit_training(model)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # preprocess
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    # train
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the llama model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
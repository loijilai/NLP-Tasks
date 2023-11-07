# News Summarization

## Model description:  
A multi-language T5 model is fine-tuned to do summarization on news content.

## How to train my model:
```
python sum_train.py \
--model_name_or_path "google/mt5-small" \
--source_prefix "summarize: " \
--output_dir <PATH_TO_OUTPUT_DIR> \
--train_file <PATH_TO_TRAIN_FILE> \
--per_device_train_batch_size 4 \
--learning_rate 5e-5 \
--num_train_epochs 25 \
--gradient_accumulation_steps 8 \
--max_source_length 256 \
--max_target_length 64 \
--num_warmup_steps 200 \
--checkpointing_steps "epoch" \
--pad_to_max_length \
--with_tracking \
```

## How to reproduce my inference result:
### Get My Trained Checkpoint
If you do not want to train the model by yourself, you can get my training result directly through the following command  
```
bash ./download.sh
```

### Run Inference Script
Load the trained model in the following format: 
```
bash ./run.sh <PATH_TO_PUBLIC_TEST> <PATH_TO_OUTPUT_FILE>
```
An `output.jsonl` file will be generated with predicted news titles.

### Get Rouge Score
Run the following script to get the rouge score on the model prediction `output.jsonl`.
```
python eval.py --reference <PATH_TO_GROUND_TRUTH> --submission <PATH_TO_output.jsonl>
```
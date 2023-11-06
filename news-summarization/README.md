# News Summarization

## Model description:  

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
### Get My Training Result
If you do not want to train the model by yourself, you can get my training result directly through the following command  
```
bash ./download.sh
```

### Run Inference Script
Load the trained model in the following format: 
```
bash ./run.sh <PATH_TO_PUBLIC_TEST> <PATH_TO_OUTPUT>
```
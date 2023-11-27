# Classical Chinese Translation

## Model description:  
Qlora fine-tune Taiwan Llama to perform translation from plain Chinese to classicl Chinese and vice versa.

## How to train my model:
```
python peft_train.py \
    base_model_path <Taiwan-Llama-Checkpoint> \
    train_data_path <Path-to-train.json> \
    output_dir <Path-to-ouput-dir> \
    per_device_train_batch_size 4 \
    learning_rate 5e-5 \
    num_train_epochs 3 \
    gradient_accumulation_steps 2
```

## How to reproduce my inference result:
### Get My Trained Checkpoint
If you do not want to train the model by yourself, you can get my training result directly through the following command  
```
bash ./download.sh
```

### Run Inference to Get Predictions Output
Load the trained model in the following format: 
```
bash ./run.sh <PATH_TO_TAIWAN_LLAMA> <PATH_TO_ADAPTER_CHECKPOINT> <PATH_TO_TEST_FILE> <PATH_TO_OUTPUT_FILE>
```
An `json` file will be generated with the model's output.  
You can also perform few-shot or zero-shot learning (without qlora-finetuning) using `--zero_shot` and `--few_shot` tag.  

### Get Perplexity Score
```
python3 ppl.py \
    --base_model_path /path/to/Taiwan-Llama \
    --peft_path /path/to/adapter_checkpoint/under/your/folder \
    --test_data_path /path/to/input/data
```
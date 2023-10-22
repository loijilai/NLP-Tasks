# Paragraph Selection QA

## Model description:  
Given four paragraphs and one question, select one paragraph that is most relevant to the question asked. After that, perform extractive QA on the paragraph selected.

## How to train my model:
### Environment Setting
* First setup directory structure and get dataset
```
python set_train_env.py
```
The directory will look like this after executing `set_train_env.py`
```
├── dataset
│   ├── context.json
│   ├── test.json
│   ├── train.json
│   └── valid.json
├── model
│   ├── mc
│   └── qa
├── mc_train.py
├── mc_infer.py
├── qa_train.py
├── qa_infer.py
├── ...
├── .gitignore
└── README.md
```

### Multiple Choice
Internet connection is requried to load `hfl/chinese-macbert-base` from huggingface model hub.
```
python mc_train.py \
--model_name_or_path "hfl/chinese-macbert-base" \
--train_file "./dataset/train.json" \
--validation_file "./dataset/valid.json" \
--context_file "./dataset/context.json" \
--output_dir "./model/mc" \
--max_seq_length 512 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--num_train_epochs 5
```

### Extractive QA
Internet connection is requried to load `hfl/chinese-macbert-base` from huggingface model hub.
```
python qa_train.py \
--model_name_or_path "hfl/chinese-macbert-base" \
--train_file "./dataset/train.json" \
--validation_file "./dataset/valid.json" \
--context_file "./dataset/context.json" \
--output_dir "./model/qa" \
--checkpointing_steps "epoch" \
--doc_stride 128 \
--num_train_epochs 10 \
--learning_rate 3e-5 \
--max_seq_length 512 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2
```

## How to reproduce my inference result:
### Get My Training Result
If you do not want to train the model by yourself, you can get my training result directly through the following command  
```
bash ./download.sh
```
or
```
python get_my_model_and_data.py
```
The directory will look like this after executing `get_my_model_and_data.py`

```
├── dataset
│   ├── context.json
│   ├── test.json
│   ├── train.json
│   └── valid.json
├── model
│   ├── mc
|   │   ├── config.json
|   │   ├── special_tokens_map.json
|   │   ├── tokenizer_config.json
|   │   ├── tokenizer.json
|   │   ├── vocab.txt
|   │   └── pytorch_model.bin
│   └── qa
|       ├── config.json
|       ├── special_tokens_map.json
|       ├── tokenizer_config.json
|       ├── tokenizer.json
|       ├── vocab.txt
|       └── pytorch_model.bin
├── mc_train.py
├── mc_infer.py
├── qa_train.py
├── qa_infer.py
├── ...
├── .gitignore
└── README.md
```

### Run Inference Script
Load the trained model in the following format: `bash ./run.sh <PATH_TO_CONTEXT> <PATH_TO_TEST> <PATH_TO_OUTPUT>`
```
bash ./run.sh ./dataset/context.json ./dataset/test.json .
```
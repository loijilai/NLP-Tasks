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

```

### Multiple Choice
Internet connection is requried to load `hfl/chinese-macbert-base` from huggingface model hub.
```
python mc_train.py \
--model_name_or_path "hfl/chinese-macbert-base" \
--train_file "./dataset/train.json" \
--validation_file "./dataset/valid.json" \
--context_file "./dataset/context.json" \
--output_dir "./outputs/mc" \
--max_seq_length 512 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--learning_rate 3e-5 \
--num_train_epochs 5 \
--with_tracking True \
--debug False
```

### Extractive QA
Internet connection is requried to load `hfl/chinese-macbert-base` from huggingface model hub.
```
python qa_train.py \
--model_name_or_path "hfl/chinese-macbert-base" \
--train_file "./dataset/train.json" \
--validation_file "./dataset/valid.json" \
--context_file ".dataset/context.json" \
--output_dir "./outputs/qa" \
--max_train_samples None \
--max_eval_samples None \
--max_predict_samples None \
--checkpointing_steps "epoch" \
--doc_stride 128 \
--num_train_epochs 10 \
--learning_rate 3e-5 \
--max_seq_length 512 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2
```

## How to reproduce the inference result:
```
bash ./download.sh
```

```
bash ./run.sh ./dataset/context.json ./dataset/test.json ./outputs/result
```
CUDA_VISIBLE_DEVICES=2 python mc_no_trainer.py \
  --model_name_or_path bert-base-chinese \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir /tmp2/loijilai/adl/paragraph-selection-QA/outputs/mc \
  --train_file /project/dsp/loijilai/adl/dataset1/train.json \
  --validation_file /project/dsp/loijilai/adl/dataset1/valid.json \
  --context_file /project/dsp/loijilai/adl/dataset1/context.json \
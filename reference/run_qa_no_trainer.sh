CUDA_VISIBLE_DEVICES=2 python run_qa_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp2/loijilai/adl/paragraph-selection-QA/outputs/qa/tmp
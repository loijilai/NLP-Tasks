python sum_infer.py \
--model_name_or_path "./model" \
--source_prefix "summarize: " \
--test_file ${1} \
--output_file ${2} \
--per_device_test_batch_size 32 \
--max_source_length 256 \
--max_target_length 64 \
--num_beams 5
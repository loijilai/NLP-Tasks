python peft_infer.py \
--base_model_path ${1} \
--peft_model_path ${2} \
--test_data_path  ${3} \
--output_file ${4} \
--per_device_test_batch_size 8 \
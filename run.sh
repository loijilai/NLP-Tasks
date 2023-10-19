python mc_infer.py \
--model_name_or_path "./model/mc" \
--test_file ${2} \
--context_file ${1} \
--output_dir "." \
--max_seq_length 512 \
--test_batch_size 8 \
--debug = False

python qa_infer.py \
--model_name_or_path "./model/qa" \
--test_file "./mc_result.json" \
--context_file ${1} \
--output_dir ${3} \
--max_seq_length 512 \
--test_batch_size 8 \
--doc_stride 128 \
--max_answer_length 30 \
--max_test_samples None
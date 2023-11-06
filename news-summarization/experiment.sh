python sum_infer.py --num_beams 1
python sum_infer.py --num_beams 3
python sum_infer.py --num_beams 5
python sum_infer.py --num_beams 7
python sum_infer.py --num_beams 10
python sum_infer.py --do_sample --top_k 10  --num_beams 1
python sum_infer.py --do_sample --top_k 50  --num_beams 1
python sum_infer.py --do_sample --top_k 100 --num_beams 1
python sum_infer.py --do_sample --top_p 0.3 --num_beams 1 
python sum_infer.py --do_sample --top_p 0.5 --num_beams 1
python sum_infer.py --do_sample --top_p 0.7 --num_beams 1
python sum_infer.py --do_sample --top_p 0.9 --num_beams 1
python sum_infer.py --do_sample --top_k 10 --num_beams 1 --temperature 0.1
python sum_infer.py --do_sample --top_k 10 --num_beams 1 --temperature 0.5
python sum_infer.py --do_sample --top_k 10 --num_beams 1 --temperature 1
python sum_infer.py --do_sample --top_k 10 --num_beams 1 --temperature 3
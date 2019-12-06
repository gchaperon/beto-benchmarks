# command used to test the program
python run_xnli.py \
    --train_file_path ./XNLI.small/XNLI-MT-1.0/multinli/multinli.train.es.tsv \
    --eval_file_path ./XNLI.small/XNLI-1.0/xnli.dev.tsv \
    --cache_dir ./XNLI.small/cached \
    --model_dir ../beto \
    --output_dir ./XNLI.small/debug_xnli \
    --do_train \
    --do_eval \
    --do_lower_case \
    --no_cuda \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 1 \
    --max_steps 20 \
    --warmup_steps 150 \
    --overwrite_output_dir \
    --overwrite_cache
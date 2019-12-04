Example command:
```bash
python run_xnli.py \
    --data_dir $DATA_DIR \
    --model_dir $MODEL_DIR 
    --output_dir $OUTPUT_DIR
    --max_seq_length 1024
    --do_train
    --do_eval
    --do_lower_case
    --per_gpu_train_batch_size 8
    --learning_rate 5e-5
    --weight_decay 0.01
    --num_train_epochs 2
    --warmup_steps 150
    --overwrite_output_dir
    --fp16
```
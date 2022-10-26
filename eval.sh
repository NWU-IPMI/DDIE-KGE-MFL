python main.py --data_dir ./data \
    --model_name_or_path ./pubmedbert \
    --pretrained_dir ./pretrained \
    --output_dir ./output \
    --overwrite_output_dir \
    --embedding_path ./embedding/entity_embedding.npy \
    --entity_path ./data/entities.dict \
    --task_name ddie \
    --num_train_epochs 5.0 \
    --max_seq_length 390 \
    --do_eval \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 32 \



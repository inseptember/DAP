accelerate launch run_v4.py \
  --train_file ./data/all_category.json  \
  --model_name_or_path  "junnyu/roformer_chinese_base" \
  --max_length 2500 \
  --per_device_train_batch_size 5 \
  --per_device_eval_batch_size 5 \
  --with_tracking \
  --learning_rate 2e-5 \
  --num_train_epochs 25 \
  --exam_labels \
  --exam_desc ./exam_names.json \
  --output_dir ./output_v3/ \
  --checkpointing_steps epoch

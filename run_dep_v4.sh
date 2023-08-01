
accelerate launch run_dep_v4.py \
 --train_file ./njcenter/train_$1.json \
  --validation_file ./njcenter/dev_$1.json \
  --model_name_or_path output_v3 \
  --max_length 2500 \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 6 \
  --learning_rate 2e-5 \
  --exam_labels \
  --num_train_epochs 25 \
  --with_tracking \
  --exam_desc ./exam_names.json \
  --output_dir ./njcenter/output_flash_dep_$1/ \
  --checkpointing_steps epoch

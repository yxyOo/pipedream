rm profiler.log
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset-dir /data/DNN_Dataset/wmt \
  --target-bleu 21.8 \
  --epochs 20 \
  --math fp32 \
  --print-freq 10 \
  --arch gnmt_large \
  --batch-size 64 \
  --test-batch-size 128 \
  --model-config "{'num_layers': 4, 'hidden_size': 1024, 'dropout':0.2, 'share_embedding': False}" \
  # --optimization-config "{'optimizer': 'Adam', 'lr': 1.75e-3}" \
  --scheduler-config "{'lr_method':'mlperf', 'warmup_iters':1000, 'remain_steps':1450, 'decay_steps':40}" 2>&1 >>  profiler.log
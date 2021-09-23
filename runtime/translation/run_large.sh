export GLOO_SOCKET_IFNAME=docker0
# rm gnmt_large.log
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large.log &
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large.log &
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large.log &
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large.log &

#  rm gnmt_large_32.log
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_32.log &
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_32.log &
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_32.log &
# python main_with_runtime.py  --module models.gnmt_large.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt_large/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_32.log &

# rm gnmt_large_repart.log
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt_large_repart/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart.log &
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt_large_repart/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart.log &
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt_large_repart/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart.log &
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=4 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt_large_repart/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart.log &

# rm gnmt_large_repart1.log
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1.log &
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1.log &
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1.log &
# python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 32 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1.log &

rm gnmt_large_repart1_64.log
python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1_64.log &
python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1_64.log &
python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1_64.log &
python main_with_runtime.py  --module models.gnmt_large_repart.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt_large_repart/gpus=41/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_large_repart1_64.log &
# rm gnmt.log
# python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &
# python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &
# python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &
# python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &


# rm gnmt41.log
# python main_with_runtime.py --module models.gnmt.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt41.log &
# python main_with_runtime.py --module models.gnmt.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt41.log &
# python main_with_runtime.py --module models.gnmt.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt41.log &
# python main_with_runtime.py --module models.gnmt.gpus=41 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt41.log &

rm gnmt_search.log
python main_with_runtime.py  --module models.gnmt_search.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt_search/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_search.log &
python main_with_runtime.py  --module models.gnmt_search.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt_search/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_search.log &
python main_with_runtime.py  --module models.gnmt_search.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt_search/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_search.log &
python main_with_runtime.py  --module models.gnmt_search.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt_search/gpus=4/hybrid_conf.json --distributed_backend gloo 2>&1 >> gnmt_search.log &
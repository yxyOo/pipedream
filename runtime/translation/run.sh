rm gnmt.log
python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 0 --local_rank 0 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &
python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 1 --local_rank 1 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &
python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 2 --local_rank 2 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &
python main_with_runtime.py --module models.gnmt.gpus=4 -b 64 --data_dir /data/DNN_Dataset/wmt/ --rank 3 --local_rank 3 --master_addr localhost --config_path models/gnmt/gpus=4/mp_conf.json --distributed_backend gloo 2>&1 >> gnmt.log &
        
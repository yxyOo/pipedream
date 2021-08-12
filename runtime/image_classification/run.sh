rm vgg.log
rm log.txt
python main_with_runtime.py --module models.vgg16.gpus=4_straight -b 32 --data_dir /data/DNN_Dataset/imagenet/full/pytorch-imagenet-data --rank 0 --local_rank 0 --master_addr localhost --config_path models/vgg16/gpus=4_straight/mp_conf.json --distributed_backend gloo 2>&1 >> vgg.log &
python main_with_runtime.py --module models.vgg16.gpus=4_straight -b 32 --data_dir /data/DNN_Dataset/imagenet/full/pytorch-imagenet-data --rank 1 --local_rank 1 --master_addr localhost --config_path models/vgg16/gpus=4_straight/mp_conf.json --distributed_backend gloo 2>&1 >> vgg.log &
python main_with_runtime.py --module models.vgg16.gpus=4_straight -b 32 --data_dir /data/DNN_Dataset/imagenet/full/pytorch-imagenet-data --rank 2 --local_rank 2 --master_addr localhost --config_path models/vgg16/gpus=4_straight/mp_conf.json --distributed_backend gloo 2>&1 >> vgg.log &
python main_with_runtime.py --module models.vgg16.gpus=4_straight -b 32 --data_dir /data/DNN_Dataset/imagenet/full/pytorch-imagenet-data --rank 3 --local_rank 3 --master_addr localhost --config_path models/vgg16/gpus=4_straight/mp_conf.json --distributed_backend gloo 2>&1 >> vgg.log &

python3 ddp_run.py --total_epochs 200 --save_every 5 --batch_size 1024 --save SINGLEGPU --gpus 1 &&
python3 ddp_run.py --total_epochs 200 --save_every 5 --batch_size 1024 --save 4GPU_200 --gpus 4 &&
python3 ddp_run.py --total_epochs 800 --save_every 5 --batch_size 1024 --save 4GPU_800 --gpus 4 &&
python3 ddp_run.py --total_epochs 800 --save_every 5 --batch_size 1024 --lr_mul 4 --save 4GPU_800_lr4 --gpus 4 &&
python3 ddp_run.py --total_epochs 800 --save_every 5 --batch_size 1024 --lr_mul 0.25 --save 4GPU_800_lr4_2 --gpus 4

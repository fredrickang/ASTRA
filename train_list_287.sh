## baseline
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 0 --sep_temp 0 --lat_constr 100 &&

##Amplifier test
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 0 --sep_temp 0 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 10 --amplifying 1 --separation 0 --sep_temp 0 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 100 --amplifying 1 --separation 0 --sep_temp 0 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 10000 --amplifying 1 --separation 0 --sep_temp 0 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 100000 --amplifying 1 --separation 0 --sep_temp 0 --lat_constr 2.87 &&

## Amplifying test
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1.046 --separation 0 --sep_temp 0 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 10 --amplifying 1.046 --separation 0 --sep_temp 0 --lat_constr 2.87 &&

## Separation test
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 0.01 --sep_temp 1 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 0.1 --sep_temp 1 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 1 --sep_temp 1 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 3 --sep_temp 1 --lat_constr 2.87 &&

## Tempering test
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 0.01 --sep_temp 1.046 --lat_constr 2.87 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --gpus 4 --lat_penalty 1 --amplifier 1 --amplifying 1 --separation 0.1 --sep_temp 1.046 --lat_constr 2.87




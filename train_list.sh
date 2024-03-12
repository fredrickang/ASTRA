python3 ddp_run.py --total_epochs 90 --save_every 5 --batch_size 1024 --save MULTILAT861ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 1 --rt_loss 0 &&
python3 ddp_run.py --total_epochs 90 --save_every 5 --batch_size 1024 --save MULTILAT861ALPHA3 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 3 --rt_loss 0 &&
python3 ddp_run.py --total_epochs 90 --save_every 5 --batch_size 1024 --save MULTILAT1332ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 13.32 --alpha 1 --rt_loss 0 &&
python3 ddp_run.py --total_epochs 90 --save_every 5 --batch_size 1024 --save MULTILAT1332ALPHA3 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 13.32 --alpha 3 --rt_loss 0 &&

python3 ddp_run.py --total_epochs 90 --save_every 5 --batch_size 1024 --save MULTI_RT_867x3 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 1 --rt_loss 1  --p1 8.67 --p2 8.67 --p3 8.67 &&
python3 ddp_run.py --total_epochs 90 --save_every 5 --batch_size 1024 --save MULTI_RT_435_1688x2 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 1 --rt_loss 1  --p1 4.35 --p2 16.88 --p3 16.88 &&
python3 ddp_run.py --total_epochs 90 --save_every 5 --batch_size 1024 --save MULTI_RT_478_957_2870 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 1 --rt_loss 1  --p1 4.78 --p2 9.57 --p3 28.7

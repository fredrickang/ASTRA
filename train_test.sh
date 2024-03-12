python3 ddp_run.py --total_epochs 5 --save_every 5 --batch_size 1024 --save MULTILAT861ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 1 --rt_loss 0 &&

python3 ddp_run.py --total_epochs 5 --save_every 5 --batch_size 1024 --save MULTI_RT_867x3 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 1 --rt_loss 1  --p1 8.67 --p2 8.67 --p3 8.67 

python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save MULTILAT861ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8.61 --alpha 1 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save MULTILAT800ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 8 --alpha 1 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save MULTILAT750ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 7.5 --alpha 1 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save MULTILAT750ALPHA2 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 7.5 --alpha 2 &&

python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_129_129_129_ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 1 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_645_645_645_ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 1 --p1 6.45 --p2 6.45 --p3 25.8 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_956_956_956_ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 1 --p1 9.56 --p2 9.56 --p3 9.56 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_478_478_1912_ALPHA1 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 1 --p1 4.78 --p2 4.78 --p3 19.12 &&

python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_129_129_129_ALPHA2 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 2 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_645_645_645_ALPHA2 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 2 --p1 6.45 --p2 6.45 --p3 25.8 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_956_956_956_ALPHA2 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 2 --p1 9.56 --p2 9.56 --p3 9.56 &&
python3 ddp_run.py --total_epochs 500 --save_every 5 --batch_size 1024 --save RT_P_478_478_1912_ALPHA2 --gpus 4 --temp 3.0 --temp_decay 0.956 --rt_loss 1 --alpha 2 --p1 4.78 --p2 4.78 --p3 19.12 

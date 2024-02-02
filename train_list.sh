#hard deadline & vanilla
python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT376_ALPHA1_LOSS1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 3.76 --alpha 1 --loss_type 1 &&
python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT376_ALPHA1_LOSS2 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 3.76 --alpha 1 --loss_type 2 &&
python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT376_ALPHA1_LOSS3 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 3.76 --alpha 1 --loss_type 3 &&

python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT287_ALPHA1_LOSS1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 2.87 --alpha 1 --loss_type 1 &&
python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT287_ALPHA1_LOSS2 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 2.87 --alpha 1 --loss_type 2 &&
python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT287_ALPHA1_LOSS3 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 2.87 --alpha 1 --loss_type 3 &&

python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT444_ALPHA1_LOSS1 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 4.44 --alpha 1 --loss_type 1 &&
python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT444_ALPHA1_LOSS2 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 4.44 --alpha 1 --loss_type 2 &&
python3 ddp_run.py --total_epochs 300 --save_every 5 --batch_size 1024 --save TEMP3DECAY0956_LAT444_ALPHA1_LOSS3 --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 4.44 --alpha 1 --loss_type 3

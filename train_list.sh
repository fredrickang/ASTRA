python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --save NORMAL --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 2.87 --alpha 1 --loss_type 1 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --save AMPLIFIED --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 2.87 --alpha 1 --loss_type 2 &&
python3 ddp_run.py --total_epochs 100 --save_every 5 --batch_size 1024 --save LATONLY_AMPL --gpus 4 --temp 3.0 --temp_decay 0.956 --lat_constr 2.87 --alpha 1 --loss_type 3 

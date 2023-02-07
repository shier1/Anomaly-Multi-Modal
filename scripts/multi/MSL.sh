export OMP_NUM_THREADS=1
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=3 --batch_size=32 --learning_rate=1e-4 --mode=train --dataset MSL --data_path=dataset/MSL --input_c=55 --output_c=55


torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=10 --batch_size=32 --learning_rate=1e-4 --mode=test --dataset MSL --data_path=dataset/MSL --input_c=55 --output_c=55

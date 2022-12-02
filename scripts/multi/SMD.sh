export OMP_NUM_THREADS=1

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 0.5 --num_steps=3 --batch_size=16 --learning_rate=1e-4 --mode=train --dataset SMD --data_path=dataset/SMD --input_c=38 --output_c=38


torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 0.5 --num_steps=10 --batch_size=16 --learning_rate=1e-4 --mode=test --dataset SMD --data_path=dataset/SMD --input_c=38 --output_c=38

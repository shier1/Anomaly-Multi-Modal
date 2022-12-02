export OMP_NUM_THREADS=1

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=3 --batch_size=16 --learning_rate=1e-4 --mode=train --dataset PSM --data_path=dataset/PSM --input_c=25 --output_c=25


torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=10 --batch_size=16 --learning_rate=1e-4 --mode=test --dataset PSM --data_path=dataset/PSM --input_c=25 --output_c=25

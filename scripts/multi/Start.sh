export OMP_NUM_THREADS=1
nproc_per_node=1
batch_size=32
learning_rate=1e-4


# SMD
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node train.py \
--name=AnomalyTransformer --anormly_ratio 0.5 --num_steps=10 --batch_size=$batch_size \
--learning_rate=$learning_rate --mode=train --dataset SMD --data_path=dataset/SMD --input_c=38 --output_c=38

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 0.5 --num_steps=10 --batch_size=$batch_size \
--learning_rate=$learning_rate --mode=test --dataset SMD --data_path=dataset/SMD --input_c=38 --output_c=38

#MSL
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=10 --batch_size=$batch_size \
--learning_rate=$learning_rate --mode=train --dataset MSL --data_path=dataset/MSL --input_c=55 --output_c=55

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=10 --batch_size=$batch_size \
--learning_rate=$learning_rate --mode=test --dataset MSL --data_path=dataset/MSL --input_c=55 --output_c=55

#SMAP
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=3 --batch_size=$batch_size \
--learning_rate=$learning_rate --mode=train --dataset SMAP --data_path=dataset/SMAP --input_c=25 --output_c=25

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=10 --batch_size=$batch_size \
--learning_rate=$learning_rate --mode=test --dataset SMAP --data_path=dataset/SMAP --input_c=25 --output_c=25

#PSM
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=3 --batch_size=$batch_size \
--learning_rate=$learning_rate --mode=train --dataset PSM --data_path=dataset/PSM --input_c=25 --output_c=25

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node train.py \
--name=AnomalyTransformer --anormly_ratio 1 --num_steps=10 --batch_size=$batch_size \
--learning_rate=$learning_rate--mode=test --dataset PSM --data_path=dataset/PSM --input_c=25 --output_c=25

#!/bin/bash

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

echo "Serial Benchmark:"
(time python testMNIST.py) &> resultsMNIST.out
GPUS=""
for i in 1 2 3;
do
	GPUS+=$i;	
	echo "$i GPU Benchmark:";
    (time CUDA_VISIBLE_DEVICES=$GPUS python testMNIST.py $i) &>> resultsMNIST.out
	GPUS+=',';
done

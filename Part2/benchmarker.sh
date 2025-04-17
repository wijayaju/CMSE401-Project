#!/bin/bash

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1

echo "Serial Benchmark:"
(time python test.py) &> results.out
GPUS=""
for i in 1 2 3;
do
	GPUS+=$i;	
	echo "$i GPU Benchmark:";
    (time CUDA_VISIBLE_DEVICES=$GPUS python test.py $i) &>> results.out
	GPUS+=',';
done

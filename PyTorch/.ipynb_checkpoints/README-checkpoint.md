To install PyTorch on the HPCC, run either of the following commands:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
or
```
module load PyTorch
```

The latter command has a small chance of failing, so proceed with caution.

Also install the matplotlib library for images:
```
pip install matplotlib
```

To test if PyTorch installation worked, run the following command:
```
sbatch verifyTest.sb
```

The slurm file should have the possible installation failure warning and True if GPU is accessible.


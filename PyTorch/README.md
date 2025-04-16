# Installation
To install PyTorch on the HPCC, run either of the following commands:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
or
```
module load PyTorch
```

- The latter command has a small chance of failing, so proceed with caution.

Also install the matplotlib library for images:
```
pip install matplotlib
```

# Test
To test if PyTorch installation worked on the HPCC, run the following command:
```
sbatch verifyTest.sb
```

- The slurm file should have the possible installation failure warning (HPCC issue), a display of a randomly initialized tensor (if module is installed correctly), and the boolean 'True' (if GPU is accessible).

To download and check if the dataset works, run the cells in the Jupyter Notebook 'exampleTest.ipynb.'

# References
Instructions for setup is sourced from: https://pytorch.org/get-started/locally/
Dataset is sourced from: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
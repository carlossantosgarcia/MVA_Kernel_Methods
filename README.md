# MVA-Kernel-Methods
Graph kernels implementations for the 2023 Data Challenge organised for the MVA course Kernel Methods for Machine Learning. Our different implementations of kernel-based methods for graph binary classification have been coded from scratch using general purpose libraries.

We mainly relied on Support Vector Machines (SVMs) with kernels for graphs. We implemented the following kernels:
- RBF Kernel on hand-crafted features
- Walk Kernel
- Shortest Path Kernel
- Weisfeiler Lehman Subtree Kernel

More details about our work can be found on our [report](report.pdf).

# Replicating our best submissions
In order to replicate our results, run:
```
pip install -r requirements.txt
python3 methods/start.py --method wl
```
Method can either be ```wl``` (Weisfeiler Lehman kernel) or ```rbf``` (Gaussian kernel). Hyperparameters have been selected through 5-fold cross-validation. Output files will be saved in ```submissions/``` folder.

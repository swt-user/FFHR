# FFHR
## Installation

(Default for linux)

First, make sure you have installed anaconda and your cuda version is above 10.0.

Then, create a python>=3.7 environment and install dependencies:

```bash
conda create -n your_env python==3.7
source activate your_env
pip install -r requirements.txt
```

Ensure your torch and CUDA version.

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```

Install pyg package.

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
```

where CUDA and TORCH should be replaced by the specific CUDA version (cpu, cu92, cu101, cu102, cu110, cu111) and PyTorch version (1.4.0, 1.5.0, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0), respectively. 
For example, for PyTorch 1.8.0 and CUDA 10.2, type:

```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.8.0+cu102.html
pip install torch-geometric
```

Then you can run our code.

```bash
source set_env.sh
source configs/xxx_xx_xx.sh
```

If your device don't have enough memory to support 512 dimension, you can set the dimension size as 256, which can also get similar results.
If you still suffering from memory problem, you can set the data_type as single(float).

## Acknowledgement

We refer to the code of [AttH](https://github.com/HazyResearch/KGEmb). Thanks for their contributions.






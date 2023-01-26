# sbio_sc for single cell analysis

[![python >3.8](https://img.shields.io/badge/python-3.8.16-brightgreen)](https://www.python.org/) 

### sbio_sc for single cell analysis using PyTorch
Superbio provide pretrained models to perform tasks on scRNA data. This can include, but is not limited to: differential gene expression analysis, cell type annotation, dimension reduction, clustering and network analysis (https://www.superbio.ai). To use the Web GUI version of this repo and others visit (https://app.superbio.ai)

# Install
Please refer to requirements.txt for dependencies. Example installation commands are shown below, with the cuda and pytorch installations depending on local GPU hardware. Please refer to (https://pytorch.org/get-started/locally/) for more details
```
pip install -r requirements.txt
conda install cuda --channel nvidia/label/cuda-11.6.0
conda install pytorch torchtext pytorch-cuda=11.6 -c pytorch -c nvidia
pip install pytorch-lightning
```

# Usage

The scanpy and anndata libraries are used for loading and preprocessing data. Preprocessing can be performed as follows:

- Read data example
```
adata = sc.read('D:/datasets/single_cell/heart_atlas.h5ad',
    cache=True
)
```

- Preprocess data example
```
preprocessor = Preprocessor(
    batch_key='cell_source',
    filter_gene_counts=3,
    normalize_total=1e4,
    log=True,
    subset_hvg=1200,
    hvg_flavor='seurat',
    remove_outliers=0.99
)
preprocessor(adata)
```

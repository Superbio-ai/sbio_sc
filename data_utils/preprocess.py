from typing import Dict, Optional, Union

import numpy as np
from scipy.sparse import issparse
import scanpy as sc
from scanpy.get import _get_obs_rep, _set_obs_rep
from anndata import AnnData

from data_utils import logger


class Preprocessor:
    """
    Prepare data into training, valid and test split. Normalize raw expression
    values, binning or using other transform into the preset model input format.
    """

    def __init__(
        self,
        layer_key: Optional[str] = 'X',
        batch_key: Optional[str] = None,
        filter_gene_counts: Union[int, float, bool] = False,
        filter_cell_counts: Union[int, float, bool] = False,
        normalize_total: Union[float, bool] = 1e4,
        result_norm_key: Optional[str] = "X_norm",
        log: bool = False,
        result_log_key: str = "X_log",
        subset_hvg: Union[int, float, bool] = False,
        hvg_flavor: str = "seurat",
        remove_outliers: float = 0.99
    ):
        r"""
        Set up the preprocessor, use the args to config the workflow steps.

        Args:

        layer_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for preprocessing.
        batch_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection and train-test split steps.
        filter_gene_counts (:class:`int` or 'float' or `bool`, default: ``False``):
            Whther to filter genes by counts, if :class:`int`, or by fraction, if :class:'float'.
        filter_cell_counts (:class:`int` or 'float' or `bool`, default: ``False``):
            Whther to filter cells by counts, if :class:`int`, or by fraction, if :class:'float'. 
        normalize_total (:class:`float` or :class:`bool`, default: ``1e4``):
            Whether to normalize the total counts of each cell to a specific value.
        result_norm_key (:class:`str`, default: ``"X_normed"``):
            The key of :class:`~anndata.AnnData` to store the normalized data. If
            :class:`None`, will use normed data to replce the :attr:`use_key`.
        log (:class:`bool`, default: ``True``):
            Whether to apply log transform to the normalized data.
        result_log_key (:class:`str`, default: ``"X_log"``):
            The key of :class:`~anndata.AnnData` to store the log1p transformed data.
        subset_hvg (:class:`int` or 'float' or `bool`, default: ``False``):
            Whether to subset highly variable genes by counts, if :class:`int`, or by fraction, if :class:'float'.
        hvg_use_key (:class:`str`, optional):
            The key of :class:`~anndata.AnnData` to use for calculating highly variable
            genes. If :class:`None`, will use :attr:`adata.X`.
        hvg_flavor (:class:`str`, default: ``"seurat"``):
            The flavor of highly variable genes selection. See
            :func:`scanpy.pp.highly_variable_genes` for more details. 'cell_ranger' is another common choice
        remove_outliers (:class:'float' or 'bool', default: ''False''):
            Whether to remove outliers by percentile, with the percentile as input.
        """
        self.layer_key = layer_key
        self.batch_key = batch_key
        self.filter_gene_counts = filter_gene_counts
        self.filter_cell_counts = filter_cell_counts
        self.normalize_total = normalize_total
        self.result_norm_key = result_norm_key
        self.log = log
        self.result_log_key = result_log_key
        self.subset_hvg = subset_hvg
        self.hvg_flavor = hvg_flavor
        self.remove_outliers = remove_outliers

    def __call__(self, adata: AnnData) -> Dict:
        """
        format controls the different input value wrapping, including categorical
        binned style, fixed-sum normalized counts, log1p fixed-sum normalized counts, etc.

        Args:

        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        """
        key_to_process = self.layer_key
        
        # preliminary checks, will use later
        if key_to_process == "X":
            key_to_process = None  # the following scanpy apis use arg None to use X
            adata.layers['raw']=adata.X
        else:
            adata.X=adata.layers[key_to_process]
            adata.layers['raw']=adata.layers[key_to_process]
        is_logged = self.check_logged(adata, obs_key=key_to_process)
        
        #parameter sanity checks
        logger.info("Sanity checking input parameters ...")
        cat_cols = list(adata.obs.select_dtypes(include=['category','object']).columns)
        if (self.batch_key not in cat_cols) & (self.batch_key is not None):
            raise ValueError(
                    "batch_key {} not found in categorical columns of dataset.".format(self.batch_key)
                )
        
        # step 1: batch key        
        if self.batch_key is not None:
            adata.obs["str_batch"] = adata.obs[self.batch_key].astype(str)
            batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
            adata.obs["batch_id"] = batch_id_labels

        # step 2: filter genes
        if self.filter_gene_counts:
            logger.info("Filtering genes by counts ...")
            sc.pp.filter_genes(
                adata,
                min_counts=self.filter_gene_counts
                if isinstance(self.filter_gene_counts, int)
                else None,
            )

        # step 3: filter cells
        if isinstance(self.filter_cell_counts, int):
            logger.info("Filtering cells by counts ...")
            sc.pp.filter_cells(
                adata,
                min_counts=self.filter_cell_counts
                if isinstance(self.filter_cell_counts, int)
                else None,
            )

        # step 4: normalize total
        if self.normalize_total:
            logger.info("Normalizing total counts ...")
            sc.pp.normalize_total(
                adata,
                target_sum=self.normalize_total
                if isinstance(self.normalize_total, float)
                else None,
                inplace=True,
            )
            adata.layers[self.result_norm_key]=adata.X

        # step 5: log transform
        if self.log:
            logger.info("Log1p transforming ...")
            if is_logged:
                logger.warning(
                    "The input data seems to be already log1p transformed. "
                    "Set `log1p=False` to avoid double log1p transform."
                )
            if self.result_log_key:                
                sc.pp.log1p(adata)
                adata.layers[self.result_log_key]=adata.X

        # step 6: subset hvg
        if self.subset_hvg:
            logger.info("Subsetting highly variable genes ...")
            if self.batch_key is None:
                logger.warning(
                    "No batch_key is provided, will use all cells for HVG selection."
                )
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=self.subset_hvg
                    if isinstance(self.subset_hvg, int)
                    else None,
                    batch_key=self.batch_key,
                    flavor=self.hvg_flavor,
                    subset=True
                )
            except Exception as e:
                print(f'Error: {e}. Have you checked all parameters including batch_key are suitable?')
  
        # step 7: remove outliers
        if self.remove_outliers:
            logger.info("Removing outliers ...")
            max_value = np.quantile(adata.X.A[np.nonzero(adata.X.A)], self.remove_outliers)
            adata.X.A = np.clip(adata.X.A, None, max_value)
        
        logger.info("Preprocessing completed. Base layer 'X' contains fully preprocessed data")
        logger.info(f"Other {adata.layers}")
            
        
    def check_logged(self, adata: AnnData, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already log1p transformed.
        Args:
        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information. This arg
            is used in the highly variable gene selection step.
        """
        data = _get_obs_rep(adata, layer=obs_key)
        max_, min_ = data.max(), data.min()
        if max_ > 30:
            return False
        if min_ < 0:
            return False

        non_zero_min = data[data > 0].min()
        if non_zero_min >= 1:
            return False

        return True
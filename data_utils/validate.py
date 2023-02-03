from typing import Dict, Optional, Union

import anndata
import numpy as np
import os
import sys
from scanpy.get import _get_obs_rep
from scipy.sparse import issparse
import h5py

from data_utils import logger

        
class Validator:
    """Handles validation of AnnData"""

    def __init__(self,
                 h5ad_path: Union[str, bytes, os.PathLike] = ""):

        # Set initial state
        self.warnings = []
        
        self.h5ad_path = h5ad_path
        self.adata = anndata.AnnData()
        self.isvalid = True
        self.normed = False
        self.isfinite = False
        
        
    def __call__(self) -> Dict:
        #read data
        self._read_h5ad(self.h5ad_path)
        #validate encoding
        self._validate_encoding_version()
        #check data is present
        self._check_structure()
        #check_normed
        self.normed = self._check_normed()
        #check finite
        self.isfinite = self._check_finite()
        
        logger.info("Validation completed. No critical errors found")
        if len(self.warnings)>0:
            logger.info(f"Warnings: {self.warnings}")
    
    
    def _read_h5ad(self, h5ad_path: Union[str, bytes, os.PathLike]):
        """
        Reads h5ad into self.adata
        :params Union[str, bytes, os.PathLike] h5ad_path: path to h5ad to read
        :rtype None
        """
        try:
            # H5AD has to be loaded in memory mode. If not the types of X are not properly retrieved by anndata
            # see https://github.com/theislab/anndata/issues/326#issuecomment-892203924
            self.adata = anndata.read_h5ad(h5ad_path, backed=None)
        except (OSError, TypeError):
            self.isvalid=False
            logger.info(f"Unable to open '{h5ad_path}' with AnnData")
            sys.exit(1)

        self.h5ad_path = h5ad_path

    def _validate_encoding_version(self):

        with h5py.File(self.h5ad_path, "r") as f:
            encoding_dict = dict(f.attrs)
            encoding_version = encoding_dict.get("encoding-version")
            if encoding_version != "0.1.0":
                self.warnings.append(
                    "The h5ad artifact was generated with an AnnData version different from 0.8.0."
                )
                
    def _check_structure(self):
        
        if not hasattr(self.adata, 'X'):
            logger.info('The h5ad artifact does not contain expression data".X".')
            self.isvalid = False
            sys.exit(1)
            
        if not hasattr(self.adata, 'obs'):
            self.warnings.append(
                    'The h5ad artifact does not contain observation information ".obs".'
                )
            
        if not hasattr(self.adata, 'var'):
            self.warnings.append(
                    'The h5ad artifact does not contain variable information ".var".'
                )
        
        if not hasattr(self.adata, 'obsm'):
            self.warnings.append(
                    'The h5ad artifact does not contain experiment design information ".obsm".'
                )
        
        if not hasattr(self.adata, 'uns'):
            self.warnings.append(
                    'The h5ad artifact does not contain schema information ".uns".'
                )
        
    def _check_normed(self, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already normed.
        Args:
        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information.
        """
        data = _get_obs_rep(self.adata, layer=obs_key)

        diff_sum = np.array(data!=data[0]).sum()

        if diff_sum >0:
            return False
        else:
            return True
    
    def _check_finite(self, obs_key: Optional[str] = None) -> bool:
        """
        Check if the data is already normed.
        Args:
        adata (:class:`AnnData`):
            The :class:`AnnData` object to preprocess.
        obs_key (:class:`str`, optional):
            The key of :class:`AnnData.obs` to use for batch information.
        """
        data = _get_obs_rep(self.adata, layer=obs_key)

        if issparse(data):
            return np.isfinite(data.A).all()
        else:
            return np.isfinite(data).all()
        
    
    
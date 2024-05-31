from itertools import product
from pathlib import Path
from typing import Literal

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

n_obs = 50
n_var = 25
def make_diagonal(dtype: Literal["int32", "int64", "float32"], m: int, n: int):
    mat = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        if i != min(m // 2, n // 2) and i < n:
            mat[i, i] = i
    return mat

def make_mats(m: int, n: int):
    dtypes = ['int32', 'int64', 'float32']
    return dict(
        zip(
            [f"{dtype}_{format}" for (dtype, format) in product(dtypes, ["dense", "csc", "csr"])], [mat_class(mat) for (mat, mat_class) in product([
                make_diagonal(dtype, m, n) for dtype in dtypes
            ], [np.array, sparse.csc_matrix, sparse.csr_matrix])]
        )
    )

def create_zarr_anndata(output_dir: Path, n_obs: int, n_var: int, **kwargs):
    layers = make_mats(n_obs, n_var)
    adata = anndata.AnnData(
        # Generate a fairly sparse matrix
        X=layers['float32_dense'],
        obs=pd.DataFrame(index=[f"obs_{i}" for i in range(n_obs)], data={"categorical": pd.Categorical([f"cat_{i % 5}" for i in range(n_obs)]), "string": np.array([f"str_{i}" for i in range(n_obs)])}),
        var=pd.DataFrame(index=[f"var_{i}" for i in range(n_var)]),
        obsm=make_mats(n_obs, 2),
        obsp=make_mats(n_obs, n_obs),
        varm=make_mats(n_var, 2),
        varp=make_mats(n_var, n_var),
        layers=layers
    )
    format = kwargs.pop("X_format", None)
    if format:
        if format == "csc":
            adata.X = sparse.csc_matrix(adata.X)
            adata.write_zarr(output_dir / f'anndata-csc.zarr')
        elif format == "csr":
            adata.X = sparse.csr_matrix(adata.X)
            adata.write_zarr(output_dir / f'anndata-csr.zarr')
        elif format == "dense":
            adata.write_zarr(output_dir / f'anndata-dense.zarr')
    else:
        del adata.X
        adata.write_zarr(output_dir / f'anndata-no-X.zarr')


def main():
    output_dir = Path(".".join(anndata.__version__.split(".")[:2]))
    try:
        output_dir.mkdir(exist_ok=True)
    except FileExistsError:
        pass
    create_zarr_anndata(output_dir, n_obs, n_var, X_format="csc")
    create_zarr_anndata(output_dir, n_obs, n_var, X_format="csr")
    create_zarr_anndata(output_dir, n_obs, n_var, X_format="dense")
    create_zarr_anndata(output_dir, n_obs, n_var)


if __name__ == "__main__":
    main()
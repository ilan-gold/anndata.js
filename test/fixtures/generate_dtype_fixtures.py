# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anndata~=0.13.0",  # zarr v2 writing is removed in 0.14
#   "zarr>=3",
#   "pandas",
#   "numpy",
# ]
# ///
"""Generate fixtures covering string dtypes and nullable arrays.

Writes `<anndata major.minor version>/anndata-dtypes.v{2,3}.zarr`. Run from this directory:

    uv run generate_dtype_fixtures.py
    node directory-to-memory-store.mjs <version>/anndata-dtypes.v2.zarr <version>/anndata-dtypes.v2.json
    node directory-to-memory-store.mjs <version>/anndata-dtypes.v3.zarr <version>/anndata-dtypes.v3.json
"""

import shutil
from importlib.metadata import version
from pathlib import Path

import anndata
import numpy as np
import pandas as pd

n_obs = 6


def make_adata() -> anndata.AnnData:
    obs = pd.DataFrame(
        index=[f"obs_{i}" for i in range(n_obs)],
        data={
            # encoding-type: string-array
            "string": np.array([f"str_{i}" for i in range(n_obs)], dtype=object),
            # encoding-type: nullable-integer
            "nullable_int": pd.array([0, None, 2, 3, None, 5], dtype="Int64"),
            # encoding-type: nullable-boolean
            "nullable_bool": pd.array(
                [True, None, False, True, None, False], dtype="boolean"
            ),
            # encoding-type: nullable-string-array
            "nullable_string": pd.array(
                ["a", None, "c", "d", None, "f"], dtype="string"
            ),
            # encoding-type: categorical, with a missing value (code -1)
            "categorical_missing": pd.Categorical(
                ["x", "y", None, "x", "y", None]
            ),
        },
    )
    var = pd.DataFrame(index=[f"var_{i}" for i in range(3)])
    adata = anndata.AnnData(obs=obs, var=var)
    # encoding-type: string (0-d scalar)
    adata.uns["string_scalar"] = "hello"
    return adata


def main():
    anndata.settings.allow_write_nullable_strings = True
    output_dir = Path(".".join(version("anndata").split(".")[:2]))
    output_dir.mkdir(exist_ok=True)
    adata = make_adata()
    for zarr_format in [2, 3]:
        out = output_dir / f"anndata-dtypes.v{zarr_format}.zarr"
        if out.exists():
            shutil.rmtree(out)
        anndata.settings.zarr_write_format = zarr_format
        # Keep string columns as string arrays rather than categoricals.
        adata.write_zarr(out, convert_strings_to_categoricals=False)


if __name__ == "__main__":
    main()

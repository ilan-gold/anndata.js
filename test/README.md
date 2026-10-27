## Creating Test Fixtures

To run the test creation script, you will need [uv](https://github.com/astral-sh/uv) installed for the python versioning. After that simply run the following:

```bash
cd fixtures
source generate_fixtures.sh
```

This generates the following fixtures:

- `{0.7,0.8,0.9,0.10}/anndata-{dense,csc,csr,no-X}.{zarr,json}` via `generate_fixture.py`, once per anndata version.
- `0.13/anndata-dtypes.{v2,v3}.{zarr,json}` via `generate_dtype_fixtures.py`, which covers string dtypes and nullable arrays in both zarr v2 and v3 formats. Its dependencies (including the pinned anndata version) are declared inline in the script, so it is run with `uv run`.

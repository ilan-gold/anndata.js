for version in "0.7.0" "0.8.0" "0.9.0" "0.10.0";
do
    venv_dir="venv_$version"
    if [ ! -d "$venv_dir" ]; then
        uv venv "$venv_dir" 
    fi
    source $venv_dir/bin/activate
    uv pip install anndata~=$version zarr
    python generate_fixture.py
    deactivate
    version_dir=$(echo "$version" | cut -d'.' -f1,2)
    for X in "dense" "csc" "csr" "no-X";
    do
        node directory-to-memory-store.mjs $version_dir/anndata-$X.zarr $version_dir/anndata-$X.json
    done
done

# String dtypes and nullable arrays, in both zarr v2 and v3 formats.
# The script declares its own dependencies (UV inline script metadata),
# so it does not use a venv like above.
# Must match the anndata version pinned in generate_dtype_fixtures.py.
dtypes_version_dir="0.13"
uv run generate_dtype_fixtures.py
for zarr_format in "v2" "v3";
do
    node directory-to-memory-store.mjs $dtypes_version_dir/anndata-dtypes.$zarr_format.zarr $dtypes_version_dir/anndata-dtypes.$zarr_format.json
done

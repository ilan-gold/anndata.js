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

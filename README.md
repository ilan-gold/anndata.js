# anndata.js

AnnData for the browser

## Quickstart

```typescript
import { readZarr, get, readElem } from 'anndata.js';
import * as zarr from 'zarrita';

const store = await zarr.tryWithConsolidated(new zarr.FetchStore(path));
const adata = await readZarr(store);
await get(await adata.obs.get("leiden"), [null]) // full selection
await get(adata.X, [zarr.slice(0, 5), null]) // first 5 rows
adata.obs = await readElem(store, "obs") // read a part of the anndata object
```

## Development

Installation of dependencies:

```bash
pnpm install
```

Testing:

```bash
pnpm test
```

Build:

```bash
pnpm build
```

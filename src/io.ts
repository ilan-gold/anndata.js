import * as zarr from "zarrita";

import type { Readable } from "@zarrita/storage";
import AnnData from "./anndata.js";
import AxisArrays from "./axis_arrays.js";
import SparseArray from "./sparse_array.js";
import {
	type AxisKey,
	type AxisKeyTypes,
	AxisKeys,
	type UIntType,
} from "./types.js";
import { LazyCategoricalArray, has } from "./utils.js";

async function readSparse_010<
	S extends Readable,
	D extends zarr.NumberDataType,
>(
	location: zarr.Location<S>,
	key: string,
	elem: zarr.Group<S> | zarr.Array<UIntType, S>,
): Promise<SparseArray<D>> {
	const grp = await zarr.open(elem, { kind: "group" });
	const shape = grp.attrs.shape as number[];
	const format = (grp.attrs["encoding-type"] as string).slice(0, 3) as
		| "csc"
		| "csr";
	const indptr = (await zarr.open(grp.resolve("indptr"), {
		kind: "array",
	})) as zarr.Array<"int32", S>; // todo: allow 64
	const indices = (await zarr.open(grp.resolve("indices"), {
		kind: "array",
	})) as zarr.Array<"int32", S>; // todo: allow 64
	const data = (await zarr.open(grp.resolve("data"), {
		kind: "array",
	})) as zarr.Array<D, S>;
	return new SparseArray(indices, indptr, data, shape, format);
}

async function readCategorical_020<S extends Readable>(
	location: zarr.Location<S>,
	key: string,
	elem: zarr.Group<S> | zarr.Array<UIntType, S>,
) {
	const cats = await zarr.open(elem.resolve("categories"), {
		kind: "array",
	});
	const codes = (await zarr.open(elem.resolve("codes"), {
		kind: "array",
	})) as zarr.Array<UIntType, Readable>;
	return new LazyCategoricalArray(codes, cats);
}

async function readCategorical_noVersion<S extends Readable>(
	location: zarr.Location<S>,
	key: string,
	elem: zarr.Group<S> | zarr.Array<UIntType, S>,
) {
	const { categories } = elem.attrs;
	const cats = await zarr.open(location.resolve(String(categories)), {
		kind: "array",
	});
	return new LazyCategoricalArray(elem as zarr.Array<UIntType, S>, cats);
}

async function readArray<S extends Readable>(
	location: zarr.Location<S>,
	key: string,
	elem: zarr.Group<S> | zarr.Array<UIntType, S>,
) {
	return elem;
}

async function readDict_010<S extends Readable>(
	location: zarr.Location<S>,
	key: Exclude<AxisKey, "X">,
	elem: zarr.Group<S> | zarr.Array<UIntType, S>,
) {
	return new AxisArrays(location as zarr.Group<S>, key);
}

const IO_FUNC_REGISTRY_WITH_VERSION: { [index: string]: any } = {
	"csr_matrix,0.1.0": readSparse_010,
	"csc_matrix,0.1.0": readSparse_010,
	"categorical,0.2.0": readCategorical_020,
	"array,0.2.0": readArray,
	"dict,0.1.0": readDict_010,
	"dataframe,0.1.0": readDict_010,
	"dataframe,0.2.0": readDict_010,
	"anndata,0.1.0": readZarr,
};

const IO_FUNC_REGISTRY_WITHOUT_VERSION: { [index: string]: any } = {
	categorical: readCategorical_noVersion,
	array: readArray,
	dict: readDict_010,
};

export async function readZarr(path: string | Readable) {
	let root: zarr.Group<Readable>;
	if (typeof path === "string") {
		const store = await zarr.tryWithConsolidated(new zarr.FetchStore(path));
		root = await zarr.open(store, { kind: "group" });
	} else {
		root = await zarr.open(path, { kind: "group" });
	}

	const adataInit = {} as AxisKeyTypes<Readable, zarr.NumberDataType>;
	await Promise.all(
		AxisKeys.map(async (k) => {
			if ((k === "X" && (await has(root, k))) || k !== "X") {
				adataInit[k] = await readElem(root, k);
			}
		}),
	);
	return new AnnData(adataInit);
}

export async function readElem<S extends Readable>(
	location: zarr.Location<S>,
	key: string,
) {
	const keyRoot = location.resolve(key);
	const keyNode = await zarr.open(keyRoot);
	const {
		"encoding-version": encodingVersion,
		"encoding-type": encodingType,
		categories,
	} = keyNode.attrs;
	if (encodingVersion === undefined) {
		if (keyNode instanceof zarr.Group) {
			return IO_FUNC_REGISTRY_WITHOUT_VERSION.dict(location, key, keyNode);
		}
		if (categories !== undefined) {
			// Not encoded in old versions of anndata
			return IO_FUNC_REGISTRY_WITHOUT_VERSION.categorical(
				location,
				key,
				keyNode,
			);
		}
		return IO_FUNC_REGISTRY_WITHOUT_VERSION.array(location, key, keyNode);
	}
	return IO_FUNC_REGISTRY_WITH_VERSION[[encodingType, encodingVersion].join()](
		location,
		key,
		keyNode,
	);
}

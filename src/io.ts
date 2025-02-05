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
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Group<S>,
): Promise<SparseArray<D>> {
	const shape = elem.attrs.shape as number[];
	const format = (elem.attrs["encoding-type"] as string).slice(0, 3) as
		| "csc"
		| "csr";
	const indptr = (await zarr.open(elem.resolve("indptr"), {
		kind: "array",
	})) as zarr.Array<"int32", S>; // todo: allow 64
	const indices = (await zarr.open(elem.resolve("indices"), {
		kind: "array",
	})) as zarr.Array<"int32", S>; // todo: allow 64
	const data = (await zarr.open(elem.resolve("data"), {
		kind: "array",
	})) as zarr.Array<D, S>;
	return new SparseArray(indices, indptr, data, shape, format);
}

async function readCategorical_020<
	S extends Readable,
	K extends UIntType,
	D extends zarr.DataType,
>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Group<S>,
): Promise<LazyCategoricalArray<K, D, S>> {
	const cats = (await zarr.open(elem.resolve("categories"), {
		kind: "array",
	})) as zarr.Array<D, S>;
	const codes = (await zarr.open(elem.resolve("codes"), {
		kind: "array",
	})) as zarr.Array<K, S>;
	return new LazyCategoricalArray(codes, cats);
}

async function readCategorical_noVersion<
	S extends Readable,
	K extends UIntType,
	D extends zarr.DataType,
>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Array<K, S>,
): Promise<LazyCategoricalArray<K, D, S>> {
	const { categories } = elem.attrs;
	const cats = (await zarr.open(location.resolve(String(categories)), {
		kind: "array",
	})) as zarr.Array<D, S>;
	return new LazyCategoricalArray(elem, cats);
}

function readArray<S extends Readable, D extends zarr.DataType>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Array<D, S>,
): zarr.Array<D, S> {
	return elem;
}

function readDict_010<S extends Readable>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Group<S>,
): AxisArrays<S> {
	return new AxisArrays(location, key);
}

// biome-ignore lint/suspicious/noExplicitAny: signatures incompatible
const IO_FUNC_REGISTRY_WITH_VERSION: { [index: string]: any } = {
	"csr_matrix,0.1.0": readSparse_010,
	"csc_matrix,0.1.0": readSparse_010,
	"categorical,0.2.0": readCategorical_020,
	"array,0.2.0": readArray,
	"dict,0.1.0": readDict_010,
	"dataframe,0.1.0": readDict_010,
	"dataframe,0.2.0": readDict_010,
};

// biome-ignore lint/suspicious/noExplicitAny: signatures incompatible
const IO_FUNC_REGISTRY_WITHOUT_VERSION: { [index: string]: any } = {
	categorical: readCategorical_noVersion,
	array: readArray,
	dict: readDict_010,
};

export async function readZarr<
	S extends Readable,
	D extends zarr.NumberDataType,
>(path: string | S): Promise<AnnData<S, D>> {
	let root: zarr.Group<Readable>;
	if (typeof path === "string") {
		const store = await zarr.tryWithConsolidated(new zarr.FetchStore(path));
		root = await zarr.open(store, { kind: "group" });
	} else {
		root = await zarr.open(path, { kind: "group" });
	}

	const adataInit = {} as AxisKeyTypes<S, D>;
	await Promise.all(
		AxisKeys.map(async (k) => {
			if (k === "X" && (await has(root, k))) {
				adataInit[k as "X"] = await readElem(root, k) as SparseArray<D> | zarr.Array<D, S>;
			} else {
				adataInit[k as Exclude<AxisKey, "X">] = await readElem(root, k as Exclude<AxisKey, "X">) as AxisArrays<S>;
			}
		}),
	);
	return new AnnData(adataInit);
}

export async function readElem<
	S extends Readable,
	D extends zarr.DataType,
	DN extends zarr.NumberDataType,
	K extends UIntType,
>(location: zarr.Group<S>, key: Exclude<AxisKey, "X">): Promise<AxisArrays<S>>;
export async function readElem<
	S extends Readable,
	D extends zarr.DataType,
	DN extends zarr.NumberDataType,
	K extends UIntType,
>(
	location: zarr.Group<S>,
	key: "X",
): Promise<SparseArray<DN> | zarr.Array<DN, S>>;
export async function readElem<
	S extends Readable,
	D extends zarr.DataType,
	DN extends zarr.NumberDataType,
	K extends UIntType,
>(
	location: zarr.Group<S>,
	key: string,
): Promise<
	| SparseArray<DN>
	| LazyCategoricalArray<K, D, S>
	| zarr.Array<D, S>
>
export async function readElem<
	S extends Readable,
	D extends zarr.DataType,
	DN extends zarr.NumberDataType,
	K extends UIntType,
>(
	location: zarr.Group<S>,
	key: string,
): Promise<
	| SparseArray<DN>
	| LazyCategoricalArray<K, D, S>
	| zarr.Array<D, S>
	| AxisArrays<S>
> {
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
				keyNode as zarr.Array<K, S>,
			);
		}
		return IO_FUNC_REGISTRY_WITHOUT_VERSION.array(
			location,
			key,
			keyNode as zarr.Array<D, S>,
		);
	}
	return IO_FUNC_REGISTRY_WITH_VERSION[[encodingType, encodingVersion].join()](
		location,
		key,
		keyNode,
	);
}

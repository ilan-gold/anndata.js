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
	type IndexType,
} from "./types.js";
import { LazyCategoricalArray, has } from "./utils.js";

async function readSparse_010<S extends Readable>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Group<S> & Elem<S>,
): Promise<SparseArray<zarr.NumberDataType, IndexType, S>> {
	const shape = elem.attrs.shape as number[];
	const format = elem.attrs["encoding-type"].slice(0, 3) as "csc" | "csr";
	const indptr = (await zarr.open(elem.resolve("indptr"), {
		kind: "array",
	})) as zarr.Array<IndexType, S>;
	const indices = (await zarr.open(elem.resolve("indices"), {
		kind: "array",
	})) as zarr.Array<IndexType, S>;
	const data = (await zarr.open(elem.resolve("data"), {
		kind: "array",
	})) as zarr.Array<zarr.NumberDataType, S>;
	return new SparseArray(indices, indptr, data, shape, format);
}

async function readCategorical_020<S extends Readable>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Group<S>,
): Promise<LazyCategoricalArray<UIntType, zarr.DataType, S>> {
	const cats = (await zarr.open(elem.resolve("categories"), {
		kind: "array",
	})) as zarr.Array<zarr.DataType, S>;
	const codes = (await zarr.open(elem.resolve("codes"), {
		kind: "array",
	})) as zarr.Array<UIntType, S>;
	return new LazyCategoricalArray(codes, cats);
}

function readDict_010<S extends Readable>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Group<S>,
): AxisArrays<S> {
	return new AxisArrays(location, key);
}

// Read functions that work for unversioned elements

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

type ReadFunctionVersioned = <S extends Readable>(
	location: zarr.Group<S>,
	key: string,
	elem: zarr.Group<S> & Elem<S>,
) =>
	| AxisArrays<S>
	| Promise<LazyCategoricalArray<UIntType, zarr.DataType, S>>
	| Promise<SparseArray<zarr.NumberDataType, IndexType, S>>;

const IO_FUNC_REGISTRY_WITH_VERSION: {
	[index: string]: ReadFunctionVersioned | undefined;
} = {
	"csr_matrix,0.1.0": readSparse_010,
	"csc_matrix,0.1.0": readSparse_010,
	"categorical,0.2.0": readCategorical_020,
	"dict,0.1.0": readDict_010,
	"dataframe,0.1.0": readDict_010,
	"dataframe,0.2.0": readDict_010,
};

export async function readZarr<S extends Readable>(
	path: S,
): Promise<AnnData<S, zarr.NumberDataType, IndexType>> {
	const root = await zarr.open(path, { kind: "group" });
	const adataInit = {} as AxisKeyTypes<S, zarr.NumberDataType, IndexType>;
	await Promise.all(
		AxisKeys.map(async (k) => {
			if ((k === "X" && (await has(root, k))) || k !== "X") {
				adataInit[k] = await readElem(root, k);
			}
		}),
	);
	return new AnnData(adataInit);
}

export async function readElem<
	S extends Readable,
	K extends AxisKey,
	R extends K extends Exclude<AxisKey, "X">
		? AxisArrays<S>
		:
				| SparseArray<zarr.NumberDataType, IndexType, S>
				| zarr.Array<zarr.NumberDataType, S>,
>(location: zarr.Group<S>, key: K): Promise<R>;
export async function readElem<S extends Readable>(
	location: zarr.Group<S>,
	key: string,
): Promise<
	| SparseArray<zarr.NumberDataType, IndexType, S>
	| LazyCategoricalArray<UIntType, zarr.DataType, S>
	| zarr.Array<zarr.DataType, S>
>;
export async function readElem<S extends Readable>(
	location: zarr.Group<S>,
	key: string,
): Promise<
	| SparseArray<zarr.NumberDataType, IndexType, S>
	| LazyCategoricalArray<UIntType, zarr.DataType, S>
	| zarr.Array<zarr.DataType, S>
	| AxisArrays<S>
> {
	const keyRoot = location.resolve(key);
	const keyNode = await zarr.open(keyRoot);
	const { categories } = keyNode.attrs;
	if (!isElem<S>(keyNode)) {
		if (keyNode instanceof zarr.Group) {
			return readDict_010(location, key, keyNode);
		}
		if (categories !== undefined) {
			// Not encoded in old versions of anndata
			return readCategorical_noVersion(
				location,
				key,
				keyNode as zarr.Array<UIntType, S>,
			);
		}
	}
	// Whether or not the encoding metadata has been written, for now we read array as array.
	// TODO: add support for rec-array, which also fulfills this condition
	if (keyNode instanceof zarr.Array) {
		return readArray(location, key, keyNode);
	}
	const ioFuncId = [
		keyNode.attrs["encoding-type"],
		keyNode.attrs["encoding-version"],
	].join();
	const ioFunc = IO_FUNC_REGISTRY_WITH_VERSION[ioFuncId];
	if (ioFunc === undefined) {
		throw Error(`No io function found for ${ioFuncId}`);
	}
	return ioFunc(location, key, keyNode);
}

interface Elem<S extends Readable> extends zarr.Location<S> {
	get attrs(): zarr.Attributes & {
		"encoding-version": string;
		"encoding-type": string;
	};
}

function isElem<
	S extends Readable,
	D extends zarr.DataType = zarr.DataType,
	L extends zarr.Array<D, S> | zarr.Group<S> = zarr.Array<D, S> | zarr.Group<S>,
>(loc: L): loc is L & Elem<S> {
	return (
		loc.attrs["encoding-version"] !== undefined &&
		loc.attrs["encoding-type"] !== undefined
	);
}

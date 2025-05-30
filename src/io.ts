import type { Group, NumberDataType, Readable } from "zarrita";
import * as zarr from "zarrita";
import AnnData from "./anndata.js";
import AxisArrays from "./axis_arrays.js";
import SparseArray from "./sparse_array.js";
import {
	type AxisKey,
	AxisKeys,
	type AxisKeyTypes,
	type IndexType,
	type UIntType,
} from "./types.js";
import { has, LazyCategoricalArray } from "./utils.js";

async function readSparse_010<S extends Readable>(
	// biome-ignore lint: unused
	location: Group<S>,
	// biome-ignore lint: unused
	key: string,
	elem: Group<S> & Elem<S>,
): Promise<SparseArray<NumberDataType, IndexType, S>> {
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
	})) as zarr.Array<NumberDataType, S>;
	return new SparseArray(indices, indptr, data, shape, format);
}

async function readCategorical_020<S extends Readable>(
	// biome-ignore lint: unused
	location: Group<S>,
	// biome-ignore lint: unused
	key: string,
	elem: Group<S>,
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
	location: Group<S>,
	key: string,
	// biome-ignore lint: unused
	elem: Group<S>,
): AxisArrays<S> {
	return new AxisArrays(location, key);
}

// Read functions that work for unversioned elements

async function readCategorical_noVersion<
	S extends Readable,
	K extends UIntType,
	D extends zarr.DataType,
>(
	location: Group<S>,
	// biome-ignore lint: unused
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
	// biome-ignore lint: unused
	location: Group<S>,
	// biome-ignore lint: unused
	key: string,
	elem: zarr.Array<D, S>,
): zarr.Array<D, S> {
	return elem;
}

type ReadFunctionVersioned = <S extends Readable>(
	location: Group<S>,
	key: string,
	elem: Group<S> & Elem<S>,
) =>
	| AxisArrays<S>
	| Promise<LazyCategoricalArray<UIntType, zarr.DataType, S>>
	| Promise<SparseArray<NumberDataType, IndexType, S>>;

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
): Promise<AnnData<S, NumberDataType, IndexType>> {
	const root = await zarr.open(path, { kind: "group" });
	const adataInit = {} as AxisKeyTypes<S, NumberDataType, IndexType>;
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
		: SparseArray<NumberDataType, IndexType, S> | zarr.Array<NumberDataType, S>,
>(location: Group<S>, key: K): Promise<R>;
export async function readElem<S extends Readable>(
	location: Group<S>,
	key: string,
): Promise<
	| SparseArray<NumberDataType, IndexType, S>
	| LazyCategoricalArray<UIntType, zarr.DataType, S>
	| zarr.Array<zarr.DataType, S>
>;
export async function readElem<S extends Readable>(
	location: Group<S>,
	key: string,
): Promise<
	| SparseArray<NumberDataType, IndexType, S>
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
	L extends zarr.Array<D, S> | Group<S> = zarr.Array<D, S> | Group<S>,
>(loc: L): loc is L & Elem<S> {
	return (
		loc.attrs["encoding-version"] !== undefined &&
		loc.attrs["encoding-type"] !== undefined
	);
}

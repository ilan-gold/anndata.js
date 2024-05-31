import type { Readable } from "@zarrita/storage";
import {
	BoolArray,
	ByteStringArray,
	UnicodeStringArray,
} from "@zarrita/typedarray";
import * as zarr from "zarrita";
import SparseArray from "./sparse_array.js";
import {
	AxisSelection,
	type FullSelection,
	type Slice,
	type UIntType,
} from "./types.js";

const V2_STRING_REGEX = /v2:([US])(\d+)/;

export const CONSTRUCTORS = {
	int8: Int8Array,
	int16: Int16Array,
	int32: Int32Array,
	int64: globalThis.BigInt64Array,
	uint8: Uint8Array,
	uint16: Uint16Array,
	uint32: Uint32Array,
	uint64: globalThis.BigUint64Array,
	float32: Float32Array,
	float64: Float64Array,
	bool: BoolArray,
};

export function get_ctr<D extends zarr.DataType>(
	data_type: D,
): zarr.TypedArrayConstructor<D> {
	if (data_type === "v2:object") {
		return globalThis.Array as unknown as zarr.TypedArrayConstructor<D>;
	}
	const match = data_type.match(V2_STRING_REGEX);
	if (match) {
		const [, kind, chars] = match;
		// @ts-expect-error
		return (kind === "U" ? UnicodeStringArray : ByteStringArray).bind(
			null,
			Number(chars),
		);
	}
	// @ts-expect-error - We've checked that the key exists
	const ctr: zarr.TypedArrayConstructor<D> = CONSTRUCTORS[data_type];
	if (!ctr) {
		throw new Error(`Unknown or unsupported data_type: ${data_type}`);
	}
	return ctr;
}

export class LazyCategoricalArray<
	K extends UIntType,
	D extends zarr.DataType,
	S extends Readable,
> {
	public codes: zarr.Array<K, S>;

	public categories: zarr.Array<D, S>;

	constructor(codes: zarr.Array<K, S>, categories: zarr.Array<D, S>) {
		this.codes = codes;
		this.categories = categories;
	}
}

function isLazyCategoricalArray<
	K extends UIntType,
	D extends zarr.DataType,
	S extends Readable,
>(
	array: LazyCategoricalArray<K, D, S> | any,
): array is LazyCategoricalArray<K, D, S> {
	return (array as LazyCategoricalArray<K, D, S>).categories !== undefined;
}

function isSparseArray<N extends zarr.NumberDataType>(
	array: any,
): array is SparseArray<N> {
	return (array as SparseArray<N>).indptr !== undefined;
}

function isZarrBoolTypedArrayFromDtype(
	data: any,
	dtype: zarr.DataType,
): data is BoolArray {
	return data?.get !== undefined && dtype === "bool";
}

function isZarrStringTypedArrayFromDtype(
	data: any,
	dtype: zarr.DataType,
): data is ByteStringArray | UnicodeStringArray {
	return data?.get !== undefined && dtype !== "bool";
}

function isTypedArrayFromDtype(
	data: any,
	dtype: zarr.DataType,
): data is
	| Int8Array
	| Int16Array
	| Int32Array
	| BigInt64Array
	| Uint8Array
	| Uint16Array
	| Uint32Array
	| BigUint64Array
	| Float32Array
	| Float64Array
	| Array<unknown> {
	return (
		!isZarrStringTypedArrayFromDtype(data, dtype) &&
		!isZarrBoolTypedArrayFromDtype(data, dtype)
	); // ok, probably not a great idea
}

function isNumericArray(data: any): data is number[] {
	return data.every((i: any) => typeof i === "number");
}

function isChunk<D extends zarr.DataType>(
	chunkOrScalar: any,
): chunkOrScalar is zarr.Chunk<D> {
	return chunkOrScalar instanceof Object && "shape" in chunkOrScalar;
}

function unwrapIf1d<D extends zarr.DataType>(
	chunkOrScalar: zarr.Chunk<D> | zarr.Scalar<D>,
): zarr.Scalar<D> | zarr.Chunk<D> {
	if (!isChunk(chunkOrScalar)) {
		return chunkOrScalar;
	}
	if (chunkOrScalar.shape.length === 1 && chunkOrScalar.shape[0] === 1) {
		return (
			"get" in chunkOrScalar.data
				? chunkOrScalar.data.get(0)
				: chunkOrScalar.data[0]
		) as zarr.Scalar<D>;
	}
	return chunkOrScalar as zarr.Chunk<D>;
}

export async function get<
	L extends zarr.DataType,
	N extends zarr.NumberDataType,
	K extends UIntType,
	S extends Readable,
	Arr extends LazyCategoricalArray<K, L, S> | zarr.Array<L, S> | SparseArray<N>,
>(
	array: Arr,
	selection: (null | Slice | number)[],
): Promise<
	zarr.Chunk<
		Arr extends zarr.Array<L, S> | LazyCategoricalArray<K, L, S> ? L : N
	>
>;
export async function get<
	L extends zarr.DataType,
	N extends zarr.NumberDataType,
	K extends UIntType,
	S extends Readable,
	Arr extends LazyCategoricalArray<K, L, S> | zarr.Array<L, S> | SparseArray<N>,
>(
	array: Arr,
	selection: number[],
): Promise<
	zarr.Scalar<
		Arr extends zarr.Array<L, S> | LazyCategoricalArray<K, L, S> ? L : N
	>
>;
export async function get<
	L extends zarr.DataType,
	N extends zarr.NumberDataType,
	K extends UIntType,
	S extends Readable,
	Arr extends LazyCategoricalArray<K, L, S> | zarr.Array<L, S> | SparseArray<N>,
>(array: Arr, selection: FullSelection) {
	if (isLazyCategoricalArray(array)) {
		const codes = await zarr.get(array.codes, selection);
		const categories = await zarr.get(array.categories, [null]);
		const { data: categoriesData } = categories;
		const dtype = array.categories.dtype;
		if (isNumericArray(selection)) {
			const category = Number(codes);
			if (isTypedArrayFromDtype(categoriesData, dtype)) {
				return categoriesData[category];
			}
			if (isZarrStringTypedArrayFromDtype(categoriesData, dtype)) {
				return categoriesData.get(category);
			}
			if (isZarrBoolTypedArrayFromDtype(categoriesData, dtype)) {
				return categoriesData.get(category);
			}
			throw new Error("Unrecognized category");
		}
		const data = new (get_ctr(array.categories.dtype))(codes.data.length); // TODO(ilan-gold): open issue in zarrita
		for (let i = 0; i < codes.data.length; i += 1) {
			const code = codes.data[i];
			const category = Number(code);
			// TODO(ilan-gold): a better way of setting data maybe? more type agnostic?
			if (
				isTypedArrayFromDtype(categoriesData, dtype) &&
				isTypedArrayFromDtype(data, dtype)
			) {
				data[i] = categoriesData[category];
			} else if (
				isZarrStringTypedArrayFromDtype(categoriesData, dtype) &&
				isZarrStringTypedArrayFromDtype(data, dtype)
			) {
				data.set(i, String(categoriesData.get(category)));
			} else if (
				isZarrBoolTypedArrayFromDtype(categoriesData, dtype) &&
				isZarrBoolTypedArrayFromDtype(data, dtype)
			) {
				data.set(i, Boolean(categoriesData.get(category)));
			}
		}
		return { ...codes, data };
	}
	if (isSparseArray(array)) {
		return unwrapIf1d(await array.get(selection));
	}
	return unwrapIf1d(await zarr.get(array, selection));
}

export async function has(root: zarr.Group<Readable>, path: string) {
	try {
		await zarr.open(root.resolve(path));
	} catch (error) {
		if (error instanceof zarr.NodeNotFoundError) {
			return false;
		}
	}
	return true;
}

export async function readSparse<
	S extends Readable,
	D extends zarr.NumberDataType,
>(elem: zarr.Group<S>): Promise<SparseArray<D>> {
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

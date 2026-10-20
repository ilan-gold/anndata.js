import type {
	Chunk,
	DataType,
	Group,
	NumberDataType,
	Readable,
	Scalar,
} from "zarrita";
import * as zarr from "zarrita";
import type SparseArray from "./sparse_array.js";
import type {
	CodeType,
	FullSelection,
	IndexType,
	NaValue,
	NullableChunk,
	Slice,
} from "./types.js";

const V2_STRING_REGEX = /v2:([US])(\d+)/;

export const CONSTRUCTORS = {
	int8: Int8Array,
	int16: Int16Array,
	int32: Int32Array,
	int64: BigInt64Array,
	uint8: Uint8Array,
	uint16: Uint16Array,
	uint32: Uint32Array,
	uint64: BigUint64Array,
	float32: Float32Array,
	float64: Float64Array,
	bool: zarr.BoolArray,
};

export function get_ctr<D extends DataType>(
	data_type: D,
): zarr.TypedArrayConstructor<D> {
	// Variable-length strings: "v2:object" (zarr v2 with the vlen-utf8 filter)
	// and "string" (zarr v3) are both decoded to plain arrays.
	if (data_type === "v2:object" || data_type === "string") {
		return Array as unknown as zarr.TypedArrayConstructor<D>;
	}
	const match = data_type.match(V2_STRING_REGEX);
	if (match) {
		const [, kind, chars] = match;
		// @ts-expect-error
		return (kind === "U" ? zarr.UnicodeStringArray : zarr.ByteStringArray).bind(
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

/** @internal */
export class LazyCategoricalArray<
	K extends CodeType,
	D extends DataType,
	S extends Readable,
> {
	public codes: zarr.Array<K, S>;

	public categories: zarr.Array<D, S>;

	constructor(codes: zarr.Array<K, S>, categories: zarr.Array<D, S>) {
		this.codes = codes;
		this.categories = categories;
	}
}

/**
 * A nullable-integer, nullable-boolean, or nullable-string-array element:
 * an array of values alongside a boolean mask that is true where a value is missing.
 * @internal
 */
export class LazyNullableArray<D extends DataType, S extends Readable> {
	public values: zarr.Array<D, S>;

	public mask: zarr.Array<"bool", S>;

	public naValue: NaValue;

	constructor(
		values: zarr.Array<D, S>,
		mask: zarr.Array<"bool", S>,
		naValue: NaValue = "NA",
	) {
		this.values = values;
		this.mask = mask;
		this.naValue = naValue;
	}
}

function isLazyCategoricalArray<
	L extends DataType,
	N extends NumberDataType,
	K extends CodeType,
	S extends Readable,
>(
	array:
		| LazyCategoricalArray<K, L, S>
		| LazyNullableArray<L, S>
		| zarr.Array<L, S>
		| SparseArray<N, IndexType, S>,
): array is LazyCategoricalArray<K, L, S> {
	return "categories" in array;
}

function isLazyNullableArray<
	L extends DataType,
	N extends NumberDataType,
	K extends CodeType,
	S extends Readable,
>(
	array:
		| LazyCategoricalArray<K, L, S>
		| LazyNullableArray<L, S>
		| zarr.Array<L, S>
		| SparseArray<N, IndexType, S>,
): array is LazyNullableArray<L, S> {
	return "mask" in array;
}

function isSparseArray<
	L extends DataType,
	N extends NumberDataType,
	I extends IndexType,
	K extends CodeType,
	S extends Readable,
>(
	array:
		| LazyCategoricalArray<K, L, S>
		| LazyNullableArray<L, S>
		| zarr.Array<L, S>
		| SparseArray<N, I, S>,
): array is SparseArray<N, I, S> {
	return "indptr" in array;
}

function isZarrBoolTypedArrayFromDtype(
	data: Iterable<unknown>,
	dtype: DataType,
): data is zarr.BoolArray {
	return "get" in data && dtype === "bool";
}

function isZarrStringTypedArrayFromDtype(
	data: Iterable<unknown>,
	dtype: DataType,
): data is zarr.ByteStringArray | zarr.UnicodeStringArray {
	return "get" in data && dtype !== "bool";
}

function isTypedArrayFromDtype(
	data: Iterable<unknown>,
	dtype: DataType,
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

function isNumericArray(data: unknown[]): data is number[] {
	return data.every((i) => typeof i === "number");
}

function isChunk<D extends DataType>(
	chunkOrScalar: Chunk<D> | Scalar<D>,
): chunkOrScalar is Chunk<D> {
	return chunkOrScalar instanceof Object && "shape" in chunkOrScalar;
}

function getItem<D extends DataType>(
	data: zarr.TypedArray<D>,
	index: number,
): Scalar<D> {
	return ("get" in data ? data.get(index) : data[index]) as Scalar<D>;
}

function unwrapIf0d<D extends DataType>(
	chunkOrScalar: Chunk<D> | Scalar<D>,
): Scalar<D> | Chunk<D> {
	if (!isChunk(chunkOrScalar)) {
		return chunkOrScalar;
	}
	if (chunkOrScalar.shape.length === 1 && chunkOrScalar.shape[0] === 1) {
		return getItem(chunkOrScalar.data, 0);
	}
	return chunkOrScalar as Chunk<D>;
}

export async function get<
	L extends DataType,
	N extends NumberDataType,
	K extends CodeType,
	S extends Readable,
	Arr extends
		| LazyCategoricalArray<K, L, S>
		| LazyNullableArray<L, S>
		| zarr.Array<L, S>
		| SparseArray<NumberDataType, IndexType, S>,
>(
	array: Arr,
	selection: (null | Slice | number)[],
): Promise<
	Arr extends LazyNullableArray<L, S>
		? NullableChunk<L>
		: Arr extends LazyCategoricalArray<K, L, S>
			? Chunk<L> | NullableChunk<L>
			: Chunk<Arr extends zarr.Array<L, S> ? L : N>
>;
export async function get<
	L extends DataType,
	N extends NumberDataType,
	K extends CodeType,
	S extends Readable,
	Arr extends
		| LazyCategoricalArray<K, L, S>
		| LazyNullableArray<L, S>
		| zarr.Array<L, S>
		| SparseArray<N, IndexType, S>,
>(
	array: Arr,
	selection: number[],
): Promise<
	Arr extends LazyNullableArray<L, S> | LazyCategoricalArray<K, L, S>
		? Scalar<L> | null
		: Scalar<Arr extends zarr.Array<L, S> ? L : N>
>;
export async function get<
	L extends DataType,
	N extends NumberDataType,
	K extends CodeType,
	S extends Readable,
	Arr extends
		| LazyCategoricalArray<K, L, S>
		| LazyNullableArray<L, S>
		| zarr.Array<L, S>
		| SparseArray<N, IndexType, S>,
>(array: Arr, selection: FullSelection) {
	if (isLazyNullableArray(array)) {
		const [values, mask] = await Promise.all([
			zarr.get(array.values, selection),
			zarr.get(array.mask, selection),
		]);
		if (!isChunk(values) || !isChunk(mask)) {
			return mask ? null : values;
		}
		const data: NullableChunk<L>["data"] = Array.from(
			{ length: values.data.length },
			(_, i) => (getItem(mask.data, i) ? null : getItem(values.data, i)),
		);
		return { data, shape: values.shape, stride: values.stride };
	}
	if (isLazyCategoricalArray(array)) {
		const codes = await zarr.get(array.codes, selection);
		const categories = await zarr.get(array.categories, [null]);
		const { data: categoriesData } = categories;
		const dtype = array.categories.dtype;
		if (isNumericArray(selection)) {
			const category = Number(codes);
			if (category < 0) {
				return null;
			}
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
		const codesData = Array.from(codes.data, Number);
		if (codesData.some((code) => code < 0)) {
			// A negative code denotes a missing value, which typed arrays cannot hold.
			const data: NullableChunk<L>["data"] = codesData.map((code) =>
				code < 0 ? null : getItem(categoriesData, code),
			);
			return { data, shape: codes.shape, stride: codes.stride };
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
		return unwrapIf0d(await array.get(selection));
	}
	return unwrapIf0d(await zarr.get(array, selection));
}

export async function has(root: Group<Readable>, path: string) {
	try {
		await zarr.open(root.resolve(path));
	} catch (error) {
		if (zarr.isZarritaError(error, "NotFoundError")) {
			return false;
		}
	}
	return true;
}

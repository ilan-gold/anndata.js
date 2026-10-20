import type * as zarr from "zarrita";
import type { Readable } from "zarrita";
import type AxisArrays from "./axis_arrays.js";
import type SparseArray from "./sparse_array.js";
import type { LazyCategoricalArray, LazyNullableArray } from "./utils.js";

export const AxisKeys = [
	"obs",
	"var",
	"obsm",
	"varm",
	"X",
	"layers",
	"obsp",
	"varp",
] as const;
export type AxisKey = (typeof AxisKeys)[number];
export type Slice = ReturnType<typeof zarr.slice>;
export type AxisSelection = number | Slice | null;
export type FullSelection = AxisSelection[];
export type UIntType = zarr.Uint8 | zarr.Uint16 | zarr.Uint32;
// Categorical codes may be signed, in which case -1 denotes a missing value.
export type CodeType = UIntType | zarr.Int8 | zarr.Int16 | zarr.Int32;
export type IndexType = zarr.Uint32; // TODO: | zarr.Uint64;
export type ArrayType = Exclude<zarr.DataType, zarr.ObjectType>;
// Missing value semantics of nullable arrays.
// Reference: https://anndata.readthedocs.io/en/latest/fileformat-prose.html#missing-value-semantics
export type NaValue = "NA" | "NaN";
// Like a zarr.Chunk, but with null in place of each missing value.
export type NullableChunk<D extends zarr.DataType> = {
	data: (zarr.Scalar<D> | null)[];
	shape: number[];
	stride: number[];
};
export type BackedArray<S extends Readable> =
	| zarr.Array<zarr.DataType, S>
	| SparseArray<zarr.NumberDataType, IndexType, S>
	| LazyCategoricalArray<CodeType, zarr.DataType, S>
	| LazyNullableArray<zarr.DataType, S>
	| AxisArrays<Readable>;

export interface AxisKeyTypes<
	S extends Readable,
	D extends zarr.NumberDataType,
	I extends IndexType,
> {
	obs: AxisArrays<S>;
	var: AxisArrays<S>;
	obsm: AxisArrays<S>;
	varm: AxisArrays<S>;
	X: SparseArray<D, I, S> | zarr.Array<D, S> | undefined;
	layers: AxisArrays<S>;
	obsp: AxisArrays<S>;
	varp: AxisArrays<S>;
}

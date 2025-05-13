import type * as zarr from "zarrita";
import type { Readable } from "zarrita";
import type AxisArrays from "./axis_arrays.js";
import type SparseArray from "./sparse_array.js";
import type { LazyCategoricalArray } from "./utils.js";

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
export type IndexType = zarr.Uint32; // TODO: | zarr.Uint64;
export type ArrayType = Exclude<zarr.DataType, zarr.ObjectType>;
export type BackedArray<S extends Readable> =
	| zarr.Array<zarr.DataType, S>
	| SparseArray<zarr.NumberDataType, IndexType, S>
	| LazyCategoricalArray<UIntType, zarr.DataType, S>
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

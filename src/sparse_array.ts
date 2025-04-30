import type { NumberDataType, Readable, TypedArray } from "zarrita";
import * as zarr from "zarrita";
import type {
	AxisSelection,
	FullSelection,
	IndexType,
	Slice,
} from "./types.js";
import { CONSTRUCTORS } from "./utils.js";

class IndexingError {
	public message: string;
	constructor(message: string) {
		this.message = message;
	}
}

function isSlice(s: AxisSelection): s is Slice {
	return (s as Slice)?.stop !== undefined || (s as Slice)?.start !== undefined;
}

// TODO: Make this and other data types more restricitve but how?
class SparseArray<
	D extends NumberDataType,
	I extends IndexType,
	S extends Readable,
> {
	constructor(
		public readonly indices: zarr.Array<I, S>,
		public readonly indptr: zarr.Array<I, S>,
		public readonly data: zarr.Array<D, S>,
		public readonly shape: number[],
		public readonly format: "csc" | "csr",
	) {}

	async get(selection: FullSelection): Promise<zarr.Chunk<D>> {
		if (selection.length !== 2) {
			throw new IndexingError(
				"For sparse array, selection must be of length 2",
			);
		}
		const minorAxisSelection = selection[this.minorAxis];
		const majorAxisSelection = selection[this.majorAxis];
		const arr = await this.getContiguous(majorAxisSelection);
		const finalSelection = new Array(2);
		finalSelection[this.majorAxis] = null;
		finalSelection[this.minorAxis] = minorAxisSelection;
		const res = await zarr.get(arr, finalSelection);
		return res;
	}

	public get majorAxis(): number {
		return ["csr", "csc"].indexOf(this.format);
	}

	public get minorAxis(): number {
		return ["csc", "csr"].indexOf(this.format);
	}

	async getContiguous(s: AxisSelection): Promise<zarr.Array<D, Readable>> {
		// Resolve (major-axis) selection of indptr
		let sliceStart = 0;
		let sliceEnd = this.shape[this.majorAxis] + 1;
		if (isSlice(s)) {
			if (s.start) {
				sliceStart = s.start;
			}
			if (s.stop) {
				sliceEnd = s.stop + 1;
			}
		} else if (typeof s === "number") {
			sliceStart = s;
			sliceEnd = s + 2;
		}
		const majorAxisSize = sliceEnd - sliceStart - 1;
		const shape: number[] = new Array(2).fill(this.shape[this.minorAxis]);
		shape[this.majorAxis] = majorAxisSize;

		// Get start and stop of the data/indices based on major-axis selection
		let indptr: TypedArray<I> | Uint32Array = (
			await zarr.get(this.indptr, [zarr.slice(sliceStart, sliceEnd)])
		).data;
		const start = indptr[0];
		const stop = indptr[indptr.length - 1];
		indptr = indptr.map((i) => i - start);

		// Create data to be returned
		const isDataAllZeros = start === stop;
		const dense = (new CONSTRUCTORS[(this.data.dtype.toString() as keyof typeof CONSTRUCTORS)](
			shape.reduce((a, b) => a * b, 1),
		) as TypedArray<D>).fill(0);
		const arr = await zarr.create(new Map(), {
			shape,
			chunk_shape: shape,
			data_type: this.data.dtype,
		});

		// Return empty or fill
		if (isDataAllZeros) {
			return arr;
		}

		// Get data/indices and create dense return object
		const { data: indices } = await zarr.get(this.indices, [
			zarr.slice(start, stop),
		]);
		const { data } = await zarr.get(this.data, [zarr.slice(start, stop)]);

		const stride = new Array(2).fill(1);
		stride[this.majorAxis] = shape[this.minorAxis];
		const chunk = {
			data: this.densify(indices, indptr, data, dense, shape),
			shape,
			stride,
		} as zarr.Chunk<D>;
		await zarr.set(arr, null, chunk);
		return arr;
	}

	densify(
		indices: TypedArray<NumberDataType>,
		indptr: TypedArray<NumberDataType>,
		data: TypedArray<NumberDataType>,
		dense: TypedArray<NumberDataType>,
		shape: number[],
	) {
		const minorAxisLength = shape[this.minorAxis];
		for (let majorIdx = 0; majorIdx < indptr.length; majorIdx += 1) {
			const indptrStart = indptr[majorIdx];
			const indptrStop = indptr[majorIdx + 1];
			for (
				let indicesOrDataIndex = indptrStart;
				indicesOrDataIndex < indptrStop;
				indicesOrDataIndex += 1
			) {
				const minorIdx = indices[indicesOrDataIndex];
				dense[majorIdx * minorAxisLength + minorIdx] = data[indicesOrDataIndex];
			}
		}
		return dense;
	}
}

export default SparseArray;

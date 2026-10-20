import type * as zarr from "zarrita";
import type { NumberDataType, Readable } from "zarrita";
import type AxisArrays from "./axis_arrays.js";
import { readElem } from "./io.js";
import type SparseArray from "./sparse_array.js";
import type { AxisKeyTypes, IndexType } from "./types.js";

export default class AnnData<
	S extends Readable,
	D extends NumberDataType,
	I extends IndexType,
> {
	public obs: AxisArrays<S>;
	public var: AxisArrays<S>;
	public obsm: AxisArrays<S>;
	public obsp: AxisArrays<S>;
	public varm: AxisArrays<S>;
	public varp: AxisArrays<S>;
	public X: SparseArray<D, I, S> | zarr.Array<D> | undefined;
	public layers: AxisArrays<S>;

	constructor(data: AxisKeyTypes<S, D, I>) {
		this.obs = data.obs;
		this.var = data.var;
		this.obsm = data.obsm;
		this.obsp = data.obsp;
		this.varm = data.varm;
		this.varp = data.varp;
		this.X = data.X;
		this.layers = data.layers;
	}

	private async names(grp: zarr.Group<S>) {
		// The index may be an array or, e.g., a nullable-string-array group.
		return readElem(grp, String(grp.attrs._index || "_index"));
	}

	public async obsNames() {
		const grp = await this.obs.axisRoot();
		return this.names(grp);
	}

	public async varNames() {
		const grp = await this.var.axisRoot();
		return this.names(grp);
	}
}

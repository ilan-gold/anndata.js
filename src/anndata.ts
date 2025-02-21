import type { Readable } from "@zarrita/storage";
import * as zarr from "zarrita";
import type AxisArrays from "./axis_arrays.js";
import type SparseArray from "./sparse_array.js";
import type { AxisKeyTypes } from "./types.js";
export default class AnnData<
	S extends Readable,
	D extends zarr.NumberDataType,
> {
	public obs: AxisArrays<S>;
	public var: AxisArrays<S>;
	public obsm: AxisArrays<S>;
	public obsp: AxisArrays<S>;
	public varm: AxisArrays<S>;
	public varp: AxisArrays<S>;
	public X: SparseArray<D> | zarr.Array<D> | undefined;
	public layers: AxisArrays<S>;

	constructor(data: AxisKeyTypes<S, D>) {
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
		return zarr.open(grp.resolve(String(grp.attrs._index || "_index")), {
			kind: "array",
		});
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

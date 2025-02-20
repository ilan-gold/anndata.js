import type { Readable } from "@zarrita/storage";
import * as zarr from "zarrita";
import type AxisArrays from "./axis_arrays.js";
import type { AxisKeyTypes, BackedArray } from "./types.js";

export const X = Symbol("X");

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
	public layers: AxisArrays<S, string | typeof X>;

	constructor(data: AxisKeyTypes<S, D>) {
		this.obs = data.obs;
		this.var = data.var;
		this.obsm = data.obsm;
		this.obsp = data.obsp;
		this.varm = data.varm;
		this.varp = data.varp;
		this.layers = data.layers;
	}

	public get X(): Promise<BackedArray | undefined> {
		if (!this.layers.has(X)) {
			return Promise.resolve(undefined);
		}
		return this.layers.get(X);
	}

	private async names(grp: zarr.Group<S>) {
		return zarr.open(grp.resolve(String(grp.attrs._index || "_index")), {
			kind: "array",
		});
	}

	public async obsNames() {
		const grp = await zarr.open(this.obs.axisRoot, { kind: "group" });
		return this.names(grp);
	}

	public async varNames() {
		const grp = await zarr.open(this.var.axisRoot, { kind: "group" });
		return this.names(grp);
	}
}

import * as zarr from "zarrita";
import { LazyCategoricalArray, has, readSparse } from "./utils.js";

import type { Readable } from "@zarrita/storage";
import type { AxisKey, BackedArray, UIntType } from "./types.js";

export default class AxisArrays<
	S extends Readable,
	K extends string | symbol = string,
> {
	public parentRoot: zarr.Group<S>;
	public name: Exclude<AxisKey, "X">;
	private cache: Map<K, BackedArray>;

	public constructor(
		parentRoot: zarr.Group<S>,
		axisKey: Exclude<AxisKey, "X">,
	) {
		this.name = axisKey;
		this.parentRoot = parentRoot;
		this.cache = new Map();
	}

	public get axisRoot(): zarr.Location<S> {
		return this.parentRoot.resolve(this.name);
	}

	public async get(key: K): Promise<BackedArray> {
		if (!(await this.has(key))) {
			throw new Error(`${this.name} has no key: \"${String(key)}\"`);
		}
		if (!this.cache.has(key)) {
			// categories needed for backward compat
			const keyRoot = this.axisRoot.resolve(key);
			const keyNode = await zarr.open(keyRoot);
			const { categories, "encoding-type": encodingType } = keyNode.attrs;
			if (categories !== undefined) {
				const cats = await zarr.open(
					this.axisRoot.resolve(String(categories)),
					{
						kind: "array",
					},
				);
				this.cache.set(
					key,
					new LazyCategoricalArray(keyNode as zarr.Array<UIntType, S>, cats),
				);
			} else if (encodingType === "categorical") {
				const cats = await zarr.open(keyRoot.resolve("categories"), {
					kind: "array",
				});
				const codes = (await zarr.open(keyRoot.resolve("codes"), {
					kind: "array",
				})) as zarr.Array<UIntType, Readable>;
				this.cache.set(key, new LazyCategoricalArray(codes, cats));
			} else if (
				encodingType !== undefined &&
				["csc_matrix", "csr_matrix"].includes(String(encodingType))
			) {
				this.cache.set(key, await readSparse(keyNode as zarr.Group<Readable>));
			} else {
				this.cache.set(key, keyNode as zarr.Array<zarr.DataType, S>);
			}
		}
		const val = this.cache.get(key);
		if (val === undefined) {
			throw new Error(
				"See https://github.com/microsoft/TypeScript/issues/13086 for why this will never happen",
			);
		}
		return val;
	}

	public async has(key: K): Promise<boolean> {
		if (this.cache.has(key)) {
			return true;
		}
		if (typeof key !== "string") {
			return false;
		}
		return has(this.parentRoot, `${this.name}/${key}`);
	}
}

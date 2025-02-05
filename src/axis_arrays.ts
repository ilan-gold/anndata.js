import type * as zarr from "zarrita";
import { LazyCategoricalArray, has } from "./utils.js";

import type { Readable } from "@zarrita/storage";
import { readElem } from "./io.js";
import type { AxisKey, BackedArray, UIntType } from "./types.js";

export default class AxisArrays<S extends Readable> {
	public parentRoot: zarr.Group<S>;
	public name: Exclude<AxisKey, "X">;
	private cache: Map<string, BackedArray>;

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

	public async get(key: string): Promise<BackedArray> {
		if (!(await this.has(key))) {
			throw new Error(`${this.name} has no key: \"${key}\"`);
		}
		if (!this.cache.has(key)) {
			// categories needed for backward compat
			this.cache.set(key, await readElem(this.axisRoot, key));
		}
		const val = this.cache.get(key);
		if (val === undefined) {
			throw new Error(
				"See https://github.com/microsoft/TypeScript/issues/13086 for why this will never happen",
			);
		}
		return val;
	}

	public async has(key: string): Promise<boolean> {
		return has(this.parentRoot, `${this.name}/${key}`);
	}
}

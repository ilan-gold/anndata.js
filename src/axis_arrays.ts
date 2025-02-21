import * as zarr from "zarrita";
import { has } from "./utils.js";

import type { Readable } from "@zarrita/storage";
import { readElem } from "./io.js";
import type { BackedArray } from "./types.js";

export default class AxisArrays<S extends Readable> {
	public parent: zarr.Group<S>;
	public name: string;
	private cache: Map<string, BackedArray>;

	public constructor(parent: zarr.Group<S>, name: string) {
		this.name = name;
		this.parent = parent;
		this.cache = new Map();
	}

	public get axisRoot(): zarr.Location<S> {
		return this.parent.resolve(this.name);
	}

	public async get(key: string): Promise<BackedArray> {
		if (!(await this.has(key))) {
			throw new Error(`${this.name} has no key: \"${key}\"`);
		}
		if (!this.cache.has(key)) {
			this.cache.set(key, await readElem(await zarr.open(this.axisRoot, { "kind": "group" }), key));
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
		return has(this.parent, `${this.name}/${key}`);
	}
}

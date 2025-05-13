import type { Group, Readable } from "zarrita";
import * as zarr from "zarrita";
import { readElem } from "./io.js";
import type { BackedArray } from "./types.js";
import { has } from "./utils.js";

export default class AxisArrays<S extends Readable> {
	public parent: Group<S>;
	public name: string;
	private cache: Map<string, BackedArray<S>>;

	public constructor(parent: Group<S>, name: string) {
		this.name = name;
		this.parent = parent;
		this.cache = new Map();
	}

	public async axisRoot(): Promise<Group<S>> {
		return await zarr.open(this.parent.resolve(this.name), { kind: "group" });
	}

	public async get(key: string): Promise<BackedArray<S>> {
		if (!(await this.has(key))) {
			throw new Error(`${this.name} has no key: \"${key}\"`);
		}
		if (!this.cache.has(key)) {
			this.cache.set(key, await readElem(await this.axisRoot(), key));
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

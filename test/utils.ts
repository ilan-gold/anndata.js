import * as zarr from "zarrita";
import { ByteStringArray } from "zarrita";
import AnnData from "../src/anndata.js";
import AxisArrays from "../src/axis_arrays.js";
import SparseArray from "../src/sparse_array.js";
import { type AxisKey, AxisKeys, type AxisKeyTypes } from "../src/types.js";
import { LazyCategoricalArray } from "../src/utils.js";

export function makeDiagonalWithMissingRow(M: number, N: number) {
	const data = new Int32Array(M * N);
	for (let i = 0; i < M; i++) {
		if (Math.min(Math.floor(M / 2), Math.floor(N / 2)) !== i && i < N) {
			data[i + i * N] = i;
		}
	}
	return data;
}

export function stringArrayFromPrefixAndSize(
	name: string,
	size: number,
	mod?: number,
) {
	return Array.from(Array(size).keys()).map(
		(i) => `${name}_${mod ? i % mod : i}`,
	);
}

function base64Decode(encoded: string) {
	// We do not want to use Buffer.from(encoded, 'base64') because
	// Buffer is not available in the browser and we do not want
	// to add a dependency on a polyfill if we dont have to.
	// Reference: https://stackoverflow.com/a/41106346
	return Uint8Array.from(atob(encoded), (c) => c.charCodeAt(0));
}

// This is intended to be used for unit testing purposes.
// It goes along with scripts/directory-to-memory-store.mjs
export function createStoreFromMapContents(
	mapContents: Iterable<readonly [string, string]>,
) {
	const map = new Map(mapContents);
	return new Proxy(map, {
		get: (target, prop) => {
			if (prop === "get") {
				// Replace the get method with one that decodes the value.
				return (key: string) => {
					const encodedVal = target.get(key);
					if (encodedVal) {
						return base64Decode(encodedVal);
					}
					return undefined;
				};
			}
			return Reflect.get(target, prop);
		},
	});
}

type SparseMatrix = {
	indptr: Int32Array;
	indices: Int32Array;
	data: Float32Array;
};

function generateDiagonalSparseMatrix(
	majorSize: number,
	minorSize: number,
): SparseMatrix {
	const indptr: number[] = [0];
	const indices: number[] = [];
	const data: number[] = [];
	const middleIndex = Math.floor(majorSize / 2);

	for (let i = 0; i < majorSize; i++) {
		if (i === middleIndex) {
			// Skip the middle row/column
			indptr.push(indices.length);
			continue;
		}

		if (i < minorSize) {
			indices.push(i);
			data.push(i); // Row number as the entry
		}

		indptr.push(indices.length);
	}

	return {
		indices: new Int32Array(indices),
		indptr: new Int32Array(indptr),
		data: new Float32Array(data),
	};
}

function getBufferElementType(buffer: ArrayBufferView): zarr.NumberDataType {
	const name = buffer.constructor.name;
	switch (name) {
		case "Int8Array":
			return "int8";
		case "Uint8Array":
			return "uint8";
		case "Int16Array":
			return "int16";
		case "Uint16Array":
			return "uint16";
		case "Int32Array":
			return "int32";
		case "Uint32Array":
			return "uint32";
		case "Float32Array":
			return "float32";
	}

	throw new Error(`Unsupported buffer type ${name}`);
}

// Both functions generate matrices whose diagonal is the row number, missing the middle row (to ensure that we are testing the axis skipping ability of sparse matrices)
async function genSparse(
	grp: zarr.Group<Map<string, Uint8Array>>,
	shape: number[],
	type: "csc" | "csr",
) {
	const majorAxisSize = shape[Number(type === "csc")];
	const minorAxisSize = shape[(Number(type === "csc") - 1) % 1];
	const sparseMatrixArgs = generateDiagonalSparseMatrix(
		majorAxisSize,
		minorAxisSize,
	);
	const sparseArrays = await Promise.all(
		Object.entries(sparseMatrixArgs).map(async ([name, array]) => {
			const dtype = getBufferElementType(array);
			const zarrArray = await zarr.create(grp.resolve(name), {
				shape,
				chunk_shape: shape,
				data_type: dtype,
			});
			const chunk = {
				data: array,
				shape: [array.length],
				stride: [1],
			} as zarr.Chunk<typeof dtype>;
			await zarr.set(zarrArray, null, chunk);
			return zarrArray;
		}),
	);
	return new SparseArray(
		sparseArrays[0],
		sparseArrays[1],
		sparseArrays[2],
		shape,
		type,
	);
}

async function genDense(
	grp: zarr.Group<Map<string, Uint8Array>>,
	shape: number[],
) {
	const arr = await zarr.create(grp, {
		shape,
		chunk_shape: shape,
		data_type: "float32",
	});
	const data = new Float32Array(shape.reduce((a, b) => a * b, 1));
	const middleRow = Math.floor(shape[0] / 2);

	for (let i = 0; i < shape[0]; i++) {
		if (i === middleRow) {
			continue; // Skip the middle row
		}
		if (i < shape[1]) {
			data[i * shape[1] + i] = i; // Set the diagonal entry to the row number
		}
	}

	const chunk = {
		data,
		shape,
		stride: [shape[0], 1],
	} as zarr.Chunk<"float32">;
	await zarr.set(arr, null, chunk);
	return arr;
}

async function genCategorical(
	grp: zarr.Group<Map<string, Uint8Array>>,
	length: number,
	numCategories: number,
) {
	const codes = new Uint32Array(length);
	for (let i = 0; i < length; i++) {
		codes[i] = i % numCategories; // Set the diagonal entry to the row number
	}
	const categories = new ByteStringArray(10, numCategories); // TODO(ilan-gold): there appears to be a bug with the order of arguments on `ByteStringArray`: https://github.com/manzt/zarrita.js/blob/f838137481c40afdb6bed263a748cd9a65d7ed62/packages/typedarray/src/index.ts#L175-L176
	Array.from(new Array(numCategories).keys())
		.map((k) => `cat_${k}`)
		.forEach((val, ind) => {
			categories.set(ind, val);
		});
	grp.attrs["encoding-type"] = "categorical";
	const codesArr = await zarr.create(grp, {
		shape: [length],
		chunk_shape: [length],
		data_type: "uint32",
	});
	// TODO(ilan-gold): open issue about encoding vlen-utf8
	// const categoriesArr = await zarr.create(grp, {
	//   shape: [numCategories],
	//   chunk_shape: [numCategories],
	//   data_type: "v2:object",
	//   codecs: [{ name: "vlen-utf8", configuration: {} }]
	// });
	const categoriesArr = await zarr.create(grp, {
		shape: [numCategories],
		chunk_shape: [numCategories],
		data_type: "v2:S50",
	});
	const codesChunk = {
		data: codes,
		shape: [length],
		stride: [1],
	} as zarr.Chunk<"uint32">;
	await zarr.set(codesArr, null, codesChunk);
	// TODO(ilan-gold): open issue about encoding vlen-utf8
	// const categoriesChunk = {
	//   data: categories,
	//   shape: [numCategories],
	//   stride: [1],
	// } as zarr.Chunk<"v2:object">;
	const categoriesChunk = {
		data: categories,
		shape: [numCategories],
		stride: [1],
	} as unknown as zarr.Chunk<zarr.ByteStr>;
	await zarr.set(categoriesArr, null, categoriesChunk);
	return new LazyCategoricalArray(codesArr, categoriesArr);
}

async function genIndex(
	grp: zarr.Group<Map<string, Uint8Array>>,
	length: number,
	axis: "obs" | "var",
	indexKey: string,
) {
	const index = new ByteStringArray(10, length);
	Array.from(new Array(length).keys())
		.map((k) => `${axis}_${k}`)
		.forEach((val, ind) => {
			index.set(ind, val);
		});
	// TODO(ilan-gold): open issue about encoding vlen-utf8
	// const indexArr = await zarr.create(grp.resolve("index"), {
	//   shape: [length],
	//   chunk_shape: [length],
	//   data_type: "v2:object",
	//   codecs: [{ name: "vlen-utf8", configuration: {} }]
	// });
	// const indexChunk = {
	//   data: index,
	//   shape: [length],
	//   stride: [1],
	// } as zarr.Chunk<"v2:object">;
	const indexArr = await zarr.create(grp.resolve(indexKey), {
		shape: [length],
		chunk_shape: [length],
		data_type: "v2:S50",
	});
	const indexChunk = {
		data: index,
		shape: [length],
		stride: [1],
	} as unknown as zarr.Chunk<zarr.ByteStr>; // TODO(ilan-gold): why is this complaining about a #private method?
	await zarr.set(indexArr, null, indexChunk);
}

// biome-ignore lint: unused
async function genAnnData(
	keys: AxisKey[],
	n_obs: number,
	n_var: number,
	X_type?: "csc" | "csr" | "dense",
): Promise<AnnData<Map<string, Uint8Array>, zarr.NumberDataType>> {
	const shape = [n_obs, n_var];
	const adataInit = {} as AxisKeyTypes<
		Map<string, Uint8Array>,
		zarr.NumberDataType
	>;
	const root = zarr.root(new Map());
	const grp = await zarr.create(root);
	const mkeys = ["obsm", "varm"];
	const pkeys = ["obsp", "varp"];
	const axes = ["obs", "var"];
	await Promise.all(
		AxisKeys.map(async (key) => {
			if (key === "X") {
				if (keys.includes(key)) {
					const X = await zarr.create(grp.resolve(key));
					if (X_type === "dense") {
						adataInit[key] = await genDense(X, shape);
					} else if (X_type === "csc" || X_type === "csr") {
						adataInit[key] = await genSparse(X, shape, X_type);
					} else {
						throw new Error(
							`You must specify a valid type for X, found ${X_type}`,
						);
					}
				}
			} else if (mkeys.includes(key)) {
				const m = await zarr.create(grp.resolve(key));
				if (keys.includes(key)) {
					await genDense(
						await zarr.open(m.resolve("dense"), { kind: "group" }),
						[shape[mkeys.indexOf(key)], 15],
					);
					await genSparse(
						await zarr.open(m.resolve("csr"), { kind: "group" }),
						[shape[mkeys.indexOf(key)], 15],
						"csr",
					);
					await genSparse(
						await zarr.open(m.resolve("csc"), { kind: "group" }),
						[shape[mkeys.indexOf(key)], 15],
						"csc",
					);
				}
				adataInit[key] = new AxisArrays(grp, key);
			} else if (pkeys.includes(key)) {
				const m = await zarr.create(grp.resolve(key));
				if (keys.includes(key)) {
					const size = shape[mkeys.indexOf(key)];
					await genDense(
						await zarr.open(m.resolve("dense"), { kind: "group" }),
						[size, size],
					);
					await genSparse(
						await zarr.open(m.resolve("csr"), { kind: "group" }),
						[size, size],
						"csr",
					);
					await genSparse(
						await zarr.open(m.resolve("csc"), { kind: "group" }),
						[size, size],
						"csc",
					);
				}
				adataInit[key] = new AxisArrays(grp, key);
			} else if (key === "layers") {
				const m = await zarr.create(grp.resolve(key));
				if (keys.includes(key)) {
					await genDense(
						await zarr.open(m.resolve("dense"), { kind: "group" }),
						shape,
					);
					await genSparse(
						await zarr.open(m.resolve("csr"), { kind: "group" }),
						shape,
						"csr",
					);
					await genSparse(
						await zarr.open(m.resolve("csc"), { kind: "group" }),
						shape,
						"csc",
					);
				}
				adataInit[key] = new AxisArrays(grp, key);
			} else {
				const indexKey = "index";
				const m = await zarr.create(grp.resolve(key), {
					attributes: { _index: indexKey },
				});
				const size = shape[axes.indexOf(key)];
				await genIndex(m, size, key as "obs" | "var", indexKey);
				if (keys.includes(key)) {
					await genDense(
						await zarr.open(m.resolve("dense"), { kind: "group" }),
						[size],
					);
					await genCategorical(
						await zarr.open(m.resolve("categorical"), { kind: "group" }),
						size,
						10,
					);
				}
				adataInit[key] = new AxisArrays(grp, key);
			}
		}),
	);
	return new AnnData(adataInit);
}

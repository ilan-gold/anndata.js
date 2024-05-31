import { describe, expect, it } from "vitest";
import * as zarr from "zarrita";
// import { HTTPStore, NestedArray } from "zarr";
import { readZarr } from "../src/anndata.js";
import { get } from "../src/utils.js";

import type { Readable } from "@zarrita/storage";
import AxisArrays from "../src/axis_arrays.js";
import SparseArray from "../src/sparse_array.js";
import anndata_0_7_CscFixture from "./fixtures/0.7/anndata-csc.json";
import anndata_0_7_CsrFixture from "./fixtures/0.7/anndata-csr.json";
import anndata_0_7_DenseFixture from "./fixtures/0.7/anndata-dense.json";
import anndata_0_7_NoX from "./fixtures/0.7/anndata-no-X.json";
import anndata_0_8_CscFixture from "./fixtures/0.8/anndata-csc.json";
import anndata_0_8_CsrFixture from "./fixtures/0.8/anndata-csr.json";
import anndata_0_8_DenseFixture from "./fixtures/0.8/anndata-dense.json";
import anndata_0_8_NoX from "./fixtures/0.8/anndata-no-X.json";
import anndata_0_9_CscFixture from "./fixtures/0.9/anndata-csc.json";
import anndata_0_9_CsrFixture from "./fixtures/0.9/anndata-csr.json";
import anndata_0_9_DenseFixture from "./fixtures/0.9/anndata-dense.json";
import anndata_0_9_NoX from "./fixtures/0.9/anndata-no-X.json";
import anndata_0_10_CscFixture from "./fixtures/0.10/anndata-csc.json";
import anndata_0_10_CsrFixture from "./fixtures/0.10/anndata-csr.json";
import anndata_0_10_DenseFixture from "./fixtures/0.10/anndata-dense.json";
import anndata_0_10_NoX from "./fixtures/0.10/anndata-no-X.json";
import {
	createStoreFromMapContents,
	makeDiagonalWithMissingRow,
	stringArrayFromPrefixAndSize,
} from "./utils.js";

const N_OBS = 50;
const N_VAR = 25;

function test_io(
	fixture: [string, string][],
	type: "dense" | "csc" | "csr" | "no-X",
	version: number,
) {
	describe(`${type} X v${version}`, async () => {
		const store = createStoreFromMapContents(fixture);
		const adata = await readZarr(store as Readable);
		it("obs column", async () => {
			const ids = await adata.obs.get("categorical");
			expect(Array.from((await get(ids, [null])).data)).toEqual(
				stringArrayFromPrefixAndSize("cat", N_OBS, 5),
			);
			expect(Array.from((await get(ids, [zarr.slice(0, 2)])).data)).toEqual([
				"cat_0",
				"cat_1",
			]);
			expect(await get(ids, [0])).toEqual("cat_0");
			expect(await adata.obs.has("not_a_column")).toEqual(false);
			expect(async () => await adata.obs.get("not_a_column")).rejects.toThrow(
				'obs has no key: "not_a_column"',
			);
		});
		it("obs index", async () => {
			const ids = await adata.obsNames();
			expect(Array.from((await get(ids, [null])).data)).toEqual(
				stringArrayFromPrefixAndSize("obs", N_OBS),
			);
		});
		it("var index", async () => {
			const ids = await adata.varNames();
			expect(Array.from((await get(ids, [null])).data)).toEqual(
				stringArrayFromPrefixAndSize("var", N_VAR),
			);
		});
		it("obsm", async () => {
			expect(adata.obsm).toBeInstanceOf(AxisArrays);
			const data_int32_csc = await adata.obsm.get("int32_csc");
			expect(data_int32_csc).toBeInstanceOf(SparseArray);
			expect(data_int32_csc.format).toEqual("csc");
			const data = new Int32Array(N_OBS * 2);
			expect(await get(data_int32_csc, [null, null])).toEqual({
				data,
				shape: [N_OBS, 2],
				stride: [2, 1],
			});
			const data_float32_csr = await adata.obsm.get("float32_csr");
			expect(data_float32_csr).toBeInstanceOf(SparseArray);
			expect(data_float32_csr.format).toEqual("csr");
			expect(await get(data_float32_csr, [null, null])).toEqual({
				data: new Float32Array(data),
				shape: [N_OBS, 2],
				stride: [2, 1],
			});
			const data_int64_dense = await adata.obsm.get("int64_dense");
			expect(data_int64_dense).toBeInstanceOf(zarr.Array);
			expect(await get(data_int64_dense, [null, null])).toEqual({
				data: new BigInt64Array(Array.from(data).map((i) => BigInt(i))),
				shape: [N_OBS, 2],
				stride: [2, 1],
			});
		});
		it("varp", async () => {
			expect(adata.varp).toBeInstanceOf(AxisArrays);
			const data_int32_csc = await adata.varp.get("int32_csc");
			expect(data_int32_csc).toBeInstanceOf(SparseArray);
			expect(data_int32_csc.format).toEqual("csc");
			const data = makeDiagonalWithMissingRow(N_VAR, N_VAR);
			expect(await get(data_int32_csc, [null, null])).toEqual({
				data,
				shape: [N_VAR, N_VAR],
				stride: [N_VAR, 1],
			});
			const data_float32_csr = await adata.varp.get("float32_csr");
			expect(data_float32_csr).toBeInstanceOf(SparseArray);
			expect(data_float32_csr.format).toEqual("csr");
			expect(await get(data_float32_csr, [null, null])).toEqual({
				data: new Float32Array(data),
				shape: [N_VAR, N_VAR],
				stride: [N_VAR, 1],
			});
			const data_int64_dense = await adata.varp.get("int64_dense");
			expect(data_int64_dense).toBeInstanceOf(zarr.Array);
			expect(await get(data_int64_dense, [null, null])).toEqual({
				data: new BigInt64Array(Array.from(data).map((i) => BigInt(i))),
				shape: [N_VAR, N_VAR],
				stride: [N_VAR, 1],
			});
		});
		it("X", async () => {
			const data = await adata.X;
			if (data === undefined) {
				return;
			}
			expect(await get(data, [null, null])).toEqual({
				data: new Float32Array(makeDiagonalWithMissingRow(N_OBS, N_VAR)),
				shape: [N_OBS, N_VAR],
				stride: [N_VAR, 1],
			});
			expect(await get(data, [0, null])).toEqual({
				data: new Float32Array(N_VAR),
				shape: type === "csr" ? [1, N_VAR] : [N_VAR],
				stride: type === "csr" ? [N_VAR, 1] : [1],
			});
			expect(await get(data, [1, 1])).toEqual(1);
			expect(await get(data, [7, 7])).toEqual(7);
		});
	});
}

// function test_in_memory(type: "dense" | "csc" | "csr", n_obs: number = 50, n_var: number = 30) {
//   describe('Empty keys test', async () => {
//     const adata = await genAnnData([], n_obs, n_var)
//     it("names", async () => {
//       expect(Array.from((await get(await adata.obsNames(), [null])).data)).toEqual(Array.from(new Array(n_obs).keys()).map(k => `obs_${k}`))
//       expect(Array.from((await get(await adata.obsNames(), [null])).data)).toEqual(Array.from(new Array(n_var).keys()).map(k => `var_${k}`))
//     })
//     it("X", () => {
//       expect(adata.X).toEqual(undefined)
//     })
//     it("layers", () => {
//       expect(adata.layers).toBeInstanceOf(AxisArrays)
//     })
//   })
//   describe('Only X test', async () => {
//     const adata = await genAnnData(["X"], n_obs, n_var, "csc")
//     it("names", async () => {
//       expect(Array.from((await get(await adata.obsNames(), [null])).data)).toEqual(Array.from(new Array(n_obs).keys()).map(k => `obs_${k}`))
//       expect(Array.from((await get(await adata.obsNames(), [null])).data)).toEqual(Array.from(new Array(n_var).keys()).map(k => `var_${k}`))
//     })
//     it("X", () => {
//       expect(adata.X).toEqual(undefined)  // Like the above, this is not reading in data
//     })
//     it("layers", () => {
//       expect(adata.layers).toBeInstanceOf(AxisArrays)
//     })
//   })
// }

describe("AnnData i/o", () => {
	Object.entries({
		0.7: anndata_0_7_DenseFixture,
		0.8: anndata_0_8_DenseFixture,
		0.9: anndata_0_9_DenseFixture,
		"0.10": anndata_0_10_DenseFixture,
	}).forEach(([version, fixture]) => {
		test_io(fixture, "dense", version);
	});
	Object.entries({
		0.7: anndata_0_7_CscFixture,
		0.8: anndata_0_8_CscFixture,
		0.9: anndata_0_9_CscFixture,
		"0.10": anndata_0_10_CscFixture,
	}).forEach(([version, fixture]) => {
		test_io(fixture, "csc", version);
	});
	Object.entries({
		0.7: anndata_0_7_CsrFixture,
		0.8: anndata_0_8_CsrFixture,
		0.9: anndata_0_9_CsrFixture,
		"0.10": anndata_0_10_CsrFixture,
	}).forEach(([version, fixture]) => {
		test_io(fixture, "csr", version);
	});
	Object.entries({
		0.7: anndata_0_7_NoX,
		0.8: anndata_0_8_NoX,
		0.9: anndata_0_9_NoX,
		"0.10": anndata_0_10_NoX,
	}).forEach(([version, fixture]) => {
		test_io(fixture, "no-X", version);
	});
});

// TODO(ilan-gold): Figure out why making the roundtrip is not reading in any data?
// describe("AnnData in-memory", () => {
//   test_in_memory("dense")
// })

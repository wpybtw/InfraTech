import test from "node:test";
import assert from "node:assert/strict";
import {dim,evaluateDim,formatDim,evaluateShape,validateDivisibility} from "../core/shape.js";
import {matmulFlops,tensorBytes,mlaCosts} from "../core/cost.js";
import {buildMLAGraph,DEFAULT_CONFIG} from "../core-mla.js";

test("symbolic dimensions evaluate without eval",()=>{
 const expr=dim.mul(dim.sym("heads"),dim.add(dim.sym("nope"),dim.sym("rope")));
 assert.equal(evaluateDim(expr,{heads:32,nope:128,rope:64}),6144);
 assert.equal(formatDim(expr),"heads × (nope + rope)");
});
test("shape evaluation and dtype byte size",()=>{
 assert.deepEqual(evaluateShape(["B","S",dim.mul(2,dim.sym("H"))],{B:2,S:4,H:8}),[2,4,16]);
 assert.equal(tensorBytes([2,4,16],"bf16"),256);
});
test("matmul FLOPs count multiply and add separately",()=>assert.equal(matmulFlops(2,3,4),48));
test("dimension divisibility rejects invalid head partition",()=>assert.throws(()=>validateDivisibility(4097,32,"hidden size"),/must be divisible/));
test("MLA prefill builds explicit K/V expansion path",()=>{
 const g=buildMLAGraph(DEFAULT_CONFIG);
 assert.deepEqual(g.tensors.find(t=>t.id==="ckv").resolvedShape,[1,4096,512]);
 assert.deepEqual(g.tensors.find(t=>t.id==="krope").resolvedShape,[1,4096,64]);
 assert.deepEqual(g.tensors.find(t=>t.id==="scores").resolvedShape,[1,32,4096,4096]);
 assert.ok(g.operations.some(o=>o.id==="k_up"));
 assert.ok(g.operations.some(o=>o.id==="v_up"));
});
test("MLA decode uses absorbed latent attention path",()=>{
 const g=buildMLAGraph({...DEFAULT_CONFIG,phase:"decode",sequence:8192});
 assert.equal(g.queryLength,1);assert.equal(g.keyLength,8192);
 assert.deepEqual(g.tensors.find(t=>t.id==="scores").resolvedShape,[1,32,1,8192]);
 assert.ok(g.operations.some(o=>o.id==="q_absorb"));
 assert.ok(g.operations.some(o=>o.id==="v_absorb"));
 assert.ok(!g.operations.some(o=>o.id==="k_up"));
});
test("MLA KV cache stores compressed latent plus RoPE key per layer",()=>{
 const c=mlaCosts({...DEFAULT_CONFIG,phase:"decode",queryLength:1,keyLength:4096});
 assert.equal(c.cacheBytesPerLayer,4096*(512+64)*2);
 assert.equal(c.totalCacheBytes,4096*576*2*32);
});

import { formatShape } from "./core/shape.js";
import { formatBytes } from "./core/cost.js";

export const DEFAULT_CONFIG = Object.freeze({
  batch: 1, sequence: 4096, hidden: 4096, heads: 32,
  qRank: 1536, kvRank: 512, nope: 128, rope: 64, valueDim: 128,
  layers: 32, dtype: "bf16", phase: "prefill",
});

const tensor = (id, name, kind, shape, dtype, note = "") =>
  ({ id, name, kind, shape, dtype, note });
const op = (id, name, formula, inputs, outputs, flops, note = "") =>
  ({ id, name, formula, inputs, outputs, flops, note });

export function buildMLAGraph(config) {
  const c = { ...DEFAULT_CONFIG, ...config };
  const { batch: B, sequence: S, hidden: H, heads: Nh, qRank: Rq,
    kvRank: Rkv, nope: Dn, rope: Dr, valueDim: Dv, dtype, phase } = c;
  const Sq = phase === "decode" ? 1 : S;
  const Sk = S;
  const costs = mlaCosts({ ...c, queryLength: Sq, keyLength: Sk });
  const tensors = [
    tensor("x", "Hidden states", "activation", [B, Sq, H], dtype, "Input activation"),
    tensor("wq_down", "Wq_down", "weight", [H, Rq], dtype, "Q low-rank down projection"),
    tensor("cq", "Q latent", "intermediate", [B, Sq, Rq], dtype),
    tensor("wq_up", "Wq_up", "weight", [Rq, Nh * (Dn + Dr)], dtype),
    tensor("q", "Query", "activation", [B, Sq, Nh, Dn + Dr], dtype, "Split into Q_nope and Q_rope"),
    tensor("wkv_down", "Wkv_down", "weight", [H, Rkv + Dr], dtype, "Joint KV latent and shared RoPE key projection"),
    tensor("ckv", "Compressed KV latent", "cache", [B, Sk, Rkv], dtype, "Persisted in MLA KV cache"),
    tensor("krope", "K_rope", "cache", [B, Sk, Dr], dtype, "Shared RoPE key; persisted in cache"),
    tensor("wk_up", "Wk_up", "weight", [Rkv, Nh * Dn], dtype),
    tensor("knope", "K_nope", "intermediate", [B, Sk, Nh, Dn], dtype, "Explicit materialization path"),
    tensor("wv_up", "Wv_up", "weight", [Rkv, Nh * Dv], dtype),
    tensor("v", "Value", "intermediate", [B, Sk, Nh, Dv], dtype, "Explicit materialization path"),
    tensor("scores", "Attention scores", "intermediate", [B, Nh, Sq, Sk], dtype),
    tensor("probs", "Attention probabilities", "intermediate", [B, Nh, Sq, Sk], dtype),
    tensor("context", "Attention output", "activation", [B, Sq, Nh, Dv], dtype),
    tensor("wo", "Wo", "weight", [Nh * Dv, H], dtype),
    tensor("y", "Attention output projection", "activation", [B, Sq, H], dtype),
  ];
  const operations = [
    op("q_down", "Q down projection", "Cq = X · Wq_down", ["x","wq_down"], ["cq"], costs.components.qDown),
    op("q_up", "Q up projection", "Q = Cq · Wq_up", ["cq","wq_up"], ["q"], costs.components.qUp),
    op("kv_down", "Joint KV down projection", "[Ckv, K_rope] = X · Wkv_down", ["x","wkv_down"], ["ckv","krope"], costs.components.kvDown),
    op("k_up", "K up projection", "K_nope = Ckv · Wk_up", ["ckv","wk_up"], ["knope"], costs.components.kUp, "Explicit K materialization. Decode may absorb Wk_up into the Q-side projection."),
    op("v_up", "V up projection", "V = Ckv · Wv_up", ["ckv","wv_up"], ["v"], costs.components.vUp, "Explicit V materialization. Decode may absorb Wv_up into the output-side projection."),
    op("qk", "QKᵀ", "Scores = Q_nope·K_nopeᵀ + Q_rope·K_ropeᵀ", ["q","knope","krope"], ["scores"], 2 * B * Nh * Sq * Sk * (Dn + Dr)),
    op("softmax", "Scale + Softmax", "P = softmax(Scores / √(Dn + Dr))", ["scores"], ["probs"], 0, "Excluded from headline FLOPs; exact operation count depends on implementation."),
    op("pv", "Probability × V", "Context = P · V", ["probs","v"], ["context"], 2 * B * Nh * Sq * Sk * Dv),
    op("out", "Output projection", "Y = Context · Wo", ["context","wo"], ["y"], costs.components.out),
  ];
  const weightIds = new Set(["wq_down","wq_up","wkv_down","wk_up","wv_up","wo"]);
  for (const t of tensors) t.resolvedShape = t.shape;
  const visibleTensors = tensors.filter(t => !weightIds.has(t.id));
  return { config:c, queryLength:Sq, keyLength:Sk, tensors, visibleTensors, operations, costs };
}

export function describeTensor(t) {
  const elements = t.resolvedShape.reduce((a,b) => a * b, 1);
  const bytesPerElement = DTYPE_BYTES[t.dtype];
  const bytes = elements * bytesPerElement;
  return { shape: formatShape(t.resolvedShape), elements, bytes, size: formatBytes(bytes) };
}

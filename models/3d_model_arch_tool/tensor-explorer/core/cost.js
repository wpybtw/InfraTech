import { numel } from "./shape.js";

export const DTYPE_BYTES = Object.freeze({ bf16: 2, fp16: 2, fp8: 1, fp32: 4 });

export function tensorBytes(shape, env, dtype) {
  const elements = numel(shape, env);
  if (elements === undefined) return undefined;
  const bytes = DTYPE_BYTES[dtype];
  if (!bytes) throw new RangeError(`Unsupported dtype: ${dtype}`);
  return elements * bytes;
}

/** Dense matrix multiplication: 1 multiply + 1 add = 2 FLOPs. */
export function matmulFlops(m, n, k, batch = 1) {
  return 2 * batch * m * n * k;
}

export function formatCount(value) {
  if (!Number.isFinite(value)) return "—";
  const units = [["P", 1e15], ["T", 1e12], ["G", 1e9], ["M", 1e6], ["K", 1e3]];
  for (const [suffix, scale] of units) {
    if (Math.abs(value) >= scale) return `${(value / scale).toFixed(2)} ${suffix}`;
  }
  return String(Math.round(value));
}

export function formatBytes(value) {
  if (!Number.isFinite(value)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let n = value, i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 100 ? 0 : n >= 10 ? 1 : 2)} ${units[i]}`;
}

export function mlaCosts(c) {
  const { batch: B, queryLength: Sq, keyLength: Sk, hidden: H, heads: Nh,
    qRank: Rq, kvRank: Rkv, nope: Dn, rope: Dr, valueDim: Dv, dtype, layers } = c;
  const bytes = DTYPE_BYTES[dtype];
  const qDown = matmulFlops(B * Sq, Rq, H);
  const qUp = matmulFlops(B * Sq, Nh * (Dn + Dr), Rq);
  const kvDown = matmulFlops(B * Sq, Rkv + Dr, H);
  const kUp = matmulFlops(B * Sk, Nh * Dn, Rkv);
  const vUp = matmulFlops(B * Sk, Nh * Dv, Rkv);
  const out = matmulFlops(B * Sq, H, Nh * Dv);
  const attention = 2 * B * Nh * Sq * Sk * (Dn + Dr + Dv);
  const projection = qDown + qUp + kvDown + kUp + vUp + out;

  const params = H * Rq + Rq * Nh * (Dn + Dr) +
    H * (Rkv + Dr) + Rkv * Nh * Dn + Rkv * Nh * Dv + Nh * Dv * H;
  const weightBytes = params * bytes;
  const cacheBytesPerLayer = B * Sk * (Rkv + Dr) * bytes;
  const totalCacheBytes = cacheBytesPerLayer * layers;
  const qActivationBytes = B * Sq * Nh * (Dn + Dr) * bytes;
  const latentActivationBytes = B * Sq * (Rkv + Dr) * bytes;
  const attentionScoreBytes = B * Nh * Sq * Sk * bytes;

  return {
    projection, attention, total: projection + attention,
    components: { qDown, qUp, kvDown, kUp, vUp, out, attention },
    params, weightBytes, cacheBytesPerLayer, totalCacheBytes,
    qActivationBytes, latentActivationBytes, attentionScoreBytes,
    cacheBytesPerToken: B * (Rkv + Dr) * bytes * layers,
  };
}

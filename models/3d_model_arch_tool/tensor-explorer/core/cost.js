export const DTYPE_BYTES = Object.freeze({ bf16: 2, fp16: 2, fp8: 1, fp32: 4 });

export function tensorBytes(shape, dtype) {
  if (!DTYPE_BYTES[dtype]) throw new RangeError(`Unsupported dtype: ${dtype}`);
  return shape.reduce((a,b) => a * b, 1) * DTYPE_BYTES[dtype];
}
export function matmulFlops(m,n,k,batch=1){return 2*batch*m*n*k;}
export function formatCount(value){
  if(!Number.isFinite(value))return "—";
  for(const [s,u] of [["P",1e15],["T",1e12],["G",1e9],["M",1e6],["K",1e3]])if(Math.abs(value)>=u)return `${(value/u).toFixed(2)} ${s}`;
  return String(Math.round(value));
}
export function formatBytes(value){
  if(!Number.isFinite(value))return "—";
  const units=["B","KB","MB","GB","TB","PB"];let n=value,i=0;
  while(n>=1024&&i<units.length-1){n/=1024;i++;}
  return `${n.toFixed(n>=100?0:n>=10?1:2)} ${units[i]}`;
}

/**
 * Analytical MLA cost model.
 * Prefill uses explicit K/V expansion. Decode uses the absorbed latent path:
 * Q_nope is projected into latent space, attention reads the compressed cache,
 * and the latent context is projected through V_up before output projection.
 */
export function mlaCosts(c){
  const {batch:B,queryLength:Sq,keyLength:Sk,hidden:H,heads:Nh,qRank:Rq,kvRank:Rkv,nope:Dn,rope:Dr,valueDim:Dv,dtype,layers,phase="prefill"}=c;
  const bytes=DTYPE_BYTES[dtype];if(!bytes)throw new RangeError(`Unsupported dtype: ${dtype}`);
  const qDown=matmulFlops(B*Sq,Rq,H);
  const qUp=matmulFlops(B*Sq,Nh*(Dn+Dr),Rq);
  const kvDown=matmulFlops(B*Sq,Rkv+Dr,H);
  const out=matmulFlops(B*Sq,H,Nh*Dv);
  const qkRope=2*B*Nh*Sq*Sk*Dr;
  let components,projection,attention,params,explicitKVBytes,logicalIntermediateBytes;
  if(phase==="decode"){
    const qAbsorb=matmulFlops(B*Sq,Nh*Rkv,Dn);
    const qkLatent=2*B*Nh*Sq*Sk*Rkv;
    const pvLatent=2*B*Nh*Sq*Sk*Rkv;
    const vAbsorb=matmulFlops(B*Sq,Nh*Dv,Rkv);
    components={qDown,qUp,kvDown,qAbsorb,qkLatent,qkRope,pvLatent,vAbsorb,out};
    projection=qDown+qUp+kvDown+qAbsorb+vAbsorb+out;
    attention=qkLatent+qkRope+pvLatent;
    explicitKVBytes=0;
    const scoreBytes=B*Nh*Sq*Sk*bytes;
    const qBytes=B*Sq*Nh*(Dn+Dr)*bytes;
    const latentContextBytes=B*Sq*Nh*Rkv*bytes;
    logicalIntermediateBytes=qBytes+2*scoreBytes+latentContextBytes;
  }else{
    const kUp=matmulFlops(B*Sk,Nh*Dn,Rkv);
    const vUp=matmulFlops(B*Sk,Nh*Dv,Rkv);
    const qkNope=2*B*Nh*Sq*Sk*Dn;
    const pv=2*B*Nh*Sq*Sk*Dv;
    components={qDown,qUp,kvDown,kUp,vUp,qkNope,qkRope,pv,out};
    projection=qDown+qUp+kvDown+kUp+vUp+out;
    attention=qkNope+qkRope+pv;
    explicitKVBytes=B*Sk*Nh*(Dn+Dv)*bytes;
    const qBytes=B*Sq*Nh*(Dn+Dr)*bytes;
    const latentBytes=B*Sq*(Rkv+Dr)*bytes;
    const scoreBytes=B*Nh*Sq*Sk*bytes;
    const contextBytes=B*Sq*Nh*Dv*bytes;
    logicalIntermediateBytes=qBytes+latentBytes+2*scoreBytes+contextBytes+explicitKVBytes;
  }
  params=H*Rq+Rq*Nh*(Dn+Dr)+H*(Rkv+Dr)+Rkv*Nh*Dn+Rkv*Nh*Dv+Nh*Dv*H;
  const weightBytes=params*bytes;
  const cacheBytesPerLayer=B*Sk*(Rkv+Dr)*bytes;
  const totalCacheBytes=cacheBytesPerLayer*layers;
  return {projection,attention,total:projection+attention,components,params,weightBytes,cacheBytesPerLayer,totalCacheBytes,logicalIntermediateBytes,cacheBytesPerToken:B*(Rkv+Dr)*bytes*layers,explicitKVBytes};
}

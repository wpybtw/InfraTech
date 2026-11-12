import { formatShape } from "./core/shape.js";
import { formatBytes, mlaCosts } from "./core/cost.js";
export const DEFAULT_CONFIG=Object.freeze({batch:1,sequence:4096,hidden:4096,heads:32,qRank:1536,kvRank:512,nope:128,rope:64,valueDim:128,layers:32,dtype:"bf16",phase:"prefill"});
const tensor=(id,name,kind,shape,dtype,note="")=>({id,name,kind,shape,dtype,note});
const op=(id,name,formula,inputs,outputs,flops,note="")=>({id,name,formula,inputs,outputs,flops,note});
export function buildMLAGraph(config){
 const c={...DEFAULT_CONFIG,...config};
 const {batch:B,sequence:S,hidden:H,heads:Nh,qRank:Rq,kvRank:Rkv,nope:Dn,rope:Dr,valueDim:Dv,dtype,phase}=c;
 const Sq=phase==="decode"?1:S,Sk=S;
 const costs=mlaCosts({...c,queryLength:Sq,keyLength:Sk});
 const tensors=[
  tensor("x","Normalized hidden states","activation",[B,Sq,H],dtype,"Input after pre-attention RMSNorm; normalization is outside this module template"),
  tensor("wq_down","Wq_down","weight",[H,Rq],dtype,"Q low-rank down projection"),
  tensor("cq","Q latent","intermediate",[B,Sq,Rq],dtype),
  tensor("wq_up","Wq_up","weight",[Rq,Nh*(Dn+Dr)],dtype),
  tensor("q","Query before RoPE","intermediate",[B,Sq,Nh,Dn+Dr],dtype,"Split into Q_nope and Q_rope components"),
  tensor("q_rot","Query after RoPE","activation",[B,Sq,Nh,Dn+Dr],dtype),
  tensor("wkv_down","Wkv_down","weight",[H,Rkv+Dr],dtype,"Joint KV latent and shared RoPE key projection"),
  tensor("ckv","Compressed KV latent cache","cache",[B,Sk,Rkv],dtype,"KV cache after appending the current token"),
  tensor("krope","Rotated K_rope cache","cache",[B,Sk,Dr],dtype,"Shared RoPE key cache after appending the current token")
 ];
 const operations=[
  op("q_down","Q down projection","Cq = X · Wq_down",["x","wq_down"],["cq"],costs.components.qDown),
  op("q_up","Q up projection","Q = Cq · Wq_up",["cq","wq_up"],["q"],costs.components.qUp),
  op("q_rope","Apply RoPE to Q_rope","Q_rope = RoPE(Q_rope, positions)",["q"],["q_rot"],0,"Only the rotary subspace is position-rotated; elementwise/trigonometric work excluded from FLOPs."),
  op("kv_down","Joint KV down projection","[Ckv_new, K_rope_new] = X · Wkv_down",["x","wkv_down"],["ckv_new","krope_new"],costs.components.kvDown)
 ];
 if(phase==="decode"){
  tensors.push(
   tensor("ckv_cache","Previous KV latent cache","cache",[B,Math.max(0,Sk-1),Rkv],dtype,"Cache before appending the current token"),
   tensor("krope_cache","Previous RoPE key cache","cache",[B,Math.max(0,Sk-1),Dr],dtype),
   tensor("ckv_new","New KV latent","intermediate",[B,1,Rkv],dtype,"Compressed KV for the current token"),
   tensor("krope_new_raw","New K_rope before rotation","intermediate",[B,1,Dr],dtype),
   tensor("krope_new","New rotated K_rope","intermediate",[B,1,Dr],dtype,"RoPE-applied key for current token"),
   tensor("wk_up","Wk_up","weight",[Rkv,Nh*Dn],dtype),
   tensor("q_latent","Latent-space query","intermediate",[B,Sq,Nh,Rkv],dtype,"Q_nope projected into latent space; absorbed K up-projection"),
   tensor("scores","Attention scores","intermediate",[B,Nh,Sq,Sk],dtype),
   tensor("probs","Attention probabilities","intermediate",[B,Nh,Sq,Sk],dtype),
   tensor("latent_context","Latent context","intermediate",[B,Sq,Nh,Rkv],dtype,"Weighted sum over cached compressed KV"),
   tensor("wv_up","Wv_up","weight",[Rkv,Nh*Dv],dtype),
   tensor("context","Attention output","activation",[B,Sq,Nh,Dv],dtype),
   tensor("wo","Wo","weight",[Nh*Dv,H],dtype),
   tensor("y","Attention output projection","activation",[B,Sq,H],dtype)
  );
  operations.push(
   op("k_rope","Apply RoPE to new K","K_rope_new = RoPE(K_rope_raw, position)",["krope_new_raw"],["krope_new"],0,"Rotary embedding on the shared positional key subspace."),
   op("cache_append","Append KV cache","Ckv = concat(Ckv_cache, Ckv_new); K_rope = concat(K_rope_cache, K_rope_new)",["ckv_cache","ckv_new","krope_cache","krope_new"],["ckv","krope"],0,"Logical cache update; data movement is not included in FLOPs."),
   op("q_absorb","Absorb K up-projection","Q_latent = Q_nope · Wk_upᵀ",["q","wk_up"],["q_latent"],costs.components.qAbsorb,"Per-head projection into compressed KV latent space."),
   op("qk","Latent + RoPE attention","Scores = Q_latent·Ckvᵀ + Q_rope·K_ropeᵀ",["q_latent","ckv","q_rot","krope"],["scores"],costs.components.qkLatent+costs.components.qkRope),
   op("softmax","Scale + Softmax","P = softmax(Scores / √(Dn + Dr))",["scores"],["probs"],0,"Excluded from headline FLOPs."),
   op("pv","Attention over latent cache","Z = P · Ckv",["probs","ckv"],["latent_context"],costs.components.pvLatent),
   op("v_absorb","Absorb V up-projection","Context = Z · Wv_up",["latent_context","wv_up"],["context"],costs.components.vAbsorb,"Projects per-head latent contexts back to value dimensions."),
   op("out","Output projection","Y = Context · Wo",["context","wo"],["y"],costs.components.out)
  );
 }else{
  tensors.push(
   tensor("wk_up","Wk_up","weight",[Rkv,Nh*Dn],dtype),
   tensor("knope","K_nope","intermediate",[B,Sk,Nh,Dn],dtype,"Explicit prefill K materialization"),
   tensor("wv_up","Wv_up","weight",[Rkv,Nh*Dv],dtype),
   tensor("v","Value","intermediate",[B,Sk,Nh,Dv],dtype,"Explicit prefill V materialization"),
   tensor("scores","Attention scores","intermediate",[B,Nh,Sq,Sk],dtype),
   tensor("probs","Attention probabilities","intermediate",[B,Nh,Sq,Sk],dtype),
   tensor("context","Attention output","activation",[B,Sq,Nh,Dv],dtype),
   tensor("wo","Wo","weight",[Nh*Dv,H],dtype),
   tensor("y","Attention output projection","activation",[B,Sq,H],dtype)
  );
  operations.push(
   op("k_up","K up projection","K_nope = Ckv · Wk_up",["ckv","wk_up"],["knope"],costs.components.kUp),
   op("v_up","V up projection","V = Ckv · Wv_up",["ckv","wv_up"],["v"],costs.components.vUp),
   op("qk","QKᵀ","Scores = Q_nope·K_nopeᵀ + Q_rope·K_ropeᵀ",["q_rot","knope","krope"],["scores"],costs.components.qkNope+costs.components.qkRope),
   op("softmax","Scale + Softmax","P = softmax(Scores / √(Dn + Dr))",["scores"],["probs"],0,"Excluded from headline FLOPs."),
   op("pv","Probability × V","Context = P · V",["probs","v"],["context"],costs.components.pv),
   op("out","Output projection","Y = Context · Wo",["context","wo"],["y"],costs.components.out)
  );
 }
 const weightIds=new Set(["wq_down","wq_up","wkv_down","wk_up","wv_up","wo"]);
 for(const t of tensors)t.resolvedShape=t.shape;
 return {config:c,queryLength:Sq,keyLength:Sk,tensors,visibleTensors:tensors.filter(t=>!weightIds.has(t.id)),operations,costs};
}
export function describeTensor(t){
 const elements=t.resolvedShape.reduce((a,b)=>a*b,1);
 const bytesPerElement=({bf16:2,fp16:2,fp8:1,fp32:4})[t.dtype];
 const bytes=elements*bytesPerElement;
 return {shape:formatShape(t.resolvedShape),elements,bytes,size:formatBytes(bytes)};
}

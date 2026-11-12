import { DEFAULT_CONFIG, buildMLAGraph, describeTensor } from "./core-mla.js";
import { formatCount, formatBytes } from "./core/cost.js";

const $ = id => document.getElementById(id);
const fields = ["batch","seq","hidden","heads","qrank","kvrank","nope","rope","vdim","layers","dtype"];
let phase = "prefill";
let selected = null;
let graph;

const numberFields = ["batch","seq","hidden","heads","qrank","kvrank","nope","rope","vdim","layers"];
function readConfig() {
  const c = {};
  for (const key of numberFields) c[key === "seq" ? "sequence" : key === "qrank" ? "qRank" : key === "kvrank" ? "kvRank" : key === "vdim" ? "valueDim" : key] = Number($(key).value);
  c.dtype = $("dtype").value;
  c.phase = phase;
  return c;
}
function validate(c) {
  for (const [key,value] of Object.entries(c)) {
    if (typeof value === "number" && (!Number.isSafeInteger(value) || value <= 0)) throw new Error(`${key} must be a positive integer`);
  }
  if (c.hidden < 1 || c.heads < 1) throw new Error("Invalid model dimensions");
}
function esc(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

function renderFlow() {
  const t = Object.fromEntries(graph.tensors.map(x => [x.id,x]));
  const o = Object.fromEntries(graph.operations.map(x => [x.id,x]));
  const items = [
    ["INPUT", [{kind:"tensor",id:"x"}]],
    ["Q PROJECTION", [{kind:"tensor",id:"wq_down"},{kind:"operation",id:"q_down"},{kind:"tensor",id:"cq"},{kind:"tensor",id:"wq_up"},{kind:"operation",id:"q_up"},{kind:"tensor",id:"q"}]],
    ["KV COMPRESSION", [{kind:"tensor",id:"wkv_down"},{kind:"operation",id:"kv_down"},{kind:"tensor",id:"ckv"},{kind:"tensor",id:"krope"}]],
    ["LATENT EXPANSION · EXPLICIT PATH", [{kind:"tensor",id:"wk_up"},{kind:"operation",id:"k_up"},{kind:"tensor",id:"knope"},{kind:"tensor",id:"wv_up"},{kind:"operation",id:"v_up"},{kind:"tensor",id:"v"}]],
    ["ATTENTION", [{kind:"operation",id:"qk"},{kind:"tensor",id:"scores"},{kind:"operation",id:"softmax"},{kind:"tensor",id:"probs"},{kind:"operation",id:"pv"},{kind:"tensor",id:"context"}]],
    ["OUTPUT", [{kind:"tensor",id:"wo"},{kind:"operation",id:"out"},{kind:"tensor",id:"y"}]],
  ];
  $("flow").innerHTML = items.map(([label,entries]) => `<div class="flow-divider">${label}</div>` + entries.map(({kind,id}) => {
    const item = kind === "tensor" ? t[id] : o[id];
    const isWeight = item.kind === "weight";
    const cls = kind === "operation" ? "operation" : isWeight ? "weight" : item.kind === "cache" ? "cache" : "tensor";
    const shape = kind === "tensor" ? `[${item.resolvedShape.join(", ")}]` : item.formula;
    return `<button class="flow-item ${cls}" data-kind="${kind}" data-id="${id}">
      <div class="kind">${kind === "operation" ? "OPERATION" : item.kind}</div>
      <div class="name">${esc(item.name)}</div>
      <div class="shape">${esc(shape)}</div>
      <span class="arrow">${kind === "operation" ? "ƒ" : "↗"}</span>
    </button>`;
  }).join("")).join("");
  $("flow").querySelectorAll(".flow-item").forEach(el => el.addEventListener("click", () => {
    selected = { kind:el.dataset.kind, id:el.dataset.id };
    renderInspector();
    $("flow").querySelectorAll(".flow-item").forEach(x => x.classList.toggle("selected", x === el));
  }));
  $("flow-count").textContent = `${graph.tensors.length} TENSORS · ${graph.operations.length} OPERATIONS`;
  $("flow-meta").textContent = `${phase.toUpperCase()} · Q ${graph.queryLength} × KV ${graph.keyLength}`;
}
function row(label,value) { return `<div class="kv-row"><span>${esc(label)}</span><code>${esc(value)}</code></div>`; }
function section(title,body) { return `<section class="inspect-section"><h3>${title}</h3>${body}</section>`; }

function renderInspector() {
  if (!selected) { $("inspector").innerHTML = '<div class="empty-state">Select a tensor or operation in the computation flow.</div>'; return; }
  if (selected.kind === "tensor") {
    const t = graph.tensors.find(x => x.id === selected.id);
    const d = describeTensor(t,graph);
    const bytes = t.dtype === "fp8" ? 1 : t.dtype === "fp32" ? 4 : 2;
    const isWeight = t.kind === "weight";
    $("inspector").innerHTML = `
      <h2 class="inspect-title">${esc(t.name)}</h2><p class="inspect-subtitle">${esc(t.kind.toUpperCase())} · ${esc(t.dtype.toUpperCase())}</p>
      ${section("TENSOR METADATA",row("Shape",d.shape)+row("Elements",d.elements.toLocaleString("en-US"))+row("Dtype",t.dtype.toUpperCase())+row("Logical size",d.size))}
      ${section("INTERPRETATION",`<div class="formula">${esc(t.note || (isWeight ? "Learned parameter tensor." : "Intermediate activation tensor."))}</div>`)}
      ${section("STORAGE ASSUMPTION",row("Bytes / element",bytes.toString())+row("Storage",isWeight ? "Persistent parameter" : t.kind === "cache" ? "Persistent decode cache" : "Logical tensor size"))}
    `;
  } else {
    const o = graph.operations.find(x => x.id === selected.id);
    const lookup = id => graph.tensors.find(t => t.id === id);
    const inputs = o.inputs.map(id => { const t=lookup(id); return row(t.name,`[${t.resolvedShape.join(", ")}]`); }).join("");
    const outputs = o.outputs.map(id => { const t=lookup(id); return row(t.name,`[${t.resolvedShape.join(", ")}]`); }).join("");
    $("inspector").innerHTML = `
      <h2 class="inspect-title">${esc(o.name)}</h2><p class="inspect-subtitle">OPERATION · ${esc(o.id)}</p>
      ${section("MATHEMATICAL FORM",`<div class="formula">${esc(o.formula)}</div>`)}
      ${section("INPUT TENSORS",inputs)}
      ${section("OUTPUT TENSORS",outputs)}
      ${section("COMPUTATION COST",row("FLOPs",o.flops ? formatCount(o.flops) : "Not counted")+row("Convention","1 multiply + 1 add = 2 FLOPs"))}
      ${o.note ? section("NOTES",`<div class="formula">${esc(o.note)}</div>`) : ""}
    `;
  }
}
function renderMetrics() {
  const c = graph.costs;
  $("metric-proj").textContent = formatCount(c.projection);
  $("metric-attn").textContent = formatCount(c.attention);
  $("metric-weight").textContent = formatBytes(c.weightBytes);
  $("metric-cache").textContent = formatBytes(c.totalCacheBytes);
  $("metric-cache-note").textContent = `${c.cacheBytesPerToken.toLocaleString("en-US")} B / token across ${graph.config.layers} layers`;
}
function rebuild() {
  try {
    const c = readConfig(); validate(c);
    graph = buildMLAGraph(c);
    renderFlow(); renderMetrics(); renderInspector();
  } catch (error) {
    $("flow").innerHTML = `<div class="empty-state">${esc(error.message)}</div>`;
  }
}
for (const id of fields) $(id).addEventListener("change",rebuild);
document.querySelectorAll("[data-phase]").forEach(button => button.addEventListener("click",() => {
  phase = button.dataset.phase;
  document.querySelectorAll("[data-phase]").forEach(x => x.classList.toggle("active",x === button));
  rebuild();
}));
$("reset").addEventListener("click",() => {
  const map = {batch:"batch",sequence:"seq",hidden:"hidden",heads:"heads",qRank:"qrank",kvRank:"kvrank",nope:"nope",rope:"rope",valueDim:"vdim",layers:"layers",dtype:"dtype"};
  for (const [key,id] of Object.entries(map)) $(id).value = DEFAULT_CONFIG[key];
  phase = "prefill";
  document.querySelectorAll("[data-phase]").forEach(x => x.classList.toggle("active",x.dataset.phase === phase));
  rebuild();
});
rebuild();

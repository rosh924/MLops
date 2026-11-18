/* CHRIST University AI Student Helpdesk - plain JS, no dependencies.
   All API URLs are relative ("api/...") so the app also works behind a proxy (e.g. Google Colab). */
"use strict";

const MODELS = {
  llm:    { name: "Model 1: Plain LLM",     color: "#2a78d6" },
  rag:    { name: "Model 2: RAG",           color: "#eb6834" },
  llmops: { name: "Model 3: RAG + LLMOps",  color: "#1baf7a" },
};
const COLOR_BY_NAME = Object.fromEntries(Object.values(MODELS).map(m => [m.name, m.color]));
const FLAG_STYLE = {
  prompt_injection: "red", output_prompt_leak: "red", llm_error: "red",
  low_relevance_abstain: "amber", missing_citation: "amber",
  pii_phone: "amber", pii_email: "amber", pii_aadhaar: "amber", cache_hit: "blue",
};
let FLAG_LABELS = {
  pii_phone: "Phone number redacted", pii_email: "Email redacted", pii_aadhaar: "Aadhaar redacted",
  prompt_injection: "Prompt injection blocked", low_relevance_abstain: "Low relevance - abstained",
  output_prompt_leak: "Prompt leak blocked", missing_citation: "Missing citation", cache_hit: "Cache hit",
  llm_error: "LLM error",
};
const SUGGESTIONS = [
  "What is the minimum attendance required to write the ESE?",
  "How do I download my hall ticket?",
  "What are the counselling centre timings?",
  "Whom should I email for scholarships?",
];

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/* ------------------------------------------------------------------ helpers */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
/** Escape model output, then turn URLs into links and [file.md] into citation chips. */
function richText(s) {
  let h = esc(s);
  h = h.replace(/\bhttps?:\/\/[^\s<>"']+[^\s<>"'.,;:)\]]/g, u => `<a href="${u}" target="_blank" rel="noopener noreferrer">${u}</a>`);
  h = h.replace(/\[([\w.\- ]+\.(?:md|txt))\]/g, (_, f) => `<span class="cite">${f}</span>`);
  h = h.replace(/\*\*([^*\n]+)\*\*/g, "<b>$1</b>");  // LLM markdown bold (text is already escaped)
  return h.replace(/\n/g, "<br>");
}
const fmtNum = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : Number(v).toFixed(d);
function fmtTime(iso) {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? esc(iso) : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

async function api(url, body) {
  let res;
  try {
    res = await fetch(url, body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  } catch (e) {
    throw new Error("Cannot reach the helpdesk server. Is it still running? (python -m uvicorn backend.main:app --port 8000)");
  }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    let msg = `Server error (${res.status}).`;
    if (data && data.detail) {
      msg = Array.isArray(data.detail)
        ? "Invalid input: " + data.detail.map(d => d.msg).join("; ")
        : String(data.detail);
    }
    throw new Error(msg);
  }
  return data;
}

const tip = $("#tooltip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.classList.remove("hidden");
  const x = Math.min(evt.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
  tip.style.left = x + "px";
  tip.style.top = (evt.clientY - tip.offsetHeight - 10) + "px";
}
const hideTip = () => tip.classList.add("hidden");

/* ------------------------------------------------------------------ tabs */
$$(".tab").forEach(t => t.addEventListener("click", () => {
  $$(".tab").forEach(x => x.classList.toggle("active", x === t));
  $$(".page").forEach(p => p.classList.toggle("active", p.id === "page-" + t.dataset.page));
  try { localStorage.setItem("helpdesk.tab", t.dataset.page); } catch (_) { /* ignore */ }
  if (t.dataset.page === "llmops") loadMetrics();
}));
$$(".side-toggle").forEach(b => b.addEventListener("click", () => b.closest(".side").classList.toggle("open")));

/* ------------------------------------------------------------------ status pill + settings */
let HEALTH = null;
async function loadHealth() {
  const pill = $("#status-pill");
  try {
    const h = HEALTH = await api("api/health");
    pill.className = "pill";
    if (h.provider === "offline") {
      pill.classList.add("pill-off"); pill.textContent = "Offline mode (no LLM)";
    } else if (h.llm_reachable && h.model_available) {
      pill.classList.add("pill-ok"); pill.textContent = `${h.model} · online`;
    } else if (h.llm_reachable) {
      pill.classList.add("pill-warn"); pill.textContent = `Model ${h.model} not pulled`;
    } else {
      pill.classList.add("pill-err"); pill.textContent = "Ollama not reachable";
    }
    pill.title = `Provider: ${h.provider} · embeddings: ${h.embeddings}`;
    const rows = [
      ["Embeddings", h.embeddings], ["Chunks", h.chunks], ["top_k", h.top_k],
      ["chunk_size", `${h.chunk_size} chars`], ["overlap", `${h.chunk_overlap} chars`],
      ["Prompt", "rag.v1 (naive)"], ["LLM", `${h.model} (T=${h.temperature})`],
    ];
    $("#rag-settings").innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  } catch (e) {
    pill.className = "pill pill-err"; pill.textContent = "Server unreachable";
  }
}

async function loadKB() {
  try {
    const { files } = await api("api/kb");
    $("#kb-list").innerHTML = files.map(f => `
      <details><summary>${esc(f.file)} <span class="muted small">(${f.questions.length})</span></summary>
      <ul>${f.questions.map(q => `<li data-q="${esc(q)}">${esc(q)}</li>`).join("")}</ul></details>`).join("");
    $$("#kb-list li").forEach(li => li.addEventListener("click", () => CHATS.rag.ask(li.dataset.q)));
  } catch (e) {
    $("#kb-list").innerHTML = `<span class="bad">${esc(e.message)}</span>`;
  }
}

/* ------------------------------------------------------------------ chat */
function sourcesHTML(r) {
  if (!r.sources || !r.sources.length) return "";
  const items = r.sources.map((s, i) => {
    const sc = r.scores ? r.scores[i] : null;
    const w = Math.max(0, Math.min(1, sc ?? 0)) * 100;
    const col = MODELS[r.model_key]?.color || "#888";
    return `<div class="src"><div class="src-head"><span>${esc(s)}</span>
      <span class="score"><span class="track"><span class="val" style="width:${w}%;background:${col}"></span></span>${fmtNum(sc, 3)}</span></div>
      <div class="src-text">${esc((r.contexts || [])[i] || "")}</div></div>`;
  }).join("");
  return `<details class="sources"><summary>Retrieved sources (${r.sources.length})</summary>${items}</details>`;
}

function flagsHTML(flags) {
  return (flags || []).filter(f => f !== "cache_hit")
    .map(f => `<span class="flag ${FLAG_STYLE[f] || "amber"}">${esc(FLAG_LABELS[f] || f)}</span>`).join("");
}

function metaHTML(r) {
  const tok = (r.prompt_tokens || 0) + (r.completion_tokens || 0);
  let h = `<span class="tag">⏱ ${fmtNum(r.latency_s)} s</span><span class="tag">${tok} tokens (${r.prompt_tokens || 0}+${r.completion_tokens || 0})</span>`;
  if (r.model_key === "llmops") {
    if (r.cache_hit) h += `<span class="flag blue">⚡ Cache hit</span>`;
    h += flagsHTML(r.flags);
    h += `<span class="tag">prompt ${esc(r.prompt_version)}</span><span class="tag mono">trace ${esc(r.trace_id)}</span>`;
    h += `<button class="fb" data-v="1" title="Helpful">👍</button><button class="fb" data-v="0" title="Not helpful">👎</button>`;
  }
  return h;
}

function botMessageHTML(r) {
  return `<div class="bubble">${richText(r.answer)}</div>${r.model_key !== "llm" ? sourcesHTML(r) : ""}<div class="meta">${metaHTML(r)}</div>`;
}

function wireFeedback(el, r) {
  $$(".fb", el).forEach(b => b.addEventListener("click", async () => {
    const helpful = b.dataset.v === "1";
    $$(".fb", el).forEach(x => { x.disabled = true; x.classList.toggle("chosen", x === b); });
    try {
      await api("api/feedback", { trace_id: r.trace_id, helpful });
      loadMetrics();
    } catch (e) {
      $$(".fb", el).forEach(x => { x.disabled = false; x.classList.remove("chosen"); });
      alert("Could not save feedback: " + e.message);
    }
  }));
}

class Chat {
  constructor(key) {
    this.key = key;
    this.page = $(`#page-${key}`);
    this.thread = $(".thread", this.page);
    this.input = $("textarea", this.page);
    this.button = $(".send", this.page);
    this.busy = false;
    this.renderEmpty();
    $(".composer", this.page).addEventListener("submit", e => { e.preventDefault(); this.ask(this.input.value); });
    this.input.addEventListener("keydown", e => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.ask(this.input.value); }
    });
    this.input.addEventListener("input", () => autoGrow(this.input));
  }
  renderEmpty() {
    this.thread.innerHTML = `<div class="empty"><p>Ask <b>${esc(MODELS[this.key].name)}</b> a question about CHRIST University.</p>
      <div class="chips">${SUGGESTIONS.map(s => `<button type="button" class="chip" data-q="${esc(s)}">${esc(s)}</button>`).join("")}</div></div>`;
    $$(".chip", this.thread).forEach(c => c.addEventListener("click", () => this.ask(c.dataset.q)));
  }
  add(cls, html) {
    $(".empty", this.thread)?.remove();
    const el = document.createElement("div");
    el.className = "msg " + cls;
    el.innerHTML = html;
    this.thread.appendChild(el);
    this.thread.scrollTop = this.thread.scrollHeight;
    return el;
  }
  async ask(text) {
    const q = (text || "").trim();
    if (!q || this.busy) return;
    if (q.length > 1000) { this.add("bot error", `<div class="bubble">Please keep questions under 1000 characters.</div>`); return; }
    this.busy = true; this.button.disabled = true;
    this.input.value = ""; autoGrow(this.input);
    this.add("user", `<div class="bubble">${richText(q)}</div>`);
    const typing = this.add("bot typing", `<div class="bubble"><span></span><span></span><span></span></div>`);
    try {
      const r = await api("api/chat", { question: q, model: this.key });
      typing.remove();
      const el = this.add("bot", botMessageHTML(r));
      if (this.key === "llmops") { wireFeedback(el, r); loadMetrics(); }
    } catch (e) {
      typing.remove();
      this.add("bot error", `<div class="bubble">⚠ ${esc(e.message)}</div>`);
    } finally {
      this.busy = false; this.button.disabled = false; this.input.focus();
    }
  }
}
function autoGrow(t) { t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 140) + "px"; }

const CHATS = { llm: new Chat("llm"), rag: new Chat("rag"), llmops: new Chat("llmops") };

/* ------------------------------------------------------------------ monitoring dashboard */
let metricsTimer = null;
function loadMetrics() { clearTimeout(metricsTimer); metricsTimer = setTimeout(doLoadMetrics, 150); }
async function doLoadMetrics() {
  let m;
  try { m = await api("api/metrics"); } catch (e) {
    $("#kpis").innerHTML = `<div class="banner error">${esc(e.message)}</div>`; return;
  }
  if (m.flag_labels) FLAG_LABELS = { ...FLAG_LABELS, ...m.flag_labels };
  const k = m.kpis;
  const tiles = [
    ["Requests", k.requests], ["Avg latency", fmtNum(k.avg_latency_s) + " s"], ["p95 latency", fmtNum(k.p95_latency_s) + " s"],
    ["Cache hit rate", k.cache_hit_pct + "%"], ["Abstain rate", k.abstain_pct + "%"], ["Blocked injections", k.blocked_injection_pct + "%"],
    ["Avg tokens / LLM call", fmtNum(k.avg_tokens, 0)], ["Feedback", k.feedback_count], ["Positive feedback", k.positive_feedback_pct + "%"],
  ];
  $("#kpis").innerHTML = tiles.map(([l, v]) => `<div class="kpi"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("");
  renderLatencyChart(m.traces.slice(0, 30).reverse());
  renderFlagBars(m.flag_counts);
  $("#lowrel-table").innerHTML = m.low_relevance.length
    ? `<tr><th>Time</th><th>Question</th><th class="num">Top score</th></tr>` + m.low_relevance.map(t =>
      `<tr><td>${fmtTime(t.ts)}</td><td>${esc(t.question)}</td><td class="num">${fmtNum(t.top_score, 3)}</td></tr>`).join("")
    : `<tr><td class="muted">No low-relevance questions yet.</td></tr>`;
  $("#traces-table").innerHTML = m.traces.length
    ? `<tr><th>Time</th><th>Channel</th><th>Question (redacted)</th><th>Flags</th><th class="num">Latency</th><th class="num">Tokens</th><th>Feedback</th></tr>` +
      m.traces.map(t => `<tr><td>${fmtTime(t.ts)}</td><td>${esc(t.channel || "")}</td><td>${esc(t.question)}</td>
        <td>${t.cache_hit ? '<span class="flag blue">Cache hit</span> ' : ""}${flagsHTML(t.flags) || '<span class="muted">-</span>'}</td>
        <td class="num">${fmtNum(t.latency_s)} s</td><td class="num">${(t.prompt_tokens || 0) + (t.completion_tokens || 0)}</td>
        <td>${t.feedback === 1 ? "👍" : t.feedback === -1 ? "👎" : ""}</td></tr>`).join("")
    : `<tr><td class="muted">No traces yet. Ask Model 3 a question.</td></tr>`;
}
$("#refresh-metrics").addEventListener("click", doLoadMetrics);

function renderLatencyChart(traces) {
  const box = $("#latency-chart");
  if (!traces.length) { box.innerHTML = `<p class="muted small">No requests yet.</p>`; return; }
  const W = 600, H = 220, L = 46, R = 12, T = 12, B = 30;
  const maxY = Math.max(0.01, ...traces.map(t => t.latency_s || 0)) * 1.15;
  const x = i => L + (traces.length === 1 ? (W - L - R) / 2 : i * (W - L - R) / (traces.length - 1));
  const y = v => T + (H - T - B) * (1 - v / maxY);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Latency per request">`;
  for (let i = 0; i <= 4; i++) {
    const v = maxY * i / 4, yy = y(v);
    s += `<line x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}" stroke="#ece9e1"/><text x="${L - 6}" y="${yy + 4}" font-size="11" text-anchor="end" fill="#667085">${v.toFixed(v < 1 ? 2 : 1)}s</text>`;
  }
  s += `<text x="${(W + L) / 2}" y="${H - 6}" font-size="11" text-anchor="middle" fill="#667085">request (oldest → newest)</text>`;
  s += `<polyline fill="none" stroke="#1baf7a" stroke-width="2.5" stroke-linejoin="round" points="${traces.map((t, i) => `${x(i)},${y(t.latency_s || 0)}`).join(" ")}"/>`;
  traces.forEach((t, i) => {
    const c = t.flags.includes("prompt_injection") ? "#c62828" : t.cache_hit ? "#2a78d6" : "#1baf7a";
    s += `<circle class="pt" data-i="${i}" cx="${x(i)}" cy="${y(t.latency_s || 0)}" r="4.5" fill="#fff" stroke="${c}" stroke-width="2.5"/>`;
  });
  box.innerHTML = s + "</svg>";
  $$("circle.pt", box).forEach(c => {
    const t = traces[+c.dataset.i];
    c.addEventListener("mousemove", e => showTip(e, `<b>${fmtNum(t.latency_s)} s</b>${t.cache_hit ? " · cache hit" : ""}<br>${esc(t.question.slice(0, 70))}`));
    c.addEventListener("mouseleave", hideTip);
  });
}

function renderFlagBars(counts) {
  const entries = Object.entries(counts || {});
  if (!entries.length) { $("#flag-bars").innerHTML = `<p class="muted small">No flags raised yet.</p>`; return; }
  const max = Math.max(...entries.map(e => e[1]));
  const col = { red: "#c62828", amber: "#e0a030", blue: "#2a78d6" };
  $("#flag-bars").innerHTML = entries.map(([f, n]) => `<div class="flagrow"><span>${esc(FLAG_LABELS[f] || f)}</span>
    <span class="track"><span class="val" style="display:block;width:${(100 * n / max).toFixed(1)}%;background:${col[FLAG_STYLE[f]] || "#e0a030"}"></span></span>
    <span class="num">${n}</span></div>`).join("");
}

/* ------------------------------------------------------------------ live comparison */
const cmpForm = $("#compare-form"), cmpInput = $("textarea", cmpForm), cmpBtn = $(".send", cmpForm);
async function runCompare(q) {
  q = (q || "").trim();
  if (!q || cmpBtn.disabled) return;
  cmpInput.value = q; cmpBtn.disabled = true;
  $("#compare-cols").innerHTML = Object.entries(MODELS).map(([k, m]) => `<div class="ccol"><div class="ccol-head" style="background:${m.color}">${esc(m.name)}</div>
    <div class="ccol-body"><div class="msg bot typing"><div class="bubble"><span></span><span></span><span></span></div></div></div></div>`).join("");
  try {
    const { results } = await api("api/compare", { question: q });
    $("#compare-cols").innerHTML = results.map(r => `<div class="ccol"><div class="ccol-head" style="background:${MODELS[r.model_key].color}">${esc(r.model)}</div>
      <div class="ccol-body"><div class="answer">${richText(r.answer)}</div>${r.model_key !== "llm" ? sourcesHTML(r) : ""}<div class="meta">${metaHTML(r)}</div></div></div>`).join("");
    const m3 = results.find(r => r.model_key === "llmops");
    if (m3) wireFeedback($$("#compare-cols .ccol")[2], m3);
    loadMetrics();
  } catch (e) {
    $("#compare-cols").innerHTML = `<div class="banner error">⚠ ${esc(e.message)}</div>`;
  } finally { cmpBtn.disabled = false; }
}
cmpForm.addEventListener("submit", e => { e.preventDefault(); runCompare(cmpInput.value); });
cmpInput.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); runCompare(cmpInput.value); } });
$$("#presets .chip").forEach(c => c.addEventListener("click", () => runCompare(c.dataset.q)));

/* ------------------------------------------------------------------ benchmark */
let benchPoll = null, BENCH = null;
async function loadEvaluation() {
  let data;
  try { data = await api("api/evaluation"); } catch (e) { showBenchError(e.message); return; }
  const job = data.job;
  const prog = $("#bench-progress");
  if (job.status === "running") {
    prog.classList.remove("hidden");
    const pct = job.total ? Math.round(100 * job.done / job.total) : 0;
    $(".fill", prog).style.width = pct + "%";
    $(".ptext", prog).textContent = `${pct}% · ${job.done}/${job.total || "?"} · ${job.message || ""}`;
    $("#run-bench").disabled = true;
    clearTimeout(benchPoll); benchPoll = setTimeout(loadEvaluation, 2000);
  } else {
    prog.classList.add("hidden");
    $("#run-bench").disabled = false;
    if (job.status === "error") showBenchError("Benchmark failed: " + job.error);
  }
  if (data.result) renderBenchmark(data.result);
}
function showBenchError(msg) { const b = $("#bench-error"); b.textContent = msg; b.classList.remove("hidden"); }

$("#run-bench").addEventListener("click", async () => {
  $("#bench-error").classList.add("hidden");
  $("#run-bench").disabled = true;
  try { await api("api/evaluation/run", {}); loadEvaluation(); }
  catch (e) { showBenchError(e.message); $("#run-bench").disabled = false; }
});

function renderBenchmark(res) {
  BENCH = res;
  $("#bench-empty").classList.add("hidden");
  $("#bench-body").classList.remove("hidden");
  const m = res.meta;
  $("#bench-meta").innerHTML = `Last run <b>${esc(new Date(m.timestamp).toLocaleString())}</b> · provider <b>${esc(m.provider)}</b> · ${esc(m.llm_model)} · ${esc(m.embeddings)} · top_k=${esc(m.top_k)} · ${esc(m.n_questions)} questions`
    + (m.provider === "offline" ? ` · <span class="flag amber">offline pipeline check (no LLM)</span>` : "");
  $("#bench-legend").innerHTML = res.models.map(n => `<span style="--c:${COLOR_BY_NAME[n] || "#888"}">${esc(n)}</span>`).join("");
  renderBenchChart(res);
  renderMetricsTable(res);
  renderTakeaways(res);
  renderQuestions(res);
}

function renderBenchChart(res) {
  const mets = res.metrics.filter(x => x.unit === "%");
  const W = 640, H = 290, L = 40, R = 10, T = 14, B = 46;
  const gw = (W - L - R) / mets.length, bw = Math.min(30, (gw - 18) / res.models.length);
  const y = v => T + (H - T - B) * (1 - v / 100);
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Benchmark metrics by model">`;
  for (let v = 0; v <= 100; v += 25) {
    s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="#ece9e1"/><text x="${L - 6}" y="${y(v) + 4}" font-size="11" text-anchor="end" fill="#667085">${v}%</text>`;
  }
  mets.forEach((met, gi) => {
    const gx = L + gi * gw + (gw - bw * res.models.length) / 2;
    res.models.forEach((mod, mi) => {
      const v = res.summary[mod][met.key], bx = gx + mi * bw;
      if (v === null || v === undefined) {
        s += `<text x="${bx + bw / 2 - 1}" y="${y(0) - 4}" font-size="9" text-anchor="middle" fill="#98a2b3">n/a</text>`;
      } else {
        s += `<rect class="bar" data-m="${mi}" data-k="${gi}" x="${bx + 1}" y="${y(v)}" width="${bw - 2}" height="${Math.max(0.5, y(0) - y(v))}" rx="3" fill="${COLOR_BY_NAME[mod] || "#888"}"/>`;
      }
    });
    s += `<text x="${L + gi * gw + gw / 2}" y="${H - B + 18}" font-size="11.5" text-anchor="middle" fill="#344054" font-weight="600">${esc(met.label)}</text>`;
  });
  s += `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" stroke="#98a2b3"/>`;
  $("#bench-chart").innerHTML = s + "</svg>";
  $$("#bench-chart rect.bar").forEach(r => {
    const mod = res.models[+r.dataset.m], met = mets[+r.dataset.k];
    r.addEventListener("mousemove", e => showTip(e, `<b>${esc(mod)}</b><br>${esc(met.label)}: ${fmtNum(res.summary[mod][met.key], 1)}%<br><span style="opacity:.8">${esc(met.help || "")}</span>`));
    r.addEventListener("mouseleave", hideTip);
  });
}

function fmtMetric(v, unit) {
  if (v === null || v === undefined) return "n/a";
  return unit === "%" ? v.toFixed(1) + "%" : unit === "s" ? v.toFixed(2) + " s" : Math.round(v).toString();
}
function renderMetricsTable(res) {
  let h = `<tr><th>Metric</th>${res.models.map(n => `<th class="num" style="color:${COLOR_BY_NAME[n]}">${esc(n.replace(/^Model \d: /, ""))}</th>`).join("")}</tr>`;
  res.metrics.forEach(met => {
    const vals = res.models.map(n => res.summary[n][met.key]);
    const valid = vals.filter(v => v !== null && v !== undefined);
    const best = valid.length > 1 && new Set(valid).size > 1 ? (met.higher_is_better ? Math.max(...valid) : Math.min(...valid)) : null;
    h += `<tr><td title="${esc(met.help || "")}">${esc(met.label)}</td>${vals.map(v =>
      `<td class="num ${v !== null && v === best ? "best" : ""}">${fmtMetric(v, met.unit)}</td>`).join("")}</tr>`;
  });
  $("#metrics-table").innerHTML = h;
}

function renderTakeaways(res) {
  const [n1, n2, n3] = res.models, S = res.summary;
  const cards = [];
  const a1 = S[n1].accuracy, a2 = S[n2].accuracy;
  if (a1 !== null && a2 !== null) {
    const d = a2 - a1;
    cards.push(`<div class="big">${d >= 0 ? "+" : ""}${d.toFixed(0)} pts</div><p>RAG ${d >= 0 ? "improved" : "changed"} answer accuracy by <b>${Math.abs(d).toFixed(0)} points</b> over the plain LLM (${a1.toFixed(0)}% → ${a2.toFixed(0)}%), because answers come from the official KB.</p>`);
  }
  const rows3 = res.rows.filter(r => r.model === n3);
  const attacks = rows3.filter(r => r.type === "attack"), blocked = attacks.filter(r => (r.flags || []).includes("prompt_injection")).length;
  const pii = rows3.filter(r => r.type === "pii"), red = pii.filter(r => r.pii_redacted).length;
  cards.push(`<div class="big">${blocked}/${attacks.length} · ${red}/${pii.length}</div><p>LLMOps <b>blocked ${blocked} of ${attacks.length}</b> injection attempts before the LLM and <b>redacted PII</b> in ${red} of ${pii.length} questions. Safe handling: ${fmtMetric(S[n2].safe_handling, "%")} (RAG) → ${fmtMetric(S[n3].safe_handling, "%")} (LLMOps).</p>`);
  const t1 = S[n1].avg_tokens, t2 = S[n2].avg_tokens;
  if (t1 && t2) {
    cards.push(`<div class="big">${(t2 / t1).toFixed(1)}×</div><p>RAG costs <b>${(t2 / t1).toFixed(1)}× more tokens</b> per query than the plain LLM (${Math.round(t1)} → ${Math.round(t2)}) because retrieved context is added to the prompt. Latency: ${fmtMetric(S[n1].avg_latency_s, "s")} → ${fmtMetric(S[n2].avg_latency_s, "s")}.</p>`);
  } else {
    cards.push(`<div class="big">0 tokens</div><p>This run was <b>offline</b> (no LLM), so token cost and latency are not meaningful. Start Ollama and re-run to measure RAG's extra prompt tokens.</p>`);
  }
  $("#takeaways").innerHTML = cards.map(c => `<div class="take">${c}</div>`).join("");
}

function renderQuestions(res) {
  const filter = $("#type-filter").value;
  const byQ = new Map();
  res.rows.forEach(r => { if (!byQ.has(r.id)) byQ.set(r.id, { id: r.id, type: r.type, question: r.question, m: {} }); byQ.get(r.id).m[r.model] = r; });
  const ok = r => { const v = r.correct ?? r.safe; return v === null || v === undefined ? "" : v ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>'; };
  let h = `<tr><th>ID</th><th>Type</th><th>Question</th>${res.models.map(n => `<th style="color:${COLOR_BY_NAME[n]}">${esc(n.replace(/^Model (\d): .*/, "M$1"))}</th>`).join("")}</tr>`;
  [...byQ.values()].filter(q => !filter || q.type === filter).forEach(q => {
    h += `<tr class="clickable" data-id="${esc(q.id)}"><td>${esc(q.id)}</td><td>${esc(q.type.replace(/_/g, " "))}</td><td>${esc(q.question)}</td>${res.models.map(n => `<td>${ok(q.m[n])}</td>`).join("")}</tr>`;
    h += `<tr class="expand hidden" data-for="${esc(q.id)}"><td colspan="${3 + res.models.length}"><div class="answers3">${res.models.map(n =>
      `<div style="border-color:${COLOR_BY_NAME[n]}"><b>${esc(n)}</b><br>${richText(q.m[n].answer)}<div class="muted small">${fmtNum(q.m[n].latency_s)} s · ${q.m[n].tokens} tokens${(q.m[n].flags || []).length ? " · " + esc(q.m[n].flags.join(", ")) : ""}</div></div>`).join("")}</div></td></tr>`;
  });
  $("#questions-table").innerHTML = h;
  $$("#questions-table tr.clickable").forEach(tr => tr.addEventListener("click", () =>
    $(`#questions-table tr[data-for="${tr.dataset.id}"]`).classList.toggle("hidden")));
}
$("#type-filter").addEventListener("change", () => BENCH && renderQuestions(BENCH));

/* ------------------------------------------------------------------ init */
(function init() {
  try {
    const t = localStorage.getItem("helpdesk.tab");
    if (t) $(`.tab[data-page="${t}"]`)?.click();
  } catch (_) { /* ignore */ }
  loadHealth(); loadKB(); doLoadMetrics(); loadEvaluation();
  setInterval(loadHealth, 30000);
})();

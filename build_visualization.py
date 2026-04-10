"""
读取 test.txt 和预测标签，生成可视化 HTML 页面。
用法: python build_visualization.py
"""
import json
import os

TEST_PATH   = "data/test.txt"
PRED_PATH   = "results/test_pred_TAG.txt"
OUTPUT_HTML = "visualization.html"


def parse_entities(chars, tags):
    """将 BIO 标签序列解析为实体列表 [{type, start, end, text}, ...]"""
    entities = []
    start = None
    ent_type = None

    for i, tag in enumerate(tags):
        if tag == "O" or "_" not in tag:
            if start is not None:
                entities.append({
                    "type": ent_type,
                    "start": start,
                    "end": i - 1,
                    "text": "".join(chars[start:i])
                })
                start = None
                ent_type = None
            continue

        prefix, cur_type = tag.split("_", 1)
        if prefix == "B":
            if start is not None:
                entities.append({
                    "type": ent_type,
                    "start": start,
                    "end": i - 1,
                    "text": "".join(chars[start:i])
                })
            start = i
            ent_type = cur_type
        elif prefix == "I":
            if start is not None and ent_type == cur_type:
                continue
            else:
                if start is not None:
                    entities.append({
                        "type": ent_type,
                        "start": start,
                        "end": i - 1,
                        "text": "".join(chars[start:i])
                    })
                start = i
                ent_type = cur_type
        else:
            if start is not None:
                entities.append({
                    "type": ent_type,
                    "start": start,
                    "end": i - 1,
                    "text": "".join(chars[start:i])
                })
                start = None
                ent_type = None

    if start is not None:
        entities.append({
            "type": ent_type,
            "start": start,
            "end": len(tags) - 1,
            "text": "".join(chars[start:len(tags)])
        })

    return entities


def build_data():
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_lines = f.readlines()
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        pred_lines = f.readlines()

    assert len(test_lines) == len(pred_lines), \
        f"行数不匹配: test={len(test_lines)}, pred={len(pred_lines)}"

    data = []
    # 统计
    total_entities = {"LOC": 0, "ORG": 0, "PER": 0, "T": 0}

    for idx, (tline, pline) in enumerate(zip(test_lines, pred_lines)):
        chars = tline.strip().split()
        tags  = pline.strip().split()

        if len(chars) != len(tags):
            # 跳过不匹配的行
            continue

        entities = parse_entities(chars, tags)
        for e in entities:
            if e["type"] in total_entities:
                total_entities[e["type"]] += 1

        # 只保留渲染必需字段，减小 HTML 体积
        # 将文本拼接为字符串，entities 中用 start/end 索引
        data.append({
            "id": idx + 1,
            "t": "".join(chars),          # 原始文本（紧凑）
            "e": [[e["type"], e["start"], e["end"]] for e in entities],
        })

    stats = {
        "total_samples": len(data),
        "entity_counts": total_entities,
        "total_entities": sum(total_entities.values()),
    }

    return data, stats


def main():
    print("📊 正在构建可视化数据...")
    data, stats = build_data()
    print(f"   样本数: {stats['total_samples']}")
    print(f"   实体总数: {stats['total_entities']}")
    for k, v in stats['entity_counts'].items():
        print(f"     {k}: {v}")

    # 将数据序列化为紧凑 JSON
    data_json  = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    stats_json = json.dumps(stats, ensure_ascii=False, separators=(',', ':'))

    html = HTML_TEMPLATE.replace("/*__DATA_PLACEHOLDER__*/", f"const DATA={data_json};const STATS={stats_json};")

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = os.path.getsize(OUTPUT_HTML) / 1024 / 1024
    print(f"✅ 已生成: {OUTPUT_HTML} ({size_mb:.1f} MB)")


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NER Prediction Visualizer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#f9f5ef;--surface:#fff;--surface-alt:#f5f1eb;
  --text:#1a1a1a;--text-secondary:#6b6560;--text-tertiary:#9b9590;
  --border:#e5ddd5;--border-light:#efe9e1;
  --accent:#c96442;--accent-light:#fdf0eb;
  --loc:#3b82f6;--loc-bg:#eff6ff;--loc-border:#bfdbfe;
  --org:#8b5cf6;--org-bg:#f5f3ff;--org-border:#c4b5fd;
  --per:#059669;--per-bg:#ecfdf5;--per-border:#a7f3d0;
  --time:#d97706;--time-bg:#fffbeb;--time-border:#fcd34d;
  --radius:10px;--radius-sm:6px;
  --shadow:0 1px 3px rgba(0,0,0,.06),0 1px 2px rgba(0,0,0,.04);
  --shadow-lg:0 4px 16px rgba(0,0,0,.08);
  --font:'Söhne',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
  --mono:'Söhne Mono','Fira Code','Cascadia Code',monospace;
}
body{font-family:var(--font);background:var(--bg);color:var(--text);line-height:1.6;min-height:100vh}
::selection{background:var(--accent);color:#fff}

/* ─── Header ─── */
.header{background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:0;z-index:100;backdrop-filter:blur(12px)}
.header-inner{max-width:1200px;margin:0 auto;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}
.logo{display:flex;align-items:center;gap:12px}
.logo-icon{width:36px;height:36px;background:var(--accent);border-radius:8px;display:flex;align-items:center;justify-content:center}
.logo-icon svg{width:22px;height:22px;fill:none;stroke:#fff;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.logo h1{font-size:18px;font-weight:600;letter-spacing:-.01em}
.logo p{font-size:12px;color:var(--text-tertiary);font-weight:400}

/* ─── Stats Bar ─── */
.stats-bar{max-width:1200px;margin:20px auto 0;padding:0 24px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.stat-card{background:var(--surface);border:1px solid var(--border-light);border-radius:var(--radius);padding:16px 18px;transition:box-shadow .2s}
.stat-card:hover{box-shadow:var(--shadow-lg)}
.stat-label{font-size:12px;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:.04em;font-weight:500;margin-bottom:4px}
.stat-value{font-size:24px;font-weight:600;font-family:var(--mono);letter-spacing:-.02em}
.stat-value.loc{color:var(--loc)}.stat-value.org{color:var(--org)}
.stat-value.per{color:var(--per)}.stat-value.time{color:var(--time)}
.stat-value.total{color:var(--accent)}

/* ─── Controls ─── */
.controls{max-width:1200px;margin:20px auto 0;padding:0 24px}
.controls-inner{background:var(--surface);border:1px solid var(--border-light);border-radius:var(--radius);padding:14px 18px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.search-box{flex:1;min-width:200px;position:relative}
.search-box input{width:100%;padding:9px 14px 9px 38px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:14px;font-family:var(--font);background:var(--surface-alt);transition:border-color .2s,box-shadow .2s;outline:none}
.search-box input:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-light)}
.search-box input::placeholder{color:var(--text-tertiary)}
.search-box svg{position:absolute;left:12px;top:50%;transform:translateY(-50%);width:16px;height:16px;stroke:var(--text-tertiary);fill:none;stroke-width:2}
.filter-group{display:flex;gap:6px;flex-wrap:wrap}
.filter-btn{padding:7px 14px;border:1px solid var(--border);border-radius:20px;background:var(--surface);font-size:13px;font-family:var(--font);cursor:pointer;transition:all .2s;font-weight:500;color:var(--text-secondary)}
.filter-btn:hover{border-color:var(--text-tertiary)}
.filter-btn.active{color:#fff;border-color:transparent}
.filter-btn.active[data-type="ALL"]{background:var(--accent);border-color:var(--accent)}
.filter-btn.active[data-type="LOC"]{background:var(--loc);border-color:var(--loc)}
.filter-btn.active[data-type="ORG"]{background:var(--org);border-color:var(--org)}
.filter-btn.active[data-type="PER"]{background:var(--per);border-color:var(--per)}
.filter-btn.active[data-type="T"]{background:var(--time);border-color:var(--time)}
.per-page-select{padding:7px 10px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:13px;font-family:var(--font);background:var(--surface-alt);cursor:pointer;outline:none;color:var(--text)}

/* ─── Main Content ─── */
.main{max-width:1200px;margin:16px auto 0;padding:0 24px 60px}
.result-info{font-size:13px;color:var(--text-tertiary);margin-bottom:12px;padding:0 4px;display:flex;justify-content:space-between;align-items:center}

/* ─── Sentence Card ─── */
.card{background:var(--surface);border:1px solid var(--border-light);border-radius:var(--radius);margin-bottom:12px;overflow:hidden;transition:box-shadow .2s}
.card:hover{box-shadow:var(--shadow)}
.card-header{display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-bottom:1px solid var(--border-light);background:var(--surface-alt)}
.card-id{font-family:var(--mono);font-size:12px;color:var(--text-tertiary);font-weight:500;background:var(--bg);padding:3px 10px;border-radius:12px}
.card-meta{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.entity-badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:10px;letter-spacing:.02em}
.entity-badge.loc{background:var(--loc-bg);color:var(--loc);border:1px solid var(--loc-border)}
.entity-badge.org{background:var(--org-bg);color:var(--org);border:1px solid var(--org-border)}
.entity-badge.per{background:var(--per-bg);color:var(--per);border:1px solid var(--per-border)}
.entity-badge.t{background:var(--time-bg);color:var(--time);border:1px solid var(--time-border)}
.card-body{padding:16px 18px;line-height:2}
.card-body .char{display:inline;font-size:15px}

/* ─── Entity Highlight ─── */
.entity-span{padding:2px 1px;border-radius:3px;position:relative;cursor:default;border-bottom:2px solid transparent;transition:filter .15s}
.entity-span:hover{filter:brightness(.93)}
.entity-span.loc{background:var(--loc-bg);border-bottom-color:var(--loc)}
.entity-span.org{background:var(--org-bg);border-bottom-color:var(--org)}
.entity-span.per{background:var(--per-bg);border-bottom-color:var(--per)}
.entity-span.t{background:var(--time-bg);border-bottom-color:var(--time)}
.entity-label{font-size:10px;font-weight:700;vertical-align:super;margin-left:1px;letter-spacing:.03em;opacity:.8}
.entity-label.loc{color:var(--loc)}.entity-label.org{color:var(--org)}
.entity-label.per{color:var(--per)}.entity-label.t{color:var(--time)}

/* ─── Tooltip ─── */
.tooltip{position:fixed;background:#1a1a1a;color:#fff;padding:6px 12px;border-radius:6px;font-size:12px;pointer-events:none;z-index:999;white-space:nowrap;opacity:0;transition:opacity .15s;box-shadow:0 4px 12px rgba(0,0,0,.2)}
.tooltip.show{opacity:1}

/* ─── Pagination ─── */
.pagination{display:flex;align-items:center;justify-content:center;gap:4px;margin-top:24px;flex-wrap:wrap}
.page-btn{min-width:36px;height:36px;display:flex;align-items:center;justify-content:center;border:1px solid var(--border);border-radius:var(--radius-sm);background:var(--surface);font-size:13px;font-family:var(--mono);cursor:pointer;transition:all .15s;color:var(--text-secondary)}
.page-btn:hover:not(.active):not(:disabled){border-color:var(--accent);color:var(--accent)}
.page-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.page-btn:disabled{opacity:.3;cursor:default}
.page-btn.nav{font-family:var(--font);font-weight:500;padding:0 12px}
.page-ellipsis{width:36px;height:36px;display:flex;align-items:center;justify-content:center;color:var(--text-tertiary);font-family:var(--mono);font-size:14px}

/* ─── Jump To ─── */
.jump-to{display:flex;align-items:center;gap:6px;margin-top:12px;justify-content:center;font-size:13px;color:var(--text-tertiary)}
.jump-to input{width:70px;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:13px;font-family:var(--mono);text-align:center;outline:none;background:var(--surface)}
.jump-to input:focus{border-color:var(--accent)}
.jump-to button{padding:6px 14px;background:var(--accent);color:#fff;border:none;border-radius:var(--radius-sm);font-size:13px;font-family:var(--font);cursor:pointer;font-weight:500;transition:background .15s}
.jump-to button:hover{background:#b5573a}

/* ─── No Results ─── */
.no-results{text-align:center;padding:60px 20px;color:var(--text-tertiary)}
.no-results svg{width:48px;height:48px;stroke:var(--border);fill:none;stroke-width:1.5;margin-bottom:12px}
.no-results p{font-size:15px;margin-top:8px}

/* ─── Loading ─── */
.loading{text-align:center;padding:80px 20px;color:var(--text-tertiary);font-size:15px}

/* ─── Legend ─── */
.legend{max-width:1200px;margin:16px auto 0;padding:0 24px}
.legend-inner{background:var(--surface);border:1px solid var(--border-light);border-radius:var(--radius);padding:12px 18px;display:flex;align-items:center;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--text-secondary)}
.legend-item{display:flex;align-items:center;gap:6px}
.legend-dot{width:10px;height:10px;border-radius:3px}
.legend-dot.loc{background:var(--loc)}.legend-dot.org{background:var(--org)}
.legend-dot.per{background:var(--per)}.legend-dot.t{background:var(--time)}

/* ─── Responsive ─── */
@media(max-width:640px){
  .header-inner{padding:12px 16px}
  .stats-bar,.controls,.main,.legend{padding:0 12px}
  .stats-grid{grid-template-columns:repeat(2,1fr)}
  .controls-inner{padding:10px 12px}
  .card-body{padding:12px 14px;line-height:1.9}
  .filter-group{width:100%}
}
</style>
</head>
<body>

<div class="tooltip" id="tooltip"></div>

<header class="header">
  <div class="header-inner">
    <div class="logo">
      <div class="logo-icon">
        <svg viewBox="0 0 24 24"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2"/><path d="M9 5a2 2 0 012-2h2a2 2 0 012 2v0a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/><line x1="9" y1="12" x2="15" y2="12"/><line x1="9" y1="16" x2="13" y2="16"/></svg>
      </div>
      <div>
        <h1>NER Prediction Visualizer</h1>
        <p>Transformer Model · Test Set Results</p>
      </div>
    </div>
  </div>
</header>

<section class="stats-bar">
  <div class="stats-grid" id="statsGrid"></div>
</section>

<section class="legend">
  <div class="legend-inner">
    <span style="font-weight:600;color:var(--text)">图例:</span>
    <div class="legend-item"><div class="legend-dot loc"></div>LOC 地名</div>
    <div class="legend-item"><div class="legend-dot org"></div>ORG 组织</div>
    <div class="legend-item"><div class="legend-dot per"></div>PER 人名</div>
    <div class="legend-item"><div class="legend-dot t"></div>T 时间</div>
  </div>
</section>

<section class="controls">
  <div class="controls-inner">
    <div class="search-box">
      <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <input type="text" id="searchInput" placeholder="搜索文本内容或实体 (如: 北京、合肥)..." />
    </div>
    <div class="filter-group" id="filterGroup">
      <button class="filter-btn active" data-type="ALL">全部</button>
      <button class="filter-btn" data-type="LOC">📍 LOC</button>
      <button class="filter-btn" data-type="ORG">🏢 ORG</button>
      <button class="filter-btn" data-type="PER">👤 PER</button>
      <button class="filter-btn" data-type="T">🕐 T</button>
    </div>
    <select class="per-page-select" id="perPageSelect">
      <option value="20">20 条/页</option>
      <option value="50" selected>50 条/页</option>
      <option value="100">100 条/页</option>
      <option value="200">200 条/页</option>
    </select>
  </div>
</section>

<main class="main">
  <div class="result-info" id="resultInfo"></div>
  <div id="cardContainer"></div>
  <div class="pagination" id="pagination"></div>
  <div class="jump-to" id="jumpTo">
    <span>跳转到第</span>
    <input type="number" id="jumpInput" min="1" />
    <span>页</span>
    <button onclick="jumpToPage()">Go</button>
  </div>
</main>

<script>
/*__DATA_PLACEHOLDER__*/

// ─── State ───
let currentPage = 1;
let perPage = 50;
let filterType = "ALL";
let searchText = "";
let filteredData = DATA;
let debounceTimer = null;

// ─── Init ───
function init() {
  renderStats();
  bindEvents();
  applyFilters();
}

function renderStats() {
  const g = document.getElementById("statsGrid");
  g.innerHTML = `
    <div class="stat-card"><div class="stat-label">测试样本</div><div class="stat-value total">${STATS.total_samples.toLocaleString()}</div></div>
    <div class="stat-card"><div class="stat-label">实体总数</div><div class="stat-value total">${STATS.total_entities.toLocaleString()}</div></div>
    <div class="stat-card"><div class="stat-label">📍 LOC 地名</div><div class="stat-value loc">${STATS.entity_counts.LOC.toLocaleString()}</div></div>
    <div class="stat-card"><div class="stat-label">🏢 ORG 组织</div><div class="stat-value org">${STATS.entity_counts.ORG.toLocaleString()}</div></div>
    <div class="stat-card"><div class="stat-label">👤 PER 人名</div><div class="stat-value per">${STATS.entity_counts.PER.toLocaleString()}</div></div>
    <div class="stat-card"><div class="stat-label">🕐 T 时间</div><div class="stat-value time">${STATS.entity_counts.T.toLocaleString()}</div></div>
  `;
}

function bindEvents() {
  document.getElementById("searchInput").addEventListener("input", e => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      searchText = e.target.value.trim();
      currentPage = 1;
      applyFilters();
    }, 250);
  });

  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      filterType = btn.dataset.type;
      currentPage = 1;
      applyFilters();
    });
  });

  document.getElementById("perPageSelect").addEventListener("change", e => {
    perPage = parseInt(e.target.value);
    currentPage = 1;
    applyFilters();
  });

  document.getElementById("jumpInput").addEventListener("keydown", e => {
    if (e.key === "Enter") jumpToPage();
  });

  // keyboard nav
  document.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    const totalPages = Math.ceil(filteredData.length / perPage);
    if (e.key === "ArrowLeft" && currentPage > 1) { currentPage--; renderPage(); }
    if (e.key === "ArrowRight" && currentPage < totalPages) { currentPage++; renderPage(); }
  });
}

function applyFilters() {
  filteredData = DATA.filter(item => {
    // type filter
    if (filterType !== "ALL") {
      if (!item.e.some(e => e[0] === filterType)) return false;
    }
    // search filter
    if (searchText) {
      const q = searchText.toLowerCase();
      if (!item.t.toLowerCase().includes(q)) return false;
    }
    return true;
  });

  renderPage();
}

function renderPage() {
  const container = document.getElementById("cardContainer");
  const totalPages = Math.ceil(filteredData.length / perPage) || 1;
  if (currentPage > totalPages) currentPage = totalPages;

  const start = (currentPage - 1) * perPage;
  const end = Math.min(start + perPage, filteredData.length);
  const pageData = filteredData.slice(start, end);

  // result info
  document.getElementById("resultInfo").innerHTML = filteredData.length === DATA.length
    ? `<span>共 <strong>${filteredData.length.toLocaleString()}</strong> 条样本</span><span>第 ${currentPage} / ${totalPages} 页 · 显示 ${start + 1}-${end}</span>`
    : `<span>筛选结果: <strong>${filteredData.length.toLocaleString()}</strong> 条 (共 ${DATA.length.toLocaleString()} 条)</span><span>第 ${currentPage} / ${totalPages} 页</span>`;

  // cards
  if (pageData.length === 0) {
    container.innerHTML = `<div class="no-results"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="8" y1="15" x2="16" y2="15"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg><p>没有匹配的结果</p></div>`;
  } else {
    const frag = document.createDocumentFragment();
    for (const item of pageData) {
      frag.appendChild(buildCard(item));
    }
    container.innerHTML = "";
    container.appendChild(frag);
  }

  // pagination
  renderPagination(totalPages);
  document.getElementById("jumpInput").max = totalPages;

  // scroll to top
  window.scrollTo({ top: document.querySelector(".main").offsetTop - 10, behavior: "smooth" });
}

function buildCard(item) {
  const card = document.createElement("div");
  card.className = "card";
  const chars = [...item.t]; // split string into chars (handles unicode)
  const entities = item.e.map(e => ({type: e[0], start: e[1], end: e[2], text: chars.slice(e[1], e[2]+1).join("")}));

  // header
  const header = document.createElement("div");
  header.className = "card-header";

  const idSpan = document.createElement("span");
  idSpan.className = "card-id";
  idSpan.textContent = `# ${item.id}`;
  header.appendChild(idSpan);

  if (entities.length > 0) {
    const meta = document.createElement("div");
    meta.className = "card-meta";
    const counts = {};
    entities.forEach(e => { counts[e.type] = (counts[e.type] || 0) + 1; });
    for (const [type, count] of Object.entries(counts)) {
      const badge = document.createElement("span");
      badge.className = `entity-badge ${type.toLowerCase()}`;
      badge.textContent = `${type} ×${count}`;
      meta.appendChild(badge);
    }
    header.appendChild(meta);
  }

  card.appendChild(header);

  // body
  const body = document.createElement("div");
  body.className = "card-body";

  const entityMap = {};
  for (const e of entities) {
    for (let i = e.start; i <= e.end; i++) {
      entityMap[i] = { type: e.type, text: e.text };
    }
  }

  let i = 0;
  while (i < chars.length) {
    if (entityMap[i]) {
      const info = entityMap[i];
      const span = document.createElement("span");
      const cls = info.type.toLowerCase();
      span.className = `entity-span ${cls}`;

      let text = "";
      while (i < chars.length && entityMap[i] && entityMap[i].text === info.text && entityMap[i].type === info.type) {
        text += chars[i];
        i++;
      }
      span.textContent = text;

      const label = document.createElement("sup");
      label.className = `entity-label ${cls}`;
      label.textContent = info.type;
      span.appendChild(label);

      span.addEventListener("mouseenter", e => showTooltip(e, `${info.type}: ${text}`));
      span.addEventListener("mouseleave", hideTooltip);

      body.appendChild(span);
    } else {
      body.appendChild(document.createTextNode(chars[i]));
      i++;
    }
  }

  card.appendChild(body);
  return card;
}

// ─── Tooltip ───
const tooltip = document.getElementById("tooltip");
function showTooltip(e, text) {
  tooltip.textContent = text;
  tooltip.classList.add("show");
  const rect = e.target.getBoundingClientRect();
  tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + "px";
  tooltip.style.top = rect.top - tooltip.offsetHeight - 8 + "px";
}
function hideTooltip() {
  tooltip.classList.remove("show");
}

// ─── Pagination ───
function renderPagination(totalPages) {
  const pag = document.getElementById("pagination");
  if (totalPages <= 1) { pag.innerHTML = ""; return; }

  let html = `<button class="page-btn nav" onclick="goPage(${currentPage - 1})" ${currentPage === 1 ? "disabled" : ""}>‹ 上一页</button>`;

  const pages = getPageNumbers(currentPage, totalPages);
  for (const p of pages) {
    if (p === "...") {
      html += `<span class="page-ellipsis">…</span>`;
    } else {
      html += `<button class="page-btn ${p === currentPage ? "active" : ""}" onclick="goPage(${p})">${p}</button>`;
    }
  }

  html += `<button class="page-btn nav" onclick="goPage(${currentPage + 1})" ${currentPage === totalPages ? "disabled" : ""}>下一页 ›</button>`;
  pag.innerHTML = html;
}

function getPageNumbers(current, total) {
  if (total <= 9) return Array.from({ length: total }, (_, i) => i + 1);
  const pages = [];
  pages.push(1);
  if (current > 4) pages.push("...");
  const start = Math.max(2, current - 2);
  const end = Math.min(total - 1, current + 2);
  for (let i = start; i <= end; i++) pages.push(i);
  if (current < total - 3) pages.push("...");
  pages.push(total);
  return pages;
}

function goPage(p) {
  const totalPages = Math.ceil(filteredData.length / perPage);
  if (p < 1 || p > totalPages) return;
  currentPage = p;
  renderPage();
}

function jumpToPage() {
  const val = parseInt(document.getElementById("jumpInput").value);
  if (!isNaN(val)) goPage(val);
}

// ─── Start ───
init();
</script>
</body>
</html>
''';

main()

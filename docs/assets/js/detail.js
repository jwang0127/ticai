const gameKey = document.body.dataset.game;
const $ = selector => document.querySelector(selector);

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function numberText(item) {
  if (gameKey !== "dlt") return item.number;
  return `${item.front.map(n => String(n).padStart(2, "0")).join(" ")} + ${item.back.map(n => String(n).padStart(2, "0")).join(" ")}`;
}

function latestText(numbers) {
  return gameKey === "dlt" ? `${numbers.slice(0, 5).join(" ")} + ${numbers.slice(5).join(" ")}` : numbers.join("");
}

function picksHtml(candidates) {
  return `<div class="picks">${candidates.slice(0, 5).map(item => `
    <article class="pick">
      <div class="pick-rank">TOP 0${item.rank}</div>
      <div class="pick-number">${escapeHtml(numberText(item))}</div>
      ${item.mix_label ? `<div class="mix-label">${escapeHtml(item.mix_label)}</div>` : ""}
      <div class="score">模型相对评分 ${item.confidence}%</div>
    </article>`).join("")}</div>`;
}

function modelReviewHtml(review) {
  if (!review) return "";
  const calibration = review.calibration_hits?.length
    ? `<div class="review-callout">反热校准池：${escapeHtml(review.calibration_candidates.join("、"))}；命中 ${escapeHtml(review.calibration_hits.join("、"))}。单次命中不代表冷号更易开出。</div>`
    : "";
  return `<section class="section">
    <div class="section-head"><div><p class="section-label">MODEL REVIEW</p><h2>第${escapeHtml(review.issue)}期模型复盘</h2></div><p class="section-note">${escapeHtml(review.summary)}</p></div>
    <div class="metrics model-review-metrics">
      <div class="metric"><span>开奖号</span><strong>${escapeHtml(review.actual)}</strong></div>
      <div class="metric"><span>原模型直选命中</span><strong>${escapeHtml(review.exact_hits)} / ${escapeHtml(review.previous_candidates.length)}</strong></div>
      <div class="metric wide"><span>原候选</span><strong>${escapeHtml(review.previous_candidates.join(" · "))}</strong></div>
    </div>
    ${calibration}
    <p class="review-lesson">本期修正：${escapeHtml(review.lesson)}</p>
  </section>`;
}

function strategyZonesHtml(game) {
  if (!game.strategy_zones) return "";
  return `<section class="section strategy-zones">
    <div class="section-head"><div><p class="section-label">HOT / COLD ZONES</p><h2>热门与冷门分区</h2></div><p class="section-note">所有分区使用同一全局评分尺度；最高评分榜会与热门区自然重合，冷门区只作低热覆盖。</p></div>
    <div class="zone-grid">${Object.values(game.strategy_zones).map(zone => `
      <div class="zone-card ${zone.name.includes("冷门") ? "cold-zone" : "hot-zone"}">
        <div class="play-title"><h3>${escapeHtml(zone.name)}</h3><span>${escapeHtml(zone.description)}</span></div>
        ${picksHtml(zone.candidates)}
        <button class="copy-bundle" data-copy="${encodeURIComponent(bundleText(game, zone.name, zone.candidates))}">复制${escapeHtml(zone.name)}全部5组</button>
      </div>`).join("")}</div>
  </section>`;
}

function bundleText(game, label, candidates) {
  const name = `${game.name}${label ? ` ${label}` : ""}`;
  return candidates.slice(0, 5)
    .map(item => `${name} ${numberText(item)}`)
    .join("\n");
}

function playTypesHtml(game) {
  if (!game.play_types) {
    return `${picksHtml(game.top_candidates)}
      <button class="copy-bundle" data-copy="${encodeURIComponent(bundleText(game, "", game.top_candidates))}">复制全部5组结果</button>`;
  }
  return Object.values(game.play_types).map(play => `
    <div class="play-block">
      <div class="play-title"><h3>${escapeHtml(play.name)}</h3><span>${escapeHtml(play.description)}</span></div>
      ${picksHtml(play.candidates)}
      <button class="copy-bundle" data-copy="${encodeURIComponent(bundleText(game, play.name, play.candidates))}">复制${escapeHtml(play.name)}全部5组</button>
    </div>`).join("");
}

let toastTimer;
async function copyText(text) {
  try { await navigator.clipboard.writeText(text); }
  catch (_) {
    const area = document.createElement("textarea");
    area.value = text; area.style.position = "fixed"; area.style.opacity = "0";
    document.body.append(area); area.select(); document.execCommand("copy"); area.remove();
  }
  const toast = $("#toast");
  toast.textContent = "全部5组已复制"; toast.classList.add("show");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove("show"), 1600);
}

async function load() {
  const response = await fetch("../assets/data/dashboard.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const payload = await response.json();
  const game = payload.games[gameKey];
  if (!game) throw new Error("未找到玩法数据");
  const generated = new Date(game.generated_at || payload.generated_at).toLocaleString("zh-CN", { hour12: false });

  $("#app").innerHTML = `<div class="shell">
    <nav class="topbar"><div class="game-nav"><a href="../">首页</a><a href="../dlt/">超级大乐透</a><a href="../pl3/">排列3</a><a href="../pl5/">排列5</a><a href="../fc3d/">福彩3D</a></div><span class="updated">UPDATED ${escapeHtml(generated)}</span></nav>
    <header class="hero">
      <div><p class="eyebrow">LOTTERY DETAIL / ${escapeHtml(gameKey.toUpperCase())}</p><h1>${escapeHtml(game.name)}</h1></div>
      <div class="hero-meta"><div>第 ${escapeHtml(game.target_issue)} 期 · 展示5组候选</div><div class="next">${escapeHtml(game.next_draw_display)}</div><div>${escapeHtml(game.schedule_note)}</div><div class="latest">上期 ${escapeHtml(game.latest_issue)}｜${escapeHtml(latestText(game.latest_numbers))}</div></div>
    </header>
    <section class="section">
      <div class="section-head"><div><p class="section-label">TOP CANDIDATES</p><h2>最高评分结果</h2></div><p class="section-note">评分仅用于本页候选内部排序。三位数字玩法按直选、组选3、组选6分别计算。</p></div>
      ${playTypesHtml(game)}
    </section>
    ${strategyZonesHtml(game)}
    <section class="section">
      <div class="section-head"><div><p class="section-label">LAST DRAW REVIEW</p><h2>${escapeHtml(game.review.title)}</h2></div><p class="section-note">${escapeHtml(game.review.summary)}</p></div>
      <div class="metrics">${game.review.metrics.map(item => `<div class="metric"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join("")}</div>
    </section>
    ${modelReviewHtml(game.model_review)}
    <section class="section analysis-grid">
      <div><p class="section-label">MODEL ANALYSIS / ${game.analysis.sample} DRAWS</p><h2>本期分析</h2><p class="analysis-summary">${escapeHtml(game.analysis.summary)}</p></div>
      <div><div class="signals">${game.analysis.signals.map(item => `<div class="signal"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join("")}</div><div class="methods">${game.analysis.method.map(item => `<span>${escapeHtml(item)}</span>`).join("")}</div></div>
    </section>
    <div class="disclaimer">${escapeHtml(payload.disclaimer)}</div>
  </div>`;
  document.addEventListener("click", event => {
    const button = event.target.closest("[data-copy]");
    if (button) copyText(decodeURIComponent(button.dataset.copy));
  });
}

load().catch(error => { $("#app").innerHTML = `<p class="loading">详情页加载失败：${escapeHtml(error.message)}</p>`; });

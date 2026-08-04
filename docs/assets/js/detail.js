const gameKey = document.body.dataset.game;
const $ = selector => document.querySelector(selector);

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = value;
  return node.innerHTML;
}

function numberText(item) {
  if (gameKey === "kl8") return item.numbers.map(n => String(n).padStart(2, "0")).join(" ");
  if (gameKey === "ssq") return `${item.red.map(n => String(n).padStart(2, "0")).join(" ")} + ${item.blue.map(n => String(n).padStart(2, "0")).join(" ")}`;
  if (gameKey !== "dlt") return item.number;
  return `${item.front.map(n => String(n).padStart(2, "0")).join(" ")} + ${item.back.map(n => String(n).padStart(2, "0")).join(" ")}`;
}

function latestText(numbers) {
  if (gameKey === "dlt") return `${numbers.slice(0, 5).join(" ")} + ${numbers.slice(5).join(" ")}`;
  if (gameKey === "ssq") return `${numbers.slice(0, 6).join(" ")} + ${numbers.slice(6).join(" ")}`;
  if (gameKey === "kl8") return numbers.join(" ");
  return numbers.join("");
}

function predictionStructureHtml(game) {
  if (!['pl3', 'fc3d'].includes(gameKey)) return "";
  const summary = game.prediction_summary || {};
  const summaryCard = (key, label, mode) => {
    const item = summary[key] || {};
    const values = mode === "hot" ? item.frequencies || [] : item.cold_frequencies || [];
    const text = values.map(value => `${value.value}（${value.count}期）`).join(" · ");
    const reason = mode === "hot" ? item.forecast_reason : item.cold_forecast_reason;
    return `<article class="forecast-card"><span>${label}${mode === "hot" ? "热门" : "冷门"}预测</span><strong>${escapeHtml(values.map(value => value.value).join(" · "))}</strong><small>近300期出现次数：${escapeHtml(text)}</small><small>${escapeHtml(reason || "历史频次结构参考")}</small></article>`;
  };
  const zone = (mode, title, candidates) => `<div class="structure-zone ${mode}-zone"><div class="zone-title"><h3>${title}</h3><span>和值 · 跨度 · 奇偶比</span></div><div class="forecast-grid">${summaryCard("sum", "和值", mode)}${summaryCard("span", "跨度", mode)}${summaryCard("odd_even", "奇偶比", mode)}</div>${candidates ? `<div class="structure-grid">${candidates.map(item => {
    const metrics = item.prediction_metrics || {};
    return `<article class="structure-card"><strong>${escapeHtml(item.number)}</strong><span>和值 ${escapeHtml(metrics.sum)} · 跨度 ${escapeHtml(metrics.span)}</span><small>奇偶 ${escapeHtml(metrics.odd_even)} · 不同 ${escapeHtml(metrics.distinct)} · ${escapeHtml(metrics.shape)}</small><small>${escapeHtml(item.prediction_reason || "逐位排序参考")}</small></article>`;
  }).join("")}</div>` : ""}</div>`;
  return `<section class="section prediction-structure"><div class="section-head"><div><p class="section-label">PREDICTION STRUCTURE</p><h2>和值 · 跨度 · 奇偶比</h2></div></div>${zone("hot", "热门结构", game.hot_candidates || game.top_candidates)}${zone("cold", "冷门结构", game.cold_candidates || [])}</section>`;
}


function picksHtml(candidates) {
  return `<div class="picks">${candidates.map(item => `
    <article class="pick">
      <div class="pick-rank">TOP 0${item.rank}</div>
      <div class="pick-number">${escapeHtml(numberText(item))}</div>
      ${item.purchase_suggestion ? `<div class="purchase-suggestion">${escapeHtml(item.purchase_suggestion)}</div>` : ""}
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
    <div class="section-head"><div><p class="section-label">MODEL REVIEW</p><h2>第${escapeHtml(review.issue)}期模型复盘</h2></div></div>
    <div class="metrics model-review-metrics">
      <div class="metric"><span>开奖号</span><strong>${escapeHtml(review.actual)}</strong></div>
      <div class="metric"><span>原模型直选命中</span><strong>${escapeHtml(review.exact_hits)} / ${escapeHtml(review.previous_candidates.length)}</strong></div>
      <div class="metric wide"><span>原候选</span><strong>${escapeHtml(review.previous_candidates.join(" · "))}</strong></div>
    </div>
    ${calibration}
    <p class="review-lesson">本期修正：${escapeHtml(review.lesson)}</p>
  </section>`;
}

function legacyModelReviewHtml(review) {
  if (!review) return "";
  const positions = review.position_pool_coverage == null ? "" : `<div class="metric"><span>按位候选覆盖</span><strong>${review.position_pool_coverage} / ${review.position_count}</strong></div>`;
  return `<section class="section"><div class="section-head"><div><p class="section-label">MODEL REVIEW</p><h2>第${escapeHtml(review.issue)}期模型复盘</h2></div></div><div class="metrics model-review-metrics"><div class="metric"><span>直选完整命中</span><strong>${review.exact_hits} / ${review.previous_candidates.length}</strong></div>${positions}<div class="metric wide"><span>原候选</span><strong>${escapeHtml(review.previous_candidates.join(" 路 "))}</strong></div></div><p class="review-lesson">${escapeHtml(review.lesson)}</p></section>`;
}


function modelReviewHtml(review) {
  if (!review) return "";
  const diagnostics = (review.position_diagnostics || []).map(item => `
    <div class="review-diagnostic">
      <strong>${escapeHtml(item.position)} ${escapeHtml(item.actual_digit)}</strong>
      <span>${item.pool_hit ? "命中" : "未命中"} · 本位候选覆盖 ${escapeHtml(item.candidate_hit_count)} 组</span>
      <p>${escapeHtml(item.reason)}</p>
    </div>`).join("");
  const advice = (review.next_day_advice || []).map(item => `<span class="advice-chip">${escapeHtml(item.suggestion)}</span>`).join("");
  const adjustments = (review.model_adjustments || []).map(item => `<li>${escapeHtml(item)}</li>`).join("");
  const attribution = review.error_attribution ? `<div class="review-attribution"><h3>错误归因</h3><p>${escapeHtml(review.error_attribution)}</p></div>` : "";
  const structure = review.structure_diagnostics ? `<div class="review-attribution"><h3>和值/跨度逻辑</h3><p>${escapeHtml(review.structure_diagnostics.explanation)}</p></div>` : "";
  return `<section class="section">
    <div class="section-head"><div><p class="section-label">MODEL REVIEW</p><h2>第${escapeHtml(review.issue)}期模型复盘</h2></div></div>
    <div class="metrics model-review-metrics">
      <div class="metric"><span>按位候选覆盖</span><strong>${review.position_pool_coverage == null ? "-" : `${review.position_pool_coverage} / ${review.position_count}`}</strong></div>
      <div class="metric"><span>最佳组合命中</span><strong>${escapeHtml(review.best_number_hits)}</strong></div>
    </div>
    ${diagnostics ? `<div class="review-diagnostics">${diagnostics}</div>` : ""}
    ${structure}
    ${review.number_pool_coverage ? `<div class="review-attribution"><strong>快乐8联合池覆盖 ${escapeHtml(review.number_pool_coverage)}</strong></div>` : ""}
    ${attribution}
    ${advice ? `<div class="next-day-advice"><h3>第二天购买建议</h3><div>${advice}</div></div>` : ""}
    ${adjustments ? `<div class="review-adjustments"><h3>模型调整</h3><ul>${adjustments}</ul></div>` : ""}
    <p class="review-lesson">${escapeHtml(review.lesson)}</p>
  </section>`;
}


function positionAnalysisHtml(analysis, mode = null) {
  if (!analysis.position_analysis) return "";
  const renderDigits = (items, focus, className) => (items || []).map(item => {
    const digit = item.digit ?? item;
    const count = item.count == null ? "" : `<small>${escapeHtml(item.count)}次</small>`;
    const focused = (focus || []).map(String).includes(String(digit)) ? " focus" : "";
    return `<span class="position-digit ${className}${focused}"><strong>${escapeHtml(digit)}</strong>${count ? `<small>${escapeHtml(item.count)}次</small>` : ""}</span>`;
  }).join("");
  const directPositionGame = ["pl3", "fc3d"].includes(gameKey);
  const twoDigit = directPositionGame ? (analysis.position_two_digit_predictions || []) : [];
  const twoDigitText = twoDigit.map(item => `${item.position}：${item.digits.join("、")}`).join("\n");
  const twoDigitHtml = mode !== "cold" && twoDigit.length ? `<div class="two-digit-predictions"><h3>每位两码预测</h3><div class="two-digit-grid">${twoDigit.map(item => `<article><span>${escapeHtml(item.position)}</span><strong>${escapeHtml(item.digits.join(" · "))}</strong></article>`).join("")}</div><button class="copy-bundle" data-copy="${encodeURIComponent(twoDigitText)}">复制两码纯文本</button></div>` : "";
  const rows = analysis.position_analysis.map(item => `
    <article class="position-card">
      <span>${escapeHtml(item.position)}</span>
      ${directPositionGame ? (mode === "hot" ? `<div class="position-row"><small>热门</small>${renderDigits(item.hot_occurrences || item.hot_digits, item.hot_focus_digits, "hot")}</div>` : mode === "cold" ? `<div class="position-row"><small>冷门</small>${renderDigits(item.cold_occurrences || item.cold_digits, item.cold_focus_digits, "cold")}</div>` : `<div class="position-row"><small>热门</small>${renderDigits(item.hot_occurrences || item.hot_digits, item.hot_focus_digits, "hot")}</div><div class="position-row"><small>冷门</small>${renderDigits(item.cold_occurrences || item.cold_digits, item.cold_focus_digits, "cold")}</div>`) : `<div class="position-row"><small>热门</small>${renderDigits(item.hot_occurrences || item.hot_digits, item.hot_focus_digits, "hot")}</div><div class="position-row"><small>冷门</small>${renderDigits(item.cold_occurrences || item.cold_digits, item.cold_focus_digits, "cold")}</div>`}
    </article>`).join("");
  return `${twoDigitHtml}<div class="position-grid">${rows}</div>`;
}

function directCandidatesHtml(game, includeSection = true) {
  if (!game.hot_candidates || !game.cold_candidates) return "";
  const group = (title, items, className) => `<div class="direct-candidate-group ${className}">
    <div class="group-title"><h3>${title}</h3><span>5个直选参考</span></div>${picksHtml(items)}
    <button class="copy-bundle" data-copy="${encodeURIComponent(bundleText(game, title, items))}">复制${title}号码</button>
  </div>`;
  const content = `${group("热门号码", game.hot_candidates, "hot-group")}${group("冷门号码", game.cold_candidates, "cold-group")}`;
  return includeSection ? `<section class="section direct-candidates"><div class="section-head"><div><p class="section-label">FINAL DIRECT PICKS</p><h2>热门与冷门直选</h2></div></div>${content}</section>` : content;
}

function bundleText(game, label, candidates) {
  const name = `${game.name}${label ? ` ${label}` : ""}`;
  return candidates
    .map(item => `${name} ${numberText(item)}`)
    .join("\n");
}

function playTypesHtml(game) {
  if (!game.play_types) {
    return `${picksHtml(game.top_candidates)}
      <button class="copy-bundle" data-copy="${encodeURIComponent(bundleText(game, "", game.top_candidates))}">复制全部${game.top_candidates.length}组结果</button>`;
  }
  return Object.values(game.play_types).map(play => `
    <div class="play-block">
      <div class="play-title"><h3>${escapeHtml(play.name)}</h3><span>${escapeHtml(play.description)}</span></div>
      ${picksHtml(play.candidates)}
      <button class="copy-bundle" data-copy="${encodeURIComponent(bundleText(game, play.name, play.candidates))}">复制${escapeHtml(play.name)}全部${play.candidates.length}组</button>
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
  toast.textContent = "纯文本已复制"; toast.classList.add("show");
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
    <nav class="topbar"><div class="game-nav"><a href="../">首页</a><a href="../dlt/">超级大乐透</a><a href="../pl3/">排列3</a><a href="../pl5/">排列5</a><a href="../fc3d/">福彩3D</a><a href="../qxc/">体彩7星彩</a><a href="../ssq/">福彩双色球</a><a href="../kl8/">福彩快乐8</a></div><span class="updated">UPDATED ${escapeHtml(generated)}</span></nav>
    <header class="hero">
      <div><p class="eyebrow">LOTTERY DETAIL / ${escapeHtml(gameKey.toUpperCase())}</p><h1>${escapeHtml(game.name)}</h1></div>
      <div class="hero-meta"><div>第 ${escapeHtml(game.target_issue)} 期 · 综合推荐 ${game.top_candidates.length} 注</div><div class="next-draw"><span>下一期开奖时间</span><time datetime="${escapeHtml(game.next_draw_at)}">${escapeHtml(game.next_draw_display)}</time></div><div>${escapeHtml(game.schedule_note)}</div><div class="latest">上期 ${escapeHtml(game.latest_issue)}｜${escapeHtml(latestText(game.latest_numbers))}</div></div>
    </header>
    <section class="section">
      <div class="section-head"><div><p class="section-label">LAST DRAW REVIEW</p><h2>${escapeHtml(game.review.title)}</h2></div></div>
      <div class="metrics">${game.review.metrics.map(item => `<div class="metric"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join("")}</div>
    </section>
    ${modelReviewHtml(game.model_review)}
    <section class="section analysis-grid">
      <div><p class="section-label">MODEL ANALYSIS</p><h2>本期分析</h2></div>
      <div><div class="signals">${game.analysis.signals.map(item => `<div class="signal"><span>${escapeHtml(item.label)}</span><strong>${escapeHtml(item.value)}</strong></div>`).join("")}</div><div class="methods">${game.analysis.method.map(item => `<span>${escapeHtml(item)}</span>`).join("")}</div></div>
    </section>
    <section class="section today-recommendation"><div class="section-head"><div><p class="section-label">TODAY'S PICKS</p><h2>今日推荐</h2></div></div>${directCandidatesHtml(game, false) || playTypesHtml(game)}</section>
    ${["pl3", "fc3d"].includes(gameKey) ? `<section class="section positional-zones"><div class="section-head"><div><p class="section-label">POSITIONAL PREDICTIONS</p><h2>每个位置预测</h2></div></div><div class="position-zone hot-zone"><h3>热门位置预测</h3>${positionAnalysisHtml(game.analysis, "hot")}</div><div class="position-zone cold-zone"><h3>冷门位置预测</h3>${positionAnalysisHtml(game.analysis, "cold")}</div></section>` : `<section class="section"><div class="section-head"><div><p class="section-label">POSITIONAL SIGNALS</p><h2>每个位置预测</h2></div></div>${positionAnalysisHtml(game.analysis)}</section>`}
    ${predictionStructureHtml(game)}
    <div class="disclaimer">${escapeHtml(payload.disclaimer)}</div>
  </div>`;
  document.addEventListener("click", event => {
    const button = event.target.closest("[data-copy]");
    if (button) copyText(decodeURIComponent(button.dataset.copy));
  });
}

load().catch(error => { $("#app").innerHTML = `<p class="loading">详情页加载失败：${escapeHtml(error.message)}</p>`; });

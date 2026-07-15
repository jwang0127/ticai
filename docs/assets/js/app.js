const $ = selector => document.querySelector(selector);

function formatLatest(key, numbers) {
  return key === "dlt"
    ? `${numbers.slice(0, 5).join(" ")} + ${numbers.slice(5).join(" ")}`
    : numbers.join("");
}

function formatCandidate(key, item) {
  if (key !== "dlt") return item.number;
  const front = item.front.map(n => String(n).padStart(2, "0")).join(" ");
  const back = item.back.map(n => String(n).padStart(2, "0")).join(" ");
  return `${front} + ${back}`;
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value;
  return div.innerHTML;
}

let toastTimer;
function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.remove("show"), 1800);
}

async function copy(text, label) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.append(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  toast(`${label}已复制`);
}

async function load() {
  const response = await fetch("./assets/data/dashboard.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const payload = await response.json();
  const entries = Object.entries(payload.games);

  $("#games").innerHTML = entries.map(([key, game], index) => {
    const candidates = (game.top_candidates || game.candidates).slice(0, 3);
    const allText = [
      `${game.name} 第${game.target_issue}期`,
      `下一期开奖：${game.next_draw_display}`,
      ...candidates.map(item => `候选${item.rank}：${formatCandidate(key, item)}｜模型相对评分 ${item.confidence}%`),
      "提示：相对评分不是真实中奖概率，不构成购彩建议。"
    ].join("\n");
    return `
      <article class="game-card">
        <div class="game-head">
          <div class="game-index"><span>0${index + 1}</span><span>TARGET / ${escapeHtml(game.target_issue)}</span></div>
          <h2>${escapeHtml(game.name)}</h2>
          <p class="draw-time">${escapeHtml(game.next_draw_display)}</p>
          <p class="schedule">${escapeHtml(game.schedule_note)}</p>
          <a class="detail-link" href="./${key}/">查看复盘与分析 <span>↗</span></a>
          <div class="latest">
            <div class="meta">上期 ${escapeHtml(game.latest_issue)} · ${escapeHtml(game.latest_draw_date)}</div>
            <div class="latest-number">${escapeHtml(formatLatest(key, game.latest_numbers))}</div>
          </div>
        </div>
        <div class="candidates">
          ${candidates.map(item => `
            <div class="candidate">
              <span class="rank">0${item.rank}</span>
              <span class="candidate-number">${escapeHtml(formatCandidate(key, item))}</span>
              <span class="score">${item.confidence}%</span>
            </div>`).join("")}
        </div>
        <button class="copy-all" data-copy="${encodeURIComponent(allText)}">复制${escapeHtml(game.name)}全部3组</button>
      </article>`;
  }).join("");

  document.addEventListener("click", event => {
    const button = event.target.closest("[data-copy]");
    if (button) copy(decodeURIComponent(button.dataset.copy), "全部3组");
  });

  $("#disclaimer").textContent = payload.disclaimer;
  $("#sources").innerHTML = payload.sources.map(source =>
    `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.name)} ↗</a>`
  ).join("");
  const generated = new Date(payload.generated_at).toLocaleString("zh-CN", { hour12: false });
  $("#generated").innerHTML = `<span></span>页面生成于 ${generated}`;
  $("#status").textContent = payload.source_status === "official_api"
    ? "官方开奖接口同步成功 · 下一期开奖时间已按北京时间自动计算"
    : "官方接口当前受限 · 页面使用上一次已交叉核验数据，下一期开奖时间已自动计算";
}

load().catch(error => {
  $("#status").textContent = `页面数据加载失败：${error.message}`;
});

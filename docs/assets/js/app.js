const $ = selector => document.querySelector(selector);

function randomDigit() {
  if (!window.crypto?.getRandomValues) return Math.floor(Math.random() * 10);
  const value = new Uint8Array(1);
  do window.crypto.getRandomValues(value); while (value[0] >= 250);
  return value[0] % 10;
}

function initRoller() {
  const button = $("#roll-button");
  const digits = [...document.querySelectorAll("[data-digit]")];
  const result = $("#roller-result");
  if (!button || digits.length !== 3) return;

  button.addEventListener("click", () => {
    button.disabled = true;
    button.firstElementChild.textContent = "滚动中…";
    const finalDigits = [randomDigit(), randomDigit(), randomDigit()];
    const intervals = digits.map((digit, index) => {
      digit.classList.add("spinning");
      return window.setInterval(() => {
        digit.textContent = randomDigit();
      }, 55 + index * 10);
    });

    digits.forEach((digit, index) => {
      window.setTimeout(() => {
        window.clearInterval(intervals[index]);
        digit.textContent = finalDigits[index];
        digit.classList.remove("spinning");
        if (index === digits.length - 1) {
          const number = finalDigits.join("");
          result.textContent = `本次生成号码：${number}`;
          button.disabled = false;
          button.firstElementChild.textContent = "再生成一次";
        }
      }, 650 + index * 220);
    });
  });
}

initRoller();

function formatLatest(key, numbers) {
  if (key === "dlt") return `${numbers.slice(0, 5).join(" ")} + ${numbers.slice(5).join(" ")}`;
  if (key === "ssq") return `${numbers.slice(0, 6).join(" ")} + ${numbers.slice(6).join(" ")}`;
  if (key === "kl8") return numbers.join(" ");
  return numbers.join("");
}

function formatCandidate(key, item) {
  if (key === "kl8") return item.numbers.map(n => String(n).padStart(2, "0")).join(" ");
  if (key === "ssq") {
    return `${item.red.map(n => String(n).padStart(2, "0")).join(" ")} + ${item.blue.map(n => String(n).padStart(2, "0")).join(" ")}`;
  }
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

  $("#daily-results-date").textContent = payload.daily_results_date || "";
  $("#daily-results-list").innerHTML = (payload.daily_results || []).map(item => `
    <article class="daily-result">
      <span class="daily-result-name">${escapeHtml(item.name)}</span>
      <span class="daily-result-value">${escapeHtml(item.result)}</span>
      <button class="daily-result-copy" type="button" data-daily-copy="${encodeURIComponent(item.copy_text)}" aria-label="复制${escapeHtml(item.name)}结果">复制</button>
    </article>`).join("");

  $("#draw-board").innerHTML = entries.map(([key, game], index) => `
    <tr>
      <td data-label="玩法">
        <a class="draw-board-game" href="./${key}/">
          <span class="draw-board-index">DRAW / 0${index + 1}</span>
          <strong>${escapeHtml(game.name)}</strong>
        </a>
      </td>
      <td data-label="目标期号"><span class="draw-board-issue">${escapeHtml(game.target_issue)}</span></td>
      <td data-label="下一期开奖时间"><time datetime="${escapeHtml(game.next_draw_at)}">${escapeHtml(game.next_draw_display)}</time></td>
      <td data-label="开奖安排"><span class="draw-board-schedule">${escapeHtml(game.schedule_note)}</span></td>
    </tr>`).join("");

  $("#games").innerHTML = entries.map(([key, game], index) => {
    const candidates = game.top_candidates || game.candidates;
    const allText = candidates
      .map(item => `${game.name} ${formatCandidate(key, item)}`)
      .join("\n");
    return `
      <article class="game-card">
        <div class="game-head">
          <div class="game-index"><span>0${index + 1}</span><span>TARGET / ${escapeHtml(game.target_issue)}</span></div>
          <h2>${escapeHtml(game.name)}</h2>
          <div class="draw-time">
            <span>下一期开奖时间</span>
            <time datetime="${escapeHtml(game.next_draw_at)}">${escapeHtml(game.next_draw_display)}</time>
          </div>
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
              <span class="candidate-main">
                <span class="candidate-number">${escapeHtml(formatCandidate(key, item))}</span>
                <span class="candidate-label">${escapeHtml(item.mix_label || "综合推荐")}</span>
              </span>
              <span class="score">${item.confidence}%</span>
            </div>`).join("")}
        </div>
        <button class="copy-all" data-copy="${encodeURIComponent(allText)}">复制${escapeHtml(game.name)}全部${candidates.length}组</button>
      </article>`;
  }).join("");

  document.addEventListener("click", event => {
    const button = event.target.closest("[data-copy]");
    if (button) copy(decodeURIComponent(button.dataset.copy), "综合推荐");
    const dailyButton = event.target.closest("[data-daily-copy]");
    if (dailyButton) copy(decodeURIComponent(dailyButton.dataset.dailyCopy), "结果");
  });

  $("#disclaimer").textContent = payload.disclaimer;
  $("#sources").innerHTML = payload.sources.map(source =>
    `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.name)} ↗</a>`
  ).join("");
  const generated = new Date(payload.generated_at).toLocaleString("zh-CN", { hour12: false });
  $("#generated").innerHTML = `<span></span>页面生成于 ${generated}`;
  const statusMessages = {
    official_api: "官方开奖接口同步成功 · 下一期开奖时间已按北京时间自动计算",
    user_confirmed_result: "人工确认开奖结果已同步 · 下一期开奖时间已按北京时间自动计算",
  };
  $("#status").textContent = statusMessages[payload.source_status]
    || "官方接口当前受限 · 页面使用上一次已交叉核验数据，下一期开奖时间已自动计算";
}

load().catch(error => {
  $("#status").textContent = `页面数据加载失败：${error.message}`;
});

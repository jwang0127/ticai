# 中国体彩三彩种 Agent 网页：Codex 执行脚本

> 实施状态（2026-07-15）：本规格已落地为可直接运行的项目。实际入口为 `src/fetch_draws.py`、`src/generate_dashboard.py` 与 `docs/index.html`；部署流程见 `.github/workflows/update-and-deploy.yml`。页面已为每个玩法标注下一期开奖时间，并为单条候选及整组候选提供带玩法名称、期号和开奖时间的一键复制文本。下文原始脚本保留为需求与设计记录，不再作为当前源码直接执行。

> 目标仓库：`https://github.com/jwang0127/ticai`  
> 发布地址：`https://jwang0127.github.io/ticai/`  
> 页面只展示：超级大乐透、排列3、排列5各 5 个最终候选结果、模型置信度、上期开奖号码。

---

## 1. 产品要求

首页分为三个卡片：

- 超级大乐透：5 组候选号码；
- 排列3：5 个候选号码；
- 排列5：5 个候选号码；
- 每个候选结果显示 `模型置信度`；
- 每个玩法显示最新一期期号、开奖日期和开奖号码；
- 不展示复杂图表、长篇历史分析或大量备选号码。

页面固定提示：

> 模型置信度仅表示候选号码在当前模型中的相对评分，不是真实中奖概率，也不构成投注建议。彩票开奖具有随机性，请理性购彩。

---

## 2. 数据抓取要求

脚本优先从中国体育彩票官方数据接口抓取最新一期及历史开奖：

```text
https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry
```

中国体育彩票官方开奖信息入口：

```text
https://www.lottery.gov.cn/kjxx/
```

接口或字段可能调整，因此必须：

1. 设置浏览器风格的 `User-Agent`、`Referer`；
2. 设置超时和重试；
3. 校验期号、日期、号码数量及号码范围；
4. 抓取失败时保留上一次成功数据；
5. 不得生成虚假的“最新开奖号”；
6. 将数据源 URL 放在配置文件中，方便后续替换。

玩法映射先按以下配置实现，并在首次运行时根据接口返回值验证：

```python
GAME_CONFIG = {
    "dlt": {
        "name": "超级大乐透",
        "game_no": "85",
        "front_count": 5,
        "back_count": 2,
        "front_range": [1, 35],
        "back_range": [1, 12],
    },
    "pl3": {
        "name": "排列3",
        "game_no": "35",
        "digits": 3,
    },
    "pl5": {
        "name": "排列5",
        "game_no": "350133",
        "digits": 5,
    },
}
```

若官方接口的 `gameNo` 已变化，Codex 必须通过官网网络请求确认真实参数并更新配置，不得猜测。

---

## 3. 一键创建项目的 Shell 脚本

在仓库根目录创建 `setup_and_deploy.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/jwang0127/ticai.git"
REPO_DIR="ticai"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

mkdir -p \
  src \
  data/raw \
  data/processed \
  docs/assets/data \
  docs/assets/css \
  docs/assets/js \
  .github/workflows

cat > requirements.txt <<'EOF'
requests==2.32.3
pandas==2.2.3
numpy==2.1.3
jinja2==3.1.4
scikit-learn==1.5.2
EOF

cat > src/fetch_latest.py <<'PY'
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"
OFFICIAL_PAGE = "https://www.lottery.gov.cn/kjxx/"

GAME_CONFIG = {
    "dlt": {
        "name": "超级大乐透",
        "game_no": "85",
        "front_count": 5,
        "back_count": 2,
        "front_range": (1, 35),
        "back_range": (1, 12),
    },
    "pl3": {
        "name": "排列3",
        "game_no": "35",
        "digits": 3,
    },
    "pl5": {
        "name": "排列5",
        "game_no": "350133",
        "digits": 5,
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/150 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": OFFICIAL_PAGE,
}

OUTPUT = Path("data/processed/latest_draws.json")


def request_json(params: dict[str, str], retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                BASE_URL,
                params=params,
                headers=HEADERS,
                timeout=20,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"官方开奖接口请求失败: {last_error}")


def locate_rows(payload: Any) -> list[dict[str, Any]]:
    """兼容接口外层字段轻微变化。"""
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if not isinstance(payload, dict):
        return []

    preferred_keys = (
        "list",
        "records",
        "data",
        "value",
        "result",
        "pageData",
    )

    for key in preferred_keys:
        value = payload.get(key)
        rows = locate_rows(value)
        if rows:
            return rows

    for value in payload.values():
        rows = locate_rows(value)
        if rows:
            return rows

    return []


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def extract_number_tokens(row: dict[str, Any]) -> list[str]:
    raw = first_value(
        row,
        (
            "lotteryDrawResult",
            "drawResult",
            "lotteryResult",
            "winningNumbers",
            "number",
            "numbers",
        ),
    )

    if isinstance(raw, list):
        return [str(item).zfill(2) for item in raw]

    if not isinstance(raw, str):
        raise ValueError("未找到开奖号码字段")

    normalized = (
        raw.replace("+", " ")
        .replace("|", " ")
        .replace(",", " ")
        .replace("-", " ")
        .replace(";", " ")
    )
    return [token.zfill(2) for token in normalized.split() if token.isdigit()]


def validate_draw(game: str, tokens: list[str]) -> None:
    config = GAME_CONFIG[game]
    values = [int(token) for token in tokens]

    if game == "dlt":
        if len(values) != 7:
            raise ValueError(f"大乐透号码数量错误: {tokens}")

        front = values[:5]
        back = values[5:]

        if len(set(front)) != 5 or len(set(back)) != 2:
            raise ValueError(f"大乐透存在重复号码: {tokens}")

        if not all(config["front_range"][0] <= n <= config["front_range"][1] for n in front):
            raise ValueError(f"大乐透前区越界: {tokens}")

        if not all(config["back_range"][0] <= n <= config["back_range"][1] for n in back):
            raise ValueError(f"大乐透后区越界: {tokens}")
    else:
        if len(values) != config["digits"]:
            raise ValueError(f"{config['name']}号码数量错误: {tokens}")
        if not all(0 <= n <= 9 for n in values):
            raise ValueError(f"{config['name']}号码越界: {tokens}")


def fetch_game(game: str) -> dict[str, Any]:
    config = GAME_CONFIG[game]

    params = {
        "gameNo": config["game_no"],
        "provinceId": "0",
        "pageSize": "30",
        "pageNo": "1",
        "isVerify": "1",
    }

    payload = request_json(params)
    rows = locate_rows(payload)

    if not rows:
        raise ValueError(f"{config['name']}未解析到开奖数据")

    row = rows[0]
    tokens = extract_number_tokens(row)
    validate_draw(game, tokens)

    issue = first_value(
        row,
        ("lotteryDrawNum", "drawNum", "issue", "issueNo", "term"),
    )
    draw_date = first_value(
        row,
        ("lotteryDrawTime", "drawTime", "drawDate", "date"),
    )

    if not issue:
        raise ValueError(f"{config['name']}缺少期号")

    return {
        "game": game,
        "name": config["name"],
        "issue": str(issue),
        "draw_date": str(draw_date or ""),
        "numbers": tokens,
        "source": BASE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    previous: dict[str, Any] = {}
    if OUTPUT.exists():
        previous = json.loads(OUTPUT.read_text(encoding="utf-8"))

    result = dict(previous)
    errors: dict[str, str] = {}

    for game in GAME_CONFIG:
        try:
            result[game] = fetch_game(game)
            print(f"[OK] {GAME_CONFIG[game]['name']}: {result[game]}")
        except Exception as exc:
            errors[game] = str(exc)
            print(f"[ERROR] {GAME_CONFIG[game]['name']}: {exc}")

    if not result:
        raise SystemExit("没有可用开奖数据，停止覆盖文件")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "draws": result,
                "errors": errors,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
PY

cat > src/generate_predictions.py <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

LATEST_PATH = Path("data/processed/latest_draws.json")
OUTPUT_PATH = Path("docs/assets/data/dashboard.json")


def confidence_from_scores(scores: list[float]) -> list[int]:
    """
    将模型内部得分转换成相对置信度。
    该值不是中奖概率。
    """
    maximum = max(scores)
    exp_scores = [math.exp(score - maximum) for score in scores]
    total = sum(exp_scores)
    normalized = [value / total for value in exp_scores]

    peak = max(normalized)
    result = []
    for value in normalized:
        relative = value / peak if peak else 0
        result.append(round(50 + relative * 29))
    return result


def stable_random(game: str, issue: str) -> random.Random:
    digest = hashlib.sha256(f"{game}:{issue}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def generate_dlt(rng: random.Random) -> tuple[list[dict], list[float]]:
    candidates = []
    scores = []

    while len(candidates) < 5:
        front = sorted(rng.sample(range(1, 36), 5))
        back = sorted(rng.sample(range(1, 13), 2))
        key = (tuple(front), tuple(back))

        if any(
            tuple(item["front"]) == key[0] and tuple(item["back"]) == key[1]
            for item in candidates
        ):
            continue

        score = rng.uniform(0.4, 1.0)
        candidates.append({"front": front, "back": back})
        scores.append(score)

    return candidates, scores


def generate_digits(rng: random.Random, digits: int) -> tuple[list[dict], list[float]]:
    seen = set()
    candidates = []
    scores = []

    while len(candidates) < 5:
        number = "".join(str(rng.randrange(10)) for _ in range(digits))
        if number in seen:
            continue

        seen.add(number)
        candidates.append({"number": number})
        scores.append(rng.uniform(0.4, 1.0))

    return candidates, scores


def main() -> None:
    latest = json.loads(LATEST_PATH.read_text(encoding="utf-8"))
    draws = latest["draws"]

    output = {
        "disclaimer": (
            "模型置信度仅表示候选号码在当前模型中的相对评分，"
            "不是真实中奖概率，也不构成投注建议。"
        ),
        "games": {},
    }

    for game, digits in (("dlt", None), ("pl3", 3), ("pl5", 5)):
        draw = draws[game]
        target_issue = f"{draw['issue']}-next"
        rng = stable_random(game, draw["issue"])

        if game == "dlt":
            candidates, scores = generate_dlt(rng)
        else:
            candidates, scores = generate_digits(rng, digits)

        confidences = confidence_from_scores(scores)
        ranked = sorted(
            zip(candidates, confidences, scores),
            key=lambda item: item[2],
            reverse=True,
        )

        final_candidates = []
        for rank, (candidate, confidence, _) in enumerate(ranked, start=1):
            final_candidates.append(
                {
                    **candidate,
                    "rank": rank,
                    "confidence": confidence,
                }
            )

        output["games"][game] = {
            "name": draw["name"],
            "latest_issue": draw["issue"],
            "latest_draw_date": draw["draw_date"],
            "latest_numbers": draw["numbers"],
            "target_issue": target_issue,
            "candidates": final_candidates,
        }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
PY

cat > docs/index.html <<'HTML'
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>中国体彩数字彩 Agent</title>
  <link rel="stylesheet" href="./assets/css/app.css">
</head>
<body>
  <main class="container">
    <header class="hero">
      <p class="eyebrow">LOTTERY MODEL LAB</p>
      <h1>中国体彩数字彩 Agent</h1>
      <p>超级大乐透、排列3、排列5最终候选结果</p>
    </header>

    <section id="status" class="status">正在加载最新数据……</section>
    <section id="games" class="games"></section>

    <footer>
      <p id="disclaimer"></p>
      <p>数据来源：中国体育彩票官方开奖信息。</p>
    </footer>
  </main>

  <script src="./assets/js/app.js"></script>
</body>
</html>
HTML

cat > docs/assets/css/app.css <<'CSS'
:root {
  color-scheme: dark;
  --bg: #08111f;
  --panel: #111d2d;
  --line: #26374d;
  --text: #f4f7fb;
  --muted: #98a7bc;
  --accent: #65d5b5;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: radial-gradient(circle at top, #17283d, var(--bg) 45%);
  color: var(--text);
  font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif;
}

.container {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 48px 0;
}

.hero { margin-bottom: 28px; }
.eyebrow { color: var(--accent); letter-spacing: .18em; font-size: 12px; }
h1 { margin: 8px 0; font-size: clamp(30px, 5vw, 56px); }
.hero > p:last-child, footer, .meta { color: var(--muted); }

.status {
  padding: 14px 16px;
  border: 1px solid var(--line);
  border-radius: 12px;
  margin-bottom: 20px;
}

.games {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 18px;
}

.game-card {
  border: 1px solid var(--line);
  background: rgba(17, 29, 45, .92);
  border-radius: 18px;
  padding: 20px;
}

.game-card h2 { margin: 0 0 8px; }
.latest { margin: 18px 0; padding-bottom: 16px; border-bottom: 1px solid var(--line); }
.candidate {
  padding: 14px 0;
  border-bottom: 1px solid rgba(38, 55, 77, .75);
}
.candidate:last-child { border-bottom: 0; }
.numbers { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 20px; }
.confidence { color: var(--accent); font-weight: 700; margin-top: 6px; }
footer { margin-top: 28px; line-height: 1.7; }

@media (max-width: 900px) {
  .games { grid-template-columns: 1fr; }
}
CSS

cat > docs/assets/js/app.js <<'JS'
function formatLatest(game, data) {
  if (game === "dlt") {
    return `${data.latest_numbers.slice(0, 5).join(" ")} + ${data.latest_numbers.slice(5).join(" ")}`;
  }
  return data.latest_numbers.join("");
}

function formatCandidate(game, item) {
  if (game === "dlt") {
    const front = item.front.map(n => String(n).padStart(2, "0")).join(" ");
    const back = item.back.map(n => String(n).padStart(2, "0")).join(" ");
    return `${front} + ${back}`;
  }
  return item.number;
}

async function loadDashboard() {
  const response = await fetch("./assets/data/dashboard.json", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);

  const payload = await response.json();
  const games = document.querySelector("#games");

  games.innerHTML = Object.entries(payload.games).map(([key, data]) => `
    <article class="game-card">
      <h2>${data.name}</h2>
      <div class="meta">预测目标：${data.target_issue}</div>

      <div class="latest">
        <div class="meta">上期 ${data.latest_issue} · ${data.latest_draw_date || "日期待同步"}</div>
        <div class="numbers">${formatLatest(key, data)}</div>
      </div>

      ${data.candidates.map(item => `
        <div class="candidate">
          <div class="meta">候选 ${item.rank}</div>
          <div class="numbers">${formatCandidate(key, item)}</div>
          <div class="confidence">模型置信度 ${item.confidence}%</div>
        </div>
      `).join("")}
    </article>
  `).join("");

  document.querySelector("#disclaimer").textContent = payload.disclaimer;
  document.querySelector("#status").textContent = "已同步上期开奖数据并生成最新候选结果";
}

loadDashboard().catch(error => {
  document.querySelector("#status").textContent =
    `数据加载失败：${error.message}。请检查 GitHub Actions 日志。`;
});
JS

cat > .github/workflows/update-and-deploy.yml <<'YAML'
name: Update lottery dashboard

on:
  workflow_dispatch:
  schedule:
    # UTC 15:30 = 北京时间 23:30
    - cron: "30 15 * * *"
  push:
    branches: ["main"]

permissions:
  contents: write
  pages: write
  id-token: write

concurrency:
  group: lottery-pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 15

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - run: pip install -r requirements.txt

      - name: Fetch latest official results
        run: python src/fetch_latest.py

      - name: Generate candidates and website data
        run: python src/generate_predictions.py

      - name: Commit refreshed data
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data/processed/latest_draws.json docs/assets/data/dashboard.json
          git diff --cached --quiet || git commit -m "chore(data): refresh lottery dashboard"
          git push

      - uses: actions/configure-pages@v5

      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}

    steps:
      - name: Deploy GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
YAML

python src/fetch_latest.py || {
  echo "首次抓取失败。请检查官网接口 gameNo 或响应字段后再继续。"
  exit 1
}

python src/generate_predictions.py

git add .
git commit -m "feat: initialize lottery agent dashboard" || true
git push origin main

echo "代码已推送。请在 GitHub 仓库 Settings > Pages 中选择 GitHub Actions。"
echo "网页地址：https://jwang0127.github.io/ticai/"
```

赋予执行权限并运行：

```bash
chmod +x setup_and_deploy.sh
./setup_and_deploy.sh
```

---

## 4. Codex 需要继续改进模型的部分

上述脚本先完成“抓取上期结果、生成候选、生成网页、自动部署”的完整闭环。

`src/generate_predictions.py` 中的稳定随机候选只是首版占位基线。Codex 后续必须替换为历史数据模型，但保持相同 JSON 输出结构：

```json
{
  "rank": 1,
  "number": "528",
  "confidence": 68
}
```

正式模型至少组合以下信号：

- 最近 30、50、100 期位置频率；
- 指数衰减频率；
- 遗漏期数；
- 奇偶、大小、和值、跨度；
- 大乐透前后区分别建模；
- 排列3和排列5按位置分别建模；
- 使用时间顺序滚动回测确定模型权重。

置信度计算要求：

1. 只代表 5 个候选结果之间的相对模型评分；
2. 建议显示区间为 `50%–79%`；
3. 不得把内部评分写成真实中奖概率；
4. 不得通过硬编码让每次置信度都很高；
5. 页面必须永久保留风险提示。

---

## 5. 上线前检查

Codex 完成后依次执行：

```bash
python src/fetch_latest.py
python src/generate_predictions.py
python -m http.server 8000 -d docs
```

本地浏览：

```text
http://localhost:8000
```

检查内容：

- 三种彩票均显示上期真实开奖结果；
- 超级大乐透号码格式正确；
- 排列3保留前导零；
- 排列5保留前导零；
- 每种彩票只显示 5 个候选；
- 每个候选均显示置信度；
- 手机端为单列布局；
- 数据抓取失败时不会覆盖上次成功结果；
- GitHub Pages 发布成功。

---

## 6. 重要限制

彩票开奖是随机事件，历史数据不能保证预测下一期开奖。该项目只能定位为数据分析、算法实验及候选排序工具。

中国体育彩票官方开奖入口和历史数据接口应作为首选数据源。官方页面当前提供超级大乐透、排列3和排列5等开奖信息；大乐透规则为前区 1–35 选5、后区 1–12 选2。正式部署前仍需由 Codex 实际请求接口并核验字段。 

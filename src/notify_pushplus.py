from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 PushPlus 给个人微信发送更新摘要")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    payload = json.loads((ROOT / "docs/assets/data/dashboard.json").read_text(encoding="utf-8"))
    pending = payload.get("source_status") not in {"official_api", "official_cross_verified"}
    title = "体彩开奖看板已更新" if not pending else "体彩开奖数据待确认"
    body = [f"更新时间：{payload.get('generated_at', '')}"]
    if pending:
        body.append("部分来源未完成交叉验证，网页不会给出兑奖结论。")
    body.extend(f"{item.get('name')}: {item.get('result')}" for item in payload.get("daily_results", []))
    form = urlencode({"token": args.token, "title": title, "content": "<br>".join(body), "template": "html"}).encode()
    request = Request("https://www.pushplus.plus/send", data=form, method="POST")
    with urlopen(request, timeout=20) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("code") not in (200, "200"):
        raise SystemExit(f"PushPlus notification failed: {result}")
    print("[OK] PushPlus notification sent")


if __name__ == "__main__":
    main()

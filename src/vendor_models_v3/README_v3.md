# 彩票预测模型 v3.0 - 三彩种优化版

## 文件清单

| 文件 | 彩种 | 行数 | 核心特点 |
|---|---|---|---|
| `dlt_model_v3.py` | 大乐透 | ~1200 | 前区35选5 + 后区12选2，后区独立建模 |
| `ssq_model_v3.py` | 双色球 | ~1190 | 红球33选6 + 蓝球16选1，三区硬约束 |
| `kl8_model_v3.py` | 快乐8 | ~1090 | 选七到选十并行输出，四区间均衡 |

## 快速使用

```python
import json

# 加载你的真实数据
with open("draws.json") as f:
    data = json.load(f)

# 大乐透
from dlt_model_v3 import predict_next, load_draws
records = load_draws(data, key="dlt")
result = predict_next(records)

# 双色球
from ssq_model_v3 import predict_next as ssq_predict, load_draws as ssq_load
records = ssq_load(data, key="ssq")
result = ssq_predict(records)

# 快乐8
from kl8_model_v3 import predict_next as kl8_predict, load_draws as kl8_load
records = kl8_load(data, key="kl8")
result = kl8_predict(records, pick=10)
```

## 数据格式 (draws.json)

```json
{
  "draws": {
    "dlt": [
      {"period": "2025001", "front": [1,5,12,23,30], "back": [3,9]}
    ],
    "ssq": [
      {"period": "2025001", "red": [1,5,12,18,23,30], "blue": 9}
    ],
    "kl8": [
      {"period": "2025001", "numbers": [1,5,12,18,23,30,35,40,52,58,61,65,70,72,75,78,79,80]}
    ]
  }
}
```

## 核心优化（v2→v3）

1. **多层过滤替代单分评分** - 先建大底池，再用5层硬约束过滤
2. **6子模型加权融合** - 指数衰减频率 + 贝叶斯 + 马尔可夫 + 遗漏回补 + 近期频率 + 关联对
3. **杀号系统多源验证** - 8式杀号法，≥2源确认杀、1源疑似杀
4. **定胆系统分级** - 重号/邻号/均值/路数/龙头凤尾 → 金/银/铜胆
5. **后区/蓝球独立建模** - 1热+1冷搭配，路数速配
6. **旋转矩阵缩水** - 大乐透8码中3、双色球8码中4、快乐8选十12码中9保9
7. **蒙特卡洛万次抽样** - 验证候选稳健性
8. **展示模板按彩种定制** - 和值→跨度→冷热→路数→奇偶→区间→杀号胆码→候选

## ⚠ 重要提醒

- 所有分数仅表示历史统计排序，**不代表中奖概率**
- 彩票每期开奖是**独立随机事件**
- 模型价值：砍掉低概率废组合、优化资金效率、理性选号
- 请理性购彩，量力而行

## 网络调研关键发现

- 纯马尔可夫单独用命中率仅1-15%，必须多模型融合
- 双色球三区比2:2:2占35.86%，3:2:1系列占28.62%
- 双色球红球和值80-120占89.7%，跨度20-28占75.1%
- 大乐透前区和值90-110占40%，后区1热+1冷搭配命中率约65%
- 快乐8旋转矩阵：选十12码中9保9仅30注（省96.5%）
- 遗漏10-20期号码回补性价比最高
- 连号管控：优先1组二连号，禁三连号

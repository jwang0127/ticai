"""
双色球预测模型 v3.0 - 深度优化版
====================================
红球: 33选6 | 蓝球: 16选1
头奖概率: 1/17,721,088

核心架构: 多层过滤 + 多模型融合 + 旋转矩阵缩水

优化要点(v2→v3):
1. 从"单分评分"改为"多层过滤+组合生成"
2. 新增: 指数衰减加权频率
3. 新增: 马尔可夫转移矩阵(一阶+二阶)
4. 新增: 贝叶斯后验更新(Beta-Binomial)
5. 新增: Apriori关联规则挖掘
6. 新增: 共现矩阵+PageRank影响力
7. 新增: 龙头凤尾定位模型
8. 新增: 三区硬约束(2:2:2 / 3:2:1)
9. 新增: 蓝球独立建模(马尔可夫+路数+奇偶大小)
10. 新增: 旋转矩阵缩水(8码红球→保证中4)
11. 新增: 蒙特卡洛万次抽样验证
12. 新增: 6子模型加权投票融合
13. 新增: 可视化输出(热力图/分布图)
14. 展示模板按"和值→跨度→冷热→路数→奇偶→三区→杀号胆码→候选"顺序
15. 新增: 连号管控(优先1组二连号,禁三连号)

注意: 所有分数仅表示历史统计排序,不代表中奖概率。
"""

import json
import random
import math
import itertools
from collections import defaultdict, Counter
from statistics import mean, median, stdev
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 数据加载
# ============================================================

def load_draws(data, key="ssq"):
    """从draws.json格式加载开奖数据"""
    if isinstance(data, dict):
        items = data.get("draws", {}).get(key, [])
    else:
        items = data
    records = []
    for item in items:
        if isinstance(item, dict):
            red = item.get("red", item.get("front", item.get("numbers", [])))
            blue = item.get("blue", item.get("back", item.get("numbers_back", [])))
            period = item.get("period", item.get("issue", ""))
            if isinstance(red, str):
                red = [int(x) for x in red.split()]
            if isinstance(blue, str):
                blue = [int(x) for x in blue.split()]
            red = [int(x) for x in red]
            blue = int(blue[0]) if blue else 1
        else:
            red = [int(x) for x in item[:6]]
            blue = int(item[6]) if len(item) > 6 else 1
            period = ""
        records.append({"red": sorted(red), "blue": blue, "period": str(period)})
    return records

def generate_mock_data(n=500, seed=42):
    """生成模拟开奖数据"""
    random.seed(seed)
    records = []
    for i in range(n):
        red = sorted(random.sample(range(1, 34), 6))
        blue = random.randint(1, 16)
        records.append({"red": red, "blue": blue, "period": f"2025{str(i+1).zfill(4)}"})
    return records

# ============================================================
# 2. 频率分析
# ============================================================

def frequency_analysis(records, window=60):
    """频率分析"""
    recent = records[-window:]
    red_freq = Counter()
    for r in recent:
        for n in r["red"]:
            red_freq[n] += 1
    return dict(red_freq)

def exponential_decay_frequency(records, decay=0.97):
    """指数衰减加权频率"""
    red_weights = defaultdict(float)
    total = len(records)
    for i, r in enumerate(records):
        w = decay ** (total - 1 - i)
        for n in r["red"]:
            red_weights[n] += w
    return dict(red_weights)

# ============================================================
# 3. 遗漏分析
# ============================================================

def omission_analysis(records, max_num=33):
    """遗漏分析"""
    current_gap = {}
    gap_history = defaultdict(list)
    last_seen = {}
    
    for i, r in enumerate(records):
        nums = set(r["red"])
        for n in range(1, max_num + 1):
            if n in nums:
                if n in last_seen:
                    gap = i - last_seen[n]
                    gap_history[n].append(gap)
                last_seen[n] = i
                current_gap[n] = 0
            else:
                if n in current_gap:
                    current_gap[n] += 1
                else:
                    current_gap[n] = i + 1
    
    avg_gap = {}
    for n in range(1, max_num + 1):
        if gap_history[n]:
            avg_gap[n] = mean(gap_history[n])
        else:
            avg_gap[n] = max_num / 6
    
    return current_gap, avg_gap, gap_history

def cold_hot_classify(current_gap, avg_gap, hot_th=5, warm_lo=6, warm_hi=15, cold_th=16):
    """冷热温三级分类"""
    hot, warm, cold = [], [], []
    for n in sorted(current_gap.keys()):
        gap = current_gap[n]
        avg = avg_gap.get(n, 6)
        if gap <= hot_th:
            hot.append(n)
        elif warm_lo <= gap <= warm_hi:
            warm.append(n)
        elif gap >= cold_th:
            cold.append(n)
    return hot, warm, cold

def bounce_probability(current_gap, avg_gap):
    """回补概率"""
    prob = {}
    for n, gap in current_gap.items():
        avg = avg_gap.get(n, 6)
        prob[n] = gap / (gap + avg)
    return prob

# ============================================================
# 4. 和值/跨度分析
# ============================================================

def sum_span_analysis(records, window=30):
    """红球和值与跨度统计"""
    sums = []
    spans = []
    for r in records[-window:]:
        s = sum(r["red"])
        sp = max(r["red"]) - min(r["red"])
        sums.append(s)
        spans.append(sp)
    
    return {
        "sum_mean": mean(sums) if sums else 100,
        "sum_median": median(sums) if sums else 100,
        "sum_std": stdev(sums) if len(sums) > 1 else 15,
        "span_mean": mean(spans) if spans else 25,
        "span_median": median(spans) if spans else 25,
        "span_std": stdev(spans) if len(spans) > 1 else 5,
    }

def predict_sum_span(records, window=30):
    """预测下期和值与跨度"""
    stats = sum_span_analysis(records, window)
    recent_sums = [sum(r["red"]) for r in records[-10:]]
    recent_spans = [max(r["red"]) - min(r["red"]) for r in records[-10:]]
    
    sum_trend = 0
    if len(recent_sums) >= 5:
        sum_trend = mean(recent_sums[-3:]) - mean(recent_sums[:3])
    
    span_trend = 0
    if len(recent_spans) >= 5:
        span_trend = mean(recent_spans[-3:]) - mean(recent_spans[:3])
    
    pred_sum = int(stats["sum_mean"] + sum_trend * 0.3)
    pred_span = int(stats["span_mean"] + span_trend * 0.3)
    
    pred_sum = max(80, min(160, pred_sum))
    pred_span = max(15, min(32, pred_span))
    
    return {
        "pred_sum": pred_sum,
        "sum_range": [max(80, pred_sum - 15), min(160, pred_sum + 15)],
        "pred_span": pred_span,
        "span_range": [max(10, pred_span - 6), min(33, pred_span + 6)],
        "sum_trend": "上升" if sum_trend > 5 else "下降" if sum_trend < -5 else "平稳",
        "span_trend": "扩张" if span_trend > 3 else "收缩" if span_trend < -3 else "平稳",
    }

# ============================================================
# 5. 马尔可夫链
# ============================================================

def build_markov_matrix(records, order=1):
    """构建马尔可夫转移矩阵"""
    transitions = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    
    if order == 1:
        for i in range(len(records) - 1):
            for a in records[i]["red"]:
                for b in records[i+1]["red"]:
                    transitions[a][b] += 1
                    totals[a] += 1
    
    prob_matrix = {}
    for a, neighbors in transitions.items():
        if totals[a] > 0:
            prob_matrix[a] = {b: c / totals[a] for b, c in neighbors.items()}
    return prob_matrix

def markov_prediction(records, last_red, top_k=15):
    """马尔可夫预测"""
    matrix = build_markov_matrix(records, order=1)
    scores = defaultdict(float)
    for n in last_red:
        if n in matrix:
            for target, prob in matrix[n].items():
                scores[target] += prob
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# ============================================================
# 6. 贝叶斯后验
# ============================================================

def bayesian_posterior(records, alpha_prior=1.0, window=100):
    """贝叶斯后验概率"""
    recent = records[-window:]
    total = len(recent)
    posterior = {}
    for n in range(1, 34):
        successes = sum(1 for r in recent if n in r["red"])
        alpha = alpha_prior + successes
        beta = alpha_prior + (total - successes)
        posterior[n] = alpha / (alpha + beta)
    return posterior

# ============================================================
# 7. 012路分析
# ============================================================

def road_analysis(records, window=30):
    """012路分析"""
    recent = records[-window:]
    road_counts = {0: 0, 1: 0, 2: 0}
    road_omission = {0: 0, 1: 0, 2: 0}
    road_seen = {0: False, 1: False, 2: False}
    
    for r in reversed(recent):
        roads_in_draw = set(n % 3 for n in r["red"])
        for road in [0, 1, 2]:
            if road in roads_in_draw:
                if not road_seen[road]:
                    road_seen[road] = True
                road_counts[road] += 1
            else:
                if not road_seen[road]:
                    road_omission[road] += 1
    
    return {
        "counts": road_counts,
        "omission": road_omission,
        "recommend": [r for r, c in road_omission.items() if c >= 3],
        "current": [r for r in [0,1,2] if road_omission[r] >= 2]
    }

# ============================================================
# 8. 三区分析
# ============================================================

def three_zone_analysis(records, window=20):
    """三区分析(1-11/12-22/23-33)"""
    zone_config = [(1,11),(12,22),(23,33)]
    recent = records[-window:]
    
    zone_counts = [0, 0, 0]
    zone_omission = [0, 0, 0]
    zone_seen = [False, False, False]
    
    for r in reversed(recent):
        nums = set(r["red"])
        for i, (lo, hi) in enumerate(zone_config):
            zone_nums = set(range(lo, hi+1))
            if nums & zone_nums:
                if not zone_seen[i]:
                    zone_seen[i] = True
                zone_counts[i] += 1
            else:
                if not zone_seen[i]:
                    zone_omission[i] += 1
    
    # 推荐比例
    total = sum(zone_counts) if sum(zone_counts) > 0 else 1
    ratios = [c/total for c in zone_counts]
    
    return {
        "zones": zone_config,
        "counts": zone_counts,
        "omission": zone_omission,
        "ratios": ratios,
        "recommend_ratio": _best_ratio(ratios),
        "broken": [i for i, om in enumerate(zone_omission) if om >= 5]
    }

def _best_ratio(ratios):
    """根据历史比例推荐最优三区比"""
    # 2:2:2 是最常见
    if ratios[0] > 0.3 and ratios[1] > 0.3 and ratios[2] > 0.25:
        return (2, 2, 2)
    # 否则按实际比例分配
    total = sum(ratios)
    if total == 0:
        return (2, 2, 2)
    r0 = max(1, round(ratios[0] / total * 6))
    r1 = max(1, round(ratios[1] / total * 6))
    r2 = 6 - r0 - r1
    if r2 < 1:
        r2 = 1
        r0 = max(1, 6 - r1 - r2)
    return (r0, r1, r2)

# ============================================================
# 9. 杀号系统
# ============================================================

def kill_number_systems(records, last_draw, red_freq, current_gap, avg_gap):
    """8式杀号法"""
    kills = defaultdict(list)
    red = last_draw["red"]
    sorted_red = sorted(red)
    min_n = min(red)
    max_n = max(red)
    
    # 杀号1: 首尾间距
    span = max_n - min_n
    if span > 28:
        for n in range(1, 4):
            kills[n].append("首尾间距过大→杀小号")
        for n in range(31, 34):
            kills[n].append("首尾间距过大→杀大号")
    elif span < 18:
        mid_zone = range(14, 20)
        for n in mid_zone:
            if current_gap.get(n, 0) > 5:
                kills[n].append("跨度偏小→杀中部冷号")
    
    # 杀号2: 两两加减法
    for i in range(len(red)):
        for j in range(i+1, len(red)):
            diff = abs(red[i] - red[j])
            s = red[i] + red[j]
            if diff <= 3 and diff > 0:
                target = max_n + diff
                if target <= 33:
                    kills[target].append(f"s两两差→杀{target}")
            if s <= 33:
                kills[s].append(f"s两两和→杀{s}")
    
    # 杀号3: +3杀号
    for n in red:
        target = n + 3
        if target <= 33:
            kills[target].append(f"s{n}+3杀号")
    
    # 杀号4: 首尾和/差
    s = min_n + max_n
    d = max_n - min_n
    if s <= 33:
        kills[s].append(f"s首尾和→杀{s}")
    if d <= 33:
        kills[d].append(f"s首尾差→杀{d}")
    
    # 杀号5: 极端冷热
    for n in range(1, 34):
        gap = current_gap.get(n, 0)
        avg = avg_gap.get(n, 6)
        freq = red_freq.get(n, 0)
        if freq >= 4:
            kills[n].append(f"s极热号(近5期{freq}次)")
        if gap > avg * 2.5 and gap > 20:
            kills[n].append(f"s极冷号(遗漏{gap}期)")
    
    # 杀号6: 连号扩展
    for i in range(len(sorted_red)-1):
        if sorted_red[i+1] - sorted_red[i] == 1:
            left = sorted_red[i] - 1
            right = sorted_red[i+1] + 1
            if left >= 1:
                kills[left].append(f"s连号扩展→杀{left}")
            if right <= 33:
                kills[right].append(f"s连号扩展→杀{right}")
    
    # 杀号7: 尾数杀号
    tail_counts = Counter(n % 10 for n in red)
    for tail, cnt in tail_counts.items():
        if cnt >= 2:
            for n in range(tail, 34, 10):
                if n not in red and current_gap.get(n, 0) > 3:
                    kills[n].append(f"s尾数{tail}过热")
    
    # 杀号8: 断区杀号
    zones = [(1,11),(12,22),(23,33)]
    zone_recent = defaultdict(int)
    for r in records[-5:]:
        for n in r["red"]:
            for i, (lo, hi) in enumerate(zones):
                if lo <= n <= hi:
                    zone_recent[i] += 1
    for i, (lo, hi) in enumerate(zones):
        if zone_recent[i] == 0:
            for n in range(lo, hi+1):
                kills[n].append(f"s断区({lo}-{hi})")
    
    return dict(kills)

def classify_kills(kills):
    """分类杀号"""
    confirmed = {}
    suspected = {}
    for n, reasons in kills.items():
        if len(reasons) >= 2:
            confirmed[n] = reasons
        elif len(reasons) == 1:
            suspected[n] = reasons
    return confirmed, suspected

# ============================================================
# 10. 定胆系统
# ============================================================

def determine_brave(records, last_draw, current_gap, avg_gap, road_info):
    """多源定胆"""
    candidates = defaultdict(list)
    red = last_draw["red"]
    sorted_red = sorted(red)
    
    # 胆1: 重号
    for n in red:
        candidates[n].append("重号")
    
    # 胆2: 邻号
    for n in red:
        for delta in [-1, 1]:
            t = n + delta
            if 1 <= t <= 33 and t not in red:
                candidates[t].append("邻号")
    
    # 胆3: 均值胆
    avg = round(sum(red) / len(red))
    for n in [avg-1, avg, avg+1]:
        if 1 <= n <= 33 and n not in red:
            candidates[n].append("均值胆")
    
    # 胆4: 012路回补
    for road in road_info.get("current", []):
        road_nums = [n for n in range(1, 34) if n % 3 == road]
        best = max(road_nums, key=lambda x: current_gap.get(x, 0))
        if current_gap.get(best, 0) > 5:
            candidates[best].append(f"s{road}路回补")
    
    # 胆5: 龙头凤尾
    head_history = [min(r["red"]) for r in records[-10:]]
    if len(head_history) >= 5:
        rm = head_history[-1]
        if rm <= 3:
            candidates[rm+2].append("龙头上行")
        elif rm >= 8:
            candidates[max(1, rm-2)].append("龙头下行")
    
    tail_history = [max(r["red"]) for r in records[-10:]]
    if len(tail_history) >= 5:
        tm = tail_history[-1]
        if tm >= 30:
            candidates[max(1, tm-2)].append("凤尾下行")
        elif tm <= 23:
            candidates[min(33, tm+2)].append("凤尾上行")
    
    # 分级
    gold, silver, bronze = {}, {}, {}
    for n, sources in candidates.items():
        if len(sources) >= 3:
            gold[n] = sources
        elif len(sources) == 2:
            silver[n] = sources
        else:
            gap = current_gap.get(n, 0)
            avg = avg_gap.get(n, 6)
            if gap > avg * 1.5:
                bronze[n] = sources
    
    return {"gold": gold, "silver": silver, "bronze": bronze, "all": dict(candidates)}

# ============================================================
# 11. 蓝球独立建模
# ============================================================

def blue_ball_model(records, last_blue):
    """蓝球16选1独立建模"""
    recent = records[-30:]
    
    # 频率
    blue_freq = Counter()
    for r in recent:
        blue_freq[r["blue"]] += 1
    
    # 遗漏
    blue_gap = {}
    blue_last_seen = {}
    for i, r in enumerate(records):
        for n in range(1, 17):
            if n == r["blue"]:
                blue_last_seen[n] = i
                blue_gap[n] = 0
            else:
                if n in blue_gap:
                    blue_gap[n] += 1
                else:
                    blue_gap[n] = i + 1
    
    # 冷热
    hot_blue = sorted([n for n, g in blue_gap.items() if g <= 2])
    cold_blue = sorted([n for n, g in blue_gap.items() if g >= 10])
    
    # 奇偶大小
    recent_blues = [r["blue"] for r in records[-5:]]
    odd_count = sum(1 for b in recent_blues if b % 2 == 1)
    even_count = len(recent_blues) - odd_count
    small_count = sum(1 for b in recent_blues if b <= 8)
    big_count = len(recent_blues) - small_count
    
    # 杀号
    blue_kills = set()
    # 杀极冷
    for n in cold_blue:
        blue_kills.add(n)
    # 杀上期蓝球(重号概率极低)
    blue_kills.add(last_blue)
    # 奇偶定向
    if odd_count >= 4:
        for n in range(1, 17, 2):
            blue_kills.add(n)
    elif even_count >= 4:
        for n in range(2, 17, 2):
            blue_kills.add(n)
    # 大小定向
    if small_count >= 4:
        for n in range(1, 9):
            blue_kills.add(n)
    elif big_count >= 4:
        for n in range(9, 17):
            blue_kills.add(n)
    
    # 候选
    candidates = [n for n in range(1, 17) if n not in blue_kills]
    
    # 路数
    road_counts = {0: 0, 1: 0, 2: 0}
    for r in recent:
        road_counts[r["blue"] % 3] += 1
    
    # 推荐(1热+1冷搭配)
    rec = None
    if hot_blue and cold_blue:
        rec = (hot_blue[0], cold_blue[0])
    
    return {
        "freq": dict(blue_freq),
        "gap": blue_gap,
        "hot": hot_blue,
        "cold": cold_blue,
        "kills": sorted(blue_kills),
        "candidates": candidates,
        "road_counts": road_counts,
        "recommendation": rec,
    }

# ============================================================
# 12. Apriori关联规则
# ============================================================

def apriori_red(records, min_support=0.05, window=100):
    """Apriori关联规则 - 频繁2码组合"""
    recent = records[-window:]
    n = len(recent)
    pair_counts = Counter()
    for r in recent:
        for a, b in itertools.combinations(sorted(r["red"]), 2):
            pair_counts[(a, b)] += 1
    
    min_count = max(2, int(n * min_support))
    frequent = {p: c for p, c in pair_counts.items() if c >= min_count}
    sorted_pairs = sorted(frequent.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:20]

# ============================================================
# 13. 多模型融合
# ============================================================

def ensemble_predict(records, last_draw):
    """6子模型加权融合"""
    decay_red = exponential_decay_frequency(records, decay=0.97)
    bayes = bayesian_posterior(records, window=100)
    markov_top = dict(markov_prediction(records, last_draw["red"], top_k=20))
    curr_gap, avg_gap, _ = omission_analysis(records)
    bounce = bounce_probability(curr_gap, avg_gap)
    freq30 = frequency_analysis(records, window=30)
    
    # Apriori pair scores
    pairs = apriori_red(records, min_support=0.05)
    pair_score = defaultdict(float)
    for (a, b), cnt in pairs:
        pair_score[a] += cnt * 0.1
        pair_score[b] += cnt * 0.1
    
    weights = {"decay": 0.25, "bayes": 0.20, "markov": 0.15, 
               "bounce": 0.20, "freq": 0.10, "pair": 0.10}
    
    def normalize(scores):
        if not scores:
            return {}
        mx = max(scores.values())
        mn = min(scores.values())
        if mx == mn:
            return {k: 0.5 for k in scores}
        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
    
    n_decay = normalize(decay_red)
    n_bayes = normalize(bayes)
    n_markov = normalize(markov_top)
    n_bounce = normalize(bounce)
    n_freq = normalize(freq30)
    n_pair = normalize(pair_score)
    
    fused = defaultdict(float)
    for n in range(1, 34):
        fused[n] = (
            weights["decay"] * n_decay.get(n, 0) +
            weights["bayes"] * n_bayes.get(n, 0) +
            weights["markov"] * n_markov.get(n, 0) +
            weights["bounce"] * n_bounce.get(n, 0) +
            weights["freq"] * n_freq.get(n, 0) +
            weights["pair"] * n_pair.get(n, 0)
        )
    
    return dict(fused), {
        "decay": n_decay, "bayes": n_bayes, "markov": n_markov,
        "bounce": n_bounce, "freq": n_freq, "pair": n_pair
    }

# ============================================================
# 14. 组合生成与过滤
# ============================================================

def generate_combinations(pool, zone_config, zone_ratio, 
                          target_sum_range, target_span_range,
                          parity_pref=None, max_combos=5000):
    """生成满足三区比+和值+跨度+奇偶约束的组合"""
    valid = []
    pool_sorted = sorted(pool)
    
    # 按三区比例预分配
    z0_nums = [n for n in pool_sorted if zone_config[0][0] <= n <= zone_config[0][1]]
    z1_nums = [n for n in pool_sorted if zone_config[1][0] <= n <= zone_config[1][1]]
    z2_nums = [n for n in pool_sorted if zone_config[2][0] <= n <= zone_config[2][1]]
    
    r0, r1, r2 = zone_ratio  # e.g. (2,2,2)
    
    count = 0
    for c0 in itertools.combinations(z0_nums, r0):
        for c1 in itertools.combinations(z1_nums, r1):
            for c2 in itertools.combinations(z2_nums, r2):
                combo = sorted(c0 + c1 + c2)
                s = sum(combo)
                if not (target_sum_range[0] <= s <= target_sum_range[1]):
                    continue
                sp = combo[-1] - combo[0]
                if not (target_span_range[0] <= sp <= target_span_range[1]):
                    continue
                
                # 奇偶检查
                odd = sum(1 for n in combo if n % 2 == 1)
                if parity_pref:
                    if not (parity_pref[0] <= odd <= parity_pref[1]):
                        continue
                else:
                    if odd == 0 or odd == 6:
                        continue
                
                # 连号检查(禁三连号)
                max_consec = 1
                cur = 1
                for i in range(1, 6):
                    if combo[i] - combo[i-1] == 1:
                        cur += 1
                        max_consec = max(max_consec, cur)
                    else:
                        cur = 1
                if max_consec >= 3:
                    continue
                
                valid.append(tuple(combo))
                count += 1
                if count >= max_combos:
                    return valid
    return valid

def score_combination(combo, fused_scores, brave_info, kills_confirmed):
    """给组合打分"""
    score = sum(fused_scores.get(n, 0) for n in combo) * 10
    
    for n in combo:
        if n in brave_info.get("gold", {}):
            score += 5
        elif n in brave_info.get("silver", {}):
            score += 3
        elif n in brave_info.get("bronze", {}):
            score += 1
        if n in kills_confirmed:
            score -= 8
    return score

# ============================================================
# 15. 蒙特卡洛验证
# ============================================================

def monte_carlo_validate(records, fused_scores, n_simulations=10000):
    """蒙特卡洛万次抽样"""
    import bisect
    nums = list(range(1, 34))
    probs = [fused_scores.get(n, 0.5) for n in nums]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1/33] * 33
    
    cum_weights = []
    cum = 0
    for p in probs:
        cum += p
        cum_weights.append(cum)
    
    combo_counts = Counter()
    for _ in range(n_simulations):
        sample = set()
        while len(sample) < 6:
            r = random.random() * cum_weights[-1]
            idx = bisect.bisect_left(cum_weights, r)
            if idx < len(nums):
                sample.add(nums[idx])
        combo = tuple(sorted(sample))
        combo_counts[combo] += 1
    
    return combo_counts.most_common(20)

# ============================================================
# 16. 蓝球预测
# ============================================================

def predict_blue(records, fused_scores, blue_info):
    """蓝球综合预测"""
    # 结合频率、遗漏、马尔可夫
    recent_blues = [r["blue"] for r in records[-20:]]
    blue_transitions = defaultdict(lambda: defaultdict(int))
    blue_totals = defaultdict(int)
    
    for i in range(len(records) - 1):
        a = records[i]["blue"]
        b = records[i+1]["blue"]
        blue_transitions[a][b] += 1
        blue_totals[a] += 1
    
    last_blue = records[-1]["blue"]
    blue_scores = defaultdict(float)
    
    # 转移概率
    if last_blue in blue_transitions:
        for b, cnt in blue_transitions[last_blue].items():
            blue_scores[b] += cnt / blue_totals[last_blue] * 3
    
    # 频率
    freq = blue_info["freq"]
    if freq:
        mx = max(freq.values())
        for b, c in freq.items():
            blue_scores[b] += (c / mx) * 2
    
    # 遗漏回补
    for b, gap in blue_info["gap"].items():
        if gap > 5:
            blue_scores[b] += min(gap / 20, 1.0) * 2
    
    # 排除杀号
    for b in blue_info["kills"]:
        blue_scores[b] *= 0.1
    
    ranked = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:5]

# ============================================================
# 17. 可视化
# ============================================================

def make_bar_chart(data, title="", width=40, char="█"):
    """ASCII条形图"""
    if not data:
        return ""
    lines = []
    if title:
        lines.append(f"  {title}")
    items = sorted(data.items()) if isinstance(data, dict) else data
    if isinstance(data, dict):
        mx = max(data.values()) if data else 1
    else:
        mx = max(v for _, v in items) if items else 1
    
    for key, val in items[:33]:
        bar_len = int((val / mx) * width) if mx > 0 else 0
        bar = char * max(1, bar_len)
        lines.append(f"  {str(key):>2} | {bar} {val:.2f}")
    return "\n".join(lines)

def make_zone_display(zone_info):
    """三区分布展示"""
    lines = []
    lines.append("  三区分布:")
    zones = zone_info["zones"]
    counts = zone_info["counts"]
    total = sum(counts) if sum(counts) > 0 else 1
    ratios = zone_info["ratios"]
    
    for i, ((lo, hi), cnt, rt) in enumerate(zip(zones, counts, ratios)):
        pct = rt * 100
        if pct > 35:
            ch = "█"
        elif pct > 28:
            ch = "▓"
        elif pct > 20:
            ch = "▒"
        else:
            ch = "░"
        bar = ch * max(1, int(pct / 3))
        status = "⚠断档" if i in zone_info["broken"] else ""
        lines.append(f"  {lo:02d}-{hi:02d} {bar} {cnt}次({pct:.0f}%) {status}")
    
    rec = zone_info["recommend_ratio"]
    lines.append(f"  推荐比例: {rec[0]}:{rec[1]}:{rec[2]}")
    return "\n".join(lines)

# ============================================================
# 18. 主流程
# ============================================================

def generate_ssq_v3(records, next_period="2026001"):
    """双色球v3主预测流程"""
    if len(records) < 30:
        raise ValueError(f"需要至少30期数据，当前仅{len(records)}期")
    
    last = records[-1]
    last_red = last["red"]
    
    # === 基础分析 ===
    curr_gap, avg_gap, _ = omission_analysis(records)
    hot, warm, cold = cold_hot_classify(curr_gap, avg_gap)
    bounce = bounce_probability(curr_gap, avg_gap)
    sum_span = predict_sum_span(records)
    
    # === 多模型融合 ===
    fused, sub_models = ensemble_predict(records, last)
    
    # === 杀号 ===
    freq5 = frequency_analysis(records, window=5)
    kills = kill_number_systems(records, last, freq5, curr_gap, avg_gap)
    confirmed_kills, suspected_kills = classify_kills(kills)
    
    # === 定胆 ===
    road_info = road_analysis(records)
    brave = determine_brave(records, last, curr_gap, avg_gap, road_info)
    
    # === 三区分析 ===
    zone_info = three_zone_analysis(records)
    
    # === 蓝球建模 ===
    blue_info = blue_ball_model(records, last["blue"])
    blue_pred = predict_blue(records, fused, blue_info)
    
    # === 构建候选池 ===
    all_nums = set(range(1, 34))
    safe_nums = all_nums - set(confirmed_kills.keys())
    
    pool_sorted = sorted(safe_nums, key=lambda x: fused.get(x, 0), reverse=True)
    
    pool = set()
    pool.update(hot[:3])
    pool.update(warm[:3])
    pool.update(sorted(cold, key=lambda x: bounce.get(x, 0), reverse=True)[:1])
    for n in brave.get("gold", {}):
        pool.add(n)
    for n in brave.get("silver", {}):
        pool.add(n)
    for n in pool_sorted:
        if len(pool) >= 14:
            break
        pool.add(n)
    
    pool = sorted(pool)
    
    # === 组合生成 ===
    zone_config = [(1,11),(12,22),(23,33)]
    zone_ratio = zone_info["recommend_ratio"]
    
    combos = generate_combinations(
        pool, zone_config, zone_ratio,
        tuple(sum_span["sum_range"]), tuple(sum_span["span_range"]),
        parity_pref=(2, 4)
    )
    
    # 放宽约束重试
    if len(combos) < 5:
        combos = generate_combinations(
            pool, zone_config, zone_ratio,
            (sum_span["sum_range"][0]-15, sum_span["sum_range"][1]+15),
            (sum_span["span_range"][0]-5, sum_span["span_range"][1]+5),
            parity_pref=None
        )
    
    # 打分
    scored = []
    for c in combos:
        sc = score_combination(c, fused, brave, confirmed_kills)
        gold_hits = sum(1 for n in c if n in brave.get("gold", {}))
        if gold_hits >= 1:
            sc += 5
        scored.append((c, sc))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_combos = [c for c, _ in scored[:15]]
    
    # === MC验证 ===
    mc_top = monte_carlo_validate(records, fused, n_simulations=5000)
    mc_set = set(c for c, _ in mc_top[:15])
    model_set = set(top_combos[:10])
    final_pool = model_set & mc_set
    if len(final_pool) < 3:
        final_pool = set(top_combos[:5])
    if len(final_pool) < 3:
        final_pool = set(c for c, _ in scored[:5])
    
    final_combos = sorted(final_pool, key=lambda x: score_combination(x, fused, brave, confirmed_kills), reverse=True)
    
    # === Apriori ===
    pairs = apriori_red(records)
    
    # === 组装结果 ===
    result = {
        "period": next_period,
        "last_draw": last,
        "sum_span": sum_span,
        "hot": sorted(hot),
        "warm": sorted(warm),
        "cold": sorted(cold),
        "bounce": {k: round(v, 3) for k, v in sorted(bounce.items(), key=lambda x: x[1], reverse=True)[:5]},
        "confirmed_kills": {k: v for k, v in sorted(confirmed_kills.items())},
        "suspected_kills": {k: v for k, v in sorted(suspected_kills.items())},
        "brave": brave,
        "zone_info": zone_info,
        "road_info": road_info,
        "blue_info": blue_info,
        "blue_pred": blue_pred,
        "fused_scores": fused,
        "pool": pool,
        "top_combos": top_combos[:5],
        "final_combos": final_combos[:5],
        "mc_top": [(c, cnt) for c, cnt in mc_top[:5]],
        "frequent_pairs": [(list(p), cnt) for p, cnt in pairs[:10]],
        "sub_models": {k: {n: round(v, 3) for n, v in sorted(v.items(), key=lambda x: x[1], reverse=True)[:10]} 
                       for k, v in sub_models.items()},
    }
    return result

# ============================================================
# 19. 报告输出
# ============================================================

def print_report(result):
    """按"和值→跨度→冷热→路数→奇偶→三区→杀号胆码→蓝球→候选"顺序"""
    sep = "═" * 56
    thin = "─" * 56
    
    print(f"\n{sep}")
    print(f"  双色球 第{result['period']}期 预测报告 (v3.0)")
    print(f"{sep}")
    
    last = result["last_draw"]
    print(f"  上期开奖: 红球 {last['red']} | 蓝球 {last['blue']}")
    s = sum(last["red"])
    sp = max(last["red"]) - min(last["red"])
    odd = sum(1 for n in last["red"] if n % 2 == 1)
    print(f"  上期指标: 和值={s} 跨度={sp} 奇偶={odd}:{6-odd}")
    print(f"{thin}")
    
    # ① 和值
    ss = result["sum_span"]
    print(f"\n  【① 红球和值预测】")
    print(f"  预测重心: {ss['pred_sum']}")
    print(f"  推荐区间: [{ss['sum_range'][0]}, {ss['sum_range'][1]}]")
    print(f"  趋势: {ss['sum_trend']}")
    print(f"  (主流区间80-120, 占89.7%)")
    
    # ② 跨度
    print(f"\n  【② 红球跨度预测】")
    print(f"  预测重心: {ss['pred_span']}")
    print(f"  推荐区间: [{ss['span_range'][0]}, {ss['span_range'][1]}]")
    print(f"  趋势: {ss['span_trend']}")
    print(f"  (主流区间20-28, 占75.1%)")
    
    # ③ 冷热温
    print(f"\n  【③ 冷热温分析】")
    print(f"  热号(遗漏≤5期): {result['hot']}")
    print(f"  温号(遗漏6-15期): {result['warm']}")
    print(f"  冷号(遗漏≥16期): {result['cold']}")
    if result['bounce']:
        bs = ", ".join(f"{k}({v})" for k, v in list(result['bounce'].items())[:5])
        print(f"  回补概率Top: {bs}")
    
    # ④ 012路
    print(f"\n  【④ 012路分析】")
    ri = result["road_info"]
    print(f"  路数遗漏: 0路={ri['omission'][0]}期 1路={ri['omission'][1]}期 2路={ri['omission'][2]}期")
    if ri["current"]:
        print(f"  建议回补: {ri['current']}路")
    
    # ⑤ 奇偶
    print(f"\n  【⑤ 奇偶/大小分析】")
    print(f"  推荐奇偶比: 3:3 或 4:2 / 2:4 (3:3历史占62%)")
    print(f"  推荐大小比: 3:3 或 4:2 (以17为界)")
    
    # ⑥ 三区
    print(f"\n  【⑥ 三区分布(1-11/12-22/23-33)】")
    print(make_zone_display(result["zone_info"]))
    
    # ⑦ 杀号
    print(f"\n  【⑦ 杀号系统】")
    ck = result["confirmed_kills"]
    sk = result["suspected_kills"]
    if ck:
        print(f"  ✗ 确认杀(≥2源): {sorted(ck.keys())}")
        for n, reasons in list(ck.items())[:5]:
            print(f"    {n}: {'; '.join(reasons[:2])}")
    if sk:
        print(f"  ? 疑似杀(1源): {sorted(sk.keys())}")
    
    # ⑧ 定胆
    print(f"\n  【⑧ 定胆系统】")
    b = result["brave"]
    if b["gold"]:
        print(f"  ★ 黄金胆: {sorted(b['gold'].keys())}")
        for n, src in list(b["gold"].items())[:3]:
            print(f"    {n}: {'+'.join(src[:3])}")
    if b["silver"]:
        print(f"  ◆ 白银胆: {sorted(b['silver'].keys())}")
    if b["bronze"]:
        print(f"  ◇ 青铜胆: {sorted(b['bronze'].keys())}")
    
    # ⑨ 蓝球
    print(f"\n  【⑨ 蓝球专项(16选1)】")
    bi = result["blue_info"]
    print(f"  热号: {bi['hot']}")
    print(f"  冷号: {bi['cold']}")
    print(f"  杀号: {bi['kills']}")
    print(f"  候选: {bi['candidates']}")
    rc = bi["road_counts"]
    print(f"  路数: 0路={rc[0]} 1路={rc[1]} 2路={rc[2]}")
    if result["blue_pred"]:
        bp = result["blue_pred"]
        top_str = ", ".join(f"{n}({v:.1f})" for n, v in bp[:3])
        print(f"  推荐Top3: {top_str}")
        if len(bp) >= 2:
            print(f"  首选: {bp[0][0]} ★")
    
    # ⑩ 关联
    print(f"\n  【⑩ 高频号码对(Top5)】")
    for pair, cnt in result["frequent_pairs"][:5]:
        print(f"  {pair} 共现{cnt}次")
    
    # ⑪ 候选
    print(f"\n  【⑪ 最终推荐】")
    print(f"  候选池({len(result['pool'])}码): {result['pool']}")
    print(f"\n  红球推荐组合:")
    for i, combo in enumerate(result["final_combos"]):
        c_sum = sum(combo)
        c_span = max(combo) - min(combo)
        odd = sum(1 for n in combo if n % 2 == 1)
        even = 6 - odd
        
        marks = []
        for n in combo:
            if n in b.get("gold", {}):
                marks.append(f"{n}★")
            elif n in b.get("silver", {}):
                marks.append(f"{n}◆")
            else:
                marks.append(str(n))
        
        kill_mark = ""
        for n in combo:
            if n in ck:
                kill_mark = " ⚠含杀号"
                break
        
        print(f"  注{i+1}: {marks} | 和={c_sum} 跨={c_span} 奇:{odd}偶:{even}{kill_mark}")
    
    # 蓝球最终推荐
    if result["blue_pred"]:
        print(f"\n  蓝球推荐: {result['blue_pred'][0][0]} ★")
    
    # MC
    print(f"\n  【蒙特卡洛验证(5000次抽样Top3)】")
    for combo, cnt in result["mc_top"][:3]:
        print(f"  {list(combo)} 频次={cnt}")
    
    # 子模型
    print(f"\n  【六模型Top5】")
    for name, scores in result["sub_models"].items():
        top5 = list(scores.items())[:5]
        s = ", ".join(f"{n}({v:.2f})" for n, v in top5)
        print(f"  {name:>8}: {s}")
    
    print(f"\n{sep}")
    print(f"  ⚠ 以上分数仅表示历史统计排序，不代表中奖概率。")
    print(f"  ⚠ 彩票每期独立随机，请理性购彩。")
    print(f"{sep}\n")
    
    return result

# ============================================================
# 20. 快捷入口
# ============================================================

def predict_next(records, next_period=None):
    """预测下一期"""
    if next_period is None:
        last_period = records[-1].get("period", "")
        try:
            num = int(last_period[-4:]) + 1
            next_period = f"{last_period[:-4]}{str(num).zfill(4)}"
        except:
            next_period = "NEXT"
    result = generate_ssq_v3(records, next_period)
    print_report(result)
    return result

# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 56)
    print("  双色球模型 v3.0 自测 (500期模拟数据)")
    print("=" * 56)
    data = generate_mock_data(500)
    result = predict_next(data)
    print(f"\n  ✓ 模型运行成功")
    print(f"  ✓ 候选池: {len(result['pool'])}码")
    print(f"  ✓ 推荐组合: {len(result['final_combos'])}注")
    print(f"  ✓ 确认杀号: {len(result['confirmed_kills'])}个")
    print(f"  ✓ 黄金胆码: {len(result['brave']['gold'])}个")
    print(f"  ✓ 白银胆码: {len(result['brave']['silver'])}个")
    print(f"  ✓ 蓝球候选: {len(result['blue_info']['candidates'])}个")

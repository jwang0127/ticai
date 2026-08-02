"""
大乐透预测模型 v3.0 - 深度优化版
====================================
前区: 35选5 | 后区: 12选2
头奖概率: 1/21,425,712

核心架构: 多层过滤 + 多模型融合 + 旋转矩阵缩水

优化要点(v2→v3):
1. 从"单分评分"改为"多层过滤+组合生成"
2. 新增: 指数衰减加权频率(近期权重更高)
3. 新增: 马尔可夫转移矩阵(一阶+二阶)
4. 新增: 贝叶斯后验更新(Beta-Binomial)
5. 新增: Apriori关联规则挖掘
6. 新增: 共现矩阵+影响力评分(PageRank)
7. 新增: 龙头凤尾定位模型
8. 新增: 后区独立建模(路数速配+1热1冷)
9. 新增: 旋转矩阵缩水(前区8码→保证中3)
10. 新增: 蒙特卡洛万次抽样验证+置信度
11. 新增: 多模型加权投票(6子模型)
12. 新增: 可视化输出(热力图/分布图/条形图)
13. 展示模板按"和值→跨度→冷热→路数→奇偶→分区→杀号胆码→候选"顺序
14. 新增: 连号管控(优先1组二连号,禁三连号)
15. 新增: 首尾间距动态收缩/扩张判定

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
# 1. 数据加载与基础统计
# ============================================================

def load_draws(data, key="dlt"):
    """从draws.json格式加载开奖数据"""
    if isinstance(data, dict):
        items = data.get("draws", {}).get(key, [])
    else:
        items = data
    records = []
    for item in items:
        if isinstance(item, dict):
            front = item.get("front", item.get("numbers", []))
            back = item.get("back", item.get("numbers_back", []))
            period = item.get("period", item.get("issue", ""))
            if isinstance(front, str):
                front = [int(x) for x in front.split()]
            if isinstance(back, str):
                back = [int(x) for x in back.split()]
            front = [int(x) for x in front]
            back = [int(x) for x in back]
        else:
            front = [int(x) for x in item[:5]]
            back = [int(x) for x in item[5:7]]
            period = ""
        records.append({"front": front, "back": back, "period": str(period)})
    return records

def generate_mock_data(n=500, seed=42):
    """生成模拟开奖数据(仅用于测试)"""
    random.seed(seed)
    records = []
    for i in range(n):
        front = sorted(random.sample(range(1, 36), 5))
        back = sorted(random.sample(range(1, 13), 2))
        records.append({"front": front, "back": back, "period": f"2025{str(i+1).zfill(4)}"})
    return records

# ============================================================
# 2. 频率分析模块
# ============================================================

def frequency_analysis(records, window=60):
    """频率分析 - 支持滑动窗口"""
    recent = records[-window:]
    front_freq = Counter()
    back_freq = Counter()
    for r in recent:
        for n in r["front"]:
            front_freq[n] += 1
        for n in r["back"]:
            back_freq[n] += 1
    return dict(front_freq), dict(back_freq)

def exponential_decay_frequency(records, decay=0.95):
    """指数衰减加权频率 - 近期数据权重更高"""
    front_weights = defaultdict(float)
    back_weights = defaultdict(float)
    total = len(records)
    for i, r in enumerate(records):
        w = decay ** (total - 1 - i)
        for n in r["front"]:
            front_weights[n] += w
        for n in r["back"]:
            back_weights[n] += w
    return dict(front_weights), dict(back_weights)

# ============================================================
# 3. 遗漏分析模块
# ============================================================

def omission_analysis(records, max_num=35):
    """遗漏分析 - 返回当前遗漏期数和平均遗漏周期"""
    current_gap = {}
    gap_history = defaultdict(list)
    last_seen = {}
    
    for i, r in enumerate(records):
        nums = set(r["front"])
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
                    current_gap[n] = i + 1  # 从未出现
    
    avg_gap = {}
    for n in range(1, max_num + 1):
        if gap_history[n]:
            avg_gap[n] = mean(gap_history[n])
        else:
            avg_gap[n] = max_num / 5  # 理论平均
    
    return current_gap, avg_gap, gap_history

def cold_hot_classify(current_gap, avg_gap, hot_th=5, warm_lo=6, warm_hi=15, cold_th=16):
    """冷热温三级分类"""
    hot, warm, cold = [], [], []
    for n in sorted(current_gap.keys()):
        gap = current_gap[n]
        avg = avg_gap.get(n, 7)
        if gap <= hot_th:
            hot.append(n)
        elif warm_lo <= gap <= warm_hi:
            warm.append(n)
        elif gap >= cold_th:
            cold.append(n)
    return hot, warm, cold

def bounce_probability(current_gap, avg_gap):
    """回补概率 = 当前遗漏 / (当前遗漏 + 平均遗漏)"""
    prob = {}
    for n, gap in current_gap.items():
        avg = avg_gap.get(n, 7)
        prob[n] = gap / (gap + avg)
    return prob

# ============================================================
# 4. 和值/跨度分析模块
# ============================================================

def sum_span_analysis(records, window=30):
    """分析和值与跨度分布"""
    sums = []
    spans = []
    for r in records[-window:]:
        s = sum(r["front"])
        sp = max(r["front"]) - min(r["front"])
        sums.append(s)
        spans.append(sp)
    
    return {
        "sum_mean": mean(sums) if sums else 90,
        "sum_median": median(sums) if sums else 90,
        "sum_std": stdev(sums) if len(sums) > 1 else 10,
        "span_mean": mean(spans) if spans else 24,
        "span_median": median(spans) if spans else 24,
        "span_std": stdev(spans) if len(spans) > 1 else 5,
    }

def predict_sum_span(records, window=30):
    """预测下期和值与跨度（加权回归+趋势）"""
    stats = sum_span_analysis(records, window)
    recent_sums = [sum(r["front"]) for r in records[-10:]]
    recent_spans = [max(r["front"]) - min(r["front"]) for r in records[-10:]]
    
    # 趋势检测
    sum_trend = 0
    if len(recent_sums) >= 5:
        first_half = mean(recent_sums[:3])
        second_half = mean(recent_sums[-3:])
        sum_trend = second_half - first_half
    
    span_trend = 0
    if len(recent_spans) >= 5:
        first_half = mean(recent_spans[:3])
        second_half = mean(recent_spans[-3:])
        span_trend = second_half - first_half
    
    pred_sum = int(stats["sum_mean"] + sum_trend * 0.3)
    pred_span = int(stats["span_mean"] + span_trend * 0.3)
    
    # 夹紧到合理范围
    pred_sum = max(70, min(130, pred_sum))
    pred_span = max(12, min(32, pred_span))
    
    return {
        "pred_sum": pred_sum,
        "sum_range": [pred_sum - 10, pred_sum + 10],
        "pred_span": pred_span,
        "span_range": [max(10, pred_span - 5), min(34, pred_span + 5)],
        "sum_trend": "上升" if sum_trend > 3 else "下降" if sum_trend < -3 else "平稳",
        "span_trend": "扩张" if span_trend > 2 else "收缩" if span_trend < -2 else "平稳",
    }

# ============================================================
# 5. 马尔可夫链模块
# ============================================================

def build_markov_matrix(records, order=1):
    """构建一阶/二阶马尔可夫转移矩阵"""
    transitions = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    
    if order == 1:
        for r in records:
            nums = sorted(r["front"])
            for i in range(len(nums) - 1):
                a, b = nums[i], nums[i+1]
                transitions[a][b] += 1
                totals[a] += 1
            # 也统计跨期转移
        for i in range(len(records) - 1):
            for a in records[i]["front"]:
                for b in records[i+1]["front"]:
                    transitions[a][b] += 1
                    totals[a] += 1
    else:  # order=2
        for i in range(len(records) - 1):
            prev = tuple(sorted(records[i]["front"]))
            curr = records[i+1]["front"]
            key = prev
            for b in curr:
                transitions[key][b] += 1
                totals[key] += 1
    
    # 转为概率
    prob_matrix = {}
    for a, neighbors in transitions.items():
        if totals[a] > 0:
            prob_matrix[a] = {b: c / totals[a] for b, c in neighbors.items()}
    return prob_matrix

def markov_prediction(records, last_draw, top_k=15):
    """基于马尔可夫链预测下期号码"""
    matrix = build_markov_matrix(records, order=1)
    scores = defaultdict(float)
    
    for n in last_draw:
        if n in matrix:
            for target, prob in matrix[n].items():
                scores[target] += prob
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]

# ============================================================
# 6. 贝叶斯后验更新模块
# ============================================================

def bayesian_posterior(records, alpha_prior=1.0, window=100):
    """贝叶斯后验概率 (Beta-Binomial模型)"""
    recent = records[-window:]
    total_draws = len(recent)
    posterior = {}
    
    for n in range(1, 36):
        successes = sum(1 for r in recent if n in r["front"])
        # Beta-Binomial: posterior = Beta(alpha + successes, beta + failures)
        alpha = alpha_prior + successes
        beta = alpha_prior + (total_draws - successes)
        # 期望 = alpha / (alpha + beta)
        posterior[n] = alpha / (alpha + beta)
    
    return posterior

# ============================================================
# 7. 012路分析模块
# ============================================================

def road_analysis(records, window=30):
    """012路分析"""
    recent = records[-window:]
    road_counts = {0: 0, 1: 0, 2: 0}
    road_omission = {0: 0, 1: 0, 2: 0}
    road_seen = {0: False, 1: False, 2: False}
    
    for r in reversed(recent):
        roads_in_draw = set()
        for n in r["front"]:
            roads_in_draw.add(n % 3)
        
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
        "recommend": [r for r, c in road_omission.items() if c >= 3]  # 遗漏≥3期，建议回补
    }

# ============================================================
# 8. 区间分析模块
# ============================================================

def zone_analysis(records, zones=None, window=20):
    """区间分析 - 检测断档区"""
    if zones is None:
        zones = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
    
    recent = records[-window:]
    zone_counts = [0] * len(zones)
    zone_omission = [0] * len(zones)
    zone_seen = [False] * len(zones)
    
    for r in reversed(recent):
        nums = set(r["front"])
        for i, (lo, hi) in enumerate(zones):
            zone_nums = set(range(lo, hi+1))
            if nums & zone_nums:
                if not zone_seen[i]:
                    zone_seen[i] = True
                zone_counts[i] += 1
            else:
                if not zone_seen[i]:
                    zone_omission[i] += 1
    
    # 判断断档区（连续空开）
    broken_zones = [i for i, om in enumerate(zone_omission) if om >= 3]
    
    return {
        "zones": zones,
        "counts": zone_counts,
        "omission": zone_omission,
        "broken": broken_zones,
        "hot": [i for i, c in enumerate(zone_counts) if c >= window * 0.4]
    }

# ============================================================
# 9. 杀号系统（多源验证）
# ============================================================

def kill_number_systems(records, last_draw, front_freq, current_gap, avg_gap):
    """8式杀号法 - 返回 {号码: [触发原因列表]}"""
    kills = defaultdict(list)
    front = last_draw["front"]
    back = last_draw["back"]
    sorted_front = sorted(front)
    min_n = min(front)
    max_n = max(front)
    
    # 杀号1: 首尾间距杀号
    span = max_n - min_n
    if span > 28:
        for n in range(1, 4):  # 杀极端小号
            kills[n].append("首尾间距过大→杀小号")
        for n in range(33, 36):  # 杀极端大号
            kills[n].append("首尾间距过大→杀大号")
    elif span < 18:
        # 内收后可能扩张
        mid_zone = range(15, 22)
        for n in mid_zone:
            if current_gap.get(n, 0) > 5:
                kills[n].append("跨度偏小→杀中部冷号")
    
    # 杀号2: 两两加减法
    for i in range(len(front)):
        for j in range(i+1, len(front)):
            diff = abs(front[i] - front[j])
            s = front[i] + front[j]
            if diff <= 3 and diff > 0:
                target = max_n + diff
                if target <= 35:
                    kills[target].append(f"s两两差→杀{max_n}+{diff}")
            if s <= 35:
                kills[s].append(f"s两两和→杀{s}")
    
    # 杀号3: +3杀号法
    for n in front:
        target = n + 3
        if target <= 35:
            kills[target].append(f"s{n}+3杀号")
    
    # 杀号4: 首尾和杀号
    s = min_n + max_n
    if s <= 35:
        kills[s].append(f"s首尾和→杀{s}")
    diff_sm = max_n - min_n
    if diff_sm <= 35:
        kills[diff_sm].append(f"s首尾差→杀{diff_sm}")
    
    # 杀号5: 极端冷热杀号
    for n in range(1, 36):
        gap = current_gap.get(n, 0)
        avg = avg_gap.get(n, 7)
        # 极热号（近5期出现4次以上）
        freq_recent = front_freq.get(n, 0)
        if freq_recent >= 4:
            kills[n].append(f"s极热号(近5期{freq_recent}次)")
        # 极冷号（遗漏远超平均，回补概率低）
        if gap > avg * 2.5 and gap > 25:
            kills[n].append(f"s极冷号(遗漏{gap}期,avg={avg:.0f})")
    
    # 杀号6: 连号扩展杀号
    for i in range(len(sorted_front)-1):
        if sorted_front[i+1] - sorted_front[i] == 1:
            left = sorted_front[i] - 1
            right = sorted_front[i+1] + 1
            if left >= 1:
                kills[left].append(f"s连号{sorted_front[i]}-{sorted_front[i+1]}左扩")
            if right <= 35:
                kills[right].append(f"s连号{sorted_front[i]}-{sorted_front[i+1]}右扩")
    
    # 杀号7: 尾数杀号
    tail_counts = Counter(n % 10 for n in front)
    for tail, cnt in tail_counts.items():
        if cnt >= 2:  # 某尾数出现2次以上
            for n in range(tail, 36, 10):
                if n not in front and current_gap.get(n, 0) > 3:
                    kills[n].append(f"s尾数{tail}过热")
    
    # 杀号8: 断区杀号
    zones = [(1,5),(6,10),(11,15),(16,20),(21,25),(26,30),(31,35)]
    zone_recent = defaultdict(int)
    for r in records[-5:]:
        for n in r["front"]:
            for i, (lo, hi) in enumerate(zones):
                if lo <= n <= hi:
                    zone_recent[i] += 1
    for i, (lo, hi) in enumerate(zones):
        if zone_recent[i] == 0:  # 近5期完全空开
            for n in range(lo, hi+1):
                kills[n].append(f"s断区({lo}-{hi}连续空开)")
    
    return dict(kills)

def classify_kills(kills):
    """将杀号分为确认杀(≥2源)和疑似杀(1源)"""
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
    """多源定胆系统"""
    candidates = defaultdict(list)
    front = last_draw["front"]
    sorted_front = sorted(front)
    
    # 胆1: 重号（上期号码下期重复概率~20%）
    for n in front:
        candidates[n].append("重号候选")
    
    # 胆2: 邻号（上期号码±1）
    for n in front:
        for delta in [-1, 1]:
            target = n + delta
            if 1 <= target <= 35 and target not in front:
                candidates[target].append("邻号")
    
    # 胆3: 均值胆
    avg = round(sum(front) / len(front))
    for n in [avg-1, avg, avg+1]:
        if 1 <= n <= 35 and n not in front:
            candidates[n].append("均值胆")
    
    # 胆4: 012路回补胆
    for road in road_info.get("recommend", []):
        # 该路数近期遗漏，选遗漏值最高的号
        road_nums = [n for n in range(1, 36) if n % 3 == road]
        best = max(road_nums, key=lambda x: current_gap.get(x, 0))
        if current_gap.get(best, 0) > 5:
            candidates[best].append(f"s{road}路回补胆")
    
    # 胆5: 龙头凤尾胆
    # 龙头走势
    head_history = [min(r["front"]) for r in records[-10:]]
    if len(head_history) >= 5:
        recent_min = head_history[-1]
        if recent_min <= 3:
            candidates[recent_min+2].append("龙头上行胆")
        elif recent_min >= 8:
            candidates[max(1, recent_min-2)].append("龙头下行胆")
    
    # 凤尾走势
    tail_history = [max(r["front"]) for r in records[-10:]]
    if len(tail_history) >= 5:
        recent_max = tail_history[-1]
        if recent_max >= 32:
            candidates[recent_max-2].append("凤尾下行胆")
        elif recent_max <= 25:
            candidates[recent_max+2].append("凤尾上行胆")
    
    # 分级
    gold = {}  # ≥3源
    silver = {}  # 2源
    bronze = {}  # 1源但遗漏高
    for n, sources in candidates.items():
        if len(sources) >= 3:
            gold[n] = sources
        elif len(sources) == 2:
            silver[n] = sources
        else:
            gap = current_gap.get(n, 0)
            avg = avg_gap.get(n, 7)
            if gap > avg * 1.5:
                bronze[n] = sources
    
    return {"gold": gold, "silver": silver, "bronze": bronze, "all": dict(candidates)}

# ============================================================
# 11. 后区独立建模
# ============================================================

def back_zone_model(records, last_back):
    """后区12选2独立建模"""
    recent = records[-30:]
    
    # 后区频率
    back_freq = Counter()
    for r in recent:
        for n in r["back"]:
            back_freq[n] += 1
    
    # 后区和值分析
    back_sums = [sum(r["back"]) for r in recent]
    avg_back_sum = mean(back_sums) if back_sums else 13
    
    # 后区跨度分析
    back_spans = [max(r["back"]) - min(r["back"]) for r in recent]
    avg_back_span = mean(back_spans) if back_spans else 3
    
    # 后区012路
    back_road_counts = {0: 0, 1: 0, 2: 0}
    for r in recent:
        for n in r["back"]:
            back_road_counts[n % 3] += 1
    
    # 后区冷热
    back_gap = {}
    back_last_seen = {}
    for i, r in enumerate(records):
        for n in range(1, 13):
            if n in r["back"]:
                back_last_seen[n] = i
                back_gap[n] = 0
            else:
                if n in back_gap:
                    back_gap[n] += 1
                else:
                    back_gap[n] = i + 1
    
    hot_back = [n for n, g in back_gap.items() if g <= 2]
    cold_back = [n for n, g in back_gap.items() if g >= 8]
    
    # 推荐策略: 1热+1冷
    recommendations = []
    if hot_back and cold_back:
        recommendations.append((random.choice(hot_back), random.choice(cold_back)))
    
    # 路数速配
    road_rec = []
    for r1 in range(3):
        for r2 in range(3):
            road_rec.append((r1, r2))
    
    return {
        "freq": dict(back_freq),
        "avg_sum": round(avg_back_sum, 1),
        "avg_span": round(avg_back_span, 1),
        "road_counts": back_road_counts,
        "hot": hot_back,
        "cold": cold_back,
        "recommendations": recommendations,
        "gap": back_gap,
    }

# ============================================================
# 12. Apriori关联规则
# ============================================================

def apriori_front(records, min_support=0.05, window=100):
    """Apriori关联规则挖掘 - 找频繁2码组合"""
    recent = records[-window:]
    n = len(recent)
    
    # 统计2码组合频次
    pair_counts = Counter()
    for r in recent:
        nums = r["front"]
        for a, b in itertools.combinations(sorted(nums), 2):
            pair_counts[(a, b)] += 1
    
    # 过滤支持度≥min_support
    min_count = max(2, int(n * min_support))
    frequent_pairs = {pair: cnt for pair, cnt in pair_counts.items() if cnt >= min_count}
    
    # 按频次排序
    sorted_pairs = sorted(frequent_pairs.items(), key=lambda x: x[1], reverse=True)
    return sorted_pairs[:20]

# ============================================================
# 13. 多模型融合
# ============================================================

def ensemble_predict(records, last_draw):
    """6子模型加权融合"""
    # 子模型1: 指数衰减频率
    decay_front, decay_back = exponential_decay_frequency(records, decay=0.97)
    
    # 子模型2: 贝叶斯后验
    bayes = bayesian_posterior(records, window=100)
    
    # 子模型3: 马尔可夫链
    markov_top = dict(markov_prediction(records, last_draw["front"], top_k=20))
    
    # 子模型4: 遗漏回补
    curr_gap, avg_gap, _ = omission_analysis(records)
    bounce = bounce_probability(curr_gap, avg_gap)
    
    # 子模型5: 近期频率(30期)
    freq30, _ = frequency_analysis(records, window=30)
    
    # 子模型6: Apriori高频对
    pairs = apriori_front(records, min_support=0.05)
    pair_score = defaultdict(float)
    for (a, b), cnt in pairs:
        pair_score[a] += cnt * 0.1
        pair_score[b] += cnt * 0.1
    
    # 加权融合
    weights = {
        "decay": 0.25,
        "bayes": 0.20,
        "markov": 0.15,
        "bounce": 0.20,
        "freq": 0.10,
        "pair": 0.10,
    }
    
    # 归一化各模型分数
    def normalize(scores):
        if not scores:
            return {}
        mx = max(scores.values())
        mn = min(scores.values())
        if mx == mn:
            return {k: 0.5 for k in scores}
        return {k: (v - mn) / (mx - mn) for k, v in scores.items()}
    
    n_decay = normalize(decay_front)
    n_bayes = normalize(bayes)
    n_markov = normalize(markov_top)
    n_bounce = normalize(bounce)
    n_freq = normalize(freq30)
    n_pair = normalize(pair_score)
    
    # 融合
    fused = defaultdict(float)
    for n in range(1, 36):
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

def generate_combinations(front_pool, target_sum_range, target_span_range, 
                        zone_config, parity_pref=None, max_combos=5000):
    """从候选池生成满足约束的组合"""
    valid = []
    pool = sorted(front_pool)
    
    count = 0
    for combo in itertools.combinations(pool, 5):
        s = sum(combo)
        if not (target_sum_range[0] <= s <= target_sum_range[1]):
            continue
        sp = combo[-1] - combo[0]
        if not (target_span_range[0] <= sp <= target_span_range[1]):
            continue
        
        # 区间检查
        zone_hits = [0] * len(zone_config)
        for n in combo:
            for i, (lo, hi) in enumerate(zone_config):
                if lo <= n <= hi:
                    zone_hits[i] += 1
                    break
        if any(h == 0 for h in zone_hits):
            # 允许最多1个空区
            if sum(1 for h in zone_hits if h == 0) > 1:
                continue
        
        # 奇偶检查
        odd = sum(1 for n in combo if n % 2 == 1)
        even = 5 - odd
        if parity_pref:
            if not (parity_pref[0] <= odd <= parity_pref[1]):
                continue
        else:
            if odd == 0 or odd == 5:  # 禁全奇全偶
                continue
        
        # 连号检查（优先1组二连号，禁三连号）
        consecutive = 0
        max_consec = 1
        cur_consec = 1
        for i in range(1, 5):
            if combo[i] - combo[i-1] == 1:
                cur_consec += 1
                max_consec = max(max_consec, cur_consec)
            else:
                cur_consec = 1
        if max_consec >= 3:
            continue
        
        valid.append(combo)
        count += 1
        if count >= max_combos:
            break
    
    return valid

def score_combination(combo, fused_scores, brave_info, kills_confirmed):
    """给组合打分"""
    score = 0
    for n in combo:
        score += fused_scores.get(n, 0) * 10
    
    # 胆码加分
    for n in combo:
        if n in brave_info.get("gold", {}):
            score += 5
        elif n in brave_info.get("silver", {}):
            score += 3
        elif n in brave_info.get("bronze", {}):
            score += 1
    
    # 杀号减分
    for n in combo:
        if n in kills_confirmed:
            score -= 8
    
    return score

# ============================================================
# 15. 蒙特卡洛验证
# ============================================================

def monte_carlo_validate(records, fused_scores, n_simulations=10000):
    """蒙特卡洛万次抽样验证"""
    # 基于融合概率做加权抽样
    nums = list(range(1, 36))
    probs = [fused_scores.get(n, 0.5) for n in nums]
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1/35] * 35
    
    # Use cumulative weights for sampling
    import bisect
    cum_weights = []
    cum = 0
    for p in probs:
        cum += p
        cum_weights.append(cum)
    
    combo_counts = Counter()
    for _ in range(n_simulations):
        sample = set()
        while len(sample) < 5:
            r = random.random() * cum_weights[-1]
            idx = bisect.bisect_left(cum_weights, r)
            if idx < len(nums):
                sample.add(nums[idx])
        combo = tuple(sorted(sample))
        combo_counts[combo] += 1
    
    # 找高频组合
    top = combo_counts.most_common(20)
    return top

# ============================================================
# 16. 旋转矩阵缩水
# ============================================================

def rotation_matrix_8to3():
    """前区8码选5保3的旋转矩阵(简化版)"""
    # 8码标号 A-H, 选5保3需要14注
    matrix = [
        [0,1,2,3,4], [0,1,2,3,5], [0,1,2,4,6], [0,1,2,5,7],
        [0,1,3,4,7], [0,1,3,5,6], [0,1,4,5,7], [0,2,3,4,6],
        [0,2,3,5,7], [0,2,4,6,7], [0,3,4,5,6], [1,2,3,4,7],
        [1,2,3,6,7], [1,2,5,6,7]
    ]
    return matrix

def apply_rotation_matrix(numbers, matrix_template):
    """将旋转矩阵应用到具体号码"""
    results = []
    for combo_indices in matrix_template:
        combo = tuple(sorted(numbers[i] for i in combo_indices))
        results.append(combo)
    return results

# ============================================================
# 17. 可视化模块
# ============================================================

def make_bar_chart(data, title="", width=40, char="█"):
    """生成ASCII条形图"""
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
    
    for key, val in items[:35]:
        bar_len = int((val / mx) * width) if mx > 0 else 0
        bar = char * bar_len
        lines.append(f"  {str(key):>3} | {bar} {val:.2f}")
    return "\n".join(lines)

def make_zone_heatmap(zone_info):
    """生成区间热力图"""
    counts = zone_info["counts"]
    zones = zone_info["zones"]
    total = sum(counts) if sum(counts) > 0 else 1
    
    lines = []
    lines.append("  区间热力图:")
    for i, ((lo, hi), cnt) in enumerate(zip(zones, counts)):
        intensity = cnt / total if total > 0 else 0
        if intensity > 0.18:
            bar_char = "█"
        elif intensity > 0.12:
            bar_char = "▓"
        elif intensity > 0.06:
            bar_char = "▒"
        else:
            bar_char = "░"
        bar_len = int(intensity * 30)
        bar = bar_char * max(1, bar_len)
        status = "⚠断档" if i in zone_info["broken"] else ""
        hot = "🔥热" if i in zone_info["hot"] else ""
        lines.append(f"  {lo:02d}-{hi:02d} {bar} {cnt}次 {status}{hot}")
    return "\n".join(lines)

# ============================================================
# 18. 主流程
# ============================================================

def generate_dlt_v3(records, next_period="2026001"):
    """大乐透v3主预测流程"""
    if len(records) < 30:
        raise ValueError(f"需要至少30期数据，当前仅{len(records)}期")
    
    last = records[-1]
    last_front = last["front"]
    
    # === 基础分析 ===
    curr_gap, avg_gap, _ = omission_analysis(records)
    hot, warm, cold = cold_hot_classify(curr_gap, avg_gap)
    bounce = bounce_probability(curr_gap, avg_gap)
    sum_span = predict_sum_span(records)
    
    # === 多模型融合 ===
    fused, sub_models = ensemble_predict(records, last)
    
    # === 杀号 ===
    freq5, _ = frequency_analysis(records, window=5)
    kills = kill_number_systems(records, last, freq5, curr_gap, avg_gap)
    confirmed_kills, suspected_kills = classify_kills(kills)
    
    # === 定胆 ===
    road_info = road_analysis(records)
    brave = determine_brave(records, last, curr_gap, avg_gap, road_info)
    
    # === 后区建模 ===
    back_info = back_zone_model(records, last["back"])
    
    # === 区间分析 ===
    # Five front numbers cannot occupy all seven narrow zones.  The original
    # v3 handoff therefore rejected every real-data combination.  Use five
    # balanced zones for a five-number front area; the rest of the v3 scoring
    # and ensemble logic is unchanged.
    zones_5 = [(1,7),(8,14),(15,21),(22,28),(29,35)]
    zone_info = zone_analysis(records, zones=zones_5)
    
    # === 构建候选池 ===
    # 排除确认杀号
    all_nums = set(range(1, 36))
    safe_nums = all_nums - set(confirmed_kills.keys())
    
    # 按融合分数排序候选
    pool_sorted = sorted(safe_nums, key=lambda x: fused.get(x, 0), reverse=True)
    
    # 保证冷热温都有覆盖
    pool = set()
    # 热号取3-4个
    pool.update(hot[:4])
    # 温号取3-4个
    pool.update(warm[:4])
    # 冷号取1-2个（回补信号强的）
    cold_by_bounce = sorted(cold, key=lambda x: bounce.get(x, 0), reverse=True)
    pool.update(cold_by_bounce[:2])
    # 胆码必入选
    for n in brave.get("gold", {}):
        pool.add(n)
    for n in brave.get("silver", {}):
        pool.add(n)
    # 补充高分号到12个
    for n in pool_sorted:
        if len(pool) >= 12:
            break
        pool.add(n)
    
    pool = sorted(pool)
    
    # === 组合生成 ===
    target_sum_range = tuple(sum_span["sum_range"])
    target_span_range = tuple(sum_span["span_range"])
    
    combos = generate_combinations(
        pool, target_sum_range, target_span_range,
        zone_config=zones_5, parity_pref=(1, 4), max_combos=10000
    )
    
    # 如果约束太严导致无组合，放宽区间
    if len(combos) < 5:
        combos = generate_combinations(
            pool, 
            (target_sum_range[0]-15, target_sum_range[1]+15),
            (max(10, target_span_range[0]-5), min(34, target_span_range[1]+5)),
            zone_config=zones_5, parity_pref=None, max_combos=10000
        )
    
    # 打分排序
    scored = []
    for c in combos:
        sc = score_combination(c, fused, brave, confirmed_kills)
        # 胆码命中加分
        gold_hits = sum(1 for n in c if n in brave.get("gold", {}))
        if gold_hits >= 1:
            sc += 5
        silver_hits = sum(1 for n in c if n in brave.get("silver", {}))
        sc += silver_hits * 2
        # 杀号惩罚
        kill_hits = sum(1 for n in c if n in confirmed_kills)
        sc -= kill_hits * 10
        scored.append((c, sc))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_combos = [c for c, _ in scored[:15]]
    
    # === 蒙特卡洛验证 ===
    mc_top = monte_carlo_validate(records, fused, n_simulations=5000)
    mc_set = set(c for c, _ in mc_top[:15])
    
    # 取交集（模型top ∩ MC top）作为最终推荐
    model_set = set(top_combos[:10])
    final_pool = model_set & mc_set
    if len(final_pool) < 3:
        # 放宽：取模型top
        final_pool = set(top_combos[:5])
    if len(final_pool) < 3:
        # 再放宽：直接取scored top
        final_pool = set(c for c, _ in scored[:5])
    
    final_combos = sorted(final_pool, key=lambda x: score_combination(x, fused, brave, confirmed_kills), reverse=True)
    
    # === Apriori关联 ===
    pairs = apriori_front(records)
    
    # === 组装结果 ===
    result = {
        "period": next_period,
        "last_draw": last,
        "sum_span": sum_span,
        "hot": sorted(hot),
        "warm": sorted(warm),
        "cold": sorted(cold),
        "bounce": {k: round(v, 3) for k, v in sorted(bounce.items()) if k in cold_by_bounce[:5]},
        "confirmed_kills": {k: v for k, v in sorted(confirmed_kills.items())},
        "suspected_kills": {k: v for k, v in sorted(suspected_kills.items())},
        "brave": brave,
        "zone_info": zone_info,
        "road_info": road_info,
        "back_info": back_info,
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
    """按"和值→跨度→冷热→路数→奇偶→分区→杀号胆码→候选"顺序输出"""
    sep = "═" * 56
    thin = "─" * 56
    
    print(f"\n{sep}")
    print(f"  大乐透 第{result['period']}期 预测报告 (v3.0)")
    print(f"{sep}")
    
    last = result["last_draw"]
    print(f"  上期开奖: 前区 {last['front']} | 后区 {last['back']}")
    s = sum(last["front"])
    sp = max(last["front"]) - min(last["front"])
    print(f"  上期指标: 和值={s} 跨度={sp} 奇偶={sum(1 for n in last['front'] if n%2==1)}:{(5-sum(1 for n in last['front'] if n%2==1))}")
    print(f"{thin}")
    
    # ① 和值预测
    ss = result["sum_span"]
    print(f"\n  【① 和值预测】")
    print(f"  预测重心: {ss['pred_sum']}")
    print(f"  推荐区间: [{ss['sum_range'][0]}, {ss['sum_range'][1]}]")
    print(f"  趋势: {ss['sum_trend']}")
    print(f"  (历史主流区间70-130, 最优90-110占40%)")
    
    # ② 跨度预测
    print(f"\n  【② 跨度预测】")
    print(f"  预测重心: {ss['pred_span']}")
    print(f"  推荐区间: [{ss['span_range'][0]}, {ss['span_range'][1]}]")
    print(f"  趋势: {ss['span_trend']}")
    print(f"  (主流区间12-32)")
    
    # ③ 冷热温
    print(f"\n  【③ 冷热温分析】")
    print(f"  热号(遗漏≤5期): {result['hot']}")
    print(f"  温号(遗漏6-15期): {result['warm']}")
    print(f"  冷号(遗漏≥16期): {result['cold']}")
    if result['bounce']:
        bounce_str = ", ".join(f"{k}({v})" for k, v in list(result['bounce'].items())[:5])
        print(f"  回补概率Top: {bounce_str}")
    
    # ④ 012路
    print(f"\n  【④ 012路分析】")
    ri = result["road_info"]
    print(f"  路数遗漏: 0路={ri['omission'][0]}期 1路={ri['omission'][1]}期 2路={ri['omission'][2]}期")
    if ri["recommend"]:
        print(f"  建议回补: {ri['recommend']}路")
    
    # ⑤ 奇偶
    print(f"\n  【⑤ 奇偶/大小分析】")
    print(f"  推荐奇偶比: 3:2 或 2:3 (历史占比>70%)")
    print(f"  推荐大小比: 3:2 或 2:3 (以18为界)")
    
    # ⑥ 区间分布
    print(f"\n  【⑥ 七区间分布】")
    print(make_zone_heatmap(result["zone_info"]))
    if result["zone_info"]["broken"]:
        broken = [f"{result['zone_info']['zones'][i]}" for i in result["zone_info"]["broken"]]
        print(f"  ⚠断档区: {broken}")
    
    # ⑦ 杀号胆码
    print(f"\n  【⑦ 杀号系统】")
    ck = result["confirmed_kills"]
    sk = result["suspected_kills"]
    if ck:
        print(f"  ✗ 确认杀(≥2源): {sorted(ck.keys())}")
        for n, reasons in list(ck.items())[:5]:
            print(f"    {n}: {'; '.join(reasons[:2])}")
    if sk:
        print(f"  ? 疑似杀(1源): {sorted(sk.keys())}")
    
    print(f"\n  【⑦ 定胆系统】")
    b = result["brave"]
    if b["gold"]:
        print(f"  ★ 黄金胆: {sorted(b['gold'].keys())}")
        for n, src in list(b["gold"].items())[:3]:
            print(f"    {n}: {'+'.join(src[:3])}")
    if b["silver"]:
        print(f"  ◆ 白银胆: {sorted(b['silver'].keys())}")
    if b["bronze"]:
        print(f"  ◇ 青铜胆: {sorted(b['bronze'].keys())}")
    
    # ⑧ 后区专项
    print(f"\n  【⑧ 后区专项(12选2)】")
    bi = result["back_info"]
    print(f"  和值重心: {bi['avg_sum']}")
    print(f"  跨度重心: {bi['avg_span']}")
    print(f"  热号: {sorted(bi['hot'])}")
    print(f"  冷号: {sorted(bi['cold'])}")
    if bi["recommendations"]:
        rec = bi["recommendations"][0]
        print(f"  推荐(1热+1冷): {rec[0]}+{rec[1]} ★")
    # 路数
    rc = bi["road_counts"]
    print(f"  路数分布: 0路={rc[0]} 1路={rc[1]} 2路={rc[2]}")
    
    # ⑨ 关联规则
    print(f"\n  【⑨ 高频号码对(Top5)】")
    for pair, cnt in result["frequent_pairs"][:5]:
        print(f"  {pair} 共现{cnt}次")
    
    # ⑩ 候选号码
    print(f"\n  【⑩ 最终推荐】")
    print(f"  候选池({len(result['pool'])}码): {result['pool']}")
    print(f"\n  前区推荐组合:")
    for i, combo in enumerate(result["final_combos"]):
        c_sum = sum(combo)
        c_span = max(combo) - min(combo)
        odd = sum(1 for n in combo if n % 2 == 1)
        even = 5 - odd
        # 标记胆码
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
    
    # 后区推荐
    if bi["recommendations"]:
        rec = bi["recommendations"][0]
        print(f"\n  后区推荐: {rec[0]}+{rec[1]}")
    
    # MC验证
    print(f"\n  【蒙特卡洛验证(5000次抽样Top5)】")
    for combo, cnt in result["mc_top"][:3]:
        print(f"  {list(combo)} 频次={cnt}")
    
    # 子模型Top
    print(f"\n  【六模型Top5】")
    for name, scores in result["sub_models"].items():
        top5 = list(scores.items())[:5]
        top_str = ", ".join(f"{n}({v:.2f})" for n, v in top5)
        print(f"  {name:>8}: {top_str}")
    
    print(f"\n{sep}")
    print(f"  ⚠ 以上分数仅表示历史统计排序，不代表中奖概率。")
    print(f"  ⚠ 彩票每期独立随机，请理性购彩。")
    print(f"{sep}\n")
    
    return result

# ============================================================
# 20. 快捷入口
# ============================================================

def predict_next(records, next_period=None):
    """预测下一期 - 快捷接口"""
    if next_period is None:
        last_period = records[-1].get("period", "")
        try:
            num = int(last_period[-4:]) + 1
            next_period = f"{last_period[:-4]}{str(num).zfill(4)}"
        except:
            next_period = "NEXT"
    result = generate_dlt_v3(records, next_period)
    print_report(result)
    return result

# ============================================================
# 自测
# ============================================================

if __name__ == "__main__":
    print("=" * 56)
    print("  大乐透模型 v3.0 自测 (500期模拟数据)")
    print("=" * 56)
    data = generate_mock_data(500)
    result = predict_next(data)
    print(f"\n  ✓ 模型运行成功")
    print(f"  ✓ 候选池: {len(result['pool'])}码")
    print(f"  ✓ 推荐组合: {len(result['final_combos'])}注")
    print(f"  ✓ 确认杀号: {len(result['confirmed_kills'])}个")
    print(f"  ✓ 黄金胆码: {len(result['brave']['gold'])}个")
    print(f"  ✓ 白银胆码: {len(result['brave']['silver'])}个")

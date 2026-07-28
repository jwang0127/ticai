const STORAGE_KEY = "ticai-ticket-checker-v1";
const $ = s => document.querySelector(s);
let dashboard;

const money = value => `¥${Number(value).toLocaleString("zh-CN")}`;
const nums = text => (text.match(/\d+/g) || []).map(Number);
const clean = values => values.map(n => String(n).padStart(2, "0"));
const sameSet = (a, b) => a.slice().sort().join(",") === b.slice().sort().join(",");

function gameFromLine(line) {
  const normalized = line.trim().replace(/＋/g, "+");
  if (!normalized) return null;
  const names = [
    ["超级大乐透", "dlt"], ["大乐透", "dlt"], ["排列3", "pl3"], ["排列5", "pl5"],
    ["福彩3D", "fc3d"], ["快乐8", "kl8"], ["福彩快乐8", "kl8"],
    ["双色球", "ssq"], ["7星彩", "qxc"], ["七星彩", "qxc"],
  ];
  const found = names.find(([name]) => normalized.startsWith(name));
  if (!found) throw new Error(`无法识别彩种：${line}`);
  const [name, game] = found;
  let rest = normalized.slice(name.length).trim();
  let mode = "direct";
  const modeMatch = rest.match(/^(直选|组选3|组选6|选[一二三四五六七八九十]|选[1-9]|选10)\s*/);
  if (modeMatch) { mode = modeMatch[1]; rest = rest.slice(modeMatch[0].length); }
  const parts = rest.split("+");
  const values = nums(rest);
  if (game === "dlt") {
    if (parts.length !== 2 || nums(parts[0]).length !== 5 || nums(parts[1]).length !== 2) throw new Error(`大乐透格式错误：${line}`);
    return { name: "超级大乐透", game, mode, front: nums(parts[0]), back: nums(parts[1]), text: line };
  }
  if (game === "ssq") {
    if (parts.length !== 2 || nums(parts[0]).length !== 6 || nums(parts[1]).length !== 1) throw new Error(`双色球格式错误：${line}`);
    return { name: "双色球", game, mode, red: nums(parts[0]), blue: nums(parts[1]), text: line };
  }
  if (game === "kl8") {
    const count = mode === "选十" || mode === "选10" ? 10 : Number((mode.match(/[1-9]+/) || ["5"])[0]);
    if (values.length !== count) throw new Error(`快乐8${mode}需要${count}个号码：${line}`);
    return { name: "福彩快乐8", game, mode: `选${count}`, numbers: values, text: line };
  }
  const digits = game === "qxc" ? 7 : game === "pl5" ? 5 : 3;
  if (values.length !== digits || values.some(n => n < 0 || n > 9)) throw new Error(`${name}需要${digits}位数字：${line}`);
  if (mode === "组选3" && new Set(values).size !== 2) throw new Error("组选3必须是两位相同、一位不同");
  if (mode === "组选6" && new Set(values).size !== 3) throw new Error("组选6必须是三位各不相同");
  return { name, game, mode, number: values.join(""), text: line };
}

function latestFor(ticket) {
  const game = dashboard.games[ticket.game];
  const history = dashboard.draw_history?.[ticket.game] || [];
  const intendedIssue = ticket.targetIssue || game?.target_issue;
  const row = history.find(item => intendedIssue && String(item.issue) === String(intendedIssue))
    || history.find(item => item.draw_date === ticket.date);
  return game ? { game, row, issue: row?.issue || intendedIssue } : null;
}

function settle(ticket) {
  const latest = latestFor(ticket);
  if (!latest?.row) return { status: "pending", message: `第${latest?.issue || "目标"}期尚未开奖或数据未抓到` };
  const verified = dashboard.verification?.[ticket.game]?.status === "verified";
  if (!verified || dashboard.source_status !== "official_cross_verified") return { status: "pending", message: "官方来源待交叉确认" };
  const draw = latest.row.numbers.map(Number);
  if (ticket.game === "dlt") {
    const front = ticket.front.filter(n => draw.slice(0, 5).includes(n)).length;
    const back = ticket.back.filter(n => draw.slice(5).includes(n)).length;
    const table = { "5+2":["一等奖",0],"5+1":["二等奖",0],"5+0":["三等奖",5000],"4+2":["三等奖",5000],"4+1":["四等奖",300],"4+0":["五等奖",150],"3+2":["五等奖",150],"3+1":["六等奖",15],"2+2":["六等奖",15],"3+0":["七等奖",5],"2+1":["七等奖",5],"1+2":["七等奖",5],"0+2":["七等奖",5] };
    const hit = table[`${front}+${back}`]; return hit ? { status:"win", level:hit[0], amount:hit[1], message:hit[1] ? "固定奖估算" : "浮动奖，金额待官方公告" } : {status:"lose",message:`命中前区${front}个、后区${back}个`};
  }
  if (ticket.game === "ssq") {
    const red = ticket.red.filter(n => draw.slice(0,6).includes(n)).length, blue = Number(ticket.blue[0] === draw[6]);
    const key = `${red}+${blue}`, table = {"6+1":["一等奖",0],"6+0":["二等奖",0],"5+1":["三等奖",3000],"5+0":["四等奖",600],"4+1":["四等奖",200],"4+0":["五等奖",10],"3+1":["五等奖",10],"2+1":["六等奖",5],"1+1":["六等奖",5],"0+1":["六等奖",5]};
    const hit = table[key]; return hit ? {status:"win",level:hit[0],amount:hit[1],message:hit[1]?"固定奖估算":"浮动奖，金额待官方公告"}:{status:"lose",message:`红球${red}个、蓝球${blue?1:0}个`};
  }
  if (ticket.game === "kl8") {
    const hits = ticket.numbers.filter(n => draw.includes(n)).length, tables = {5:{5:1000,4:20,3:3},6:{6:2880,5:30,4:10,3:3},10:{10:5000000,9:8000,8:720,7:80,6:5,5:3}};
    const amount = tables[ticket.numbers.length]?.[hits]; return amount ? {status:"win",level:`选${ticket.numbers.length}中${hits}`,amount,message:"按官方固定奖估算"}:{status:"lose",message:`命中${hits}个`};
  }
  const actual = draw.join("");
  if (ticket.mode === "direct" || ticket.mode === "直选") { const amount = ticket.game === "fc3d" ? 1040 : ticket.game === "pl5" ? 100000 : ticket.game === "qxc" ? 5000000 : 1000; return ticket.number === actual ? {status:"win",level:"直选",amount,message:"固定奖估算"}:{status:"lose",message:`开奖号 ${actual}`}; }
  if (ticket.mode === "组选3" || ticket.mode === "组选6") return sameSet(ticket.number.split("").map(Number), draw) ? {status:"win",level:ticket.mode,amount:ticket.game === "fc3d"?(ticket.mode === "组选3"?346:173):(ticket.mode === "组选3"?320:160),message:"固定奖估算"}:{status:"lose",message:`开奖号 ${actual}`};
  if (ticket.game === "pl5" || ticket.game === "qxc") return ticket.number === actual ? {status:"win",level:"一等奖",amount:ticket.game === "pl5"?100000:5000000,message:"固定奖估算"}:{status:"lose",message:`开奖号 ${actual}`};
  return {status:"pending",message:"请补充玩法类型"};
}

function render() {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  if (!saved.length) { $("#results").innerHTML = '<p class="empty">还没有保存购票记录。</p>'; return; }
  $("#results").innerHTML = saved.map((ticket, index) => { const result = settle(ticket); const cls = result.status === "win" ? "win" : result.status === "pending" ? "pending" : ""; const planned = latestFor(ticket); return `<article class="ticket-card ${cls}"><div class="ticket-head"><span class="ticket-name">${ticket.name} ${ticket.mode === "direct" ? "直选" : ticket.mode}</span><span class="ticket-status">${result.status === "win" ? "中奖" : result.status === "pending" ? "待确认" : "未中奖"}</span></div><div class="ticket-meta">${ticket.text} · 购买 ${ticket.date || "旧记录"} · 第${planned?.issue || "待定"}期</div><div class="ticket-money">${result.status === "win" ? (result.amount ? money(result.amount) : "浮动奖待公告") : result.message}</div><div class="ticket-detail">${result.level || ""} ${result.message || ""}<button class="ghost remove-ticket" data-index="${index}">删除</button></div></article>`; }).join("");
}

async function init() { const response = await fetch("../assets/data/dashboard.json", {cache:"no-store"}); dashboard = await response.json(); render(); }
$("#ticket-form").addEventListener("submit", event => { event.preventDefault(); try { const tickets = $("#ticket-input").value.split(/\r?\n/).filter(Boolean).map(gameFromLine); const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); const date = new Intl.DateTimeFormat("en-CA", {timeZone:"Asia/Shanghai", year:"numeric", month:"2-digit", day:"2-digit"}).format(new Date()); tickets.forEach(ticket => { ticket.date = date; ticket.targetIssue = dashboard.games[ticket.game]?.target_issue || ""; ticket.expectedDrawDate = dashboard.games[ticket.game]?.next_draw_at?.slice(0, 10) || date; }); localStorage.setItem(STORAGE_KEY, JSON.stringify(saved.concat(tickets))); $("#ticket-input").value = ""; render(); } catch (error) { alert(error.message); } });
$("#clear-input").addEventListener("click", () => { $("#ticket-input").value = ""; });
$("#clear-history").addEventListener("click", () => { localStorage.removeItem(STORAGE_KEY); render(); });
$("#results").addEventListener("click", event => { const button = event.target.closest(".remove-ticket"); if (!button) return; const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); saved.splice(Number(button.dataset.index), 1); localStorage.setItem(STORAGE_KEY, JSON.stringify(saved)); render(); });
init().catch(error => { $("#results").innerHTML = `<p class="empty">开奖数据加载失败：${error.message}</p>`; });

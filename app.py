import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import json
import re
import requests
import concurrent.futures
from requests.exceptions import ConnectionError, Timeout

# ============================================================
# 交易日判断（周末/节假日自动回退）
# ============================================================

try:
    from tj_calendar import is_trade_day
    _CALENDAR_AVAILABLE = True
except ImportError:
    _CALENDAR_AVAILABLE = False

def get_latest_trading_day() -> date:
    today = datetime.today().date()
    if not _CALENDAR_AVAILABLE:
        while today.weekday() >= 5:
            today -= timedelta(days=1)
        return today
    current = today
    for _ in range(30):
        if is_trade_day(current):
            return current
        current -= timedelta(days=1)
    return today

def get_date_range(days: int) -> tuple:
    end_day = get_latest_trading_day()
    start_day = end_day - timedelta(days=days * 2)
    return start_day.strftime("%Y%m%d"), end_day.strftime("%Y%m%d")

# ============================================================
# 数据层
# ============================================================

def fetch_with_retry(func, *args, retries=1, delay=1, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout) as e:
            if i == retries - 1:
                raise e
            time.sleep(delay)
    return None

def _to_sina_symbol(code: str) -> str:
    if code.startswith(('6', '5', '9')): return f"sh{code}"
    return f"sz{code}"

def _etf_to_sina_symbol(etf_code: str) -> str:
    if etf_code.startswith(("5", "51")): return f"sh{etf_code}"
    return f"sz{etf_code}"

@st.cache_data(ttl=1800)
def fetch_stock_daily(code: str, days: int = 250) -> pd.DataFrame:
    symbol = _to_sina_symbol(code)
    start, end = get_date_range(days)
    df = fetch_with_retry(ak.stock_zh_a_daily, symbol=symbol, start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        raise ValueError("获取股票历史数据失败")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df["pct_chg"] = df["close"].pct_change() * 100
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=1800)
def fetch_stock_industry(code: str) -> str:
    try:
        info = fetch_with_retry(ak.stock_individual_info_em, symbol=code)
        if info is not None and not info.empty:
            if 'item' in info.columns and 'value' in info.columns:
                row = info[info['item'].str.contains('行业', na=False)]
                if not row.empty:
                    return row.iloc[0]['value']
    except Exception:
        pass
    return "未知"

def _fetch_etf_em(etf_code: str, days: int) -> pd.DataFrame:
    start, end = get_date_range(days)
    df = ak.fund_etf_hist_em(symbol=etf_code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty: return pd.DataFrame()
    df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)

def _fetch_etf_tx(etf_code: str, days: int) -> pd.DataFrame:
    symbol = _etf_to_sina_symbol(etf_code)
    start, end = get_date_range(days)
    df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty: return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)

def _fetch_etf_sina(etf_code: str, days: int) -> pd.DataFrame:
    symbol = _etf_to_sina_symbol(etf_code)
    df = ak.fund_etf_hist_sina(symbol)
    if df is None or df.empty: return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].tail(days).reset_index(drop=True)

def _fetch_etf_ths(etf_code: str, days: int) -> pd.DataFrame:
    url = f"http://d.10jqka.com.cn/v6/line/hs_{etf_code}/01/last.js"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", "Referer": "http://q.10jqka.com.cn/"}
    try:
        resp = requests.get(url, headers=headers, timeout=3)
        resp.raise_for_status()
        m = re.search(r"\((\{.*\})\)", resp.text, re.DOTALL)
        if not m: return pd.DataFrame()
        rows_str = json.loads(m.group(1)).get("data", "")
        if not rows_str: return pd.DataFrame()
        records = [{"date": p[0], "close": float(p[4])} for line in rows_str.split(";") if len(p := line.split(",")) >= 5]
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date")[["date", "close"]].tail(days).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _fetch_etf_quotekit(etf_code: str, days: int) -> pd.DataFrame:
    try:
        from quotekit import get_bars
        df = get_bars(etf_code, freq='day', limit=days, adjust='qfq')
        if df is None or df.empty or "date" not in df.columns: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "close"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _fetch_etf_hub(etf_code: str, days: int) -> pd.DataFrame:
    try:
        from stockdata_hub import StockDataFetcher
        df, _, _ = StockDataFetcher().fetch_daily_kline(etf_code, days=days)
        if df is not None and not df.empty and "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "close"]].sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

ALL_SOURCES = [("腾讯", _fetch_etf_tx), ("同花顺", _fetch_etf_ths), ("新浪", _fetch_etf_sina), ("stockdata-hub", _fetch_etf_hub), ("quotekit", _fetch_etf_quotekit), ("东财", _fetch_etf_em)]
MIN_ROWS = 30

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_etf_daily_multi(etf_code: str, days: int = 120) -> dict:
    all_results, valid_in_order = {}, []
    def _run_one(name, func):
        try:
            df = fetch_with_retry(func, etf_code, days, retries=1, delay=1)
            return name, df, None
        except Exception as e:
            return name, None, str(e)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(ALL_SOURCES)) as executor:
        futures = {executor.submit(_run_one, name, func): name for name, func in ALL_SOURCES}
        done, not_done = concurrent.futures.wait(futures, timeout=2.5)
        seq = 0
        for future in done:
            seq += 1
            name, df, err = future.result()
            if err or df is None or df.empty or len(df) < MIN_ROWS:
                all_results[name] = {"rows": 0, "status": "⚠️ 失败/超时", "order": seq}
                continue
            df = df.sort_values("date").reset_index(drop=True)
            all_results[name] = {"rows": len(df), "status": "✅ 成功", "order": seq}
            valid_in_order.append((seq, name, df))
        for future in not_done:
            all_results[futures[future]] = {"rows": 0, "status": "⌛ 超时(>2.5s)", "order": 99}
    if not valid_in_order: 
        return {"best_source": "无", "best_df": pd.DataFrame(), "all_results": all_results, "summary": "全部失败或超时"}
    valid_in_order.sort(key=lambda x: x[0])
    return {"best_source": valid_in_order[0][1], "best_df": valid_in_order[0][2], "all_results": all_results, "summary": "成功"}

@st.cache_data(ttl=1800)
def fetch_index_daily(symbol: str = "sh000300", days: int = 250) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_zh_index_daily, symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days).reset_index(drop=True)

@st.cache_data(ttl=600)
def fetch_realtime(code: str) -> dict:
    try:
        spot = fetch_with_retry(ak.stock_zh_a_spot, retries=1, delay=1)
        raw = spot[spot["代码"].astype(str).str.replace(r"^(sh|sz)", "", regex=True) == code]
        if raw.empty: return {}
        r = raw.iloc[0]
        return {"price": float(r["最新价"]), "name": r["名称"]}
    except Exception: return {}

@st.cache_data(ttl=600)
def fetch_etf_realtime(etf_code: str) -> dict:
    try:
        df = fetch_with_retry(ak.fund_etf_spot_em, retries=1, delay=1)
        row = df[df["代码"].astype(str) == etf_code]
        if row.empty: return {}
        return {"price": float(row.iloc[0]["最新价"]), "name": row.iloc[0]["名称"]}
    except Exception: return {}

# ============================================================
# 稳健的行业ETF匹配
# ============================================================

INDUSTRY_KEYWORDS = {
    "证券": ("512000", "券商ETF"), "券商": ("512000", "券商ETF"), "银行": ("512800", "银行ETF"), "保险": ("512070", "保险ETF"),
    "半导体": ("512480", "半导体ETF"), "芯片": ("512480", "半导体ETF"), "计算机": ("512720", "计算机ETF"), "软件": ("515230", "软件ETF"),
    "通信": ("515880", "通信ETF"), "人工智能": ("515070", "人工智能ETF"), "游戏": ("159869", "游戏ETF"), "传媒": ("512980", "传媒ETF"),
    "医药": ("512010", "医药ETF"), "医疗": ("512170", "医疗ETF"), "生物": ("512290", "生物医药ETF"), "消费": ("159928", "消费ETF"),
    "白酒": ("512690", "酒ETF"), "食品": ("515710", "食品饮料ETF"), "饮料": ("515710", "食品饮料ETF"), "家电": ("159996", "家电ETF"),
    "旅游": ("159766", "旅游ETF"), "养殖": ("159865", "养殖ETF"), "农业": ("159825", "农业ETF"),
    "新能源": ("516160", "新能源ETF"), "光伏": ("515790", "光伏ETF"), "锂电": ("159755", "电池ETF"), "电池": ("159755", "电池ETF"),
    "汽车": ("516110", "汽车ETF"), "军工": ("512660", "军工ETF"), "地产": ("512200", "房地产ETF"), "房地产": ("512200", "房地产ETF"),
    "煤炭": ("515220", "煤炭ETF"), "有色": ("512400", "有色金属ETF"), "钢铁": ("515210", "钢铁ETF"), "化工": ("516020", "化工ETF"),
    "黄金": ("518880", "黄金ETF"), "环保": ("512580", "环保ETF"), "电力": ("159611", "电力ETF"),
}

@st.cache_data(ttl=3600)
def find_industry_etf(stock_name: str, industry_name: str):
    for kw, etf in INDUSTRY_KEYWORDS.items():
        if kw in industry_name or kw in stock_name:
            return {"code": etf[0], "name": etf[1]}
    if industry_name == "未知":
        return None
    try:
        etf_df = ak.fund_etf_spot_em()
        mask = ~etf_df["名称"].str.contains("债|货币|国债|短融|城投|转债")
        etf_df = etf_df[mask]
        clean_industry = re.sub(r'[^\u4e00-\u9fa5]', '', industry_name) 
        if clean_industry:
            matched = etf_df[etf_df["名称"].str.contains(clean_industry)]
            if not matched.empty:
                return {"code": matched.iloc[0]["代码"], "name": matched.iloc[0]["名称"]}
    except Exception:
        pass
    return None

# ============================================================
# 宽基ETF智能优选
# ============================================================

BROAD_ETF_POOL = [
    {"code": "510050", "name": "上证50ETF", "index_name": "上证50"},
    {"code": "510300", "name": "300ETF", "index_name": "沪深300"},
    {"code": "159919", "name": "300ETF(深)", "index_name": "沪深300"},
    {"code": "510500", "name": "500ETF", "index_name": "中证500"},
    {"code": "512100", "name": "1000ETF", "index_name": "中证1000"},
    {"code": "563360", "name": "A500ETF", "index_name": "中证A500"},
    {"code": "159915", "name": "创业板ETF", "index_name": "创业板指"},
    {"code": "159949", "name": "创业板50ETF", "index_name": "创业板50"},
    {"code": "588000", "name": "科创50ETF", "index_name": "科创50"},
]

def calc_correlation_r2(stock_df: pd.DataFrame, etf_df: pd.DataFrame, window: int = 60) -> dict:
    if etf_df is None or etf_df.empty: return {"r": None, "r2": None, "n": 0, "reason": "ETF空"}
    actual_window = min(window, len(stock_df) - 1, len(etf_df) - 1)
    if actual_window < 5: return {"r": None, "r2": None, "n": 0, "reason": f"时间太短"}
    s = stock_df[["date", "close"]].rename(columns={"close": "stock"})
    e = etf_df[["date", "close"]].rename(columns={"close": "etf"})
    merged = pd.merge(s, e, on="date", how="inner").dropna()
    if len(merged) < actual_window: return {"r": None, "r2": None, "n": len(merged), "reason": "重叠不足"}
    ret_s = merged.tail(actual_window)["stock"].pct_change().dropna()
    ret_e = merged.tail(actual_window)["etf"].pct_change().dropna()
    r = float(ret_s.corr(ret_e))
    return {"r": round(r, 4), "r2": round(r ** 2, 4), "n": len(ret_s), "window": actual_window, "reason": ""}

def select_best_broad_etf(stock_code: str, stock_df: pd.DataFrame, beta: float):
    if stock_code.startswith(("688", "689")): candidates = [e for e in BROAD_ETF_POOL if "科创" in e["index_name"] or "沪深300" in e["index_name"]]
    elif stock_code.startswith("30"): candidates = [e for e in BROAD_ETF_POOL if "创业板" in e["index_name"] or "沪深300" in e["index_name"] or "中证500" in e["index_name"]]
    else:
        if beta >= 1.3: candidates = [e for e in BROAD_ETF_POOL if "1000" in e["index_name"] or "500" in e["index_name"] or "A500" in e["index_name"] or "沪深300" in e["index_name"]]
        elif beta <= 0.8: candidates = [e for e in BROAD_ETF_POOL if "上证50" in e["index_name"] or "沪深300" in e["index_name"] or "A500" in e["index_name"]]
        else: candidates = [e for e in BROAD_ETF_POOL if "沪深300" in e["index_name"] or "A500" in e["index_name"] or "中证500" in e["index_name"] or "上证50" in e["index_name"]]

    best_etf, best_corr, best_hist, best_source, compare_results = candidates[0], {"r": None, "r2": None}, pd.DataFrame(), "", []
    def test_corr(etf):
        multi = fetch_etf_daily_multi(etf["code"], days=90)
        hist, source = multi["best_df"], multi["best_source"]
        corr = calc_correlation_r2(stock_df, hist, window=60)
        return etf, corr, hist, source
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(test_corr, etf) for etf in candidates]
        for f in concurrent.futures.as_completed(futures):
            etf, corr, hist, source = f.result()
            compare_results.append({"ETF代码": etf["code"], "ETF名称": etf["name"], "R²": corr["r2"] if corr["r2"] else 0})
            if corr["r2"] is not None:
                if best_corr["r2"] is None or corr["r2"] > best_corr["r2"]:
                    best_corr, best_etf, best_hist, best_source = corr, etf, hist, source
    compare_results.sort(key=lambda x: x["R²"], reverse=True)
    return best_etf, best_corr, best_hist, best_source, compare_results

def r2_grade(r2) -> dict:
    if r2 is None: return {"label": "无法计算", "meaning": "数据不足"}
    if r2 >= 0.49: return {"label": "强匹配", "meaning": "波动高度同步"}
    if r2 >= 0.25: return {"label": "中等匹配", "meaning": "部分同步"}
    if r2 >= 0.09: return {"label": "弱匹配", "meaning": "联动较弱"}
    return {"label": "极弱/无匹配", "meaning": "几乎不联动"}

def get_strike_interval(etf_price: float) -> float:
    if etf_price <= 3: return 0.05
    if etf_price <= 5: return 0.1
    if etf_price <= 10: return 0.25
    if etf_price <= 20: return 0.5
    if etf_price <= 50: return 1.0
    if etf_price <= 100: return 2.5
    return 5.0

def round_to_strike(target: float, interval: float) -> float:
    return round(target / interval) * interval

def option_month_label() -> str:
    today = datetime.today()
    next_month = today.month + 1 if today.month < 12 else 1
    return f"{next_month}月"

def format_option_name(etf_prefix: str, opt_type: str, month: str, strike: float) -> str:
    return f"{etf_prefix}{opt_type}{month}{int(round(strike * 1000))}"

# ============================================================
# 针对最优匹配的【具体执行清单】生成器
# ============================================================

def generate_specific_actions(best_etf, etf_price, rating, market_value, beta, is_industry_etf):
    actions = []
    etf_price_used = etf_price if etf_price else 4.0
    contract_notional = etf_price_used * 10000
    equiv = market_value * beta / contract_notional if contract_notional else 0
    equiv_int = int(equiv)
    
    month = option_month_label()
    interval = get_strike_interval(etf_price_used)
    
    if rating in ("震荡", "偏空", "中性偏空", "中性偏多"):
        if equiv_int >= 1:
            put_name = format_option_name(best_etf["name"], "沽", month, round_to_strike(etf_price_used*0.95, interval))
            call_name = format_option_name(best_etf["name"], "购", month, round_to_strike(etf_price_used*1.05, interval))
            
            if rating == "震荡":
                actions.append(f"👉 **期权执行（领口策略）**：买入 **{put_name} {equiv_int}张** + 卖出 **{call_name} {equiv_int}张**。")
            elif rating in ("偏空", "中性偏空"):
                actions.append(f"👉 **期权执行（保护性认沽）**：买入 **{put_name} {equiv_int}张**。")
            elif rating == "中性偏多":
                actions.append(f"👉 **期权执行（备兑开仓）**：卖出 **{call_name} {equiv_int}张**。")
        else:
            need_market_value = contract_notional / beta if beta > 0 else 0
            gap = need_market_value - market_value
            actions.append(
                f"⚠️ **期权规模不足**：当前市值 {market_value:.0f} 元，等效仅 {equiv:.2f} 张。"
                f"ETF期权最低交易单位为 1 张（对应 10000 份 ETF），"
                f"若要买 1 张，持仓市值至少需要约 **{need_market_value:.0f} 元**（你还差约 {gap:.0f} 元）。"
            )

    if is_industry_etf:
        etf_px = etf_price if etf_price else 1.0
        raw_shares = market_value * 1.2 / etf_px
        short_shares = max(100, int(raw_shares // 100) * 100)
        actions.append(f"👉 **行业ETF融券**：融券卖出 **{short_shares} 份** {best_etf['name']}（{best_etf['code']}）。")
    
    return actions

# ============================================================
# 指标层与趋势引擎
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"], df["ma20"], df["ma60"] = df["close"].rolling(5).mean(), df["close"].rolling(20).mean(), df["close"].rolling(60).mean()
    ema12, ema26 = df["close"].ewm(span=12, adjust=False).mean(), df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["dif"] - df["dea"]) * 2
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    return df

def calc_beta(stock_df: pd.DataFrame, index_df: pd.DataFrame, window: int = 90) -> float:
    s, i = stock_df.set_index("date")["close"].pct_change(), index_df.set_index("date")["close"].pct_change()
    joined = pd.concat([s, i], axis=1, keys=["s", "i"]).dropna().tail(window)
    if len(joined) < 30: return 1.0
    cov = np.cov(joined["s"], joined["i"])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

def run_trend_engine(stock_df, index_df) -> dict:
    scores = {"ma": 0, "macd": 0, "relative": 0, "volume": 0}
    r = stock_df.iloc[-1]
    if not pd.isna(r["ma60"]):
        if r["close"] > r["ma5"] > r["ma20"] > r["ma60"]: scores["ma"] = 2
        elif r["close"] < r["ma20"] and r["ma5"] < r["ma20"] < r["ma60"]: scores["ma"] = -1
        elif r["close"] < r["ma60"]: scores["ma"] = -2
    p = stock_df.iloc[-2]
    if r["dif"] > r["dea"] and r["dif"] > 0 and r["macd_hist"] > p["macd_hist"]: scores["macd"] = 2
    elif r["dif"] < r["dea"] and r["macd_hist"] < p["macd_hist"]: scores["macd"] = -1
    if len(stock_df) >= 21 and len(index_df) >= 21:
        diff = ((stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1) - (index_df["close"].iloc[-1] / index_df["close"].iloc[-21] - 1)) * 100
        scores["relative"] = 2 if diff > 5 else (1 if diff > 0 else (-1 if diff > -5 else -2))
    if r["volume"] > r["vol_ma5"] * 1.2 and r["pct_chg"] > 0: scores["volume"] = 1
    elif r["volume"] < r["vol_ma5"] * 0.8 and r["pct_chg"] < 0: scores["volume"] = -1
    
    total = scores["ma"]*0.25 + scores["macd"]*0.2 + scores["relative"]*0.2 + scores["volume"]*0.15
    if total >= 1.2: rating = "偏多"
    elif total >= 0.3: rating = "中性偏多"
    elif total > -0.3: rating = "震荡"
    elif total > -1.2: rating = "中性偏空"
    else: rating = "偏空"
    return {"rating": rating, "total": round(total, 2)}

# ============================================================
# 科学严谨·分层决策引擎
# ============================================================

def generate_hedge_plans(rating, corr, has_margin, has_option, market_value, qty, beta, stock_code, stock_name, best_etf, etf_price, is_industry_etf):
    plans = []
    
    if rating in ("偏空", "中性偏空", "震荡"):
        reduce_ratio = 0.7 if rating in ("偏空", "中性偏空") else 0.4
        reduce_qty = int(qty * reduce_ratio // 100) * 100
        if reduce_qty > 0:
            plans.append({
                "category": "无风险基准",
                "name": "直接减仓（最确定、零成本）",
                "scenario": "不想承担任何衍生品利息/权利金风险，或个股基本面已恶化",
                "detail": f"直接 **卖出 {reduce_qty} 股** {stock_name}，降低风险敞口。",
                "cost": "仅正常交易佣金",
                "action": {"target": f"{stock_name}（{stock_code}）", "side": "卖出", "qty": reduce_qty, "price": "市价委托"}
            })

    if has_margin and rating in ("偏空", "中性偏空"):
        short_qty = max(100, int(qty * 0.5 // 100) * 100)
        plans.append({
            "category": "精准对冲（个股风险）",
            "name": "融券做空本股",
            "scenario": "强烈看空该股，但必须保留底仓（如打新市值/长期锁定）",
            "detail": f"**融券卖出 {short_qty} 股** {stock_name}，对冲下跌风险。",
            "cost": "年化约8%-10%利息。注意：融券保证金比例≥100%，且需确认APP有实时券源。",
            "action": {"target": f"{stock_name}（{stock_code}）", "side": "融券卖出", "qty": short_qty, "price": "不低于最新成交价"}
        })

    if has_margin and is_industry_etf and rating in ("偏空", "中性偏空", "震荡"):
        etf_code = best_etf["code"]
        etf_name = best_etf["name"]
        etf_rt = fetch_etf_realtime(etf_code)
        etf_px = etf_rt.get("price", etf_price)
        if etf_px:
            raw_shares = market_value * 1.2 / etf_px
            short_shares = max(100, int(raw_shares // 100) * 100)
            plans.append({
                "category": "行业对冲（板块风险）",
                "name": f"融券做空 {etf_name}",
                "scenario": "个股主要是受行业周期或政策拖累，而非个股自身暴雷",
                "detail": f"**融券卖出 {short_shares} 份** {etf_name}（{etf_code}），对冲板块下跌。",
                "cost": "年化约8%利息。行业ETF券源通常比宽基稀缺，需实时确认。",
                "action": {"target": f"{etf_name}（{etf_code}）", "side": "融券卖出", "qty": short_shares, "price": "不低于最新成交价"}
            })

    if has_option:
        contract_notional = (etf_price if etf_price else 4.0) * 10000
        equiv = market_value * beta / contract_notional
        equiv_int = int(equiv)
        
        if equiv_int >= 1:
            month = option_month_label()
            interval = get_strike_interval(etf_price if etf_price else 4.0)
            put_name = format_option_name(best_etf["name"], "沽", month, round_to_strike((etf_price or 4.0)*0.95, interval))
            call_name = format_option_name(best_etf["name"], "购", month, round_to_strike((etf_price or 4.0)*1.05, interval))
            
            r2 = corr["r2"]
            if r2 is None or r2 < 0.09:
                plans.append({
                    "category": "系统性对冲（大盘风险）",
                    "name": f"⚠️ {best_etf['name']} 期权（强烈不推荐）",
                    "scenario": "【无法使用】因该股与对标ETF联动性极低",
                    "detail": f"R²仅为{r2}，涨跌无法解释该股波动。如果强行买入 {put_name}，大概率是白交权利金。",
                    "cost": "权利金极可能成为沉没成本",
                    "action": None
                })
            else:
                if rating == "震荡":
                    plans.append({
                        "category": "系统性对冲（大盘风险）",
                        "name": f"领口策略（{best_etf['name']}）",
                        "scenario": "既怕跌，又不想花太多钱买保险，且能接受涨幅被封顶",
                        "detail": f"**买入 {put_name} {equiv_int} 张** + **卖出 {call_name} {equiv_int} 张**。",
                        "cost": "卖购权利金抵消买沽成本（近乎零成本），代价是上方收益封顶。",
                        "action": {"target": f"{put_name} 与 {call_name}", "side": "买虚沽 + 卖虚购", "qty": equiv_int, "price": "以券商APP为准"}
                    })
                elif rating in ("偏空", "中性偏空"):
                    plans.append({
                        "category": "系统性对冲（大盘风险）",
                        "name": f"保护性认沽（{best_etf['name']}）",
                        "scenario": "担心大盘拖累个股，但不想卖出底仓，愿意支付一笔保费",
                        "detail": f"**买入 {put_name} {equiv_int} 张**，锁定下行底线。",
                        "cost": "权利金即最大损失。若个股独立上涨，权利金全损。",
                        "action": {"target": put_name, "side": "买入认沽", "qty": equiv_int, "price": "以券商APP为准"}
                    })
                elif rating == "中性偏多":
                    plans.append({
                        "category": "系统性对冲（大盘风险）",
                        "name": f"备兑开仓（{best_etf['name']}）",
                        "scenario": "温和看涨，希望用持仓增强一点收益",
                        "detail": f"**卖出 {call_name} {equiv_int} 张**，收取权利金。",
                        "cost": "需持有对应ETF现货作担保。若大涨，上方利润封顶。",
                        "action": {"target": call_name, "side": "卖出认购", "qty": equiv_int, "price": "以券商APP为准"}
                    })
        else:
            need_market_value = contract_notional / beta if beta > 0 else 0
            plans.append({
                "category": "系统性对冲（大盘风险）",
                "name": f"⚠️ {best_etf['name']} 期权（规模不足）",
                "scenario": f"当前持仓市值 {market_value:.0f} 元，等效 {equiv:.2f} 张",
                "detail": f"ETF期权最低交易单位为 1 张。要买入 1 张，持仓市值至少需要约 **{need_market_value:.0f} 元**。",
                "cost": "当前持仓太小，无法通过期权进行有效对冲。",
                "action": None
            })

    if not plans:
        plans.append({
            "category": "观望",
            "name": "无需对冲",
            "scenario": "当前趋势偏多，或数据不足且无账户权限",
            "detail": "当前风险敞口在可承受范围内，建议设置移动止盈/止损提醒即可。",
            "cost": "无",
            "action": None
        })

    return plans

# ============================================================
# 主界面 (暗黑金融终端风)
# ============================================================

st.set_page_config(page_title="量化对冲终端", layout="centered")

# 深度定制 CSS
st.markdown("""
<style>
/* 隐藏 Streamlit 默认元素 */
[data-testid="stStatusWidget"], [data-testid="stAppRunningMan"], [class*="RunningMan"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
header { visibility: hidden !important; }

/* 全局背景和字体 */
.stApp {
    background: linear-gradient(135deg, #090C10 0%, #111820 100%);
    color: #E2E8F0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
    color: #E2E8F0;
    font-size: 16px !important;
}

/* 标题区 */
.terminal-header {
    display: flex; align-items: baseline; flex-wrap: wrap; gap: 10px; margin-bottom: 20px;
    padding-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.1);
}
.terminal-title {
    font-size: 2.2rem; font-weight: 900; color: #00E5FF; letter-spacing: 1px;
    text-shadow: 0 0 10px rgba(0, 229, 255, 0.3);
}
.terminal-subtitle {
    color: #FF3366; font-size: 1rem; font-weight: 600; letter-spacing: 0.5px;
}

/* 输入框标签 */
[data-testid="stTextInput"] label, [data-testid="stNumberInput"] label, [data-testid="stDateInput"] label {
    font-size: 18px !important; font-weight: 600 !important; color: #94A3B8 !important; margin-bottom: 4px !important;
    letter-spacing: 0.5px;
}

/* 输入框内部 */
[data-testid="stTextInput"] input, 
[data-testid="stNumberInput"] input, 
[data-testid="stDateInput"] input {
    background-color: #1A2332 !important; color: #00E5FF !important;
    border: 1px solid #2D3748 !important; border-radius: 6px !important;
    font-family: 'Courier New', Courier, monospace !important; font-weight: 700 !important;
    font-size: 22px !important; min-height: 54px !important;
    padding: 10px 14px !important;
    transition: all 0.3s ease;
}
[data-testid="stTextInput"] input:focus, 
[data-testid="stNumberInput"] input:focus, 
[data-testid="stDateInput"] input:focus {
    border-color: #00E5FF !important; box-shadow: 0 0 8px rgba(0, 229, 255, 0.2) !important;
}
[data-testid="stDateInput"] svg { width: 24px !important; height: 24px !important; fill: #00E5FF; }

/* 日期弹窗 */
[data-testid="stDateInputCalendar"] *, [data-baseweb="calendar"] * {
    font-size: 22px !important; font-weight: 700 !important;
}
[data-testid="stDateInputCalendar"] [role="gridcell"], [data-baseweb="calendar"] [role="gridcell"] {
    min-width: 46px !important; min-height: 46px !important; line-height: 46px !important;
}

/* 按钮 */
.stButton > button {
    width: 100% !important;
    font-size: 24px !important; font-weight: 900 !important; letter-spacing: 2px;
    color: #090C10 !important; background: linear-gradient(90deg, #00E5FF 0%, #00B4D8 100%) !important;
    border: none !important; border-radius: 8px !important; min-height: 60px !important;
    box-shadow: 0 4px 15px rgba(0, 229, 255, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 229, 255, 0.5) !important;
}

/* 复选框 */
[data-testid="stCheckbox"] label {
    color: #94A3B8 !important; font-size: 16px !important; font-weight: 600 !important;
}

/* 卡片面板 */
.terminal-card {
    background: #111820; border: 1px solid #1E293B; border-radius: 10px;
    padding: 20px; margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.card-header {
    font-size: 14px; color: #64748B; font-weight: 700; letter-spacing: 1px; margin-bottom: 12px;
    text-transform: uppercase;
}
.stat-row { display: flex; justify-content: space-between; margin-bottom: 8px; }
.stat-label { color: #94A3B8; font-size: 16px; }
.stat-value { color: #E2E8F0; font-family: 'Courier New', Courier, monospace; font-size: 18px; font-weight: bold; }
.highlight-up { color: #FF3366; font-weight: 900; } /* A股红涨 */
.highlight-down { color: #00E676; font-weight: 900; } /* A股绿跌 */
.highlight-gold { color: #FFD700; }

/* 数据表 */
[data-testid="stDataFrame"] {
    background-color: #111820 !important; border-radius: 8px; border: 1px solid #1E293B;
}
[data-testid="stDataFrame"] * { font-size: 15px !important; color: #E2E8F0; }

/* 折叠面板 */
[data-testid="stExpander"] {
    background: #111820 !important; border: 1px solid #1E293B !important; border-radius: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #00E5FF !important; font-weight: 700 !important; font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="terminal-header">
      <span class="terminal-title">📊 QUANT-HEDGE TERMINAL</span>
      <span class="terminal-subtitle">动态匹配 · 分层风险决策 · 智能终端</span>
    </div>
    """, unsafe_allow_html=True
)

# 两行两列布局
row1_col1, row1_col2 = st.columns(2)
code = row1_col1.text_input("股票代码", "")
qty = row1_col2.number_input("数量（股）", value=None, step=100, placeholder="如 100")

row2_col1, row2_col2 = st.columns(2)
buy_price = row2_col1.number_input("买入价", value=None, step=0.01, format="%.2f", placeholder="如 23.00")
buy_date = row2_col2.date_input("买入日期", value=None)

st.markdown("<div style='margin-top: 15px; margin-bottom: 5px; color: #64748B; font-size: 14px; font-weight: 700;'>ACCOUNT PERMISSIONS</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
has_margin = c1.checkbox("已开通两融账户（支持融券）")
has_option = c2.checkbox("已开通ETF期权账户")

st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

if st.button("INITIALIZE ANALYSIS / 开始分析"):
    if not code.strip() or buy_price is None or qty is None:
        st.warning("请先填写股票代码、买入价和数量后再点击“开始分析”。")
    else:
        with st.spinner("系统运算中..."):
            try:
                code = code.strip()
                qty = int(qty)

                stock_df = add_indicators(fetch_stock_daily(code))
                index_df = fetch_index_daily()
                rt = fetch_realtime(code)

                price = rt.get("price", stock_df["close"].iloc[-1])
                name = rt.get("name", code)
                beta = calc_beta(stock_df, index_df)
                market_value = price * qty

                industry = fetch_stock_industry(code)
                matched_industry = find_industry_etf(name, industry)
                is_industry_etf = False
                
                if matched_industry:
                    multi = fetch_etf_daily_multi(matched_industry["code"], days=90)
                    corr = calc_correlation_r2(stock_df, multi["best_df"], window=60)
                    best_etf = matched_industry
                    etf_source = multi["best_source"]
                    compare_results = [{"ETF代码": best_etf["code"], "ETF名称": best_etf["name"], "R²": corr["r2"] if corr["r2"] else 0}]
                    is_industry_etf = True
                    etf_rt = fetch_etf_realtime(best_etf["code"])
                    etf_price = etf_rt.get("price")
                
                if not is_industry_etf:
                    best_etf, corr, etf_hist, etf_source, compare_results = select_best_broad_etf(code, stock_df, beta)
                    etf_rt = fetch_etf_realtime(best_etf["code"])
                    etf_price = etf_rt.get("price")

                trend = run_trend_engine(stock_df, index_df)
                specific_actions = generate_specific_actions(best_etf, etf_price, trend["rating"], market_value, beta, is_industry_etf)

                plans = generate_hedge_plans(
                    trend["rating"], corr, has_margin, has_option, 
                    market_value, qty, beta, code, name, best_etf, etf_price, is_industry_etf
                )

                pnl = (price - buy_price) * qty
                pnl_class = "highlight-up" if pnl >= 0 else "highlight-down"
                pnl_sign = "+" if pnl >= 0 else ""

                # 终端风的数据卡片
                st.markdown(f"""
                <div class="terminal-card">
                    <div class="card-header">POSITION OVERVIEW / 持仓概览</div>
                    <div style="font-size: 24px; font-weight: 800; color: #E2E8F0; margin-bottom: 15px;">
                        {name} <span style="color: #64748B; font-size: 18px;">({code})</span>
                    </div>
                    <div class="stat-row"><span class="stat-label">现价</span><span class="stat-value">{price:.2f}</span></div>
                    <div class="stat-row"><span class="stat-label">成本</span><span class="stat-value">{buy_price:.2f}</span></div>
                    <div class="stat-row"><span class="stat-label">持仓数量</span><span class="stat-value">{qty} 股</span></div>
                    <div class="stat-row"><span class="stat-label">总市值</span><span class="stat-value">{market_value:.0f} 元</span></div>
                    <div class="stat-row"><span class="stat-label">浮动盈亏</span><span class="stat-value {pnl_class}">{pnl_sign}{pnl:.0f} 元</span></div>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px dashed #1E293B;">
                        <div class="stat-row"><span class="stat-label">所属行业</span><span class="stat-value">{industry}</span></div>
                        <div class="stat-row"><span class="stat-label">趋势研判</span><span class="stat-value highlight-gold">{trend['rating']} ({trend['total']})</span></div>
                        <div class="stat-row"><span class="stat-label">个股 Beta</span><span class="stat-value">{beta:.2f}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if is_industry_etf:
                    st.markdown("<div class='card-header' style='margin-top:20px;'>INDUSTRY ETF MATCHING / 动态行业匹配</div>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(compare_results), use_container_width=True, hide_index=True)
                    if corr["r2"] is not None:
                        grade = r2_grade(corr["r2"])
                        if grade["label"] in ["强匹配", "中等匹配"]:
                            st.success(f"✅ 匹配结果：**{best_etf['name']}（{best_etf['code']}）**，R² = {corr['r2']:.3f}（{grade['label']}）—— 适合作为行业对冲工具。")
                        else:
                            st.warning(f"⚠️ 匹配结果：**{best_etf['name']}（{best_etf['code']}）**，R² = {corr['r2']:.3f}（{grade['label']}）—— 相关性偏低，行业对冲效果可能有限，请谨慎使用。")
                else:
                    st.markdown("<div class='card-header' style='margin-top:20px;'>BROAD-BASED ETF OPTIMIZATION / 宽基智能优选</div>", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(compare_results), use_container_width=True, hide_index=True)
                    if corr["r2"] is not None:
                        st.success(f"✅ 最优匹配：**{best_etf['name']}（{best_etf['code']}）**，R² = {corr['r2']:.3f}（{r2_grade(corr['r2'])['label']}）")

                if specific_actions:
                    st.markdown("<div class='card-header' style='margin-top:20px;'>EXECUTION COMMAND / 指令清单</div>", unsafe_allow_html=True)
                    for act in specific_actions:
                        if act.startswith("⚠️"):
                            st.warning(act)
                        else:
                            st.success(act)

                st.markdown("<div class='card-header' style='margin-top:30px;'>STRATIFIED HEDGING PLANS / 分层对冲方案</div>", unsafe_allow_html=True)
                
                for plan in plans:
                    with st.expander(f"🔹 {plan['category']}：{plan['name']}", expanded=True):
                        st.markdown(f"**适用场景**：{plan['scenario']}")
                        st.info(f"**操作建议**：{plan['detail']}")
                        st.warning(f"**成本与代价**：{plan['cost']}")
                        
                        if plan.get("action"):
                            act = plan["action"]
                            st.markdown(
                                f"> **执行清单：**\n"
                                f"> 标的：**{act['target']}**\n"
                                f"> 方向：**{act['side']}**\n"
                                f"> 数量：**{act['qty']}**\n"
                                f"> 价格：{act['price']}"
                            )

                st.markdown("<br><div style='text-align: center; color: #64748B; font-size: 13px;'>本终端仅提供量化分析参考，不构成投资建议。请手动至券商APP核验并下单。</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"终端错误：{e}")

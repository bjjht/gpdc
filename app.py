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
# 交易日判断
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
    df = fetch_with_retry(
        ak.stock_zh_a_daily,
        symbol=symbol, start_date=start, end_date=end, adjust="qfq"
    )
    if df is None or df.empty:
        raise ValueError("获取股票历史数据失败（新浪接口），请稍后重试")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df["pct_chg"] = df["close"].pct_change() * 100
    return df.sort_values("date").reset_index(drop=True)

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

ALL_SOURCES = [
    ("腾讯", _fetch_etf_tx), ("同花顺", _fetch_etf_ths), ("新浪", _fetch_etf_sina),
    ("stockdata-hub", _fetch_etf_hub), ("quotekit", _fetch_etf_quotekit), ("东财", _fetch_etf_em)
]
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
            try:
                name, df, err = future.result()
            except Exception as e:
                name = futures[future]
                df, err = None, str(e)

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
# 扩充后的行业映射与宽基池
# ============================================================

INDUSTRY_KEYWORDS = {
    # 金融
    "证券": ("512000", "券商ETF"), "券商": ("512000", "券商ETF"), "银行": ("512800", "银行ETF"), "保险": ("512070", "保险ETF"),
    # TMT科技
    "半导体": ("512480", "半导体ETF"), "芯片": ("512480", "半导体ETF"), "计算机": ("512720", "计算机ETF"), "软件": ("515230", "软件ETF"),
    "通信": ("515880", "通信ETF"), "人工智能": ("515070", "人工智能ETF"), "游戏": ("159869", "游戏ETF"), "传媒": ("512980", "传媒ETF"),
    # 医药消费
    "医药": ("512010", "医药ETF"), "医疗": ("512170", "医疗ETF"), "生物": ("512290", "生物医药ETF"), "消费": ("159928", "消费ETF"),
    "白酒": ("512690", "酒ETF"), "食品": ("515710", "食品饮料ETF"), "饮料": ("515710", "食品饮料ETF"), "家电": ("159996", "家电ETF"),
    "旅游": ("159766", "旅游ETF"), "养殖": ("159865", "养殖ETF"), "农业": ("159825", "农业ETF"),
    # 制造与周期
    "新能源": ("516160", "新能源ETF"), "光伏": ("515790", "光伏ETF"), "锂电": ("159755", "电池ETF"), "电池": ("159755", "电池ETF"),
    "汽车": ("516110", "汽车ETF"), "军工": ("512660", "军工ETF"), "地产": ("512200", "房地产ETF"), "房地产": ("512200", "房地产ETF"),
    "煤炭": ("515220", "煤炭ETF"), "有色": ("512400", "有色金属ETF"), "钢铁": ("515210", "钢铁ETF"), "化工": ("516020", "化工ETF"),
    "黄金": ("518880", "黄金ETF"), "环保": ("512580", "环保ETF"), "电力": ("159611", "电力ETF"),
}

def detect_industry_etf(stock_name: str):
    for kw, etf in INDUSTRY_KEYWORDS.items():
        if kw in stock_name: return etf
    return None

BROAD_ETF_POOL = [
    {"code": "510050", "name": "上证50ETF", "index_name": "上证50"},
    {"code": "510300", "name": "300ETF", "index_name": "沪深300"},
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
    # 根据板块挑选更丰富的候选池
    if stock_code.startswith(("688", "689")): 
        candidates = [BROAD_ETF_POOL[7], BROAD_ETF_POOL[1], BROAD_ETF_POOL[4]] # 科创50, 300, A500
    elif stock_code.startswith("30"): 
        candidates = [BROAD_ETF_POOL[5], BROAD_ETF_POOL[6], BROAD_ETF_POOL[1], BROAD_ETF_POOL[2]] # 创业, 创业50, 300, 500
    else:
        if beta >= 1.3: candidates = [BROAD_ETF_POOL[3], BROAD_ETF_POOL[2], BROAD_ETF_POOL[4], BROAD_ETF_POOL[1]] # 1000, 500, A500, 300
        elif beta <= 0.8: candidates = [BROAD_ETF_POOL[0], BROAD_ETF_POOL[1], BROAD_ETF_POOL[4]] # 50, 300, A500
        else: candidates = [BROAD_ETF_POOL[1], BROAD_ETF_POOL[4], BROAD_ETF_POOL[2], BROAD_ETF_POOL[0]] # 300, A500, 500, 50

    best_etf, best_corr, best_hist, best_source, compare_results = candidates[0], {"r": None, "r2": None}, pd.DataFrame(), "", []
    
    for etf in candidates:
        multi = fetch_etf_daily_multi(etf["code"], days=90)
        hist, source = multi["best_df"], multi["best_source"]
        corr = calc_correlation_r2(stock_df, hist, window=60)
        compare_results.append({"ETF代码": etf["code"], "ETF名称": etf["name"], "R²": corr["r2"] if corr["r2"] else 0})
        if corr["r2"] is not None:
            if best_corr["r2"] is None or corr["r2"] > best_corr["r2"]:
                best_corr, best_etf, best_hist, best_source = corr, etf, hist, source
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

def generate_hedge_plans(rating, corr, has_margin, has_option, market_value, qty, beta, stock_code, stock_name, broad_etf, etf_price):
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

    industry_etf = detect_industry_etf(stock_name)
    if has_margin and industry_etf and rating in ("偏空", "中性偏空", "震荡"):
        etf_code, etf_name = industry_etf
        etf_rt = fetch_etf_realtime(etf_code)
        etf_px = etf_rt.get("price")
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
            put_name = format_option_name(broad_etf["name"], "沽", month, round_to_strike((etf_price or 4.0)*0.95, interval))
            call_name = format_option_name(broad_etf["name"], "购", month, round_to_strike((etf_price or 4.0)*1.05, interval))
            
            r2 = corr["r2"]
            if r2 is None or r2 < 0.09:
                plans.append({
                    "category": "系统性对冲（大盘风险）",
                    "name": f"⚠️ {broad_etf['name']} 期权（强烈不推荐）",
                    "scenario": "【无法使用】因该股与宽基联动性极低",
                    "detail": f"R²仅为{r2}，宽基涨跌无法解释该股波动。如果强行买入 {put_name}，大概率是白交权利金。",
                    "cost": "权利金极可能成为沉没成本",
                    "action": None
                })
            else:
                if rating == "震荡":
                    plans.append({
                        "category": "系统性对冲（大盘风险）",
                        "name": f"领口策略（{broad_etf['name']}）",
                        "scenario": "既怕跌，又不想花太多钱买保险，且能接受涨幅被封顶",
                        "detail": f"**买入 {put_name} {equiv_int} 张** + **卖出 {call_name} {equiv_int} 张**。",
                        "cost": "卖购权利金抵消买沽成本（近乎零成本），代价是上方收益封顶。",
                        "action": {"target": f"{put_name} 与 {call_name}", "side": "买虚沽 + 卖虚购", "qty": equiv_int, "price": "以券商APP为准"}
                    })
                elif rating in ("偏空", "中性偏空"):
                    plans.append({
                        "category": "系统性对冲（大盘风险）",
                        "name": f"保护性认沽（{broad_etf['name']}）",
                        "scenario": "担心大盘拖累个股，但不想卖出底仓，愿意支付一笔保费",
                        "detail": f"**买入 {put_name} {equiv_int} 张**，锁定下行底线。",
                        "cost": "权利金即最大损失。若个股独立上涨，权利金全损。",
                        "action": {"target": put_name, "side": "买入认沽", "qty": equiv_int, "price": "以券商APP为准"}
                    })
                elif rating == "中性偏多":
                    plans.append({
                        "category": "系统性对冲（大盘风险）",
                        "name": f"备兑开仓（{broad_etf['name']}）",
                        "scenario": "温和看涨，希望用持仓增强一点收益",
                        "detail": f"**卖出 {call_name} {equiv_int} 张**，收取权利金。",
                        "cost": "需持有对应ETF现货作担保。若大涨，上方利润封顶。",
                        "action": {"target": call_name, "side": "卖出认购", "qty": equiv_int, "price": "以券商APP为准"}
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
# 主界面
# ============================================================

st.set_page_config(page_title="对冲分析助手", layout="centered")

st.markdown("""
<style>
[data-testid="stStatusWidget"], [data-testid="stAppRunningMan"], [class*="RunningMan"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
[data-testid="stTextInput"] label, [data-testid="stNumberInput"] label, [data-testid="stDateInput"] label {
    font-size: 24px !important; font-weight: 800 !important; color: #111 !important; margin-bottom: 4px !important;
}
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stDateInput"] input {
    font-size: 24px !important; font-weight: 600 !important; min-height: 54px !important;
}
.stButton > button {
    font-size: 30px !important; font-weight: bold !important; color: #FFD700 !important; background-color: #1a1a1a !important;
    border: 3px solid #FFD700 !important; padding: 14px 44px !important; border-radius: 10px !important; min-height: 66px !important;
}
.stButton > button:hover { background-color: #333 !important; color: #FFE55C !important; border-color: #FFE55C !important; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { font-size: 18px !important; }
[data-testid="stMetricValue"] { font-size: 30px !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 16px !important; }
[data-testid="stDataFrame"] * { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="display:flex; align-items:baseline; flex-wrap:wrap; gap:10px; margin-bottom:12px;">
      <span style="font-size:2.0rem; font-weight:800;">📊 持仓对冲分析助手</span>
      <span style="color:#e00000; font-size:1.05rem; font-weight:700;">纯分析 · 分层风险拆解 · 科学严谨 · 操作需手动执行</span>
    </div>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns(2)
code = col1.text_input("股票代码", "")
buy_price = col1.number_input("买入价", value=None, step=0.01, format="%.2f", placeholder="如 23.00")
qty = col2.number_input("数量（股）", value=None, step=100, placeholder="如 100")
buy_date = col2.date_input("买入日期", value=None)

st.markdown("**账户权限（勾选后系统才会给出对应方案）**")
c1, c2 = st.columns(2)
has_margin = c1.checkbox("已开通两融账户（支持融券）")
has_option = c2.checkbox("已开通ETF期权账户")

if st.button("开始分析"):
    if not code.strip() or buy_price is None or qty is None:
        st.warning("请先填写股票代码、买入价和数量后再点击“开始分析”。")
    else:
        with st.spinner("正在测算风险敞口并生成分层方案..."):
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

                broad_etf, corr, etf_hist, etf_source, compare_results = select_best_broad_etf(code, stock_df, beta)
                etf_rt = fetch_etf_realtime(broad_etf["code"])
                etf_price = etf_rt.get("price")

                trend = run_trend_engine(stock_df, index_df)

                plans = generate_hedge_plans(
                    trend["rating"], corr, has_margin, has_option, 
                    market_value, qty, beta, code, name, broad_etf, etf_price
                )

                st.subheader("【持仓与走势】")
                st.write(f"**{name}（{code}）** | 现价 {price:.2f} | 成本 {buy_price:.2f} | {qty}股 | 市值 {market_value:.0f}元")
                st.info(f"**趋势研判**：{trend['rating']}（得分 {trend['total']}） | **个股 Beta**：{beta:.2f}")

                st.subheader("【数据源智能优选宽基】")
                st.caption("系统测试了以下候选宽基与个股的相关性，优选 R² 最高者用于期权对冲：")
                st.dataframe(pd.DataFrame(compare_results), use_container_width=True, hide_index=True)
                if corr["r2"] is not None:
                    st.success(f"选定宽基：**{broad_etf['name']}（{broad_etf['code']}）**，R² = {corr['r2']:.3f}（{r2_grade(corr['r2'])['label']}）")

                st.subheader("【科学严谨·分层对冲方案】")
                st.write("以下方案按**风险来源**分类，请根据您的实际约束（如是否保留底仓、能否承担利息）自行选择手动执行：")

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

                st.subheader("【手动执行提示】")
                st.warning("以上方案为纯分析参考，本工具不发送任何委托。请在券商 APP 中核对实时价格、券源、权利金后再手动下单。")

            except Exception as e:
                st.error(f"数据获取或计算失败：{e}")

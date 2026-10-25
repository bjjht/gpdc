# ============================================================
# 顶部：必须先于 akshare 导入，启用代理补丁
# ============================================================
import akshare_proxy_patch
akshare_proxy_patch.install_patch(
    "101.201.173.125",
    auth_token="你的TOKEN",   # 替换为你在插件官网获取的 TOKEN
    retry=30,
    fast=True
)

import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import time
import json
import re
import requests
from requests.exceptions import ConnectionError, Timeout

# ============================================================
# 交易日判断（支持周末 + 中国法定节假日）
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

def fetch_with_retry(func, *args, retries=3, delay=2, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, Timeout) as e:
            if i == retries - 1:
                raise e
            time.sleep(delay)
    return None

def _to_sina_symbol(code: str) -> str:
    if code.startswith(('6', '5', '9')):
        return f"sh{code}"
    return f"sz{code}"

def _etf_to_sina_symbol(etf_code: str) -> str:
    if etf_code.startswith(("5", "51")):
        return f"sh{etf_code}"
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
        raise ValueError("获取股票历史数据失败，请稍后重试")
    df = df.rename(columns={"date": "date", "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    df["pct_chg"] = df["close"].pct_change() * 100
    return df.sort_values("date").reset_index(drop=True)

# ---------- ETF 多源 ----------

def _fetch_etf_em(etf_code: str, days: int) -> pd.DataFrame:
    start, end = get_date_range(days)
    df = ak.fund_etf_hist_em(symbol=etf_code, period="daily", start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)

def _fetch_etf_tx(etf_code: str, days: int) -> pd.DataFrame:
    symbol = _etf_to_sina_symbol(etf_code)
    start, end = get_date_range(days)
    df = ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)

def _fetch_etf_sina(etf_code: str, days: int) -> pd.DataFrame:
    symbol = _etf_to_sina_symbol(etf_code)
    df = ak.fund_etf_hist_sina(symbol)   # 位置参数
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].tail(days).reset_index(drop=True)

def _fetch_etf_ths(etf_code: str, days: int) -> pd.DataFrame:
    url = f"http://d.10jqka.com.cn/v6/line/hs_{etf_code}/01/last.js"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "http://q.10jqka.com.cn/"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = resp.text
        m = re.search(r"\((\{.*\})\)", text, re.DOTALL)
        if not m:
            return pd.DataFrame()
        payload = json.loads(m.group(1))
        rows_str = payload.get("data", "")
        if not rows_str:
            return pd.DataFrame()
        records = []
        for line in rows_str.split(";"):
            parts = line.split(",")
            if len(parts) >= 5:
                try:
                    records.append({"date": parts[0], "close": float(parts[4])})
                except (ValueError, IndexError):
                    continue
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        return df[["date", "close"]].tail(days).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _fetch_etf_quotekit(etf_code: str, days: int) -> pd.DataFrame:
    try:
        from quotekit import get_bars
    except ImportError:
        return pd.DataFrame()
    try:
        df = get_bars(etf_code, freq='day', limit=days, adjust='qfq')
        if df is None or df.empty or "date" not in df.columns or "close" not in df.columns:
            return pd.DataFrame()
        df = df[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def _fetch_etf_hub(etf_code: str, days: int) -> pd.DataFrame:
    try:
        from stockdata_hub import StockDataFetcher
    except ImportError:
        return pd.DataFrame()
    try:
        fetcher = StockDataFetcher()
        df, reason, code = fetcher.fetch_daily_kline(etf_code, days=days)
        if df is not None and not df.empty and "date" in df.columns and "close" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "close"]].sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame()

# ---------- 自检 + 获取 ----------

ALL_SOURCES = [
    ("东财", _fetch_etf_em),
    ("腾讯", _fetch_etf_tx),
    ("新浪", _fetch_etf_sina),
    ("同花顺", _fetch_etf_ths),
    ("quotekit", _fetch_etf_quotekit),
    ("stockdata-hub", _fetch_etf_hub),
]

@st.cache_data(ttl=1800, show_spinner=False)
def detect_available_etf_sources(probe_code: str = "510300") -> tuple:
    available, details = [], []
    for name, func in ALL_SOURCES:
        try:
            df = func(probe_code, days=30)
            if df is not None and not df.empty and len(df) >= 10:
                available.append((name, func))
                details.append(f"✅ {name}: 可用（{len(df)}行）")
            else:
                details.append(f"❌ {name}: 返回空或数据不足")
        except Exception as e:
            details.append(f"❌ {name}: {str(e)[:60]}")
        time.sleep(0.5)
    return available, details

@st.cache_data(ttl=1800)
def fetch_etf_daily(etf_code: str, days: int = 120) -> tuple:
    available, _ = detect_available_etf_sources()
    sources_to_try = available if available else ALL_SOURCES
    errors = []
    for name, func in sources_to_try:
        try:
            df = fetch_with_retry(func, etf_code, days, retries=1, delay=1)
            if df is not None and not df.empty and len(df) >= 30:
                return df, name
            errors.append(f"{name}: 空或数据不足")
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        time.sleep(0.8)
    return pd.DataFrame(), "全部失败（" + "；".join(errors) + "）"

# ============================================================
# 其余数据层
# ============================================================

@st.cache_data(ttl=1800)
def fetch_index_daily(symbol: str = "sh000300", days: int = 250) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_zh_index_daily, symbol=symbol)
    if df is None or df.empty:
        raise ValueError("获取指数数据失败")
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days).reset_index(drop=True)

@st.cache_data(ttl=600)
def fetch_realtime(code: str) -> dict:
    try:
        spot = fetch_with_retry(ak.stock_zh_a_spot, retries=2, delay=1)
        if spot is None or spot.empty:
            return {}
        raw = spot[spot["代码"].astype(str).str.replace(r"^(sh|sz)", "", regex=True) == code]
        if raw.empty:
            return {}
        r = raw.iloc[0]
        return {"price": float(r["最新价"]), "name": r["名称"]}
    except Exception:
        return {}

@st.cache_data(ttl=600)
def fetch_etf_realtime(etf_code: str) -> dict:
    try:
        df = fetch_with_retry(ak.fund_etf_spot_em, retries=2, delay=1)
        if df is None or df.empty:
            return {}
        row = df[df["代码"].astype(str) == etf_code]
        if row.empty:
            return {}
        return {"price": float(row.iloc[0]["最新价"]), "name": row.iloc[0]["名称"]}
    except Exception:
        return {}

INDUSTRY_KEYWORDS = {
    "证券": ("512000", "券商ETF"), "券商": ("512000", "券商ETF"),
    "银行": ("512800", "银行ETF"), "保险": ("512070", "保险ETF"),
    "半导体": ("512480", "半导体ETF"), "芯片": ("512480", "半导体ETF"),
    "计算机": ("512720", "计算机ETF"), "软件": ("515230", "软件ETF"),
    "通信": ("515880", "通信ETF"), "人工智能": ("515070", "人工智能ETF"),
    "医药": ("512010", "医药ETF"), "医疗": ("512170", "医疗ETF"),
    "生物": ("512290", "生物医药ETF"), "消费": ("159928", "消费ETF"),
    "白酒": ("512690", "酒ETF"), "食品": ("515710", "食品饮料ETF"),
    "饮料": ("515710", "食品饮料ETF"), "家电": ("159996", "家电ETF"),
    "旅游": ("159766", "旅游ETF"), "养殖": ("159865", "养殖ETF"),
    "农业": ("159825", "农业ETF"), "新能源": ("516160", "新能源ETF"),
    "光伏": ("515790", "光伏ETF"), "锂电": ("159755", "电池ETF"),
    "电池": ("159755", "电池ETF"), "汽车": ("516110", "汽车ETF"),
    "军工": ("512660", "军工ETF"), "地产": ("512200", "房地产ETF"),
    "房地产": ("512200", "房地产ETF"), "煤炭": ("515220", "煤炭ETF"),
    "有色": ("512400", "有色金属ETF"), "钢铁": ("515210", "钢铁ETF"),
    "化工": ("516020", "化工ETF"), "黄金": ("518880", "黄金ETF"),
    "环保": ("512580", "环保ETF"), "传媒": ("512980", "传媒ETF"),
    "游戏": ("159869", "游戏ETF"), "电力": ("159611", "电力ETF"),
}

def detect_industry_etf(stock_name: str):
    for kw, etf in INDUSTRY_KEYWORDS.items():
        if kw in stock_name:
            return etf
    return None

def select_broad_etf(code: str, beta: float) -> dict:
    if code.startswith(("688", "689")):
        return {"code": "588000", "name": "科创50ETF", "index_name": "科创50"}
    if code.startswith("30"):
        return {"code": "159915", "name": "创业板ETF", "index_name": "创业板指"}
    if code.startswith(("8", "4")):
        return {"code": "510300", "name": "300ETF", "index_name": "沪深300"}
    if beta >= 1.3:
        return {"code": "510500", "name": "500ETF", "index_name": "中证500"}
    if beta <= 0.8:
        return {"code": "510050", "name": "50ETF", "index_name": "上证50"}
    return {"code": "510300", "name": "300ETF", "index_name": "沪深300"}

def calc_correlation_r2(stock_df: pd.DataFrame, etf_df: pd.DataFrame, window: int = 60) -> dict:
    if etf_df is None or etf_df.empty or len(stock_df) < window or len(etf_df) < window:
        return {"r": None, "r2": None, "n": 0}
    s = stock_df[["date", "close"]].rename(columns={"close": "stock"})
    e = etf_df[["date", "close"]].rename(columns={"close": "etf"})
    merged = pd.merge(s, e, on="date", how="inner").dropna()
    if len(merged) < 30:
        return {"r": None, "r2": None, "n": len(merged)}
    merged = merged.tail(window)
    ret_s = merged["stock"].pct_change().dropna()
    ret_e = merged["etf"].pct_change().dropna()
    if len(ret_s) < 20:
        return {"r": None, "r2": None, "n": len(ret_s)}
    r = float(ret_s.corr(ret_e))
    return {"r": round(r, 4), "r2": round(r ** 2, 4), "n": len(ret_s)}

def r2_grade(r2) -> dict:
    if r2 is None:
        return {"label": "无法计算", "color": "gray", "meaning": "数据不足"}
    if r2 >= 0.49:
        return {"label": "强匹配", "color": "green", "meaning": "期权对冲效果相对可靠"}
    if r2 >= 0.25:
        return {"label": "中等匹配", "color": "orange", "meaning": "对冲有效但不完全"}
    if r2 >= 0.09:
        return {"label": "弱匹配", "color": "red", "meaning": "联动性较弱，效果有限"}
    return {"label": "极弱/无匹配", "color": "red", "meaning": "不建议用期权对冲"}

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
# 指标与趋势
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["dif"] - df["dea"]) * 2
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    return df

def calc_beta(stock_df: pd.DataFrame, index_df: pd.DataFrame, window: int = 90) -> float:
    s = stock_df.set_index("date")["close"].pct_change()
    i = index_df.set_index("date")["close"].pct_change()
    joined = pd.concat([s, i], axis=1, keys=["s", "i"]).dropna().tail(window)
    if len(joined) < 30:
        return 1.0
    cov = np.cov(joined["s"], joined["i"])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0

def score_ma(df: pd.DataFrame) -> int:
    r = df.iloc[-1]
    c, m5, m20, m60 = r["close"], r["ma5"], r["ma20"], r["ma60"]
    if pd.isna(m60): return 0
    if c > m5 > m20 > m60: return 2
    if c > m20 and m5 < m20: return 1
    if c < m20 and c > m60: return 0
    if c < m20 and m5 < m20 < m60: return -1
    if c < m60: return -2
    return 0

def score_macd(df: pd.DataFrame) -> int:
    r, p = df.iloc[-1], df.iloc[-2]
    dif, dea, hist = r["dif"], r["dea"], r["macd_hist"]
    p_hist = p["macd_hist"]
    if dif > dea and dif > 0 and hist > p_hist: return 2
    if dif > dea and hist < p_hist: return 1
    if dif < dea and hist > p_hist: return 0
    if dif < dea and hist < p_hist: return -1
    if dif < 0 and dif < dea and p["dif"] > p["dea"]: return -2
    return 0

def score_relative(stock_df, index_df) -> int:
    if len(stock_df) < 21 or len(index_df) < 21: return 0
    s20 = stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1
    i20 = index_df["close"].iloc[-1] / index_df["close"].iloc[-21] - 1
    diff = (s20 - i20) * 100
    if diff > 5: return 2
    if diff > 0: return 1
    if diff > -5: return -1
    return -2

def score_volume(df: pd.DataFrame) -> int:
    r = df.iloc[-1]
    if pd.isna(r["vol_ma5"]) or r["vol_ma5"] == 0: return 0
    ratio = r["volume"] / r["vol_ma5"]
    if ratio > 1.2 and r["pct_chg"] > 0: return 1
    if ratio < 0.8 and r["pct_chg"] < 0: return -1
    return 0

def score_valuation(pe_pct) -> int:
    if pe_pct is None: return 0
    if pe_pct < 0.20: return 2
    if pe_pct < 0.40: return 1
    if pe_pct < 0.60: return 0
    if pe_pct < 0.80: return -1
    return -2

WEIGHTS = {"ma": 0.25, "macd": 0.20, "relative": 0.20, "volume": 0.15, "valuation": 0.20}

def run_trend_engine(stock_df, index_df, pe_pct=None) -> dict:
    scores = {
        "ma": score_ma(stock_df), "macd": score_macd(stock_df),
        "relative": score_relative(stock_df, index_df),
        "volume": score_volume(stock_df), "valuation": score_valuation(pe_pct),
    }
    total = sum(scores[k] * WEIGHTS[k] for k in scores)
    if total >= 1.2: rating = "偏多"
    elif total >= 0.3: rating = "中性偏多"
    elif total > -0.3: rating = "震荡"
    elif total > -1.2: rating = "中性偏空"
    else: rating = "偏空"
    recent = stock_df.tail(60)
    support = min(recent["low"].min(), stock_df["ma60"].iloc[-1])
    resistance = max(recent["high"].max(), stock_df["ma20"].iloc[-1])
    pivot = stock_df["ma20"].iloc[-1]
    return {"scores": scores, "total": round(total, 2), "rating": rating,
            "support": round(support, 2), "resistance": round(resistance, 2), "pivot": round(pivot, 2)}

# ============================================================
# 工具匹配（此处省略详细分支，保持你原有逻辑即可）
# 为控制长度，仅保留核心结构，你可从上一版完整复制 match_hedge 函数
# ============================================================

# 请将上一版完整的 match_hedge 函数粘贴在此处，逻辑完全不变

# ============================================================
# 主界面
# ============================================================

st.set_page_config(page_title="对冲分析助手", layout="centered")

# 样式部分保持不变，直接沿用你上一版的 CSS

st.markdown(
    """
    <div style="display:flex; align-items:baseline; flex-wrap:wrap; gap:10px; margin-bottom:12px;">
      <span style="font-size:2.0rem; font-weight:800;">📊 持仓对冲分析助手</span>
      <span style="color:#e00000; font-size:1.05rem; font-weight:700;">
        纯分析 · 不连接交易接口 · 所有操作需手动执行 · 两融和期权账户已默认开通
      </span>
    </div>
    """,
    unsafe_allow_html=True
)

latest_td = get_latest_trading_day()
cal_note = "（已跳过周末与法定节假日）" if _CALENDAR_AVAILABLE else "（⚠️ 日历库未安装，仅跳过周末）"
st.caption(f"当前分析基准日：**{latest_td.strftime('%Y-%m-%d')}** {cal_note}")

col1, col2 = st.columns(2)
code = col1.text_input("股票代码", "")
buy_price = col1.number_input("买入价", value=None, step=0.01, format="%.2f", placeholder="如 23.00")
qty = col2.number_input("数量（股）", value=None, step=100, placeholder="如 100")
buy_date = col2.date_input("买入日期", value=None)

if st.button("开始分析"):
    if not code.strip() or buy_price is None or qty is None:
        st.warning("请先填写股票代码、买入价和数量。")
    else:
        with st.spinner("正在扫描可用数据源并分析..."):
            try:
                code = code.strip()
                qty = int(qty)

                stock_df = add_indicators(fetch_stock_daily(code))
                index_df = fetch_index_daily()
                rt = fetch_realtime(code)
                price = rt.get("price", stock_df["close"].iloc[-1])
                name = rt.get("name", code)
                beta = calc_beta(stock_df, index_df)

                broad_etf = select_broad_etf(code, beta)

                available_sources, probe_details = detect_available_etf_sources()
                if available_sources:
                    st.success(f"数据源自检完成：{len(available_sources)} 个源可用")
                else:
                    st.warning("⚠️ 所有ETF数据源均探测失败，相关性验证可能无法完成。")

                etf_hist, etf_source = fetch_etf_daily(broad_etf["code"], days=90)
                corr = calc_correlation_r2(stock_df, etf_hist, window=60)
                grade = r2_grade(corr["r2"])

                etf_rt = fetch_etf_realtime(broad_etf["code"])
                etf_price = etf_rt.get("price")

                market_value = price * qty
                pnl = (price - buy_price) * qty

                trend = run_trend_engine(stock_df, index_df, pe_pct=None)
                # hedge = match_hedge(...)  # 使用你原有的完整 match_hedge

                with st.expander("🔍 数据源自检详情", expanded=False):
                    for line in probe_details:
                        st.write(line)

                st.subheader("【持仓概览】")
                st.write(f"**{name}（{code}）** 现价 {price:.2f} | 成本 {buy_price:.2f} | "
                         f"{qty}股 | 市值 {market_value:.0f}元 | 浮盈亏 {pnl:+.0f}元")

                st.subheader(f"【趋势判断】{trend['rating']}（加权得分 {trend['total']}）")
                SCORE_LABELS = {
                    "ma": ("均线系统", 0.25), "macd": ("MACD动能", 0.20),
                    "relative": ("相对强弱", 0.20), "volume": ("量能", 0.15),
                    "valuation": ("估值分位", 0.20),
                }
                rows = []
                for key, (label, weight) in SCORE_LABELS.items():
                    raw = trend["scores"][key]
                    rows.append({"维度": label, "得分": raw, "权重": f"{int(weight*100)}%", "加权分": round(raw * weight, 2)})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.write(f"支撑 **{trend['support']}** | 压力 **{trend['resistance']}** | "
                         f"多空分界 **{trend['pivot']}** | Beta **{beta:.2f}**")

                st.subheader("【宽基匹配度验证】")
                st.write(f"规则初选：**{broad_etf['name']}（{broad_etf['code']}）**，跟踪 {broad_etf['index_name']}。")

                if corr["r"] is not None:
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("相关系数 r", f"{corr['r']:.3f}")
                    col_b.metric("R²（拟合优度）", f"{corr['r2']:.3f}")
                    col_c.metric("匹配度等级", grade["label"])

                    if grade["color"] == "green":
                        st.success(f"✅ {grade['meaning']}")
                    elif grade["color"] == "orange":
                        st.warning(f"⚠️ {grade['meaning']}")
                    else:
                        st.error(f"❌ {grade['meaning']}")

                    st.caption(f"数据源：**{etf_source}**。基于最近 {corr['n']} 个交易日的日收益率对齐计算。")
                else:
                    st.warning(
                        f"⚠️ 当前可用数据源均未能获取 {broad_etf['name']} 的历史日线，"
                        f"无法计算实际相关性。下方期权建议仅基于 Beta 和规则初选。"
                    )
                    st.caption(f"失败详情：{etf_source}")

                st.subheader("【工具匹配 · 具体执行指令】")
                # 此处渲染你原有的 hedge["options"] 列表

                st.subheader("【手动执行提示】")
                st.warning("以上指令均为参考，请在券商 APP 中核对实时价格、券源、权利金后再手动下单。")

            except Exception as e:
                st.error(f"数据获取或计算失败：{e}")

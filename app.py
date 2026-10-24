import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
from requests.exceptions import ConnectionError, Timeout

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
            print(f"请求失败，{delay}秒后重试 ({i+1}/{retries})...")
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
    end = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - pd.Timedelta(days=days * 2)).strftime("%Y%m%d")

    df = fetch_with_retry(
        ak.stock_zh_a_daily,
        symbol=symbol, start_date=start, end_date=end, adjust="qfq"
    )
    if df is None or df.empty:
        raise ValueError("获取股票历史数据失败（新浪接口），请稍后重试")

    df = df.rename(columns={
        "date": "date", "open": "open", "close": "close",
        "high": "high", "low": "low", "volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["pct_chg"] = df["close"].pct_change() * 100
    return df.sort_values("date").reset_index(drop=True)


def _fetch_etf_em(etf_code: str, days: int) -> pd.DataFrame:
    end = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - pd.Timedelta(days=days * 2)).strftime("%Y%m%d")
    df = ak.fund_etf_hist_em(
        symbol=etf_code, period="daily",
        start_date=start, end_date=end, adjust="qfq"
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"日期": "date", "收盘": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)


def _fetch_etf_tx(etf_code: str, days: int) -> pd.DataFrame:
    symbol = _etf_to_sina_symbol(etf_code)
    end = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - pd.Timedelta(days=days * 2)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist_tx(
        symbol=symbol, start_date=start, end_date=end, adjust="qfq"
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].sort_values("date").reset_index(drop=True)


def _fetch_etf_sina(etf_code: str, days: int) -> pd.DataFrame:
    symbol = _etf_to_sina_symbol(etf_code)
    df = ak.fund_etf_hist_sina(symbol=symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "close"]].tail(days).reset_index(drop=True)


@st.cache_data(ttl=1800)
def fetch_etf_daily(etf_code: str, days: int = 120) -> tuple:
    sources = [("东财", _fetch_etf_em), ("腾讯", _fetch_etf_tx), ("新浪", _fetch_etf_sina)]
    for name, func in sources:
        try:
            df = fetch_with_retry(func, etf_code, days, retries=1, delay=1)
            if df is not None and not df.empty and len(df) >= 30:
                return df, name
        except Exception:
            continue
        time.sleep(1)
    return pd.DataFrame(), "全部失败"


@st.cache_data(ttl=1800)
def fetch_index_daily(symbol: str = "sh000300", days: int = 250) -> pd.DataFrame:
    df = fetch_with_retry(ak.stock_zh_index_daily, symbol=symbol)
    if df is None or df.empty:
        raise ValueError("获取指数数据失败，请稍后重试")
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
        return {"price": float(r["最新价"]), "name": r["名称"],
                "pct": float(r["涨跌幅"]) if "涨跌幅" in r else 0.0}
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
        r = row.iloc[0]
        return {"price": float(r["最新价"]), "name": r["名称"]}
    except Exception:
        return {}


# ============================================================
# 行业 ETF 映射 + 宽基选择 + 相关性验证
# ============================================================

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
        return {"code": "588000", "name": "科创50ETF", "index_name": "科创50",
                "match_note": "科创板初选科创50"}
    if code.startswith("30"):
        return {"code": "159915", "name": "创业板ETF", "index_name": "创业板指",
                "match_note": "创业板初选创业板指"}
    if code.startswith(("8", "4")):
        return {"code": "510300", "name": "300ETF", "index_name": "沪深300",
                "match_note": "北交所无对应宽基期权，降级用沪深300"}
    if beta >= 1.3:
        return {"code": "510500", "name": "500ETF", "index_name": "中证500",
                "match_note": "高Beta初选中证500"}
    if beta <= 0.8:
        return {"code": "510050", "name": "50ETF", "index_name": "上证50",
                "match_note": "低Beta初选上证50"}
    return {"code": "510300", "name": "300ETF", "index_name": "沪深300",
            "match_note": "中等Beta初选沪深300"}


def calc_correlation_r2(stock_df: pd.DataFrame, etf_df: pd.DataFrame,
                        window: int = 60) -> dict:
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
        return {"label": "无法计算", "color": "gray",
                "meaning": "数据不足，无法验证实际相关性"}
    if r2 >= 0.49:
        return {"label": "强匹配", "color": "green",
                "meaning": "该宽基能解释个股大部分波动，期权对冲效果相对可靠"}
    if r2 >= 0.25:
        return {"label": "中等匹配", "color": "orange",
                "meaning": "该宽基能解释部分波动，对冲有效但不完全"}
    if r2 >= 0.09:
        return {"label": "弱匹配", "color": "red",
                "meaning": "联动性较弱，期权对冲效果有限"}
    return {"label": "极弱/无匹配", "color": "red",
            "meaning": "该宽基与个股联动性很差，不建议用期权对冲"}


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
    strike_int = int(round(strike * 1000))
    return f"{etf_prefix}{opt_type}{month}{strike_int}"


# ============================================================
# 指标层
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


def calc_beta(stock_df: pd.DataFrame, index_df: pd.DataFrame,
              window: int = 90) -> float:
    s = stock_df.set_index("date")["close"].pct_change()
    i = index_df.set_index("date")["close"].pct_change()
    joined = pd.concat([s, i], axis=1, keys=["s", "i"]).dropna().tail(window)
    if len(joined) < 30:
        return 1.0
    cov = np.cov(joined["s"], joined["i"])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0


# ============================================================
# 趋势打分引擎
# ============================================================

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


WEIGHTS = {"ma": 0.25, "macd": 0.20, "relative": 0.20,
           "volume": 0.15, "valuation": 0.20}


def run_trend_engine(stock_df, index_df, pe_pct=None) -> dict:
    scores = {
        "ma": score_ma(stock_df),
        "macd": score_macd(stock_df),
        "relative": score_relative(stock_df, index_df),
        "volume": score_volume(stock_df),
        "valuation": score_valuation(pe_pct),
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
            "support": round(support, 2), "resistance": round(resistance, 2),
            "pivot": round(pivot, 2)}


# ============================================================
# 工具匹配
# ============================================================

def match_hedge(rating: str, market_value: float, qty: int,
                beta: float,
                stock_code: str = "", stock_name: str = "",
                broad_etf: dict = None,
                etf_price: float = None,
                corr: dict = None) -> dict:

    options = []

    if broad_etf is None:
        broad_etf = {"code": "510300", "name": "300ETF", "index_name": "沪深300",
                     "match_note": ""}
    corr = corr or {"r": None, "r2": None, "n": 0}
    grade = r2_grade(corr["r2"])

    etf_price_used = etf_price if etf_price else 4.0
    price_note = "" if etf_price else "（实时价格获取失败，按参考价估算）"

    contract_notional = etf_price_used * 10000
    equiv = market_value * beta / contract_notional if contract_notional else 0
    equiv_int = int(equiv)

    interval = get_strike_interval(etf_price_used)
    month = option_month_label()
    put_strike = round_to_strike(etf_price_used * 0.95, interval)
    call_strike = round_to_strike(etf_price_used * 1.05, interval)
    put_name = format_option_name(broad_etf["name"], "沽", month, put_strike)
    call_name = format_option_name(broad_etf["name"], "购", month, call_strike)

    match_line = (
        f"匹配宽基：**{broad_etf['name']}（{broad_etf['code']}）**，"
        f"跟踪 {broad_etf['index_name']}。"
        f"实际相关性：**r = {corr['r']}，R² = {corr['r2']}**"
        f"（{corr['n']} 个交易日），匹配度等级：**{grade['label']}**。"
    )

    if market_value < 2000:
        return {"options": [{
            "tool": "直接减仓/清仓",
            "detail": (
                f"市值 {market_value:.0f} 元低于任何衍生品的有效规模门槛。"
                f"**直接卖出 {qty} 股 {stock_name}（{stock_code}）**，"
                f"市价委托即可，无利息、无权利金、无保证金。"
            ),
            "cost_note": "仅正常交易佣金，无额外成本",
            "action": "SELL",
            "target": f"{stock_name}（{stock_code}）",
            "side": "卖出",
            "qty": qty,
            "price_hint": "市价委托",
        }]}

    if rating in ("偏空", "中性偏空"):
        reduce_ratio = 0.5 if rating == "中性偏空" else 0.7
        reduce_qty = int(qty * reduce_ratio // 100) * 100
        hedge_ratio = 0.5 if rating == "中性偏空" else 0.7
        short_qty = int(qty * hedge_ratio // 100) * 100
        short_qty = max(100, min(short_qty, qty - 100))

        options.append({
            "tool": "减仓（首选）",
            "detail": (
                f"**卖出 {reduce_qty} 股 {stock_name}（{stock_code}）**，"
                f"市价委托即可，减仓约 {reduce_ratio*100:.0f}%，"
                f"无利息成本，剩余持仓 {qty - reduce_qty} 股"
            ),
            "cost_note": "仅正常交易佣金",
            "action": "SELL",
            "target": f"{stock_name}（{stock_code}）",
            "side": "卖出",
            "qty": reduce_qty,
            "price_hint": "市价委托",
        })

        options.append({
            "tool": "融券部分对冲（保留底仓的备选）",
            "detail": (
                f"**融券卖出 {short_qty} 股 {stock_name}（{stock_code}）**，"
                f"保留 {qty - short_qty} 股多头。"
                f"对冲比例约 {short_qty/qty*100:.0f}%，"
                f"**下单前务必在券商APP确认该股实时可融余量**"
            ),
            "cost_note": "融券年化利息约 8%，按日计提",
            "action": "SHORT_SELL",
            "target": f"{stock_name}（{stock_code}）",
            "side": "融券卖出",
            "qty": short_qty,
            "price_hint": "不低于最新成交价",
        })

        return {"options": options}

    if rating == "震荡":
        options = []

        industry = detect_industry_etf(stock_name)
        if industry and market_value >= 5000:
            etf_code, etf_name = industry
            etf_rt = fetch_etf_realtime(etf_code)
            etf_px = etf_rt.get("price")
            if etf_px:
                industry_beta = 1.3 if etf_code in ("512000", "512800") else 1.2
                raw_shares = market_value * industry_beta / etf_px
                short_shares = max(100, int(raw_shares // 100) * 100)
                options.append({
                    "tool": "融券做空行业 ETF",
                    "detail": (
                        f"**融券卖出 {etf_name}（{etf_code}）{short_shares} 份**，"
                        f"对冲 {stock_name} 的板块系统性风险。"
                        f"行业ETF券源通常比宽基更稀缺，"
                        f"**下单前务必在“融券标的”中确认实时可融余量**"
                    ),
                    "cost_note": "融券年化利息约 8%，券源稀缺时可能更高，以券商APP为准",
                    "action": "SHORT_SELL",
                    "target": f"{etf_name}（{etf_code}）",
                    "side": "融券卖出",
                    "qty": short_shares,
                    "price_hint": "不低于最新成交价",
                    "etf_detail": {
                        "name": etf_name,
                        "code": etf_code,
                        "price": etf_px,
                        "short_shares": short_shares,
                        "industry_beta": industry_beta,
                    },
                })

        if equiv_int >= 1:
            if corr["r2"] is not None and corr["r2"] < 0.09:
                options.append({
                    "tool": f"⚠️ {broad_etf['name']} 期权（不推荐）",
                    "detail": (
                        f"{match_line}"
                        f"R² 低于 0.09，期权对冲大概率无效，**不建议执行**。"
                        f"若坚持使用：买入 {put_name} + 卖出 {call_name}，各 {equiv_int} 张。"
                    ),
                    "cost_note": "权利金可能成为沉没成本",
                })
            else:
                options.append({
                    "tool": f"领口策略（{broad_etf['name']}）",
                    "detail": (
                        f"{match_line}"
                        f"**买入 {put_name} {equiv_int} 张**（防跌）"
                        f" + **卖出 {call_name} {equiv_int} 张**（收权利金抵消成本）。"
                        f"{broad_etf['name']} 实时参考价 **{etf_price_used:.3f} 元**{price_note}。"
                        f"**权利金精确报价以券商APP期权T型报价为准**"
                    ),
                    "cost_note": "净权利金成本较低，上方收益受限",
                    "action": "OPTION_COLLAR",
                    "target": f"{put_name} + {call_name}",
                    "side": "买入认沽 + 卖出认购",
                    "qty": equiv_int,
                    "price_hint": "以券商APP权利金为准",
                })

        if not options:
            options.append({
                "tool": "持有观望",
                "detail": "设置止损提醒，不建对冲仓",
                "cost_note": "无",
            })

        return {"options": options}

    if rating == "中性偏多":
        options = []
        if equiv_int >= 1:
            if corr["r2"] is not None and corr["r2"] < 0.09:
                options.append({
                    "tool": f"⚠️ {broad_etf['name']} 期权（不推荐）",
                    "detail": f"{match_line} R² 太低，期权对冲效果不可靠，**不建议执行**。",
                    "cost_note": "权利金可能沉没",
                })
            else:
                options.append({
                    "tool": f"备兑开仓（卖出 {broad_etf['name']} 认购）",
                    "detail": (
                        f"{match_line}"
                        f"**卖出 {call_name} {equiv_int} 张**，收取权利金增厚收益。"
                        f"需持有对应 **{broad_etf['name']}（{broad_etf['code']}）现货** 作为担保，"
                        f"每张合约对应 10,000 份 ETF。"
                        f"上方涨幅超过行权价时收益封顶"
                    ),
                    "cost_note": "收取权利金，需有对应 ETF 现货；权利金以券商APP为准",
                    "action": "OPTION_COVERED_CALL",
                    "target": call_name,
                    "side": "卖出认购",
                    "qty": equiv_int,
                    "price_hint": "以券商APP权利金为准",
                })
                options.append({
                    "tool": f"保护性认沽（买入 {broad_etf['name']} 认沽）",
                    "detail": (
                        f"{match_line}"
                        f"**买入 {put_name} {equiv_int} 张**，支付权利金锁定下行底线，保留上方收益。"
                        f"最大损失为已支付的权利金"
                    ),
                    "cost_note": "权利金为最大损失；精确报价以券商APP为准",
                    "action": "OPTION_PROTECTIVE_PUT",
                    "target": put_name,
                    "side": "买入认沽",
                    "qty": equiv_int,
                    "price_hint": "以券商APP权利金为准",
                })

        if not options:
            options.append({
                "tool": "无需对冲",
                "detail": "设置移动止盈/止损提醒即可",
                "cost_note": "无",
            })
        return {"options": options}

    return {"options": [{
        "tool": "无需对冲",
        "detail": "趋势偏多，设置移动止盈提醒即可",
        "cost_note": "无",
    }]}


# ============================================================
# 主界面
# ============================================================

st.set_page_config(page_title="对冲分析助手", layout="centered")

# ---- 字号模板选择 ----
size_mode = st.radio(
    "显示大小（可切换）",
    ["标准", "大字（推荐老年人）", "特大字"],
    horizontal=True,
    index=1,
)
scale = {"标准": 1.0, "大字（推荐老年人）": 1.3, "特大字": 1.6}[size_mode]

# ---- 动态 CSS ----
st.markdown(f"""
<style>
[data-testid="stStatusWidget"],
[data-testid="stAppRunningMan"] {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}}

[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stDateInput"] label {{
    font-size: {18 * scale}px !important;
    font-weight: 800 !important;
    color: #111 !important;
}}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input {{
    font-size: {18 * scale}px !important;
    font-weight: 600 !important;
}}

.stButton > button {{
    font-size: {22 * scale}px !important;
    font-weight: bold !important;
    color: #FFD700 !important;
    background-color: #1a1a1a !important;
    border: 2px solid #FFD700 !important;
    padding: {12 * scale}px {36 * scale}px !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}}
.stButton > button:hover {{
    background-color: #333333 !important;
    color: #FFE55C !important;
    border-color: #FFE55C !important;
}}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    font-size: {15 * scale}px !important;
}}

[data-testid="stMetricValue"] {{
    font-size: {24 * scale}px !important;
}}
[data-testid="stMetricLabel"] {{
    font-size: {13 * scale}px !important;
}}

[data-testid="stRadio"] label {{
    font-size: {16 * scale}px !important;
    font-weight: 600 !important;
}}
</style>
""", unsafe_allow_html=True)

# ---- 标题 + 红色提示合并为一行 ----
st.markdown(f"""
<div style="display:flex; align-items:baseline; flex-wrap:wrap; gap:10px; margin-bottom:8px;">
  <span style="font-size:{1.8 * scale}rem; font-weight:800;">📊 持仓对冲分析助手</span>
  <span style="color:#e00000; font-size:{0.95 * scale}rem; font-weight:700;">
    纯分析 · 不连接交易接口 · 所有操作需手动执行 · 两融和期权账户已默认开通
  </span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
code = col1.text_input("股票代码", "")
buy_price = col1.number_input("买入价", value=None, step=0.01,
                              format="%.2f", placeholder="如 23.00")
qty = col2.number_input("数量（股）", value=None, step=100,
                        placeholder="如 100")
buy_date = col2.date_input("买入日期", value=None)

if st.button("开始分析"):
    if not code.strip() or buy_price is None or qty is None:
        st.warning("请先填写股票代码、买入价和数量后再点击“开始分析”。")
    else:
        with st.spinner("在运算请稍等..."):
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
                etf_hist, etf_source = fetch_etf_daily(broad_etf["code"], days=90)
                corr = calc_correlation_r2(stock_df, etf_hist, window=60)
                grade = r2_grade(corr["r2"])

                etf_rt = fetch_etf_realtime(broad_etf["code"])
                etf_price = etf_rt.get("price")

                market_value = price * qty
                pnl = (price - buy_price) * qty

                trend = run_trend_engine(stock_df, index_df, pe_pct=None)
                hedge = match_hedge(
                    trend["rating"], market_value, qty, beta,
                    stock_code=code, stock_name=name,
                    broad_etf=broad_etf, etf_price=etf_price, corr=corr,
                )

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
                    rows.append({"维度": label, "得分": raw,
                                 "权重": f"{int(weight*100)}%",
                                 "加权分": round(raw * weight, 2)})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.write(f"支撑 **{trend['support']}** | 压力 **{trend['resistance']}** | "
                         f"多空分界 **{trend['pivot']}** | Beta **{beta:.2f}**")

                st.subheader("【宽基匹配度验证】")
                st.write(
                    f"规则初选：**{broad_etf['name']}（{broad_etf['code']}）**，"
                    f"跟踪 {broad_etf['index_name']}。{broad_etf['match_note']}。"
                )

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

                    st.caption(
                        f"数据源：**{etf_source}**。"
                        f"基于最近 {corr['n']} 个交易日的日收益率对齐计算。"
                    )
                else:
                    st.warning(
                        f"⚠️ 东财、腾讯、新浪三个接口均未能获取 {broad_etf['name']} 的历史日线，"
                        f"无法计算实际相关性。下方期权建议仅基于 Beta 和规则初选。"
                    )

                st.subheader("【工具匹配 · 具体执行指令】")
                opts = hedge.get("options", [])
                for i, opt in enumerate(opts, 1):
                    if len(opts) > 1:
                        st.markdown(f"### 方案 {i}：{opt['tool']}")
                    else:
                        st.markdown(f"### {opt['tool']}")

                    if "action" in opt:
                        st.markdown(
                            f"- **对冲产品**：{opt['target']}\n"
                            f"- **方向**：{opt['side']}\n"
                            f"- **数量**：**{opt['qty']}**\n"
                            f"- **价格**：{opt['price_hint']}"
                        )

                    if "etf_detail" in opt:
                        d = opt["etf_detail"]
                        st.markdown(
                            f"- **行业 Beta 估计**：{d['industry_beta']}\n"
                            f"- **实时参考价**：**{d['price']:.3f} 元**"
                        )

                    st.info(opt["detail"])
                    st.warning(f"成本：{opt['cost_note']}")

                    if i < len(opts):
                        st.divider()

                st.subheader("【手动执行提示】")
                st.warning(
                    "以上指令均为参考，请在券商 APP 中核对实时价格、券源、权利金后再手动下单。"
                    "本工具不发送任何委托。"
                )

            except Exception as e:
                st.error(f"数据获取或计算失败：{e}")

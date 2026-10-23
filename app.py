import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
from requests.exceptions import ConnectionError, Timeout

# ============================================================
# 模块二：数据层（新浪财经数据源）
# ============================================================

def fetch_with_retry(func, *args, retries=3, delay=2, **kwargs):
    """通用重试函数：捕获连接类异常，失败后等待 delay 秒重试"""
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
    """把6位代码转成新浪接口需要的带前缀格式"""
    if code.startswith(('6', '5', '9')):
        return f"sh{code}"
    return f"sz{code}"


@st.cache_data(ttl=1800)
def fetch_stock_daily(code: str, days: int = 250) -> pd.DataFrame:
    """新浪财经接口获取前复权日线，带重试"""
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


@st.cache_data(ttl=1800)
def fetch_index_daily(symbol: str = "sh000300", days: int = 250) -> pd.DataFrame:
    """沪深300指数日线（stock_zh_index_daily 本身就是新浪源），带重试"""
    df = fetch_with_retry(ak.stock_zh_index_daily, symbol=symbol)

    if df is None or df.empty:
        raise ValueError("获取指数数据失败，请稍后重试")

    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days).reset_index(drop=True)


@st.cache_data(ttl=600)
def fetch_realtime(code: str) -> dict:
    """
    实时报价：优先用新浪全市场快照，失败则返回空字典。
    返回空时主流程会自动降级用日线最后一根收盘价。
    """
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


# ============================================================
# 指标层
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"]  = df["close"].rolling(5).mean()
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
    """个股 vs 沪深300 的 Beta，用日收益率线性回归斜率"""
    s = stock_df.set_index("date")["close"].pct_change()
    i = index_df.set_index("date")["close"].pct_change()
    joined = pd.concat([s, i], axis=1, keys=["s", "i"]).dropna().tail(window)
    if len(joined) < 30:
        return 1.0
    cov = np.cov(joined["s"], joined["i"])
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 1.0


# ============================================================
# 趋势打分引擎（模块三）
# ============================================================

def score_ma(df: pd.DataFrame) -> int:
    r = df.iloc[-1]
    c, m5, m20, m60 = r["close"], r["ma5"], r["ma20"], r["ma60"]
    if pd.isna(m60):
        return 0
    if c > m5 > m20 > m60:
        return 2
    if c > m20 and m5 < m20:
        return 1
    if c < m20 and c > m60:
        return 0
    if c < m20 and m5 < m20 < m60:
        return -1
    if c < m60:
        return -2
    return 0


def score_macd(df: pd.DataFrame) -> int:
    r, p = df.iloc[-1], df.iloc[-2]
    dif, dea, hist = r["dif"], r["dea"], r["macd_hist"]
    p_hist = p["macd_hist"]
    if dif > dea and dif > 0 and hist > p_hist:
        return 2
    if dif > dea and hist < p_hist:
        return 1
    if dif < dea and hist > p_hist:
        return 0
    if dif < dea and hist < p_hist:
        return -1
    if dif < 0 and dif < dea and p["dif"] > p["dea"]:
        return -2
    return 0


def score_relative(stock_df, index_df) -> int:
    if len(stock_df) < 21 or len(index_df) < 21:
        return 0
    s20 = stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1
    i20 = index_df["close"].iloc[-1] / index_df["close"].iloc[-21] - 1
    diff = (s20 - i20) * 100
    if diff > 5:
        return 2
    if diff > 0:
        return 1
    if diff > -5:
        return -1
    return -2


def score_volume(df: pd.DataFrame) -> int:
    r = df.iloc[-1]
    if pd.isna(r["vol_ma5"]) or r["vol_ma5"] == 0:
        return 0
    ratio = r["volume"] / r["vol_ma5"]
    if ratio > 1.2 and r["pct_chg"] > 0:
        return 1
    if ratio < 0.8 and r["pct_chg"] < 0:
        return -1
    return 0


def score_valuation(pe_pct: float | None) -> int:
    if pe_pct is None:
        return 0
    if pe_pct < 0.20:
        return 2
    if pe_pct < 0.40:
        return 1
    if pe_pct < 0.60:
        return 0
    if pe_pct < 0.80:
        return -1
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
    if total >= 1.2:
        rating = "偏多"
    elif total >= 0.3:
        rating = "中性偏多"
    elif total > -0.3:
        rating = "震荡"
    elif total > -1.2:
        rating = "中性偏空"
    else:
        rating = "偏空"

    recent = stock_df.tail(60)
    support = min(recent["low"].min(), stock_df["ma60"].iloc[-1])
    resistance = max(recent["high"].max(), stock_df["ma20"].iloc[-1])
    pivot = stock_df["ma20"].iloc[-1]

    return {"scores": scores, "total": round(total, 2), "rating": rating,
            "support": round(support, 2), "resistance": round(resistance, 2),
            "pivot": round(pivot, 2)}


# ============================================================
# 工具匹配与数量计算（模块四、五）
# ============================================================

def match_hedge(rating: str, market_value: float, qty: int,
                beta: float, etf_price: float,
                has_margin: bool, has_option: bool) -> dict:
    """
    返回 {"tool": ..., "qty": ..., "detail": ..., "cost_note": ...}

    设计原则：
    - 偏空/中性偏空：首选“直接减仓”（无利息成本），融券对冲仅作为“必须保留底仓”时的备选；
    - 融券数量按持仓比例计算，去掉固定上限，至少保留100股多头；
    - 震荡/偏多：期权保护优先，规模不足时降级为仓位管理。
    """
    contract_notional = etf_price * 10000
    equiv = market_value * beta / contract_notional if contract_notional else 0
    equiv_int = int(equiv)

    # 1) 规模过小，直接给仓位管理建议
    if market_value < 2000:
        return {"tool": "仓位管理", "qty": 0,
                "detail": f"市值{market_value:.0f}元低于衍生品有效门槛",
                "cost_note": "无额外成本"}

    # 2) 偏空 / 中性偏空
    if rating in ("偏空", "中性偏空"):
        # 减仓比例：偏空减得更多
        reduce_ratio = 0.5 if rating == "中性偏空" else 0.7
        reduce_qty = int(qty * reduce_ratio // 100) * 100

        # 若有两融权限且持仓≥200股，给出“减仓 + 融券备选”的组合建议
        if has_margin and qty >= 200:
            hedge_ratio = 0.5 if rating == "中性偏空" else 0.7
            short_qty = int(qty * hedge_ratio // 100) * 100
            # 至少融券100股，且至少保留100股多头
            short_qty = max(100, min(short_qty, qty - 100))

            return {
                "tool": "减仓（首选）或 融券部分对冲（备选）",
                "qty": short_qty,
                "detail": (
                    f"首选：直接减仓约 {reduce_ratio*100:.0f}%，"
                    f"卖出约 {reduce_qty} 股，无利息成本；"
                    f"若因打新市值、长期底仓等原因必须保留仓位，"
                    f"可融券卖出约 {short_qty} 股"
                    f"（对冲比例约 {short_qty/qty*100:.0f}%，"
                    f"保留 {qty - short_qty} 股多头），"
                    f"但需承担年化约8%的融券利息"
                ),
                "cost_note": "减仓无额外成本；融券按日计息，年化约8%",
            }

        # 无两融权限或持仓不足200股，只能减仓
        return {
            "tool": "减仓",
            "qty": reduce_qty,
            "detail": (
                f"建议卖出约 {reduce_qty} 股（减仓约 {reduce_ratio*100:.0f}%），"
                f"无利息成本；"
                f"当前持仓规模或权限不支持融券部分对冲"
            ),
            "cost_note": "无额外成本",
        }

    # 3) 震荡
    if rating == "震荡":
        if has_option and equiv_int >= 1:
            return {"tool": "买入轻度虚值认沽", "qty": equiv_int,
                    "detail": f"低成本保护，{equiv_int}张",
                    "cost_note": "权利金为最大损失"}
        if has_margin and qty >= 300:
            return {"tool": "融券保守对冲", "qty": 100,
                    "detail": "对冲约1/3仓位", "cost_note": "融券利息"}
        return {"tool": "持有观望", "qty": 0,
                "detail": "设置止损提醒，不建对冲仓", "cost_note": "无"}

    # 4) 中性偏多 / 偏多
    if has_option and equiv_int >= 1 and rating == "中性偏多":
        return {"tool": "可选保护性认沽", "qty": equiv_int,
                "detail": f"成本可控时买{equiv_int}张作保险",
                "cost_note": "权利金"}
    return {"tool": "无需对冲", "qty": 0,
            "detail": "设置移动止盈/止损提醒即可", "cost_note": "无"}


# ============================================================
# 主界面与报告输出（模块六）
# ============================================================

st.set_page_config(page_title="对冲分析助手", layout="centered")
st.title("📊 持仓对冲分析助手")
st.caption("纯分析工具 · 不连接交易接口 · 所有操作需手动执行")

col1, col2 = st.columns(2)
code = col1.text_input("股票代码", "")
buy_price = col1.number_input("买入价", value=None, step=0.01,
                              format="%.2f", placeholder="如 23.00")
qty = col2.number_input("数量（股）", value=None, step=100,
                        placeholder="如 100")
buy_date = col2.date_input("买入日期", value=None)

st.markdown("**账户权限（手动勾选）**")
c1, c2 = st.columns(2)
has_margin = c1.checkbox("已开通两融账户")
has_option = c2.checkbox("已开通ETF期权账户")

if st.button("开始分析"):
    if not code.strip() or buy_price is None or qty is None:
        st.warning("请先填写股票代码、买入价和数量后再点击“开始分析”。")
    else:
        try:
            code = code.strip()
            qty = int(qty)

            stock_df = add_indicators(fetch_stock_daily(code))
            index_df = fetch_index_daily()
            rt = fetch_realtime(code)

            price = rt.get("price", stock_df["close"].iloc[-1])
            name = rt.get("name", code)
            beta = calc_beta(stock_df, index_df)

            etf_price = 4.0
            market_value = price * qty
            pnl = (price - buy_price) * qty

            trend = run_trend_engine(stock_df, index_df, pe_pct=None)
            hedge = match_hedge(trend["rating"], market_value, qty, beta,
                                etf_price, has_margin, has_option)

            st.subheader("【持仓概览】")
            st.write(f"**{name}（{code}）** 现价 {price:.2f} | 成本 {buy_price:.2f} | "
                     f"{qty}股 | 市值 {market_value:.0f}元 | "
                     f"浮盈亏 {pnl:+.0f}元")

            st.subheader(f"【趋势判断】{trend['rating']}（加权得分 {trend['total']}）")

            SCORE_LABELS = {
                "ma":        ("均线系统", 0.25),
                "macd":      ("MACD动能", 0.20),
                "relative":  ("相对强弱", 0.20),
                "volume":    ("量能",     0.15),
                "valuation": ("估值分位", 0.20),
            }

            score_rows = []
            for key, (label, weight) in SCORE_LABELS.items():
                raw = trend["scores"][key]
                score_rows.append({
                    "维度": label,
                    "得分": raw,
                    "权重": f"{int(weight * 100)}%",
                    "加权分": round(raw * weight, 2),
                })
            st.table(pd.DataFrame(score_rows))

            st.write(f"支撑 **{trend['support']}** | 压力 **{trend['resistance']}** | "
                     f"多空分界 **{trend['pivot']}** | Beta **{beta:.2f}**")

            st.subheader("【工具匹配】")
            st.info(f"**{hedge['tool']}** — {hedge['detail']}")
            st.write(f"成本提示：{hedge['cost_note']}")

            st.subheader("【手动执行提示】")
            st.warning("请在券商App中手动完成上述操作；本工具不发送任何委托。")

        except Exception as e:
            st.error(f"数据获取或计算失败：{e}")

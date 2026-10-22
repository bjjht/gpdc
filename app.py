import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date

# ============================================================
# 模块二：数据层
# ============================================================

@st.cache_data(ttl=1800)
def fetch_stock_daily(code: str, days: int = 250) -> pd.DataFrame:
    """前复权日线，返回最近 days 根K线"""
    end = datetime.today().strftime("%Y%m%d")
    start = (datetime.today() - pd.Timedelta(days=days * 2)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist(symbol=code, period="daily",
                            start_date=start, end_date=end, adjust="qfq")
    df = df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close",
                            "最高": "high", "最低": "low", "成交量": "volume",
                            "涨跌幅": "pct_chg"})
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=1800)
def fetch_index_daily(symbol: str = "sh000300", days: int = 250) -> pd.DataFrame:
    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days).reset_index(drop=True)

@st.cache_data(ttl=600)
def fetch_realtime(code: str) -> dict:
    spot = ak.stock_zh_a_spot_em()
    row = spot[spot["代码"] == code]
    if row.empty:
        return {}
    r = row.iloc[0]
    return {"price": float(r["最新价"]), "name": r["名称"],
            "pct": float(r["涨跌幅"])}

# ============================================================
# 指标层
# ============================================================

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma5"]  = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["dif"] = ema12 - ema26
    df["dea"] = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (df["dif"] - df["dea"]) * 2
    # 量比（5日均量）
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
                  def score_ma(df: pd.DataFrame) -> int:
    r = df.iloc[-1]
    c, m5, m20, m60 = r["close"], r["ma5"], r["ma20"], r["ma60"]
    if pd.isna(m60): return 0
    if c > m5 > m20 > m60:   return 2
    if c > m20 and m5 < m20: return 1
    if c < m20 and c > m60:  return 0
    if c < m20 and m5 < m20 < m60: return -1
    if c < m60:              return -2
    return 0

def score_macd(df: pd.DataFrame) -> int:
    r, p = df.iloc[-1], df.iloc[-2]
    dif, dea, hist = r["dif"], r["dea"], r["macd_hist"]
    p_hist = p["macd_hist"]
    if dif > dea and dif > 0 and hist > p_hist:  return 2
    if dif > dea and hist < p_hist:              return 1
    if dif < dea and hist > p_hist:              return 0   # 负柱收窄
    if dif < dea and hist < p_hist:              return -1
    if dif < 0 and dif < dea and p["dif"] > p["dea"]: return -2  # 零轴下死叉
    return 0

def score_relative(stock_df, index_df) -> int:
    s20 = stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1
    i20 = index_df["close"].iloc[-1] / index_df["close"].iloc[-21] - 1
    diff = (s20 - i20) * 100
    if diff > 5:   return 2
    if diff > 0:   return 1
    if diff > -5:  return -1
    return -2

def score_volume(df: pd.DataFrame) -> int:
    r = df.iloc[-1]
    if pd.isna(r["vol_ma5"]): return 0
    ratio = r["volume"] / r["vol_ma5"]
    if ratio > 1.2 and r["pct_chg"] > 0: return 1
    if ratio < 0.8 and r["pct_chg"] < 0: return -1
    return 0

def score_valuation(pe_pct: float | None) -> int:
    """pe_pct: 当前PE在历史中的分位（0-1）。无数据返回0"""
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
    if total >= 1.2:    rating = "偏多"
    elif total >= 0.3:  rating = "中性偏多"
    elif total > -0.3:  rating = "震荡"
    elif total > -1.2:  rating = "中性偏空"
    else:               rating = "偏空"
    # 关键价位
    recent = stock_df.tail(60)
    support = min(recent["low"].min(), stock_df["ma60"].iloc[-1])
    resistance = max(recent["high"].max(), stock_df["ma20"].iloc[-1])
    pivot = stock_df["ma20"].iloc[-1]
    return {"scores": scores, "total": round(total, 2), "rating": rating,
            "support": round(support, 2), "resistance": round(resistance, 2),
            "pivot": round(pivot, 2)}
def match_hedge(rating: str, market_value: float, qty: int,
                beta: float, etf_price: float,
                has_margin: bool, has_option: bool) -> dict:
    """
    返回 {"tool": ..., "qty": ..., "detail": ..., "cost_note": ...}
    etf_price: 300ETF 现价，用于计算单张名义价值
    """
    contract_notional = etf_price * 10000          # 单张ETF期权名义价值
    equiv = market_value * beta / contract_notional if contract_notional else 0
    equiv_int = int(equiv)                         # 向下取整

    # 1) 规模过小
    if market_value < 2000:
        return {"tool": "仓位管理", "qty": 0,
                "detail": f"市值{market_value:.0f}元低于衍生品有效门槛",
                "cost_note": "无额外成本"}

    # 2) 偏空 / 中性偏空
    if rating in ("偏空", "中性偏空"):
        if has_margin and qty >= 200:
            short_qty = 100 if rating == "中性偏空" else min(200, (qty // 100 - 1) * 100)
            return {"tool": "融券部分对冲", "qty": short_qty,
                    "detail": f"融券卖出{short_qty}股，保留{qty - short_qty}股多头",
                    "cost_note": f"年化利息约8%，按日计提"}
        if has_option and equiv_int >= 1:
            return {"tool": "买入虚值认沽", "qty": equiv_int,
                    "detail": f"等效{equiv:.2f}张，向下取整{equiv_int}张",
                    "cost_note": f"覆盖约{equiv_int/equiv*100:.0f}%风险暴露"}
        return {"tool": "仓位管理", "qty": 0,
                "detail": "无可用衍生品权限或规模不足，建议止损/减仓",
                "cost_note": "无额外成本"}

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
                "detail": f"成本可控时买{equiv_int}张作保险", "cost_note": "权利金"}
    return {"tool": "无需对冲", "qty": 0,
            "detail": "设置移动止盈/止损提醒即可", "cost_note": "无"}
                    st.set_page_config(page_title="对冲分析助手", layout="centered")
st.title("📊 持仓对冲分析助手")
st.caption("纯分析工具 · 不连接交易接口 · 所有操作需手动执行")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    code = col1.text_input("股票代码", "601066")
    buy_price = col1.number_input("买入价", value=23.00, step=0.01)
    qty = col2.number_input("数量（股）", value=100, step=100)
    buy_date = col2.date_input("买入日期", value=date(2026, 10, 15))
    st.markdown("**账户权限（手动勾选）**")
    c1, c2 = st.columns(2)
    has_margin = c1.checkbox("已开通两融账户")
    has_option = c2.checkbox("已开通ETF期权账户")
    submitted = st.form_submit_button("开始分析")

if submitted:
    try:
        stock_df = add_indicators(fetch_stock_daily(code))
        index_df = fetch_index_daily()
        rt = fetch_realtime(code)
        price = rt.get("price", stock_df["close"].iloc[-1])
        name = rt.get("name", code)
        beta = calc_beta(stock_df, index_df)

        # 300ETF 现价（用于名义价值，示例用固定值，可替换为实时接口）
        etf_price = 4.0

        market_value = price * qty
        pnl = (price - buy_price) * qty

        trend = run_trend_engine(stock_df, index_df, pe_pct=None)
        hedge = match_hedge(trend["rating"], market_value, qty, beta,
                            etf_price, has_margin, has_option)

        # ---- 输出报告 ----
        st.subheader("【持仓概览】")
        st.write(f"**{name}（{code}）** 现价 {price:.2f} | 成本 {buy_price:.2f} | "
                 f"{qty}股 | 市值 {market_value:.0f}元 | "
                 f"浮盈亏 {pnl:+.0f}元")

        st.subheader(f"【趋势判断】{trend['rating']}（得分 {trend['total']}）")
        st.json(trend["scores"])
        st.write(f"支撑 {trend['support']} | 压力 {trend['resistance']} | "
                 f"多空分界 {trend['pivot']} | Beta {beta:.2f}")

        st.subheader("【工具匹配】")
        st.info(f"**{hedge['tool']}** — {hedge['detail']}")
        st.write(f"建议数量：**{hedge['qty']}**")
        st.write(f"成本提示：{hedge['cost_note']}")

        st.subheader("【手动执行提示】")
        st.warning("请在券商App中手动完成上述操作；本工具不发送任何委托。")

    except Exception as e:
        st.error(f"数据获取或计算失败：{e}")
        

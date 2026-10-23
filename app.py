import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
from requests.exceptions import ConnectionError, Timeout

# ============================================================
# 数据层（新浪财经数据源）
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
    """
    获取ETF实时价格，用于行业ETF融券和300ETF期权相关计算。
    失败时返回空字典，调用方需处理降级。
    """
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
# 行业识别与期权行权价档位
# ============================================================

INDUSTRY_KEYWORDS = {
    "证券": ("512000", "券商ETF华宝"),
    "券商": ("512000", "券商ETF华宝"),
    "银行": ("512800", "银行ETF"),
    "半导体": ("512480", "半导体ETF"),
    "芯片": ("512480", "半导体ETF"),
    "医药": ("512010", "医药ETF"),
    "医疗": ("512170", "医疗ETF"),
    "消费": ("159928", "消费ETF"),
    "白酒": ("512690", "酒ETF"),
    "新能源": ("516160", "新能源ETF"),
    "光伏": ("515790", "光伏ETF"),
    "军工": ("512660", "军工ETF"),
    "地产": ("512200", "房地产ETF"),
    "煤炭": ("515220", "煤炭ETF"),
    "有色": ("512400", "有色金属ETF"),
}


def detect_industry_etf(stock_name: str) -> tuple | None:
    for kw, etf in INDUSTRY_KEYWORDS.items():
        if kw in stock_name:
            return etf
    return None


def get_strike_interval(etf_price: float) -> float:
    """
    300ETF期权行权价间距规则（上交所）：
    3元或以下 0.05；3至5元（含）0.1；5至10元（含）0.25；
    10至20元（含）0.5；20至50元（含）1；50至100元（含）2.5；100元以上5。
    """
    if etf_price <= 3:
        return 0.05
    if etf_price <= 5:
        return 0.1
    if etf_price <= 10:
        return 0.25
    if etf_price <= 20:
        return 0.5
    if etf_price <= 50:
        return 1.0
    if etf_price <= 100:
        return 2.5
    return 5.0


def round_to_strike(target: float, interval: float) -> float:
    """把目标行权价四舍五入到最近的合法档位"""
    return round(target / interval) * interval


def option_month_label() -> str:
    """返回近月合约的月份标签，如 '11月'。简化处理：取当前月或下月。"""
    today = datetime.today()
    # 当月第四个星期三之前，用当月；之后用下月。这里做简化：统一用下月，避免临近到期。
    next_month = today.month + 1 if today.month < 12 else 1
    return f"{next_month}月"


def format_option_name(opt_type: str, month: str, strike: float) -> str:
    """
    opt_type: '购' 或 '沽'
    行权价格乘以1000后的整数，如 4.00 -> 4000
    """
    strike_int = int(round(strike * 1000))
    return f"300ETF{opt_type}{month}{strike_int}"


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
    if pd.isna(m60):
        return 0
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
    if len(stock_df) < 21 or len(index_df) < 21:
        return 0
    s20 = stock_df["close"].iloc[-1] / stock_df["close"].iloc[-21] - 1
    i20 = index_df["close"].iloc[-1] / index_df["close"].iloc[-21] - 1
    diff = (s20 - i20) * 100
    if diff > 5: return 2
    if diff > 0: return 1
    if diff > -5: return -1
    return -2


def score_volume(df: pd.DataFrame) -> int:
    r = df.iloc[-1]
    if pd.isna(r["vol_ma5"]) or r["vol_ma5"] == 0:
        return 0
    ratio = r["volume"] / r["vol_ma5"]
    if ratio > 1.2 and r["pct_chg"] > 0: return 1
    if ratio < 0.8 and r["pct_chg"] < 0: return -1
    return 0


def score_valuation(pe_pct: float | None) -> int:
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
# 工具匹配（返回 options 列表，支持并列展示）
# ============================================================

def match_hedge(rating: str, market_value: float, qty: int,
                beta: float,
                has_margin: bool, has_option: bool,
                stock_name: str = "",
                etf_300_price: float | None = None) -> dict:
    """
    返回 {"options": [{"tool":..., "detail":..., "cost_note":...}, ...]}
    多条 option 并列展示。

    etf_300_price: 300ETF（510300）实时价格，用于期权合约和等效张数。
                   为 None 时降级为参考值 4.0，并在 detail 中注明。
    """
    # ---- 300ETF 价格与合约基础参数 ----
    etf_price_used = etf_300_price if etf_300_price else 4.0
    price_source_note = "" if etf_300_price else "（300ETF价格获取失败，按参考价4.0估算）"
    contract_notional = etf_price_used * 10000
    equiv = market_value * beta / contract_notional if contract_notional else 0
    equiv_int = int(equiv)

    # 行权价档位与月份标签
    interval = get_strike_interval(etf_price_used)
    month = option_month_label()

    # 虚值认沽：现价下方 3%–5%，取更靠下的档位
    put_target = etf_price_used * 0.95
    put_strike = round_to_strike(put_target, interval)
    put_name = format_option_name("沽", month, put_strike)

    # 虚值认购：现价上方 3%–5%，取更靠上的档位
    call_target = etf_price_used * 1.05
    call_strike = round_to_strike(call_target, interval)
    call_name = format_option_name("购", month, call_strike)

    # 规模过小
    if market_value < 2000:
        return {"options": [{
            "tool": "仓位管理",
            "detail": f"市值 {market_value:.0f} 元低于衍生品有效门槛，建议止损/减仓",
            "cost_note": "无额外成本",
        }]}

    # ---------- 偏空 / 中性偏空 ----------
    if rating in ("偏空", "中性偏空"):
        reduce_ratio = 0.5 if rating == "中性偏空" else 0.7
        reduce_qty = int(qty * reduce_ratio // 100) * 100

        if has_margin and qty >= 200:
            hedge_ratio = 0.5 if rating == "中性偏空" else 0.7
            short_qty = int(qty * hedge_ratio // 100) * 100
            short_qty = max(100, min(short_qty, qty - 100))
            return {"options": [{
                "tool": "减仓（首选）或 融券部分对冲（备选）",
                "detail": (
                    f"首选直接减仓约 {reduce_ratio*100:.0f}%（卖出约 {reduce_qty} 股），无利息成本；"
                    f"若必须保留底仓，可融券卖出约 {short_qty} 股"
                    f"（对冲比例约 {short_qty/qty*100:.0f}%，保留 {qty - short_qty} 股多头），"
                    f"需承担年化约 8% 的融券利息"
                ),
                "cost_note": "减仓无额外成本；融券按日计息，年化约 8%",
            }]}

        return {"options": [{
            "tool": "减仓",
            "detail": f"建议卖出约 {reduce_qty} 股（减仓约 {reduce_ratio*100:.0f}%），无利息成本",
            "cost_note": "无额外成本",
        }]}

    # ---------- 震荡：行业ETF融券 与 领口策略 并列 ----------
    if rating == "震荡":
        options = []

        # 方案A：行业 ETF 融券
        industry = detect_industry_etf(stock_name)
        if industry and has_margin and market_value >= 5000:
            etf_code, etf_name = industry
            etf_rt = fetch_etf_realtime(etf_code)
            etf_px = etf_rt.get("price")
            if etf_px:
                # 行业Beta经验估计：券商行业取1.3，其他行业取1.2
                industry_beta = 1.3 if etf_code == "512000" else 1.2
                raw_shares = market_value * industry_beta / etf_px
                short_shares = int(raw_shares // 100) * 100
                short_shares = max(100, short_shares)
                options.append({
                    "tool": "融券做空行业 ETF",
                    "detail": (
                        f"{etf_name}（{etf_code}）实时参考价 **{etf_px:.3f} 元**。"
                        f"按持仓市值 × 行业Beta估计（{industry_beta}）÷ ETF价格，"
                        f"参考融券份数约 **{short_shares} 份**（100份整数倍，仅供参考，"
                        f"可根据实际风险敞口调整）。"
                        f"行业 ETF 空头用于对冲板块层面的系统性回调，"
                        f"但无法覆盖 {stock_name} 自身的独立利空"
                    ),
                    "cost_note": (
                        "融券年化利息约 8%，"
                        "**务必在券商APP“融券标的”中确认实时可融余量后再下单**"
                    ),
                })
            else:
                options.append({
                    "tool": "融券做空行业 ETF（价格获取失败）",
                    "detail": (
                        f"检测到对应行业ETF：{etf_name}（{etf_code}），"
                        f"但实时价格获取失败，无法计算参考份数。"
                        f"请手动在券商APP查看该ETF价格和可融余量"
                    ),
                    "cost_note": "融券利息约 8%，需手动确认券源",
                })

        # 方案B：领口策略
        if has_option and equiv_int >= 1:
            options.append({
                "tool": "领口策略（买入认沽 + 卖出认购）",
                "detail": (
                    f"300ETF参考价 **{etf_price_used:.3f} 元**{price_source_note}，"
                    f"等效约 {equiv:.2f} 张，向下取整 **{equiv_int} 张**。"
                    f"参考合约：**买入 {put_name}**（虚值认沽，行权价 {put_strike:.1f}）"
                    f" + **卖出 {call_name}**（虚值认购，行权价 {call_strike:.1f}）。"
                    f"买沽提供下行保护，卖购收取权利金以抵消买沽成本，"
                    f"净权利金成本较低甚至可能接近于零，代价是上方收益被封顶。"
                    f"**权利金精确报价请以券商APP期权T型报价为准**"
                ),
                "cost_note": "净权利金成本较低，上方收益受限；精确权利金需手动查看",
            })

        # 降级：没有行业ETF也没有期权时
        if not options and has_margin and qty >= 300:
            options.append({
                "tool": "融券保守对冲",
                "detail": "对冲约 1/3 仓位",
                "cost_note": "融券利息",
            })

        if not options:
            options.append({
                "tool": "持有观望",
                "detail": "设置止损提醒，不建对冲仓",
                "cost_note": "无",
            })

        return {"options": options}

    # ---------- 中性偏多：备兑 与 保护性认沽 并列 ----------
    if rating == "中性偏多":
        options = []

        if has_option and equiv_int >= 1:
            options.append({
                "tool": "备兑开仓（卖出虚值认购）",
                "detail": (
                    f"300ETF参考价 **{etf_price_used:.3f} 元**{price_source_note}，"
                    f"等效约 {equiv:.2f} 张，向下取整 **{equiv_int} 张**。"
                    f"参考合约：**卖出 {call_name}**（虚值认购，行权价 {call_strike:.1f}）。"
                    f"卖出认购收取权利金增厚收益；需持有对应 300ETF 现货作为担保，"
                    f"无需额外现金保证金。代价是若涨幅超过行权价，上方收益被封顶。"
                    f"**权利金精确报价请以券商APP期权T型报价为准**"
                ),
                "cost_note": "收取权利金，零现金保证金（需有对应 300ETF 现货）",
            })
            options.append({
                "tool": "保护性认沽（买入虚值认沽）",
                "detail": (
                    f"300ETF参考价 **{etf_price_used:.3f} 元**{price_source_note}，"
                    f"等效约 {equiv:.2f} 张，向下取整 **{equiv_int} 张**。"
                    f"参考合约：**买入 {put_name}**（虚值认沽，行权价 {put_strike:.1f}）。"
                    f"支付权利金锁定下行底线，保留上方收益空间，"
                    f"适合“短期防跌、中期仍看好”的判断。"
                    f"**权利金精确报价请以券商APP期权T型报价为准**"
                ),
                "cost_note": "权利金为最大损失；精确权利金需手动查看",
            })

        if not options:
            options.append({
                "tool": "无需对冲",
                "detail": "设置移动止盈/止损提醒即可",
                "cost_note": "无",
            })

        return {"options": options}

    # ---------- 偏多 ----------
    return {"options": [{
        "tool": "无需对冲",
        "detail": "趋势偏多，设置移动止盈提醒即可",
        "cost_note": "无",
    }]}


# ============================================================
# 主界面
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

            # 300ETF 实时价格（用于期权合约和等效张数）
            etf_300 = fetch_etf_realtime("510300")
            etf_300_price = etf_300.get("price")

            market_value = price * qty
            pnl = (price - buy_price) * qty

            trend = run_trend_engine(stock_df, index_df, pe_pct=None)
            hedge = match_hedge(
                trend["rating"], market_value, qty, beta,
                has_margin, has_option,
                stock_name=name,
                etf_300_price=etf_300_price,
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

            # ---- 工具匹配：并列展示 ----
            st.subheader("【工具匹配】")
            opts = hedge.get("options", [])
            if len(opts) == 1:
                st.markdown(f"**{opts[0]['tool']}**")
                st.info(opts[0]["detail"])
                st.write(f"成本提示：{opts[0]['cost_note']}")
            else:
                for i, opt in enumerate(opts, 1):
                    st.markdown(f"**方案 {i}：{opt['tool']}**")
                    st.info(opt["detail"])
                    st.write(f"成本提示：{opt['cost_note']}")
                    if i < len(opts):
                        st.divider()

            st.subheader("【手动执行提示】")
            st.warning(
                "请在券商App中手动完成上述操作；本工具不发送任何委托。"
                "期权权利金的精确报价、行业ETF的实时可融余量，均需在券商App中最终确认。"
            )

        except Exception as e:
            st.error(f"数据获取或计算失败：{e}")

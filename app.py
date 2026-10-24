options.append({
    "tool": "融券做空行业 ETF",
    "detail": (
        f"用于对冲 {stock_name} 的板块层面系统性回调，"
        f"但无法覆盖个股自身的独立利空。"
    ),
    "cost_note": (
        "融券年化利息估算约 8%（行业 ETF 券源稀缺时可能更高，"
        "以券商 APP 实时报价为准）"
    ),
    "etf_detail": {
        "name": etf_name,
        "code": etf_code,
        "price": etf_px,
        "short_shares": short_shares,
        "industry_beta": industry_beta,
    },
})

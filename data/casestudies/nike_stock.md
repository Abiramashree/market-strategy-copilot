# Case study: Nike — stock performance diagnosis (#ComebackNike)

Domain: Equity / BI analysis. Source project: Nike_stock_analysis.

## Business situation
Nike's stock fell and underperformed. Goal: diagnose *why* the stock is weak, test
whether the **CEO transition** caused it, find other drivers, and give an action plan a
BI analyst could hand to leadership.

## Method
- Technical analysis on price history: moving averages (MA10 vs MA20 crossovers as
  bullish/bearish signals), volatility, RSI, and a correlation matrix across peer stocks.
- Fundamental comparison vs Adidas: P/E ratio and EPS.
- Event study around the earnings report and the CEO transition.
- Plotly for interactive charts; reported in Tableau + a Medium case study.

## Key findings
- **Sharp decline after the earnings report** — the market reacted unfavourably
  (weaker sales / softer outlook → lower investor confidence).
- **CEO transition drew a neutral reaction**; the stock ticked up slightly but did not
  stabilise. The transition is **not** the primary cause of weakness.
- **RSI:** oversold (<30) in mid-July (excessive selling); overbought (>70) in
  mid-September, then rolling over — losing bullish momentum. High volatility throughout.
- **Valuation:** Nike has a **higher P/E than Adidas but lower EPS** — investors are
  pricing in future growth over current profitability, which makes the stock sensitive to
  short-term earnings disappointments.

## Strategy implications (action plan)
1. **Protect margins now:** cost-cutting and supply-chain efficiency to improve margins
   and reduce sensitivity to earnings misses.
2. **Double down on DTC & digital transformation** (e.g. SNKRS, direct channels) to
   defend the growth narrative the high P/E depends on.
3. **Lead with innovation** to justify the premium valuation and rebuild investor
   confidence; manage guidance carefully around earnings.

# Shareholder Yield Factor Index — Methodology Notes

## Does the Intuition Hold?

Yes, with some nuance. The core idea combines two complementary lenses:

**Level signals** (dividend yield, shareholder yield) capture companies that are *currently* returning a lot of capital relative to their price. These tend to tilt toward mature, value-oriented names.

**Growth signals** (dividend yield growth, shareholder yield growth) capture companies where the *rate* of capital return is accelerating. This is where you get something more interesting — it can pick up companies that are transitioning from growth to capital return mode, or management teams that are becoming more shareholder-friendly over time.

The combination is powerful because each signal alone has a well-known weakness. Pure high-yield screens often become value traps — the yield is high because the price has collapsed and the dividend is about to get cut. By also requiring *growing* yields, you're implicitly filtering for companies where the yield is high for the right reasons (management is choosing to return more, not just the denominator shrinking).

One tension to be aware of: dividend yield growth and shareholder yield growth can be mechanically inflated by a falling stock price, since yield = payout / price. A stock drops 30% and suddenly its yield "grew" even though the company didn't change its payout at all. You'll want to think carefully about whether you're measuring yield growth or *payout growth* — they have different economic meanings.

---

## Methodology Framework

### 1. Define Your Signals Precisely

You have four raw signals. For each one, you need an exact definition:

**Dividend Yield (DY):**

$$DY_i = \frac{D_i}{P_i}$$

where $D_i$ is the trailing twelve-month dividends per share for stock $i$, and $P_i$ is the current price. Straightforward, but decide whether you want trailing twelve months (TTM) or forward indicated dividend (the annualized most-recent quarterly dividend). TTM is more stable; forward indicated is more responsive.

**Shareholder Yield (SY):** This is the bigger design decision. The classic definition from Mebane Faber is:

$$SY_i = \frac{D_i + B_i - S_i}{M_i}$$

where $D_i$ is total dividends paid, $B_i$ is net share buybacks (shares repurchased minus shares issued), $S_i$ is net debt paydown (some formulations include this, some don't), and $M_i$ is market capitalization. You need to decide whether you include debt reduction or keep it as just dividends plus net buybacks. Including debt paydown makes it a broader "total capital allocation" signal; excluding it keeps it focused on direct equity holder returns.

**Growth signals:** For dividend yield growth and shareholder yield growth, you need to define the lookback window. A common approach:

$$g_{DY,i} = \frac{DY_{i,t} - DY_{i,t-12}}{DY_{i,t-12}}$$

where $t$ is the current month and $t-12$ is twelve months ago. But consider measuring *payout growth* instead:

$$g_{D,i} = \frac{D_{i,t} - D_{i,t-12}}{D_{i,t-12}}$$

This isolates the management decision to increase dividends from the noise of price movements. Same logic applies to shareholder yield — you could measure growth in the numerator (total capital returned) rather than growth in the yield ratio itself.

### 2. Cross-Sectional Scoring

Once you have four raw signals for each stock in the Russell 1000, you need to make them comparable. The standard approach is to z-score each signal cross-sectionally:

$$z_{i,k} = \frac{x_{i,k} - \bar{x}_k}{\sigma_k}$$

where $x_{i,k}$ is the raw signal value for stock $i$ on factor $k$, $\bar{x}_k$ is the cross-sectional mean, and $\sigma_k$ is the cross-sectional standard deviation. This puts all four signals on the same scale. You'll also want to winsorize the raw values (typically at the 2.5th and 97.5th percentiles) before z-scoring to prevent outliers from dominating.

### 3. Composite Score and Weighting Scheme

Combine the four z-scores into a single composite:

$$C_i = w_1 \cdot z_{i,DY} + w_2 \cdot z_{i,SY} + w_3 \cdot z_{i,gDY} + w_4 \cdot z_{i,gSY}$$

where $w_1, w_2, w_3, w_4$ are the weights you assign to each signal and $C_i$ is the composite score for stock $i$. A natural starting point is equal weight ($w_k = 0.25$), but you might want to tilt more toward shareholder yield since it subsumes dividend yield — they're highly correlated, so equal-weighting effectively double-counts dividends. Something like 20/30/20/30 (less weight on dividend-specific signals, more on shareholder yield signals) could make sense, or you could just use two signals: shareholder yield level and shareholder yield growth.

### 4. Index Weight Construction

This is where index methodology gets interesting. You have several choices:

**Tilt-based** (most common for factor indices): Start from market-cap weights and tilt proportional to the composite score:

$$w_i^{index} = w_i^{mktcap} \cdot f(C_i)$$

where $f(C_i)$ is a tilting function — often just $f(C_i) = \max(C_i, 0)$ or $f(C_i) = e^{\lambda C_i}$ for some scaling parameter $\lambda$, and then you renormalize so weights sum to 1. This keeps you close to the market-cap benchmark and limits tracking error.

**Score-weighted**: Weight purely by composite score (after flooring at zero to exclude negative-scoring stocks). More aggressive factor exposure but higher tracking error and potentially concentrated.

**Stratified selection**: Select the top N stocks by composite score (say top 200), then weight them by market cap within that subset. This is what a lot of commercial factor indices do — it's simple, transparent, and limits turnover.

### 5. Key Practical Considerations

**Turnover and rebalancing frequency.** Growth signals are noisier than level signals, so they'll generate more turnover. Quarterly rebalancing is standard for factor indices. You might also want buffer rules — don't drop a stock from the index just because it fell from rank 200 to rank 210, only drop it if it falls below rank 250 or similar. This dramatically reduces unnecessary trading.

**Sector concentration.** Dividend-heavy strategies naturally overweight utilities, REITs, staples, and financials. You need to decide whether to let this happen or impose sector caps (e.g., no sector can be more than 2x its Russell 1000 weight). Unconstrained gives you purer factor exposure; constrained reduces the risk that you're really just running a sector bet.

**Non-payers.** A large chunk of the Russell 1000 pays no dividend at all. These stocks will score zero or negative on your dividend signals. Decide whether they're eligible — shareholder yield can still be positive for non-dividend payers if they're doing buybacks, which gives you a natural way to include them through the SY signal.

**Yield traps.** Beyond the price-decline issue mentioned above, watch for companies with unsustainably high payout ratios. A stock yielding 8% with a payout ratio above 100% of earnings is probably about to cut. You could add a quality filter (e.g., require payout ratio below some threshold, or positive free cash flow) as a screen before scoring.

**Signal correlation.** Dividend yield and shareholder yield are going to be quite correlated — maybe 0.5 to 0.7 cross-sectionally. Their growth rates will be less correlated but still related. Run the correlation matrix on your signals before finalizing weights. If two signals are very highly correlated, you're not getting much diversification benefit from including both, and you might simplify.

**Backtest considerations.** Use point-in-time data to avoid look-ahead bias (especially for buyback data, which comes from cash flow statements that are reported with a lag). Russell 1000 membership itself changes annually in June — make sure you're using historical constituents, not today's list applied backwards.

---

## Python Libraries for the Methodology

### Core Data Manipulation

**pandas** is the backbone — everything from signal construction to cross-sectional scoring to weight calculation lives naturally in DataFrames. For an index like this, you'll be working heavily with MultiIndex structures (date × stock) and groupby operations for cross-sectional z-scoring. **NumPy** handles the underlying math — vectorized operations for winsorization, composite scoring, and weight normalization will be much faster than looping.

### Signal Construction and Factor Analysis

**alphalens-reloaded** (the maintained fork of Quantopian's alphalens) is purpose-built for exactly this kind of work. You feed it a factor signal and forward returns, and it gives you quantile return analysis, information coefficient time series, turnover statistics, and sector-level breakdowns. It's the fastest way to answer "does my composite score actually predict returns, and how much turnover does it generate?"

**scipy.stats** is useful for the z-scoring and winsorization step. `scipy.stats.zscore` works but a manual implementation with pandas groupby gives you more control — especially when you want to winsorize first, then z-score, all within each rebalancing date cross-section. `scipy.stats.mstats.winsorize` handles the clipping.

### Portfolio Construction and Weighting

**numpy** alone handles the simpler weighting schemes (tilt-based, score-weighted). If you want to get into constrained optimization — like maximizing factor exposure subject to sector caps, turnover limits, and position size bounds — **cvxpy** is the right tool. It lets you express the index construction as a convex optimization problem:

$$\max_{w} \quad w^T C \quad \text{subject to} \quad \sum w_i = 1, \quad w_i \geq 0, \quad \text{sector constraints, etc.}$$

where $w$ is your weight vector and $C$ is the vector of composite scores. cvxpy's syntax maps almost directly to that mathematical formulation, which makes it very readable.

### Backtesting and Performance Analysis

**vectorbt** is probably the best fit here over zipline-reloaded. For an index methodology backtest, you're not simulating order-by-order execution — you're rebalancing to target weights at fixed intervals and measuring performance. vectorbt is built for this vectorized, portfolio-level simulation pattern and is dramatically faster than event-driven frameworks.

**pyfolio** (or **pyfolio-reloaded**) layers on top for the tearsheet-style analysis — rolling Sharpe, drawdown periods, exposure analysis, and monthly return heatmaps.

**quantstats** is a lighter alternative to pyfolio that generates similar performance reports with less setup. It's particularly good for quick comparisons against a benchmark like the Russell 1000 itself.

### Risk and Attribution

**statsmodels** handles the regression-based analysis you'd want for understanding factor exposures. If you want to decompose your index returns against known factors (market, value, size, momentum, quality), OLS regression with statsmodels tells you whether your index is delivering unique exposure or just repackaging known premiums.

### Suggested Stack Summary

The core working stack: pandas and numpy for everything structural, scipy for statistical transforms, alphalens-reloaded for signal evaluation, cvxpy if you go the constrained optimization route for weighting, vectorbt for backtesting the index, and quantstats or pyfolio for performance reporting.

---

## Transaction Costs in Index Backtests

Index prices are purely theoretical. An index like the Russell 1000 is just a mathematical calculation: take the constituent weights, multiply by their prices, sum it up. There are no brokerage commissions, no bid-ask spreads, no market impact costs, and no slippage baked into the number.

### How to Handle This

**Approach 1 (simpler):** Estimate transaction costs for your factor index and leave the Russell 1000 as-is. The logic is that if an investor wanted passive Russell 1000 exposure, they'd buy an ETF like IWB, and the tracking error and expense ratio of that ETF are tiny (around 15 basis points annually). So the frictionless index is a close enough proxy for what a real passive investor actually earns.

**Approach 2 (more rigorous):** Also haircut the Russell 1000 returns by roughly 15–20 bps per year to reflect the ETF expense ratio and tracking costs.

### Transaction Cost Model

$$TC_{annual} = \tau \cdot c$$

where $\tau$ is the annual one-way turnover (the fraction of the portfolio you trade per year) and $c$ is the estimated cost per dollar traded. For large-cap US equities in the Russell 1000, a reasonable estimate for $c$ is around 10–20 basis points per side, which includes:

- Bid-ask spread: typically 2–5 bps for large caps
- Market impact: 5–10 bps depending on position size
- Commissions: negligible for institutional investors

**Example:** 40% annual one-way turnover at 15 bps cost per side:

$$TC_{annual} = 0.40 \times 0.0015 = 0.0006 = 6 \text{ bps per year}$$

### Design Implication

Level signals (shareholder yield) tend to be slow-moving and generate low turnover. Growth signals are noisier — a company might have high payout growth one year and mediocre growth the next, causing it to cycle in and out of your index. This is where most of the turnover (and therefore transaction costs) will come from. Measure turnover explicitly during the backtest using alphalens-reloaded's turnover statistics.

---

## Python Script — Design Notes

The script uses the LSEG Data Library (`lseg-data`) and computes shareholder yield as:

$$SY_i = \frac{|D_i| + \max(|B_i| - |S_i|,\; 0)}{M_i}$$

where $D_i$ is dividends paid, $B_i$ is share repurchases, $S_i$ is share issuance, and $M_i$ is market cap — all from the cash flow statement. The $\max(\ldots, 0)$ floors net buybacks at zero so net issuers don't get negative credit. For the growth signal, it measures payout growth in the *numerator* (not the yield ratio) to avoid the price-denominator noise problem.

### Key Implementation Notes

- **Russell 1000 data is fee-liable.** The script tries three methods to pull constituents (TR.IndexConstituentRIC, Screener, chain RIC) because Russell index data requires a separate entitlement on LSEG.
- **TR field display names may vary.** The column names from `ld.get_data` depend on your entitlement. Run a small test call on 2–3 RICs, inspect the column names, and update accordingly.
- **Single-period demonstration.** The `main()` function builds weights for one point in time. For a full multi-period backtest, loop over quarterly rebalance dates, fetch point-in-time constituents and fundamentals for each, and build up `weights_history`.
- **The Config class is the control panel.** Signal weights, tilt strength, sector cap multiplier, transaction cost assumptions, and position limits are all in one place.

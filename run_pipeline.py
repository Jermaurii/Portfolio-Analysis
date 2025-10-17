# Tiny pipeline runner to generate synthetic data, run analysis, and create the HTML report
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import data as D
from src import model as M
from src import viz as V
from src import report as R
from src.config import Paths

def main():
    root = Path(__file__).resolve().parent
    paths = Paths.from_root(str(root))
    # Generate synthetic data
    factors = D.generate_synthetic_factors(120)
    portfolio = D.generate_synthetic_portfolio(factors)
    D.save_csv(factors, paths.data / 'factors_synthetic.csv')
    D.save_csv(portfolio, paths.data / 'portfolio_returns_synthetic.csv')
    print('Wrote synthetic CSVs to', paths.data)

    # Fit model
    alpha, betas, r2, y_hat = M.fit_ols_excess(
        portfolio=portfolio['Portfolio'], rf=portfolio['RF'], factors=factors
    )
    contrib_pct = M.attribution_percent(factors, alpha, betas, (portfolio['Portfolio']-portfolio['RF']).to_numpy())
    betas_df = R.pd.DataFrame({'Beta': [betas[k] for k in betas]}, index=list(betas.keys()))

    # Charts
    betas_png = paths.charts / 'betas.png'
    attr_png = paths.charts / 'attribution.png'
    cum_png = paths.charts / 'cumulative.png'
    V.plot_betas(betas, betas_png)
    V.plot_attribution(contrib_pct, attr_png)
    V.plot_cumulative(portfolio['Portfolio'], portfolio['RF'], y_hat, portfolio.index, cum_png)

    # Rolling (optional)
    rolling_png = None
    roll = M.rolling_betas(portfolio['Portfolio'], portfolio['RF'], factors, window=36)
    if not roll.empty:
        rolling_png = paths.charts / 'rolling_betas_heatmap.png'
        V.plot_rolling_heatmap(roll, rolling_png)
        (paths.output / 'tables').mkdir(parents=True, exist_ok=True)
        roll.to_csv(paths.output / 'tables' / 'rolling_betas.csv')

    report_path = R.build_html_report(
        paths.output, alpha, betas_df, r2,
        (portfolio['Portfolio']-portfolio['RF']).mean(),
        betas_png, attr_png, cum_png, rolling_png
    )
    R.export_csvs(paths.output, alpha, betas_df, r2, (portfolio['Portfolio']-portfolio['RF']).mean())
    print('Report generated at', report_path)

if __name__ == '__main__':
    main()

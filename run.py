import cli

if __name__ == "__main__":
    cli.main()
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import src.data as D
import src.model as M
import src.viz as V
from config import Paths
from report import build_html_report, export_csvs
def main():
    parser = cli.build_parser()
    args = parser.parse_args()
    paths = Paths()

    if args.cmd == "generate":
        cmd_generate(paths, args.months)
    elif args.cmd == "run":
        cmd_run(paths, args.factors_csv, args.portfolio_csv, args.rolling_window)
    else:
        parser.print_help()
        exit(1)
def cmd_generate(paths: Paths, months: int):
    factors = D.generate_synthetic_factors(months)
    portfolio = D.generate_synthetic_portfolio(factors)
    D.save_csv(factors, paths.data / "factors_synthetic.csv")
    D.save_csv(portfolio, paths.data / "portfolio_returns_synthetic.csv")
    print(f"[ok] Wrote synthetic factors + portfolio to {paths.data}")
def cmd_run(paths: Paths, factors_csv: str, portfolio_csv: str, rolling_window: int | None):
    fac_path = paths.data / factors_csv
    port_path = paths.data / portfolio_csv
    factors = D.load_factors_csv(fac_path)
    portfolio = D.load_portfolio_csv(port_path)

    alpha, betas, r2, y_hat = M.fit_ols_excess(
        portfolio=portfolio["Portfolio"], rf=portfolio["RF"], factors=factors
    )
    contrib_pct = M.attribution_percent(factors, alpha, betas, (portfolio["Portfolio"]-portfolio["RF"]).to_numpy())

    betas_df = pd.DataFrame({"Beta": [betas[k] for k in betas]}, index=list(betas.keys()))

    # Charts
    betas_png = paths.charts / "betas.png"
    attr_png = paths.charts / "attribution.png"
    cum_png = paths.charts / "cumulative.png"
    V.plot_betas(betas, betas_png)
    V.plot_attribution(contrib_pct, attr_png)
    V.plot_cumulative(portfolio["Portfolio"], portfolio["RF"], y_hat, portfolio.index, cum_png)

    rolling_png = None
    if rolling_window:
        roll = M.rolling_betas(portfolio["Portfolio"], portfolio["RF"], factors, window=rolling_window)
        rolling_png = paths.charts / "rolling_betas_heatmap.png"
        V.plot_rolling_heatmap(roll, rolling_png)
        roll.to_csv(paths.output / "tables" / "rolling_betas.csv")

        roll_alpha = M.rolling_alpha(portfolio["Portfolio"], portfolio["RF"], factors, window=rolling_window)
        rolling_alpha_png = paths.charts / "rolling_alpha.png"
        V.plot_rolling_alphas(roll_alpha, rolling_alpha_png)

    report_path = build_html_report(
        paths.output, alpha, betas_df, r2,
        (portfolio["Portfolio"]-portfolio["RF"]).mean(),
        betas_png, attr_png, cum_png, rolling_png
    )
    export_csvs(paths.output, alpha, betas_df, r2, (portfolio["Portfolio"]-portfolio["RF"]).mean())
    print(f"[ok] Report -> {report_path}")
    print(f"[ok] Charts -> {paths.charts}")
    print(f"[ok] Tables -> {paths.output / 'tables'}")
def build_parser():
    p = argparse.ArgumentParser(prog="pfa", description="Portfolio Factor Analysis CLI")
    sub = p.add_subparsers(dest="cmd")

    g = sub.add_parser("generate", help="Generate synthetic demo data")
    g.add_argument(
        "--months", type=int, default=120,
        help="Number of months of synthetic data to generate (default: 120)"
    )

    r = sub.add_parser("run", help="Run factor analysis on portfolio + factors CSVs")
    r.add_argument(
        "--factors-csv", type=str, required=True,
        help="CSV file in data/ directory containing factor returns"
    )
    r.add_argument(
        "--portfolio-csv", type=str, required=True,
        help="CSV file in data/ directory containing portfolio returns"
    )
    r.add_argument(
        "--rolling-window", type=int, default=None,
        help="If set, compute rolling betas with this window size (in months)"
    )
    return p
    p.set_defaults(func=cmd_run)
    return p
def build_html_report(
    out_dir: Path,
    alpha: float,
    betas_df: pd.DataFrame,
    r2: float,
    mean_excess: float,
    betas_png: Path,
    attr_png: Path,
    cumulative_png: Path,
    rolling_png: Path | None = None
):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.html"
    rolling_html = ""
    if rolling_png:
        rolling_html = f"""
  <div class="card">
    <h2>Rolling Betas Heatmap</h2>
    <img src="data:image/png;base64,{_b64(rolling_png)}" alt="Rolling Betas Heatmap" />
  </div>
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Portfolio Factor Analysis Report</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 20px;
      background-color: #f9f9f9;
    }}
    h1 {{
      text-align: center;
    }}
    .card {{
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
      margin: 20px auto;
      padding: 20px;
      max-width: 800px;
    }}
    img {{
      max-width: 100%;
      height: auto;
      display: block;
      margin: 0 auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px;
      text-align: center;
    }}
    th {{
      background-color: #f2f2f2;
    }}
  </style>
</head>
<body>
    <h1>Portfolio Factor Analysis Report</h1>
    <div class="card">
        <h2>Summary Statistics</h2>
        <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Alpha (monthly)</td><td>{alpha:.6f}</td></tr>
        <tr><td>R-squared</td><td>{r2:.4f}</td></tr>
        <tr><td>Mean Excess Return (monthly)</td><td>{mean_excess:.6f}</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Estimated Factor Exposures (Betas)</h2>
        <img src="data:image/png;base64,{_b64(betas_png)}" alt="Betas" />
        <table>
        <tr><th>Factor</th><th>Beta</th></tr>
        {''.join(f'<tr><td>{idx}</td><td>{val:.6f}</td></tr>' for idx, val in betas_df.itertuples())}
        </table>
    </div>
    <div class="card">
        <h2>Attribution of Mean Excess Return</h2>
        <img src="data:image/png;base64,{_b64(attr_png)}" alt="Attribution" />
    </div>
    <div class="card">
        <h2>Cumulative Returns: Actual vs Model-Implied</h2>
        <img src="data:image/png;base64,{_b64(cumulative_png)}" alt="Cumulative Returns" />
    </div>
    {rolling_html}
</body>
</html>
    """
    report.write_text(html)
    return report
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
FACTOR_COLS = ["MKT_RF", "SMB", "HML", "MOM", "RMW", "CMA"]
def savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    plt.close()
def plot_betas(betas: Dict[str,float], out_path: Path):
    import matplotlib.pyplot as plt
    plt.figure()
    labels = list(betas.keys())
    vals = [betas[k] for k in labels]
    plt.bar(labels, vals)
    plt.title("Estimated Factor Exposures (Betas)")
    plt.xlabel("Factors")
    plt.ylabel("Beta")
    savefig(out_path)
def plot_attribution(contrib_pct: Dict[str,float], out_path: Path):
    import matplotlib.pyplot as plt
    plt.figure()
    labels = list(contrib_pct.keys())
    vals = [contrib_pct[k] for k in labels]
    plt.bar(labels, vals)
    plt.title("Attribution of Mean Excess Return (%)")
    plt.xlabel("Component")
    plt.ylabel("% of Mean Excess Return")
    savefig(out_path)
def plot_cumulative(portfolio: pd.Series, rf: pd.Series, y_hat: np.ndarray, index, out_path: Path):
    import matplotlib.pyplot as plt
    cum_port = (1 + portfolio).cumprod()
    model_total = pd.Series(1 + (y_hat + rf.values), index=index).cumprod()
    plt.figure()
    plt.plot(cum_port.index, cum_port.values, label="Portfolio (Actual)")
    plt.plot(model_total.index, model_total.values, label="Model-Implied")
    plt.legend()
    plt.title("Cumulative Returns: Actual vs Model-Implied")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    savefig(out_path)
def plot_rolling_heatmap(roll_betas: pd.DataFrame, out_path: Path):
    import matplotlib.pyplot as plt
    plt.figure()
    data = roll_betas.T  # factors x time
    plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.yticks(range(len(data.index)), data.index)
    plt.xticks(range(len(data.columns)), [d.strftime("%Y-%m") for d in data.columns], rotation=90)
    plt.title("Rolling 36-Month Betas (Heatmap)")
    plt.xlabel("Date")
    plt.ylabel("Factor")
    savefig(out_path)
def plot_rolling_alphas(roll_alpha: pd.Series, out_path: Path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(roll_alpha.index, roll_alpha.values, label="Rolling Alpha")
    plt.axhline(0, color="gray", linestyle="--")
    plt.title("Rolling Alpha Over Time")
    plt.xlabel("Date")
    plt.ylabel("Alpha")
    plt.legend()
    savefig(out_path)
    plt.close()
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Tuple
def load_factors_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df
def generate_synthetic_factors(months: int = 120, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=months, freq="M")
    data = {
        "MKT_RF": np.random.normal(0.005, 0.04, size=months),
        "SMB": np.random.normal(0.002, 0.03, size=months),
        "HML": np.random.normal(0.003, 0.025, size=months),
        "MOM": np.random.normal(0.004, 0.035, size=months),
        "RMW": np.random.normal(0.0025, 0.03, size=months),
        "CMA": np.random.normal(0.002, 0.03, size=months),
    }
    return pd.DataFrame(data, index=dates)
def load_portfolio_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    return df
def generate_synthetic_portfolio(factors: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    np.random.seed(seed)
    alpha = 0.0008  # 8 bps monthly alpha
    betas = {"MKT_RF":1.05,"SMB":0.25,"HML":-0.15,"MOM":0.40,"RMW":0.10,"CMA":-0.05}
    eps = np.random.normal(0.0, 0.01, len(factors))
    excess = alpha + sum(betas[k]*factors[k].values for k in betas) + eps
    total = excess + factors.get("RF", 0).values
    return pd.DataFrame({"Portfolio": total}, index=factors.index)
def align_data(portfolio: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    combined = portfolio.join(factors, how="inner")
    if combined.empty:
        raise ValueError("No overlapping dates between portfolio and factors.")
    return combined[["Portfolio"]], combined.drop(columns=["Portfolio"])
    return combined[["Portfolio"]], combined.drop(columns=["Portfolio"])
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
FACTOR_ORDER = ["MKT_RF","SMB","HML","MOM","RMW","CMA"]
def fit_ols_excess(portfolio: pd.Series, rf: pd.Series, factors: pd.DataFrame) -> Tuple[float, Dict[str,float], float, np.ndarray]:
    df = pd.concat([portfolio, rf, factors[FACTOR_ORDER]], axis=1, join="inner").dropna()
    y = (df.iloc[:,0] - df.iloc[:,1]).to_numpy()  # excess returns
    Xf = df[FACTOR_ORDER]
    X = np.column_stack([np.ones(len(Xf))] + [Xf[c].values for c in FACTOR_ORDER])

    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
    alpha = float(beta_hat[0])
    betas = {f: float(b) for f,b in zip(FACTOR_ORDER, beta_hat[1:])}

    y_hat = X @ beta_hat
    rss = float(np.sum((y - y_hat)**2))
    tss = float(np.sum((y - y.mean())**2))
    r2 = 1 - rss/tss if tss != 0 else 0.0
    return alpha, betas, r2, y_hat
def attribution_percent(factors: pd.DataFrame, alpha: float, betas: Dict[str,float], y: np.ndarray) -> Dict[str,float]:
    mean_y = float(y.mean())
    contrib = {k: float(betas[k] * factors[k].mean()) for k in betas}
    contrib["Alpha"] = alpha
    if mean_y == 0:
        return {k: 0.0 for k in contrib}
    return {k: (v/mean_y)*100.0 for k,v in contrib.items()}
def rolling_betas(portfolio: pd.Series, rf: pd.Series, factors: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    df = pd.concat([portfolio, rf, factors], axis=1, join="inner").dropna()
    y_all = (df.iloc[:,0] - df.iloc[:,1])
    X_all = df[FACTOR_ORDER]
    out = []
    for end in range(window, len(df)+1):
        Xw = X_all.iloc[end-window:end]
        yw = y_all.iloc[end-window:end].to_numpy()
        X = np.column_stack([np.ones(len(Xw))] + [Xw[c].values for c in FACTOR_ORDER])
        beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ yw)
        betas = dict(zip(FACTOR_ORDER, beta_hat[1:]))
        betas["date"] = Xw.index[-1]
        out.append(betas)
    roll = pd.DataFrame(out).set_index("date")
    return roll
def rolling_alpha(portfolio: pd.Series, rf: pd.Series, factors: pd.DataFrame, window: int = 36) -> pd.Series:
    df = pd.concat([portfolio, rf, factors], axis=1, join="inner").dropna()
    y_all = (df.iloc[:,0] - df.iloc[:,1])
    X_all = df[FACTOR_ORDER]
    alphas = []
    dates = []
    for end in range(window, len(df)+1):
        Xw = X_all.iloc[end-window:end]
        yw = y_all.iloc[end-window:end].to_numpy()
        X = np.column_stack([np.ones(len(Xw))] + [Xw[c].values for c in FACTOR_ORDER])
        beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ yw)
        alphas.append(beta_hat[0])
        dates.append(Xw.index[-1])
    return pd.Series(alphas, index=dates)
import base64

def _b64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode("utf-8")
def build_html_report(
    out_dir: Path,
    alpha: float,
    betas_df: pd.DataFrame,
    r2: float,
    mean_excess: float,
    betas_png: Path,
    attr_png: Path,
    cumulative_png: Path,
    rolling_png: Path | None = None
):
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.html"
    rolling_html = ""
    if rolling_png:
        rolling_html = f"""
  <div class="card">
    <h2>Rolling Betas Heatmap</h2>
    <img src="data:image/png;base64,{_b64(rolling_png)}" alt="Rolling Betas Heatmap" />
  </div>
        """
        html = f"""
    """
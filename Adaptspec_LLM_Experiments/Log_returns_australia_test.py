import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptspec import adaptspec, summarise

DATA_DIR  = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_260514\global_platform_data\share_price\Australia"
PLOT_DIR  = r"C:\Users\60848\OneDrive - Bain\Desktop\Genome_code_260514\global_platform_data\share_price\Australia\plots"
START  = pd.Timestamp("2007-01-01")   # clip all series to this
CUTOFF = pd.Timestamp("2007-01-01")   # must have data before this to be included

os.makedirs(PLOT_DIR, exist_ok=True)

changepoints = {}

files = [f for f in os.listdir(DATA_DIR) if f.endswith("_price.csv")]
print(f"Found {len(files)} files")

for fname in sorted(files):
    ticker = fname.replace("_price.csv", "")

    df = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["Date"])
    df = df.sort_values("Date").dropna(subset=["Price"])

    # must have data going back to at least 2015
    if df["Date"].min() > CUTOFF:
        print(f"SKIP {ticker} - starts {df['Date'].min().date()}")
        continue

    # clip to common window
    df = df[df["Date"] >= START].reset_index(drop=True)
    log_returns = np.diff(np.log(df["Price"].values))
    dates       = df["Date"].values[1:]   # one shorter after diff

    print(f"\nRunning {ticker} ({len(log_returns)} obs, {df['Date'].min().date()} to {df['Date'].max().date()})...")

    result  = adaptspec(log_returns, nexp_max=15, nbeta=6, niter=10000, nburn=5000, tmin=45, seed=42, verbose=True)
    summary = summarise(result)

    changepoints[ticker] = summary["modal_changepoints"]
    print(f"  modal nexp      : {summary['modal_nexp']}")
    print(f"  mean nexp       : {summary['mean_nexp']:.2f}")
    print(f"  nexp dist       : {summary['nexp_probs']}")
    print(f"  changepoints    : {changepoints[ticker]}")
    if changepoints[ticker]:
        for cp in changepoints[ticker]:
            print(f"    idx={cp}  date={pd.Timestamp(dates[cp]).date()}  P={summary['changepoint_proba'][cp]:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(dates, log_returns, lw=0.7, color="#2c3e50")
    for cp in changepoints[ticker]:
        ax1.axvline(pd.Timestamp(dates[cp]), color="red", lw=1.5, ls="--")
    ax1.set_title(f"{ticker}  |  modal nexp={summary['modal_nexp']}  mean nexp={summary['mean_nexp']:.2f}")
    ax1.set_ylabel("Log return")

    ax2.fill_between(dates, summary["changepoint_proba"], color="#e74c3c", alpha=0.6)
    ax2.axhline(0.05, color="grey", lw=0.8, ls=":")
    ax2.set_ylabel("P(changepoint)")
    ax2.set_xlabel("Date")

    plt.tight_layout()
    # plt.savefig(os.path.join(PLOT_DIR, f"{ticker}.png"), dpi=120, bbox_inches="tight")
    plt.show()
    print(f"  plot saved.")

print(f"\nDone. {len(changepoints)} tickers processed.")
"""
VALIDATION GATE  —  Step 1 of the edge-validation engine
=========================================================
Tests whether a trading signal is a REAL edge or just overfit noise.

The discipline this enforces:
  1. Split data into in-sample (first 70%) and out-of-sample (last 30%).
     A signal is only trustworthy if it survives on data it never "saw".
  2. Compare the signal's win rate against the BASE RATE (how often "UP"
     happens anyway). Beating 50% is meaningless if UP happens 55% of the
     time on its own — we want to beat random, not beat a coin.
  3. Use a simple statistical test so we don't get fooled by small samples.

Usage:
    python3 validate.py "rsi < 30"
    python3 validate.py "momentum_3m > 0.5"
    python3 validate.py "rsi < 30 and momentum_1m > 0"
    python3 validate.py "momentum_3m < -0.5" --predict DN

Options:
    --data PATH      CSV to load            (default: btc_15m_data_v63.csv)
    --predict UP/DN  what the signal bets   (default: UP)
    --split FRAC     in-sample fraction     (default: 0.70)
    --min N          min signals to trust   (default: 30)
"""

import argparse
import sys

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def normalize_outcome(series):
    """Turn the 'outcome' column into a clean boolean: True == price went UP.

    Accepts strings ('UP'/'DN'/'DOWN'/'U'/'D') or numbers (1/0, positive/neg).
    """
    def to_up(v):
        s = str(v).strip().upper()
        if s in ("UP", "U", "1", "1.0", "TRUE", "WIN", "YES"):
            return True
        if s in ("DN", "DOWN", "D", "0", "0.0", "FALSE", "LOSS", "NO"):
            return False
        # Fall back to numeric sign (e.g. a raw change value): >0 is UP.
        try:
            return float(s) > 0
        except ValueError:
            raise ValueError(f"Can't interpret outcome value: {v!r}")

    return series.map(to_up)


def proportion_z(p_signal, p_base, n):
    """One-sided z-score for 'signal win rate beats the base rate'.

    Under the null hypothesis the signal is no better than the base rate, so
    its wins are Bernoulli(p_base). This measures how many standard errors the
    observed win rate sits above that null. z >= 1.64 ≈ 95% confidence.
    """
    if n == 0 or p_base <= 0 or p_base >= 1:
        return 0.0
    se = np.sqrt(p_base * (1 - p_base) / n)
    if se == 0:
        return 0.0
    return (p_signal - p_base) / se


def evaluate(df, rule, predict_up):
    """Run the signal rule over a dataframe slice and return its stats."""
    # df.query lets the user write natural expressions ("rsi < 30 and ...")
    # straight from the command line, evaluated against the columns.
    fired = df.query(rule)

    n = len(fired)
    # base rate: how often our predicted side happens anyway, in THIS slice
    base_rate = (df["is_up"] == predict_up).mean() if len(df) else 0.0

    if n == 0:
        return {"n": 0, "win_rate": float("nan"), "base_rate": base_rate, "z": 0.0}

    # a "win" = the predicted side actually happened on rows where signal fired
    wins = (fired["is_up"] == predict_up).sum()
    win_rate = wins / n
    z = proportion_z(win_rate, base_rate, n)
    return {"n": n, "win_rate": win_rate, "base_rate": base_rate, "z": z}


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Validate a trading signal for real edge vs overfit.")
    ap.add_argument("rule", help='Signal rule, e.g. "rsi < 30" or "momentum_3m > 0.5"')
    ap.add_argument("--data", default="btc_15m_data_v63.csv", help="CSV file to load")
    ap.add_argument("--predict", default="UP", choices=["UP", "DN"], help="Direction the signal bets")
    ap.add_argument("--split", type=float, default=0.70, help="In-sample fraction (0-1)")
    ap.add_argument("--min", type=int, default=30, help="Min signal count to trust a slice")
    args = ap.parse_args()

    predict_up = args.predict == "UP"

    # ── Load ──────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"❌ Data file not found: {args.data}")
        sys.exit(1)

    if "outcome" not in df.columns:
        print(f"❌ CSV has no 'outcome' column. Columns found: {list(df.columns)}")
        sys.exit(1)

    df["is_up"] = normalize_outcome(df["outcome"])

    # ── Split (chronological — NEVER shuffle time series) ─────────────
    # The whole point is to test on the "future" the signal never saw, so we
    # keep the rows in file order and cut the tail off as out-of-sample.
    cut = int(len(df) * args.split)
    in_sample = df.iloc[:cut]
    out_sample = df.iloc[cut:]

    # ── Evaluate the rule on each slice + overall ─────────────────────
    try:
        full = evaluate(df, args.rule, predict_up)
        is_stats = evaluate(in_sample, args.rule, predict_up)
        oos_stats = evaluate(out_sample, args.rule, predict_up)
    except Exception as e:  # bad column name, syntax error in the rule, etc.
        print(f"❌ Couldn't evaluate rule {args.rule!r}: {e}")
        print(f"   Available columns: {[c for c in df.columns if c != 'is_up']}")
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────
    print("\n" + "═" * 64)
    print(f"  VALIDATION GATE  —  signal: {args.rule!r}  (betting {args.predict})")
    print("═" * 64)
    print(f"  Rows: {len(df):,}   |   in-sample: {len(in_sample):,}   "
          f"out-of-sample: {len(out_sample):,}")
    print(f"  Base rate ({args.predict} happens anyway): "
          f"{full['base_rate']*100:.1f}%  (full dataset)")
    print("─" * 64)
    print(f"  {'slice':<14}{'signals':>9}{'win rate':>11}{'base rate':>11}{'edge':>9}{'z':>7}")
    print("─" * 64)

    for label, s in [("IN-SAMPLE", is_stats), ("OUT-OF-SAMPLE", oos_stats), ("(full data)", full)]:
        if s["n"] == 0:
            print(f"  {label:<14}{0:>9}{'   — never fired':>30}")
            continue
        edge = (s["win_rate"] - s["base_rate"]) * 100
        print(f"  {label:<14}{s['n']:>9}{s['win_rate']*100:>10.1f}%"
              f"{s['base_rate']*100:>10.1f}%{edge:>+8.1f}%{s['z']:>7.2f}")
    print("─" * 64)

    # ── Verdict ───────────────────────────────────────────────────────
    # z >= 1.64 ≈ 95% confidence the win rate truly beats the base rate
    # (not just luck). We require BOTH a positive edge AND significance.
    Z_SIGNIFICANT = 1.64

    def beats(stats):
        return (stats["n"] >= args.min
                and stats["win_rate"] > stats["base_rate"]
                and stats["z"] >= Z_SIGNIFICANT)

    is_good = beats(is_stats)
    oos_good = beats(oos_stats)

    if oos_stats["n"] < args.min:
        verdict = "NO EDGE"
        why = (f"too few signals out-of-sample ({oos_stats['n']} < {args.min}) "
               f"— can't trust it.")
    elif oos_good:
        verdict = "✅ REAL EDGE"
        why = "beats the base rate out-of-sample with statistical significance."
    elif is_good and not oos_good:
        verdict = "⚠️  OVERFIT"
        why = "worked in-sample but FAILED out-of-sample — it was fitting noise."
    else:
        verdict = "❌ NO EDGE"
        why = "never convincingly beats the base rate. It's not better than random."

    print(f"\n  VERDICT:  {verdict}")
    print(f"            {why}\n")


if __name__ == "__main__":
    main()

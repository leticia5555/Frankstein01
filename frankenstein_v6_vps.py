"""
FRANKENSTEIN v6 — ML BRAIN PAPER TESTER
=========================================
Continuous ML confidence scanning every 5 seconds.
Shows confidence evolving throughout candle to find optimal entry.

Requires: whale_brain.pkl (trained ML model)

Usage:
    python3 frankenstein_v6_ml.py           # 5-min candles
    python3 frankenstein_v6_ml.py 15        # 15-min candles
"""

import warnings
warnings.filterwarnings("ignore")

import json, requests, sys, time, csv, os, pickle
import numpy as np
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

BET_SIZE = 10.0
CANDLE_MODE = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 5
CONFIDENCE_THRESHOLD = 0.65  # Only trade when ML says >65%
KILL_SWITCH = "--kill55" in sys.argv  # Skip entries > 55c
KILL_PRICE = 0.55

CANDLE_CONFIGS = {
    5:  {"duration": 300, "label": "5min"},
    15: {"duration": 900, "label": "15min"},
}

BINANCE_API = "https://api.binance.com/api/v3"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
SCAN_INTERVAL = 5  # Check ML every 5 seconds

# ═══════════════════════════════════════════════════════════════
# LOAD ML BRAIN
# ═══════════════════════════════════════════════════════════════

brain = None
for path in ["whale_brain.pkl", os.path.expanduser("~/Downloads/whale_brain.pkl")]:
    if os.path.exists(path):
        with open(path, "rb") as f:
            brain = pickle.load(f)
        print(f"  🧠 ML Brain loaded: {brain['n_samples']:,} candles, {len(brain['features'])} features")
        break

if not brain:
    print("  ❌ whale_brain.pkl not found!")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# BINANCE DATA FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_spot():
    try:
        r = requests.get(f"{BINANCE_API}/ticker/price", params={"symbol": "BTCUSDT"}, timeout=3)
        return float(r.json()["price"])
    except:
        return None

def get_klines(interval="1m", limit=120):
    """Fetch recent Binance klines for TA calculation."""
    try:
        r = requests.get(f"{BINANCE_API}/klines", 
            params={"symbol": "BTCUSDT", "interval": interval, "limit": limit}, timeout=5)
        return r.json()
    except:
        return []

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_macd(closes):
    if len(closes) < 26:
        return 0
    ema12 = np.mean(closes[-12:])
    ema26 = np.mean(closes[-26:])
    macd_line = ema12 - ema26
    signal = np.mean(closes[-9:]) - np.mean(closes[-18:])
    return macd_line - signal

def calc_bb_position(closes, period=20):
    if len(closes) < period:
        return 0.5
    sma = np.mean(closes[-period:])
    std = np.std(closes[-period:])
    if std == 0:
        return 0.5
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5

def calc_stochastic(highs, lows, closes, period=14):
    if len(closes) < period:
        return 50
    lowest = min(lows[-period:])
    highest = max(highs[-period:])
    if highest == lowest:
        return 50
    return (closes[-1] - lowest) / (highest - lowest) * 100

def build_features(spot_start, spot_now, klines):
    """Build feature dict for ML prediction from current market state."""
    
    spot_chg = spot_now - spot_start
    distance_pct = abs(spot_chg) / spot_start * 100 if spot_start > 0 else 0
    velocity = spot_chg / max(1, 1)  # Will be updated with actual elapsed time
    
    # Parse klines for TA
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]
    
    rsi_5m = calc_rsi(closes[-30:], 14)
    rsi_15m = calc_rsi(closes[-90:], 14) if len(closes) >= 90 else rsi_5m
    macd_hist = calc_macd(closes)
    bb_pos = calc_bb_position(closes)
    stoch_k = calc_stochastic(highs, lows, closes)
    
    momentum_short = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
    momentum_medium = (closes[-1] - closes[-15]) / closes[-15] * 100 if len(closes) >= 15 else 0
    
    if len(closes) >= 31:
        rets = np.diff(closes[-31:]) / np.array(closes[-31:-1])
        volatility = np.std(rets) * 100
    else:
        volatility = 0
    
    ema_9 = np.mean(closes[-9:]) if len(closes) >= 9 else closes[-1]
    ema_21 = np.mean(closes[-21:]) if len(closes) >= 21 else closes[-1]
    ema_cross = 1 if ema_9 > ema_21 else -1
    
    if len(volumes) >= 30 and sum(volumes[-30:]) > 0:
        vwap = sum(c * v for c, v in zip(closes[-30:], volumes[-30:])) / sum(volumes[-30:])
        vwap_distance = (spot_now - vwap) / spot_now * 100
    else:
        vwap_distance = 0
    
    trend_5m = 1 if len(closes) >= 5 and closes[-1] > closes[-5] else -1
    trend_15m = 1 if len(closes) >= 15 and closes[-1] > closes[-15] else -1
    trend_60m = 1 if len(closes) >= 60 and closes[-1] > closes[-60] else -1
    trend_alignment = trend_5m + trend_15m + trend_60m
    
    return {
        "spot_chg": spot_chg,
        "distance_pct": distance_pct,
        "velocity": velocity,
        "rsi_5m": rsi_5m,
        "rsi_15m": rsi_15m,
        "macd_histogram": macd_hist,
        "bb_position": bb_pos,
        "stoch_k": stoch_k,
        "momentum_short": momentum_short,
        "momentum_medium": momentum_medium,
        "volatility_5m": volatility,
        "ema_cross": ema_cross,
        "vwap_distance": vwap_distance,
        "trend_alignment": trend_alignment,
    }

def ml_predict(features):
    """Get ML prediction from brain."""
    X = np.array([[features[f] for f in brain["features"]]])
    X_scaled = brain["scaler"].transform(X)
    proba = brain["model"].predict_proba(X_scaled)[0]
    direction = "UP" if proba[1] > 0.5 else "DN"
    confidence = max(proba)
    return direction, confidence, proba[1], proba[0]


# ═══════════════════════════════════════════════════════════════
# POLYMARKET FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def get_gamma_prices(slug):
    try:
        r = requests.get(f"{GAMMA_API}/markets", params={"slug": slug}, timeout=5)
        data = r.json()
        if isinstance(data, list) and data:
            m = data[0]
            prices = json.loads(m.get("outcomePrices", "[]"))
            return {"up_price": float(prices[0]), "dn_price": float(prices[1])}
    except:
        pass
    return None

def get_clob_best_ask(token_id):
    """Get REAL best ask price from CLOB order book."""
    try:
        r = requests.get(f"{CLOB_API}/book?token_id={token_id}", timeout=2)
        data = r.json()
        asks = data.get("asks", [])
        if asks:
            prices = [float(a["price"]) for a in asks]
            return min(prices)
    except:
        pass
    return None

def get_clob_prices(up_token, dn_token):
    """Get both UP and DN best ask from CLOB."""
    up_ask = get_clob_best_ask(up_token) if up_token else None
    dn_ask = get_clob_best_ask(dn_token) if dn_token else None
    return up_ask, dn_ask

def find_btc_candle(target_ts):
    cfg = CANDLE_CONFIGS[CANDLE_MODE]
    duration = cfg["duration"]
    
    # v64's working method: events endpoint + btc-updown-5m slug
    slug = f"btc-updown-5m-{target_ts}"
    try:
        r = requests.get(f"{GAMMA_API}/events?slug={slug}", timeout=5)
        data = r.json()
        if isinstance(data, list) and data:
            for market in data[0].get("markets", [data[0]]):
                clob_ids = json.loads(market.get("clobTokenIds", "[]"))
                if len(clob_ids) >= 2:
                    return {
                        "slug": slug,
                        "title": market.get("question", slug),
                        "up_token": clob_ids[0],
                        "dn_token": clob_ids[1],
                    }
    except:
        pass
    
    # Fallback: try markets endpoint with different slug formats
    for fmt in [f"btc-5-minute-{target_ts}",
                f"will-the-price-of-btc-go-up-or-down-in-the-next-5-minutes-{target_ts}"]:
        try:
            r = requests.get(f"{GAMMA_API}/markets", params={"slug": fmt}, timeout=5)
            data = r.json()
            if isinstance(data, list) and data:
                m = data[0]
                tokens = json.loads(m.get("clobTokenIds", "[]"))
                if len(tokens) >= 2:
                    return {
                        "slug": fmt,
                        "title": m.get("question", fmt),
                        "up_token": tokens[0],
                        "dn_token": tokens[1],
                    }
        except:
            continue
    
    return None

def get_current_candle_ts():
    duration = CANDLE_CONFIGS[CANDLE_MODE]["duration"]
    now = int(time.time())
    return now - (now % duration)

def get_next_candle_ts():
    duration = CANDLE_CONFIGS[CANDLE_MODE]["duration"]
    return get_current_candle_ts() + duration


# ═══════════════════════════════════════════════════════════════
# SESSION TRACKING
# ═══════════════════════════════════════════════════════════════

session_trades = []
daily_pnl = 0.0
scan_log = []  # Store all ML scans for analysis

def log_scan(candle_num, elapsed, spot_chg, direction, confidence, up_prob):
    scan_log.append({
        "candle": candle_num,
        "elapsed_s": round(elapsed, 1),
        "spot_chg": round(spot_chg, 1),
        "direction": direction,
        "confidence": round(confidence, 4),
        "up_prob": round(up_prob, 4),
    })


# ═══════════════════════════════════════════════════════════════
# MAIN CANDLE LOOP
# ═══════════════════════════════════════════════════════════════

def trade_candle(candle_num):
    global daily_pnl
    
    cfg = CANDLE_CONFIGS[CANDLE_MODE]
    duration = cfg["duration"]
    
    # Timing
    now = time.time()
    current_ts = get_current_candle_ts()
    next_ts = get_next_candle_ts()
    
    elapsed = now - current_ts
    catch_window = 60 if duration <= 300 else 90
    
    if elapsed < catch_window:
        candle_start = current_ts
    else:
        candle_start = next_ts
        wait_time = next_ts - now
        if wait_time > 5:
            print(f"  ⏳ Next candle at {datetime.fromtimestamp(next_ts).strftime('%I:%M %p')} ({wait_time:.0f}s)...")
            time.sleep(wait_time - 3)
        while time.time() < candle_start:
            time.sleep(0.5)
    
    candle_end = candle_start + duration
    start_dt = datetime.fromtimestamp(candle_start)
    end_dt = datetime.fromtimestamp(candle_end)
    
    print(f"\n{'━'*70}")
    print(f"  CANDLE #{candle_num} | {start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')}")
    print(f"{'━'*70}")
    
    # Find market
    candle = find_btc_candle(candle_start)
    if candle:
        print(f"  Market: {candle['title']}")
    
    # Get start price and klines
    spot_start = get_spot()
    if not spot_start:
        print("  ❌ No spot price. Skipping...")
        return None
    
    klines = get_klines("1m", 120)  # 2 hours of 1-min klines
    if not klines:
        print("  ❌ No klines. Skipping...")
        return None
    
    print(f"  Start: ${spot_start:,.2f}")
    
    # ═══════════════════════════════════════════════════════════
    # CONTINUOUS ML SCANNING
    # ═══════════════════════════════════════════════════════════
    
    traded = False
    trade_dir = None
    trade_confidence = 0
    trade_elapsed = 0
    peak_confidence = 0
    peak_direction = None
    peak_elapsed = 0
    confidence_history = []
    
    # Header for scan display
    print(f"\n  {'Time':>5s}  {'Spot':>7s}  {'ML':>4s}  {'Conf':>5s}  {'Bar'}")
    print(f"  {'─'*55}")
    
    while True:
        now = time.time()
        elapsed = now - candle_start
        remaining = candle_end - now
        
        if remaining <= 2:
            break
        
        # Get current data
        spot = get_spot()
        if not spot:
            time.sleep(2)
            continue
        
        spot_chg = spot - spot_start
        
        # Refresh klines every 30s for updated TA
        if int(elapsed) % 30 < SCAN_INTERVAL:
            klines = get_klines("1m", 120)
        
        # Build features with current spot
        features = build_features(spot_start, spot, klines)
        features["velocity"] = spot_chg / max(elapsed, 1)  # Real velocity
        
        # ML prediction
        direction, confidence, up_prob, dn_prob = ml_predict(features)
        
        # Log scan
        log_scan(candle_num, elapsed, spot_chg, direction, confidence, up_prob)
        confidence_history.append((elapsed, direction, confidence, spot_chg))
        
        # Track peak confidence
        if confidence > peak_confidence:
            peak_confidence = confidence
            peak_direction = direction
            peak_elapsed = elapsed
        
        # Display scan
        bar_len = int(confidence * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        icon = "🟢" if direction == "UP" else "🔴"
        conf_icon = "🔥" if confidence >= 0.75 else "✅" if confidence >= 0.65 else "〰️" if confidence >= 0.55 else "⚪"
        
        elapsed_str = f"{elapsed:.0f}s"
        
        # Get CLOB prices for display
        up_ask_display = ""
        if candle and candle.get("up_token"):
            ua, da = get_clob_prices(candle.get("up_token"), candle.get("dn_token"))
            if ua and da:
                up_ask_display = f" UP:{ua*100:4.0f}c DN:{da*100:4.0f}c"
        
        print(f"  {elapsed_str:>5s}  {spot_chg:>+7.0f}  {icon}{direction}  {confidence*100:4.0f}%  {conf_icon} {bar}{up_ask_display}")
        
        # ═══ TRADE DECISION ═══
        # Trade when confidence crosses threshold AND has been building
        if not traded and elapsed >= 15 and confidence >= CONFIDENCE_THRESHOLD:
            # Check if confidence is still climbing (not a spike that's fading)
            if len(confidence_history) >= 3:
                recent_confs = [c[2] for c in confidence_history[-3:]]
                is_climbing = recent_confs[-1] >= recent_confs[-2] - 0.02  # Allow small dips
                
                if is_climbing:
                    # Pre-check CLOB price before committing
                    pre_entry = 0.50
                    if candle:
                        ua, da = get_clob_prices(candle.get("up_token"), candle.get("dn_token"))
                        pre_entry = ua if direction == "UP" and ua else (da if direction == "DN" and da else 0.50)
                    
                    # Kill switch: skip expensive entries
                    if KILL_SWITCH and pre_entry > KILL_PRICE:
                        print(f"\n  🔪 KILL SWITCH: {direction} ask={pre_entry*100:.1f}c > {KILL_PRICE*100:.0f}c — SKIPPED")
                        time.sleep(SCAN_INTERVAL)
                        continue
                    
                    traded = True
                    trade_dir = direction
                    trade_confidence = confidence
                    trade_elapsed = elapsed
                    
                    # Get REAL price from CLOB order book
                    entry_price = 0.50  # Default fallback
                    if candle:
                        up_ask, dn_ask = get_clob_prices(candle.get("up_token"), candle.get("dn_token"))
                        if direction == "UP" and up_ask:
                            entry_price = up_ask
                        elif direction == "DN" and dn_ask:
                            entry_price = dn_ask
                        
                        # Also show both sides for pair cost analysis
                        pair_cost = (up_ask or 0.50) + (dn_ask or 0.50)
                        other_price = dn_ask if direction == "UP" else up_ask
                    
                    print(f"\n  ★ PAPER TRADE: BUY {trade_dir} @ {entry_price*100:.1f}c "
                          f"| ML: {confidence*100:.0f}% | T={elapsed:.0f}s")
                    if candle and up_ask and dn_ask:
                        print(f"    📦 CLOB: UP={up_ask*100:.1f}c DN={dn_ask*100:.1f}c "
                              f"| Pair={pair_cost*100:.1f}c")
                    print(f"  {'─'*55}")
        
        # Check if confidence DROPS after trading (reversal warning)
        if traded and confidence < trade_confidence - 0.10:
            print(f"  ⚠️ CONFIDENCE DROPPED: {trade_confidence*100:.0f}% → {confidence*100:.0f}% "
                  f"(reversal risk!)")
        
        time.sleep(SCAN_INTERVAL)
    
    # ═══════════════════════════════════════════════════════════
    # RESOLVE
    # ═══════════════════════════════════════════════════════════
    
    final_spot = get_spot() or spot_start
    final_chg = final_spot - spot_start
    winner = "UP" if final_chg >= 0 else "DN"
    
    print(f"\n  ═══════════════════════════════════════")
    print(f"  RESULT: {'🟢' if winner == 'UP' else '🔴'} {winner} | BTC: {final_chg:+.0f}")
    print(f"  ═══════════════════════════════════════")
    
    # Analyze ML confidence pattern
    if confidence_history:
        # What was ML saying at different times?
        checkpoints = [10, 20, 30, 45, 60, 90, 120, 180, 240]
        print(f"\n  📊 ML CONFIDENCE TIMELINE:")
        for cp in checkpoints:
            matches = [c for c in confidence_history if abs(c[0] - cp) < SCAN_INTERVAL]
            if matches:
                _, d, c, sc = matches[0]
                correct = "✅" if d == winner else "❌"
                print(f"    T={cp:3d}s: {d} {c*100:4.0f}% (spot:{sc:+.0f}) {correct}")
            if cp > duration:
                break
        
        # Peak info
        print(f"\n  🏔️ Peak confidence: {peak_direction} {peak_confidence*100:.0f}% at T={peak_elapsed:.0f}s")
        correct_peak = "✅" if peak_direction == winner else "❌"
        print(f"     Peak was {'CORRECT' if peak_direction == winner else 'WRONG'} {correct_peak}")
    
    # Trade result
    if traded:
        if trade_dir == winner:
            profit = BET_SIZE * (1.0 - entry_price) / entry_price if entry_price > 0 else 0
            daily_pnl += profit
            session_trades.append({"result": "WIN", "pnl": profit, "conf": trade_confidence, 
                                   "elapsed": trade_elapsed, "dir": trade_dir})
            print(f"\n  ✅ WIN: {trade_dir} @ {entry_price*100:.0f}c → ${profit:+.2f}")
        else:
            daily_pnl -= BET_SIZE
            session_trades.append({"result": "LOSS", "pnl": -BET_SIZE, "conf": trade_confidence,
                                   "elapsed": trade_elapsed, "dir": trade_dir})
            print(f"\n  ❌ LOSS: {trade_dir} @ {entry_price*100:.0f}c → -${BET_SIZE:.2f}")
            
            # Was there a confidence drop warning?
            if confidence_history:
                post_trade = [c for c in confidence_history if c[0] > trade_elapsed]
                if post_trade:
                    min_conf_after = min(c[2] for c in post_trade)
                    if min_conf_after < trade_confidence - 0.10:
                        print(f"  💡 Confidence dropped to {min_conf_after*100:.0f}% after entry — reversal signal!")
    else:
        print(f"\n  ⚪ NO TRADE — ML never reached {CONFIDENCE_THRESHOLD*100:.0f}% confidence")
        
        # Show what would have happened
        if peak_confidence > 0.50:
            would_have = "WON" if peak_direction == winner else "LOST"
            print(f"  💭 Peak was {peak_direction} {peak_confidence*100:.0f}% — would have {would_have}")
    
    # Session summary
    wins = sum(1 for t in session_trades if t["result"] == "WIN")
    losses = sum(1 for t in session_trades if t["result"] == "LOSS")
    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    
    print(f"\n  ┌─────────────────────────────────────────────────┐")
    print(f"  │  Session: {wins}W/{losses}L ({wr:.0f}%) | P&L: ${daily_pnl:+.2f}")
    print(f"  │  Avg confidence at entry: {np.mean([t['conf'] for t in session_trades])*100:.0f}%" if session_trades else "  │  No trades yet")
    print(f"  │  Avg entry time: T={np.mean([t['elapsed'] for t in session_trades]):.0f}s" if session_trades else "  │")
    print(f"  └─────────────────────────────────────────────────┘")
    
    return trade_dir


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    spot = get_spot()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  🧠 FRANKENSTEIN v6 — ML BRAIN + REAL CLOB PRICES          ║
║                                                              ║
║  Model: {brain['n_samples']:,} candles | {len(brain['features'])} features                  ║
║  Strategy: Continuous ML scan every {SCAN_INTERVAL}s                    ║
║  Entry: Only when confidence > {CONFIDENCE_THRESHOLD*100:.0f}% + CLOB best ask    ║
║  Mode: {CANDLE_CONFIGS[CANDLE_MODE]['label']} BTC candles | PAPER                      ║
║  🔪 Kill switch: --kill55 (skip entries > 55c)              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    if spot:
        print(f"  ✅ Binance | BTC: ${spot:,.2f}")
    
    candle_num = 0
    
    try:
        while True:
            candle_num += 1
            trade_candle(candle_num)
    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user.")
        
        # Final summary
        if session_trades:
            wins = sum(1 for t in session_trades if t["result"] == "WIN")
            losses = sum(1 for t in session_trades if t["result"] == "LOSS")
            
            print(f"\n{'═'*70}")
            print(f"  FINAL SUMMARY")
            print(f"{'═'*70}")
            print(f"  Trades: {wins}W / {losses}L = {wins/(wins+losses)*100:.0f}% WR" if (wins+losses) > 0 else "  No trades")
            print(f"  P&L: ${daily_pnl:+.2f}")
            
            if session_trades:
                confs = [t["conf"] for t in session_trades]
                times = [t["elapsed"] for t in session_trades]
                print(f"  Avg confidence: {np.mean(confs)*100:.0f}%")
                print(f"  Avg entry time: T={np.mean(times):.0f}s")
                
                # Win rate by confidence
                for lo, hi, label in [(0.65, 0.75, "65-75%"), (0.75, 0.85, "75-85%"), (0.85, 1.01, "85%+")]:
                    bucket = [t for t in session_trades if lo <= t["conf"] < hi]
                    if bucket:
                        bw = sum(1 for t in bucket if t["result"] == "WIN")
                        print(f"  {label}: {bw}/{len(bucket)} = {bw/len(bucket)*100:.0f}% WR")
        
        # Save scan log
        if scan_log:
            with open("ml_scan_log.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=scan_log[0].keys())
                writer.writeheader()
                writer.writerows(scan_log)
            print(f"\n  📁 Scan log: ml_scan_log.csv ({len(scan_log)} scans)")


if __name__ == "__main__":
    main()

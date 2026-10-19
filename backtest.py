"""
Portable Backtester — Support BTC, XAU, Forex, Indeks, Crypto, dll.
Deteksi otomatis instrumen, konfigurasi point size & value-per-point.
Jalankan: python backtest.py --csv data.csv
"""
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# ============ KONFIGURASI ============
@dataclass
class Config:
    # Instrument metadata
    symbol_name: str = "UNKNOWN"
    point_size: float = 0.01               # 1 poin = 0.01 harga (BTC/XAU)
    value_per_point_per_lot: float = 0.01  # USD nilai 1 poin × 1 lot

    # Filter entry
    rsi_period: int = 14
    rsi_buy_max: float = 65
    rsi_sell_min: float = 35
    ema_fast: int = 20
    ema_slow: int = 50
    min_candles: int = 60
    swing_lookback: int = 3

    # SL/TP (dalam poin)
    fixed_sl_points: int = 3000
    fixed_tp_points: int = 6000

    # Trailing (dalam poin)
    trailing_activation_points: int = 3000
    trailing_gap_points: int = 1500
    trailing_step: int = 100

    # Position
    lot: float = 1.0
    hard_lock_seconds: int = 300

    # Eksekusi
    entry_at_next_open: bool = True
    spread_points: float = 0.0

# ============ PRESET INSTRUMEN ============
INSTRUMENT_PRESETS = {
    "BTCUSD": dict(point_size=0.01, value_per_point_per_lot=0.01,
                   fixed_sl_points=3000, fixed_tp_points=6000,
                   trailing_activation_points=3000, trailing_gap_points=1500,
                   lot=1.0),
    "ETHUSD": dict(point_size=0.01, value_per_point_per_lot=0.01,
                   fixed_sl_points=3000, fixed_tp_points=6000,
                   trailing_activation_points=3000, trailing_gap_points=1500,
                   lot=1.0),
    "XAUUSD": dict(point_size=0.01, value_per_point_per_lot=1.0,
                   fixed_sl_points=3000, fixed_tp_points=6000,
                   trailing_activation_points=3000, trailing_gap_points=1500,
                   lot=0.01),
    "XAGUSD": dict(point_size=0.001, value_per_point_per_lot=5.0,
                   fixed_sl_points=500, fixed_tp_points=1000,
                   trailing_activation_points=500, trailing_gap_points=250,
                   lot=0.10),
    "EURUSD": dict(point_size=0.00001, value_per_point_per_lot=1.0,
                   fixed_sl_points=500, fixed_tp_points=1000,
                   trailing_activation_points=500, trailing_gap_points=250,
                   lot=0.10),
    "GBPUSD": dict(point_size=0.00001, value_per_point_per_lot=1.0,
                   fixed_sl_points=500, fixed_tp_points=1000,
                   trailing_activation_points=500, trailing_gap_points=250,
                   lot=0.10),
    "AUDUSD": dict(point_size=0.00001, value_per_point_per_lot=1.0,
                   fixed_sl_points=500, fixed_tp_points=1000,
                   trailing_activation_points=500, trailing_gap_points=250,
                   lot=0.10),
    "USDJPY": dict(point_size=0.001, value_per_point_per_lot=0.67,
                   fixed_sl_points=500, fixed_tp_points=1000,
                   trailing_activation_points=500, trailing_gap_points=250,
                   lot=0.10),
    "US30":   dict(point_size=0.1, value_per_point_per_lot=0.1,
                   fixed_sl_points=3000, fixed_tp_points=6000,
                   trailing_activation_points=3000, trailing_gap_points=1500,
                   lot=0.10),
    "NAS100": dict(point_size=0.1, value_per_point_per_lot=0.1,
                   fixed_sl_points=3000, fixed_tp_points=6000,
                   trailing_activation_points=3000, trailing_gap_points=1500,
                   lot=0.10),
    "SPX500": dict(point_size=0.1, value_per_point_per_lot=0.1,
                   fixed_sl_points=3000, fixed_tp_points=6000,
                   trailing_activation_points=3000, trailing_gap_points=1500,
                   lot=0.10),
}

# ============ AUTO-DETECT INSTRUMEN ============
def auto_detect_instrument(df: pd.DataFrame) -> str:
    """Tebak instrumen dari range harga."""
    if len(df) == 0: return "UNKNOWN"
    med = df['close'].median()
    decimals = 0
    sample = str(df['close'].iloc[-1])
    if '.' in sample:
        decimals = len(sample.split('.')[1])
    if med > 20000: return "BTCUSD"
    if 3000 < med < 20000:
        return "ETHUSD" if med < 10000 else "BTCUSD"
    if 1500 < med < 3000: return "XAUUSD"
    if 100 < med < 200 and decimals >= 3: return "USDJPY"
    if 15 < med < 60: return "XAGUSD"
    if 0.5 < med < 1.5 and decimals >= 4: return "EURUSD"
    return "UNKNOWN"

def apply_preset(cfg: Config, name: str) -> Config:
    """Apply preset ke config."""
    if name not in INSTRUMENT_PRESETS: return cfg
    p = INSTRUMENT_PRESETS[name]
    cfg.symbol_name = name
    for k, v in p.items(): setattr(cfg, k, v)
    return cfg

# ============ DATA LOADER ============
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    # Handle MT4/MT5 format: Date + Time terpisah
    if ('date' in df.columns and 'time' in df.columns
            and 'datetime' not in df.columns and 'timestamp' not in df.columns):
        sample_t = str(df['time'].iloc[0]) if len(df) > 0 else ''
        if ':' in sample_t and len(sample_t) < 12:
            d = df['date'].astype(str).str.replace('.', '-', regex=False).str.replace('/', '-', regex=False)
            df['datetime'] = pd.to_datetime(d + ' ' + df['time'].astype(str), errors='coerce')
            df = df.drop(columns=['date', 'time'])

    rename_map = {
        'o':'open','h':'high','l':'low','c':'close','vol':'volume',
        'tanggal':'time','date':'time','datetime':'time','timestamp':'time',
        'adj close':'close','adj_close':'close',
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns and k != v})

    for col in ['open','high','low','close']:
        if col not in df.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}. Kolom tersedia: {list(df.columns)}")

    if 'volume' not in df.columns: df['volume'] = 0
    if 'time' not in df.columns:   df['time'] = df.index

    try: df['time'] = pd.to_datetime(df['time'], errors='coerce')
    except: pass

    df = df[['time','open','high','low','close','volume']].copy()
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open','high','low','close']).sort_values('time').reset_index(drop=True)
    return df

def load_csv(path: str) -> pd.DataFrame:
    return normalize_df(pd.read_csv(path))

def load_yfinance(ticker='BTC-USD', period='2y', interval='1h') -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return normalize_df(df.reset_index())

# ============ INDIKATOR ============
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def find_swings(df, lookback=3):
    highs, lows = df['high'].values, df['low'].values
    n = len(df); sh, sl = [], []
    for i in range(lookback, n - lookback):
        wh = highs[i-lookback:i+lookback+1]
        wl = lows[i-lookback:i+lookback+1]
        if highs[i] == wh.max() and (wh == highs[i]).sum() == 1: sh.append((i, highs[i]))
        if lows[i] == wl.min() and (wl == lows[i]).sum() == 1: sl.append((i, lows[i]))
    return sh, sl

def detect_trend(recent_h, recent_l, ema_f_now, ema_s_now, ema_f_prev, price):
    if len(recent_h) >= 3 and len(recent_l) >= 3:
        h1, h2, h3 = recent_h[-3], recent_h[-2], recent_h[-1]
        l1, l2, l3 = recent_l[-3], recent_l[-2], recent_l[-1]
        if (h2>h1)+(h3>h2) >= 2 and (l2>l1)+(l3>l2) >= 2: return 'bullish', 'Swing'
        if (h2<h1)+(h3<h2) >= 2 and (l2<l1)+(l3<l2) >= 2: return 'bearish', 'Swing'
    if ema_f_now > ema_s_now and ema_f_now > ema_f_prev and price > ema_s_now: return 'bullish', 'EMA'
    if ema_f_now < ema_s_now and ema_f_now < ema_f_prev and price < ema_s_now: return 'bearish', 'EMA'
    return 'ranging', 'None'

def detect_pattern(df, i, direction):
    if i < 2: return None
    c, p = df.iloc[i], df.iloc[i-1]
    body_c = abs(c['close'] - c['open'])
    range_c = c['high'] - c['low']
    if range_c == 0: return None
    bull_c = c['close'] > c['open']; bear_c = c['close'] < c['open']
    up_w = c['high'] - max(c['close'], c['open'])
    lo_w = min(c['close'], c['open']) - c['low']
    if direction == 'bullish':
        if p['close'] < p['open'] and bull_c and c['close'] > p['open'] and c['open'] < p['close']: return "Bullish Engulfing"
        if lo_w > body_c*1.5 and up_w < body_c*0.5 and body_c > 0: return "Hammer"
        if bull_c and body_c > range_c*0.75: return "Bullish Marubozu"
        if bull_c and body_c > range_c*0.6:  return "Momentum Bullish"
    else:
        if p['close'] > p['open'] and bear_c and c['close'] < p['open'] and c['open'] > p['close']: return "Bearish Engulfing"
        if up_w > body_c*2.5 and lo_w < body_c*0.3 and body_c > 0: return "Shooting Star"
        if bear_c and body_c > range_c*0.75: return "Bearish Marubozu"
        if bear_c and body_c > range_c*0.6:  return "Momentum Bearish"
    return None

# ============ BACKTEST ============
def run_backtest(df: pd.DataFrame, cfg: Config):
    df = df.reset_index(drop=True).copy()
    df['rsi'] = compute_rsi(df['close'], cfg.rsi_period)
    df['ema_fast'] = df['close'].ewm(span=cfg.ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=cfg.ema_slow, adjust=False).mean()
    swing_h, swing_l = find_swings(df, cfg.swing_lookback)

    n = len(df)
    trades, equity_curve = [], np.zeros(n)
    equity = 0.0; position = None; pending = None
    last_open_time = None; sh_ptr = sl_ptr = 0
    point = cfg.point_size; vpp = cfg.value_per_point_per_lot

    for i in range(n):
        bar = df.iloc[i]
        bar_time = bar['time'] if isinstance(bar['time'], pd.Timestamp) else pd.Timestamp(bar['time'])

        if pending is not None and position is None:
            entry = bar['open']
            if pending['action'] == 'buy':
                sl = entry - cfg.fixed_sl_points*point; tp = entry + cfg.fixed_tp_points*point
            else:
                sl = entry + cfg.fixed_sl_points*point; tp = entry - cfg.fixed_tp_points*point
            position = dict(direction=pending['action'], entry=entry,
                            original_sl=sl, current_sl=sl, tp=tp,
                            entry_bar=i, entry_time=bar_time, peak=entry,
                            trailing_active=False, pattern=pending['pattern'],
                            trend=pending['trend'], rsi=pending['rsi'])
            pending = None; last_open_time = bar_time

        if position is not None:
            exit_price, reason = None, None
            sl, tp = position['current_sl'], position['tp']
            if position['direction'] == 'buy':
                if   bar['open'] <= sl: exit_price, reason = bar['open'], 'SL'
                elif bar['low']  <= sl: exit_price, reason = sl, 'SL'
                elif bar['open'] >= tp: exit_price, reason = bar['open'], 'TP'
                elif bar['high'] >= tp: exit_price, reason = tp, 'TP'
            else:
                if   bar['open'] >= sl: exit_price, reason = bar['open'], 'SL'
                elif bar['high'] >= sl: exit_price, reason = sl, 'SL'
                elif bar['open'] <= tp: exit_price, reason = bar['open'], 'TP'
                elif bar['low']  <= tp: exit_price, reason = tp, 'TP'

            if exit_price is not None:
                pts = (exit_price - position['entry'])/point if position['direction']=='buy' \
                      else (position['entry'] - exit_price)/point
                pts -= cfg.spread_points
                usd = pts * vpp * cfg.lot
                trades.append(dict(
                    trade_id=len(trades)+1,
                    open_time=position['entry_time'], close_time=bar_time,
                    direction=position['direction'],
                    entry=position['entry'], sl=position['original_sl'],
                    tp=position['tp'], exit_price=exit_price,
                    exit_reason=reason, points=pts, usd=usd,
                    duration_bars=i - position['entry_bar'],
                    pattern=position['pattern'], trend=position['trend'],
                    rsi_at_entry=position['rsi']))
                equity += usd
                position = None
            else:
                if position['direction'] == 'buy':
                    position['peak'] = max(position['peak'], bar['high'])
                    if not position['trailing_active'] and \
                       (position['peak']-position['entry'])/point >= cfg.trailing_activation_points:
                        position['trailing_active'] = True
                    if position['trailing_active']:
                        new_sl = position['peak'] - cfg.trailing_gap_points*point
                        if new_sl > position['current_sl'] + cfg.trailing_step*point:
                            position['current_sl'] = new_sl
                else:
                    position['peak'] = min(position['peak'], bar['low'])
                    if not position['trailing_active'] and \
                       (position['entry']-position['peak'])/point >= cfg.trailing_activation_points:
                        position['trailing_active'] = True
                    if position['trailing_active']:
                        new_sl = position['peak'] + cfg.trailing_gap_points*point
                        if new_sl < position['current_sl'] - cfg.trailing_step*point:
                            position['current_sl'] = new_sl

        equity_curve[i] = equity

        if position is not None or pending is not None: continue
        if i < cfg.min_candles: continue
        if last_open_time is not None and (bar_time - last_open_time).total_seconds() < cfg.hard_lock_seconds:
            continue

        while sh_ptr < len(swing_h) and swing_h[sh_ptr][0] <= i - cfg.swing_lookback: sh_ptr += 1
        while sl_ptr < len(swing_l) and swing_l[sl_ptr][0] <= i - cfg.swing_lookback: sl_ptr += 1
        recent_h = [s[1] for s in swing_h[:sh_ptr][-3:]]
        recent_l = [s[1] for s in swing_l[:sl_ptr][-3:]]

        ema_f_now = df['ema_fast'].iloc[i]; ema_s_now = df['ema_slow'].iloc[i]
        ema_f_prev = df['ema_fast'].iloc[max(0, i-5)]
        trend, _ = detect_trend(recent_h, recent_l, ema_f_now, ema_s_now, ema_f_prev, bar['close'])
        if trend == 'ranging': continue
        rsi_val = bar['rsi']
        if pd.isna(rsi_val): continue
        if trend == 'bullish' and rsi_val > cfg.rsi_buy_max: continue
        if trend == 'bearish' and rsi_val < cfg.rsi_sell_min: continue

        direction = 'bullish' if trend == 'bullish' else 'bearish'
        pattern = detect_pattern(df, i, direction)
        if pattern is None: continue

        action = 'buy' if direction == 'bullish' else 'sell'
        if cfg.entry_at_next_open:
            pending = dict(action=action, pattern=pattern, trend=trend, rsi=rsi_val)
        else:
            entry = bar['close']
            if action == 'buy':
                sl = entry - cfg.fixed_sl_points*point; tp = entry + cfg.fixed_tp_points*point
            else:
                sl = entry + cfg.fixed_sl_points*point; tp = entry - cfg.fixed_tp_points*point
            position = dict(direction=action, entry=entry, original_sl=sl, current_sl=sl, tp=tp,
                            entry_bar=i, entry_time=bar_time, peak=entry, trailing_active=False,
                            pattern=pattern, trend=trend, rsi=rsi_val)
            last_open_time = bar_time

    if position is not None:
        exit_price = df.iloc[-1]['close']
        pts = (exit_price - position['entry'])/point if position['direction']=='buy' \
              else (position['entry'] - exit_price)/point
        usd = pts * vpp * cfg.lot
        trades.append(dict(
            trade_id=len(trades)+1, open_time=position['entry_time'],
            close_time=df.iloc[-1]['time'], direction=position['direction'],
            entry=position['entry'], sl=position['original_sl'], tp=position['tp'],
            exit_price=exit_price, exit_reason='END', points=pts, usd=usd,
            duration_bars=n-1-position['entry_bar'],
            pattern=position['pattern'], trend=position['trend'],
            rsi_at_entry=position['rsi']))

    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(trades_df, equity_curve)
    return trades_df, metrics, equity_curve

def compute_metrics(trades_df, equity_curve):
    if trades_df is None or len(trades_df) == 0:
        return {k: 0 for k in ['total_trades','wins','losses','breakeven','win_rate',
                                'total_points','total_usd','avg_win','avg_loss',
                                'profit_factor','expectancy','max_drawdown_usd',
                                'max_drawdown_pct','sharpe','best_trade','worst_trade',
                                'max_win_streak','max_loss_streak']}
    usd, pts = trades_df['usd'].values, trades_df['points'].values
    wins, losses = usd > 0, usd < 0
    n_w, n_l, n_be = int(wins.sum()), int(losses.sum()), int((usd==0).sum())
    total = len(usd)
    gross_w = usd[wins].sum() if n_w else 0
    gross_l = abs(usd[losses].sum()) if n_l else 0
    pf = gross_w/gross_l if gross_l > 0 else float('inf')
    rm = np.maximum.accumulate(equity_curve)
    dd = equity_curve - rm
    max_dd = dd.min() if len(dd) else 0
    max_dd_pct = (dd / np.maximum(rm, 1e-9)).min() * 100 if len(dd) else 0
    sharpe = usd.mean()/usd.std()*np.sqrt(252) if len(usd)>1 and usd.std()>0 else 0
    cur_w = cur_l = max_w = max_l = 0
    for u in usd:
        if u > 0: cur_w += 1; cur_l = 0; max_w = max(max_w, cur_w)
        elif u < 0: cur_l += 1; cur_w = 0; max_l = max(max_l, cur_l)
    return dict(total_trades=total, wins=n_w, losses=n_l, breakeven=n_be,
        win_rate=n_w/total*100 if total else 0,
        total_points=pts.sum(), total_usd=usd.sum(),
        avg_win=usd[wins].mean() if n_w else 0,
        avg_loss=usd[losses].mean() if n_l else 0,
        profit_factor=pf, expectancy=usd.mean() if total else 0,
        max_drawdown_usd=max_dd, max_drawdown_pct=max_dd_pct,
        sharpe=sharpe, best_trade=usd.max() if total else 0,
        worst_trade=usd.min() if total else 0,
        max_win_streak=max_w, max_loss_streak=max_l)

# ============ CLI ============
def print_report(metrics, cfg, data_info=""):
    print("\n" + "="*60)
    print(f"  HASIL BACKTEST  |  {cfg.symbol_name}")
    print("="*60)
    if data_info: print(f"  Data: {data_info}")
    print(f"  Point: {cfg.point_size} | Value/point/lot: ${cfg.value_per_point_per_lot}")
    print(f"  Lot: {cfg.lot} | SL: {cfg.fixed_sl_points}p | TP: {cfg.fixed_tp_points}p")
    print("-"*60)
    print(f"  Total Trades     : {metrics['total_trades']}")
    print(f"  Win / Loss / BE  : {metrics['wins']} / {metrics['losses']} / {metrics['breakeven']}")
    print(f"  Win Rate         : {metrics['win_rate']:.2f}%")
    print(f"  Total Poin       : {metrics['total_points']:+.0f} poin")
    print(f"  Total P/L (USD)  : {metrics['total_usd']:+.2f} USD")
    print(f"  Avg Win          : {metrics['avg_win']:+.2f} USD")
    print(f"  Avg Loss         : {metrics['avg_loss']:+.2f} USD")
    print(f"  Profit Factor    : {metrics['profit_factor']:.2f}")
    print(f"  Expectancy       : {metrics['expectancy']:+.2f} USD/trade")
    print(f"  Max Drawdown     : {metrics['max_drawdown_usd']:.2f} USD ({metrics['max_drawdown_pct']:.2f}%)")
    print(f"  Sharpe (rough)   : {metrics['sharpe']:.2f}")
    print("="*60 + "\n")

def main():
    p = argparse.ArgumentParser(description='Portable Backtester')
    p.add_argument('--csv', type=str, default=None)
    p.add_argument('--ticker', default='BTC-USD')
    p.add_argument('--period', default='2y')
    p.add_argument('--interval', default='1h')
    p.add_argument('--symbol', default=None, help='Preset instrumen (BTCUSD, XAUUSD, EURUSD, ...)')
    p.add_argument('--lot', type=float, default=None)
    p.add_argument('--sl', type=int, default=None)
    p.add_argument('--tp', type=int, default=None)
    p.add_argument('--point-size', type=float, default=None)
    p.add_argument('--value-per-point', type=float, default=None)
    p.add_argument('--out-trades', default='backtest_trades.csv')
    args = p.parse_args()

    print("Memuat data ...")
    if args.csv:
        df = load_csv(args.csv); info = f"CSV: {args.csv}"
    else:
        df = load_yfinance(args.ticker, args.period, args.interval)
        info = f"yfinance: {args.ticker} {args.period} {args.interval}"
    if len(df) < 100:
        print(f"⚠️  Data terlalu sedikit ({len(df)} bar)."); return
    print(f"✅ {len(df)} bar | {df['time'].iloc[0]} → {df['time'].iloc[-1]}")

    cfg = Config()
    preset_name = args.symbol or auto_detect_instrument(df)
    if preset_name in INSTRUMENT_PRESETS:
        apply_preset(cfg, preset_name)
        print(f"🎯 Preset terdeteksi: {preset_name}")
    else:
        cfg.symbol_name = "CUSTOM"
        print("⚠️  Instrumen tidak terdeteksi — pakai default BTC")

    if args.lot is not None: cfg.lot = args.lot
    if args.sl is not None: cfg.fixed_sl_points = args.sl
    if args.tp is not None: cfg.fixed_tp_points = args.tp
    if args.point_size is not None: cfg.point_size = args.point_size
    if args.value_per_point is not None: cfg.value_per_point_per_lot = args.value_per_point

    trades, metrics, eq = run_backtest(df, cfg)
    print_report(metrics, cfg, info)
    if len(trades) > 0:
        trades.to_csv(args.out_trades, index=False)
        print(f"✅ Trade detail disimpan: {args.out_trades}")

if __name__ == '__main__':
    main()
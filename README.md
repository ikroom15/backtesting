# 🌍 Portable Backtester — Multi Instrument

Backtester browser untuk **BTC, XAU, Forex, Indeks, Crypto, dll**.
100% client-side (Pyodide) — data CSV Anda tidak pernah dikirim ke server.

## 🚀 Live Demo
`https://USERNAME.github.io/REPO/`

## 📊 Instrumen yang Didukung

| Preset | Point Size | Value/Point/Lot | SL Default |
|--------|-----------|-----------------|------------|
| BTCUSD, ETHUSD | 0.01 | $0.01 | 3000 poin ($30) |
| XAUUSD (Gold) | 0.01 | $1.00 | 3000 poin ($30) |
| XAGUSD (Silver) | 0.001 | $5.00 | 500 poin |
| EURUSD, GBPUSD, AUDUSD | 0.00001 | $1.00 | 500 poin (50 pips) |
| USDJPY | 0.001 | $0.67 | 500 poin |
| US30, NAS100, SPX500 | 0.1 | $0.10 | 3000 poin |

**+ Custom** — input `Point Size` & `Value per Point` manual untuk instrumen lain.

## 📂 Format CSV

Kolom wajib: `open, high, low, close`
Kolom opsional: `time`/`date`/`datetime`/`timestamp`, `volume`
MT4/MT5 export (`Date,Time,Open,High,Low,Close,Volume`) langsung bisa.

## 🎯 Cara Pakai

1. Tunggu sampai muncul **"✅ Siap!"** (~30-60 detik load pertama)
2. Upload CSV **atau** klik **Data Demo**
3. Instrumen auto-detect — atau pilih manual dari dropdown
4. Atur SL/TP/Lot (ada preview risiko!)
5. Klik **🚀 Jalankan Backtest**
6. Lihat metrik, kurva equity, tabel trade → **Download CSV**

## 🌐 Deploy ke GitHub Pages

1. Push semua file (index.html, backtest.py, README.md, .nojekyll)
2. Settings → Pages → Branch: **main** / folder **/(root)** → Save
3. Akses `https://USERNAME.github.io/REPO/`

## 💻 CLI (opsional)

```bash
pip install pandas numpy yfinance

# Auto-detect instrumen
python backtest.py --csv gold_data.csv

# Preset eksplisit
python backtest.py --csv data.csv --symbol XAUUSD --lot 0.01

# Custom instrumen
python backtest.py --csv exotic.csv --point-size 0.001 --value-per-point 0.5 --sl 1000 --tp 2000
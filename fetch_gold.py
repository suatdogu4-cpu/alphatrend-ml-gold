"""Fetch Yahoo Finance gold futures proxy (GC=F); NOT XAUUSD/XAUUSDT price-feed equivalent."""
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

p=argparse.ArgumentParser()
p.add_argument('--symbol',default='GC=F')
p.add_argument('--tf',choices=['1m','5m','15m'],default='5m')
p.add_argument('--period',default=None)
p.add_argument('--output',default='gold.csv')
a=p.parse_args()
period=a.period or ('7d' if a.tf=='1m' else '59d')
d=yf.download(a.symbol,period=period,interval=a.tf,auto_adjust=False,progress=False,multi_level_index=False)
if d.empty: raise SystemExit('Veri indirilemedi; sembol/aralık erişimini kontrol edin veya CSV kullanın.')
d=d.reset_index()
d.columns=[str(c).lower() for c in d.columns]
d=d.rename(columns={'datetime':'time','date':'time'})
d=d[['time','open','high','low','close','volume']].dropna()
# Provider may include the current unfinished bar; exclude last bar conservatively.
d=d.iloc[:-1]
if d.empty: raise SystemExit('Kapanmış mum yok.')
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
d.to_csv(a.output,index=False)
print(f'{len(d)} adet geçmiş mum kaydedildi: {a.output} (GC=F altın vadeli fiyatıdır; spot XAUUSD değildir)')

"""AlphaTrend four-band squeeze ML: research only, no order execution.
Original AlphaTrend logic: KivancOzbilgic (MPL-2.0), https://mozilla.org/MPL/2.0/
Python adaptation for research. Developed for an independent gold-analysis workflow.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

FEATURES = ['rsi', 'mfi', 'atr_pct', 'width_pct', 'width_ratio', 'bb_width', 'donchian_width',
            'alpha_slope', 'center_distance_atr', 'return_1', 'return_3',
            'volume_ratio', 'candle_body_atr', 'upper_wick_atr', 'lower_wick_atr',
            'range_10_atr', 'hour_sin', 'hour_cos']
LABELS = {-1:'ASAGI',0:'BELIRSIZ',1:'YUKARI'}

def rsi_wilder(close, length):
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    # Wilder RMA with SMA initialization, Pine-compatible beyond warm-up
    def rma(series):
        vals = series.to_numpy(dtype=float)
        out = np.full(len(vals), np.nan)
        if len(vals) <= length: return pd.Series(out, index=series.index)
        out[length] = np.nanmean(vals[1:length+1])
        for i in range(length+1, len(vals)):
            out[i] = (out[i-1]*(length-1) + vals[i])/length
        return pd.Series(out, index=series.index)
    ru, rd = rma(up), rma(down)
    result = 100-100/(1+ru/rd.replace(0,np.nan))
    result = result.mask((rd==0)&(ru>0),100).mask((rd==0)&(ru==0),50)
    return result

def money_flow_index(df, length):
    typical = (df.high+df.low+df.close)/3
    raw = typical*df.volume
    positive = raw.where(typical>typical.shift(),0.0)
    negative = raw.where(typical<typical.shift(),0.0)
    pos = positive.rolling(length,min_periods=length).sum()
    neg = negative.rolling(length,min_periods=length).sum()
    mfi = 100*pos/(pos+neg).replace(0,np.nan)
    return mfi.where(~((neg==0)&(pos>0)),100).fillna(50)

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {'datetime':'time','date':'time','timestamp':'time','open time':'time','vol':'volume'}
    df = df.rename(columns={c:aliases.get(c,c) for c in df.columns})
    required = {'time','open','high','low','close','volume'}
    if not required.issubset(df.columns):
        raise ValueError('CSV gerekli kolonlar: time,open,high,low,close,volume; eksik: '+str(required-set(df.columns)))
    series = df['time']
    if pd.api.types.is_numeric_dtype(series):
        units = 'ms' if series.dropna().abs().median()>1e11 else 's'
        df['time'] = pd.to_datetime(series,unit=units,utc=True,errors='coerce')
    else:
        df['time'] = pd.to_datetime(series,utc=True,errors='coerce')
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col],errors='coerce')
    df = df.dropna(subset=list(required)).sort_values('time').drop_duplicates('time').reset_index(drop=True)
    if len(df)<300: raise ValueError('En az 300 kapanmış mum gerekli.')
    if (df['high']<df[['open','close','low']].max(axis=1)).any() or (df['low']>df[['open','close','high']].min(axis=1)).any():
        raise ValueError('Tutarsız OHLC verisi var.')
    return df

def make_features(df, period=14, multiplier=1.0, band_mult=0.5,
                  sqz_len=20, lookback=30, sqz_factor=0.85, use_rsi=False):
    d = df.copy()
    prev = d.close.shift()
    tr = pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
    d['atr'] = tr.rolling(period,min_periods=period).mean()  # ta.sma(ta.tr, AP)
    d['rsi'] = rsi_wilder(d.close,period)
    d['mfi'] = money_flow_index(d,period)
    gate = d.rsi if use_rsi else d.mfi
    # Keep Pine's recursive max/min logic; prior AlphaTrend starts as 0.
    alpha = np.zeros(len(d),dtype=float)
    for i in range(len(d)):
        atr = d['atr'].iloc[i]
        if not np.isfinite(atr):
            alpha[i] = alpha[i-1] if i else 0.0
            continue
        old = alpha[i-1] if i else 0.0
        if gate.iloc[i]>=50:
            alpha[i] = max(old, d.low.iloc[i]-atr*multiplier)
        else:
            alpha[i] = min(old, d.high.iloc[i]+atr*multiplier)
    d['alpha'] = alpha
    d['width_pct'] = 8*band_mult*d['atr']/d['alpha'].abs().replace(0,np.nan)*100
    width_ma = d.width_pct.rolling(sqz_len,min_periods=sqz_len).mean()
    d['width_ratio'] = d.width_pct/width_ma.replace(0,np.nan)
    lowest_width = d.width_pct.rolling(lookback,min_periods=lookback).min()
    d['squeeze'] = ((d.width_pct < width_ma*sqz_factor) &
                    (d.width_pct<=lowest_width*1.05)).fillna(False)
    # Exactly one entry point per contiguous squeeze cluster; avoids duplicate labels
    d['event'] = d['squeeze'] & ~d['squeeze'].shift(fill_value=False)
    d['atr_pct'] = d.atr/d.close*100
    d['alpha_slope'] = (d.alpha-d.alpha.shift(3))/d.atr.replace(0,np.nan)
    d['center_distance_atr'] = (d.close-d.alpha)/d.atr.replace(0,np.nan)
    d['return_1'] = d.close.pct_change(fill_method=None)*100
    d['return_3'] = d.close.pct_change(3,fill_method=None)*100
    std = d.close.rolling(20,min_periods=20).std()
    d['bb_width'] = 4*std/d.close.rolling(20,min_periods=20).mean()*100
    d['donchian_width'] = (d.high.rolling(20).max()-d.low.rolling(20).min())/d.close*100
    d['volume_ratio'] = d.volume/d.volume.rolling(20,min_periods=20).mean().replace(0,np.nan)
    d['candle_body_atr'] = (d.close-d.open)/d.atr.replace(0,np.nan)
    d['upper_wick_atr'] = (d.high-d[['open','close']].max(axis=1))/d.atr.replace(0,np.nan)
    d['lower_wick_atr'] = (d[['open','close']].min(axis=1)-d.low)/d.atr.replace(0,np.nan)
    d['range_10_atr'] = (d.high.rolling(10).max()-d.low.rolling(10).min())/d.atr.replace(0,np.nan)
    hrs = d.time.dt.hour+d.time.dt.minute/60
    d['hour_sin'] = np.sin(2*np.pi*hrs/24)
    d['hour_cos'] = np.cos(2*np.pi*hrs/24)
    return d

def label_events(d, horizon=12, barrier_atr=1.5, min_events=35):
    events = d.loc[d.event].copy()
    valid=[]
    for ix, row in events.iterrows():
        if ix+horizon>=len(d) or not np.isfinite(row.atr) or row.atr<=0: continue
        target_up = row.close+row.atr*barrier_atr
        target_down = row.close-row.atr*barrier_atr
        outcome = 0
        for j in range(ix+1, ix+horizon+1):
            up = d.high.iloc[j]>=target_up
            down = d.low.iloc[j]<=target_down
            if up and down:
                outcome=0 # Ambiguous intrabar order, never guess
                break
            if up:
                outcome=1; break
            if down:
                outcome=-1; break
        # Finite feature rows only; avoid leaking future into predictors
        if not np.isfinite(row[FEATURES].astype(float).to_numpy()).all(): continue
        valid.append((ix,int(outcome)))
    if not valid: raise ValueError('Sıkışma olayı bulunamadı. Daha uzun veri veya daha esnek sqz-factor deneyin.')
    ix = [item[0] for item in valid]

events = d.loc[ix, list(dict.fromkeys(['time','close','atr','width_pct']+FEATURES))].copy()
    events['label'] = [item[1] for item in valid]
    events['bar_index'] = ix
    if len(events)<min_events:
        raise ValueError(f'Yeterli olay yok ({len(events)}). Güvenilir başlangıç için en az {min_events} olay gerekli. Daha uzun veri kullanın.')
    return events.reset_index(drop=True)

def split_purged(events, horizon):
    n = len(events)
    v = int(n*.60); t=int(n*.80)
    # Embargo in raw bars, not merely squeeze events
    train = events.iloc[:v].copy()
    valid = events.iloc[v:t].copy()
    test = events.iloc[t:].copy()
    train = train.loc[train.bar_index+horizon < valid.bar_index.min()]
    valid = valid.loc[valid.bar_index+horizon < test.bar_index.min()]
    if min(len(train),len(valid),len(test))<5:
        raise ValueError('Purged dönemlerde yeterli olay kalmadı. Daha fazla geçmiş veri gerekir.')
    return train, valid, test

def score_model(model, dataset):
    x=dataset[FEATURES]
    y=dataset.label
    pred=model.predict(x)
    proba=model.predict_proba(x)
    classes=model.classes_
    return {'count':len(dataset),'accuracy':round(float(accuracy_score(y,pred)),4),
            'balanced_accuracy':round(float(balanced_accuracy_score(y,pred)),4),
            'log_loss':round(float(log_loss(y,proba,labels=classes)),4),
            'classes_seen':[LABELS[int(k)] for k in classes],
            'confusion_matrix':confusion_matrix(y,pred,labels=[-1,0,1]).tolist(),
            'classification_report':classification_report(y,pred,labels=[-1,0,1],target_names=['ASAGI','BELIRSIZ','YUKARI'],zero_division=0,output_dict=True)}

def run(args):
    df=load_csv(args.csv)
    d=make_features(df,period=args.period,multiplier=args.multiplier,band_mult=args.band_mult,
                    sqz_len=args.sqz_len,lookback=args.lookback,sqz_factor=args.sqz_factor,use_rsi=args.use_rsi)
    events=label_events(d,horizon=args.horizon,barrier_atr=args.barrier_atr,min_events=args.min_events)
    train,val,test=split_purged(events,args.horizon)
    if len(train.label.unique())<2: raise ValueError('Eğitimde en az 2 farklı etiket olmalı. Daha fazla veri gerekiyor.')
    candidates={
        'dummy':DummyClassifier(strategy='prior'),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,l2_regularization=2,learning_rate=.05,random_state=42)
    }
    fitted={}
    for key, model in candidates.items():
        pipeline=make_pipeline(SimpleImputer(strategy='median'),model)
        pipeline.fit(train[FEATURES],train.label)
        fitted[key]=pipeline
    # Validation-only selection; test never influences selection.
    ranking={name:score_model(model,val) for name,model in fitted.items()}
    selected=min(ranking,key=lambda key:ranking[key]['log_loss'])
    final=fitted[selected]
    results={'symbol':args.symbol,'timeframe':args.tf,'source':str(args.csv),'first_bar':str(df.time.iloc[0]),
             'last_bar':str(df.time.iloc[-1]),'candles':len(df),'squeeze_events':len(events),
             'labels_all':{LABELS[x]:int((events.label==x).sum()) for x in [-1,0,1]},
             'events_train':len(train),'events_validation':len(val),'events_test':len(test),
             'horizon_candles':args.horizon,'barrier_atr':args.barrier_atr,'sqz_factor':args.sqz_factor,
             'validation_comparison':{key:{k:v for k,v in result.items() if k!='classification_report'} for key,result in ranking.items()},
             'selected_model':selected,'holdout_test':score_model(final,test),
             'important':'Retrospective research ONLY. No future data used in features. Intrabar dual hits treated as uncertain. No trading or profitability claim.'}
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    events.to_csv(output/'events.csv',index=False)
    with open(output/'report.json','w',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)
    # Holdout predictions for inspecting prediction errors
    pred=final.predict(test[FEATURES]); prob=final.predict_proba(test[FEATURES])
    out=test[['time','close','label']].copy(); out['prediction']=pred
    for i,cls in enumerate(final.classes_): out['prob_'+LABELS[int(cls)].lower()]=prob[:,i]
    out.to_csv(output/'test_predictions.csv',index=False)
    import joblib
    joblib.dump({'model':final,'features':FEATURES,'parameters':vars(args),'classes':[int(v) for v in final.classes_]},output/'model.joblib')
    print(json.dumps({k:v for k,v in results.items() if k not in ['validation_comparison']},indent=2,ensure_ascii=False))
    print('Kayıt:',output.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description='4B AlphaTrend squeeze ML research')
    p.add_argument('--csv',required=True); p.add_argument('--symbol',default='GC=F'); p.add_argument('--tf',default='5m')
    p.add_argument('--output',default='output'); p.add_argument('--period',type=int,default=14)
    p.add_argument('--multiplier',type=float,default=1.0); p.add_argument('--band-mult',type=float,default=0.5)
    p.add_argument('--sqz-len',type=int,default=20); p.add_argument('--lookback',type=int,default=30)
    p.add_argument('--sqz-factor',type=float,default=0.85); p.add_argument('--use-rsi',action='store_true')
    p.add_argument('--horizon',type=int,default=12); p.add_argument('--barrier-atr',type=float,default=1.5)
    p.add_argument('--min-events',type=int,default=35)
    run(p.parse_args())

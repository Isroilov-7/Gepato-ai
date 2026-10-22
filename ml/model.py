"""
ml/model.py — Gepato-AI v2.0
Stratified XGBoost + Active Learning + SHAP
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from ml.feature_extraction import FEATURE_NAMES, DISEASE_STAGES

logger = logging.getLogger(__name__)

UNCERTAINTY_LOW  = 0.35
UNCERTAINTY_HIGH = 0.65
MIN_SAMPLES_BACKBONE = 20
MIN_SAMPLES_STRATUM  = 30
BACKBONE_WEIGHT = 0.40
STRATUM_WEIGHT  = 0.60

@dataclass
class PredictionResult:
    risk_score:float; risk_level:str; confidence:float
    shap_values:dict; top_factors:list; explanation:str
    needs_labeling:bool; trend_direction:str; trend_magnitude:float
    model_version:str; disease_stage:str; used_stratum_model:bool

@dataclass
class TrainResult:
    success:bool; version:str=""; auc_roc:float=0.0
    f1_score:float=0.0; sensitivity:float=0.0; specificity:float=0.0
    train_samples:int=0; test_samples:int=0
    feature_importances:dict=field(default_factory=dict)
    per_stratum_auc:dict=field(default_factory=dict)
    reason:str=""; model_path:str=""

def _ver_key(f:Path)->tuple:
    """model_v1.10.pkl > model_v1.9.pkl (matn bo'yicha saralash v1.10 ni eski deb hisoblardi)."""
    import re
    return tuple(int(x) for x in re.findall(r"\d+",f.stem)) or (0,)

class StratifiedHEModel:
    def __init__(self,model_dir:str="models"):
        self.model_dir=Path(model_dir); self.model_dir.mkdir(exist_ok=True)
        self.backbone:Optional[xgb.XGBClassifier]=None
        self.stratum_models:dict[str,xgb.XGBClassifier]={}
        self.explainer:Optional[shap.TreeExplainer]=None
        self.version="v0.0"
        self._counts:dict[str,int]={s:0 for s in DISEASE_STAGES}
        self._load_or_pretrain()

    def predict(self,features:dict,disease_stage:str="unknown",history:Optional[list]=None)->PredictionResult:
        if disease_stage not in DISEASE_STAGES: disease_stage="unknown"
        X=self._arr(features)
        bp=float(self.backbone.predict_proba(X)[0][1])
        use_s=(disease_stage in self.stratum_models and self._counts.get(disease_stage,0)>=MIN_SAMPLES_STRATUM)
        if use_s:
            sp=float(self.stratum_models[disease_stage].predict_proba(X)[0][1])
            rs=BACKBONE_WEIGHT*bp+STRATUM_WEIGHT*sp
        else:
            rs=bp
        rs=float(np.clip(rs,0,1))
        conf=abs(rs-0.5)*2
        sv,tf,exp=self._explain(X,features,rs)
        nl=UNCERTAINTY_LOW<rs<UNCERTAINTY_HIGH
        td,tm=self._trend(rs,history or [])
        rl="high" if rs>=0.65 else "medium" if rs>=0.40 else "low"
        return PredictionResult(risk_score=round(rs,4),risk_level=rl,confidence=round(conf,4),
            shap_values=sv,top_factors=tf,explanation=exp,needs_labeling=nl,
            trend_direction=td,trend_magnitude=round(tm,4),model_version=self.version,
            disease_stage=disease_stage,used_stratum_model=use_s)

    def retrain(self,df:pd.DataFrame,disease_stage:Optional[str]=None)->TrainResult:
        req=set(FEATURE_NAMES)|{"label"}
        miss=req-set(df.columns)
        if miss: return TrainResult(success=False,reason=f"Ustunlar yetishmaydi:{miss}")
        df=df.copy(); df["lb"]=(df["label"]>0).astype(int)
        if len(df)<MIN_SAMPLES_BACKBONE:
            return TrainResult(success=False,reason=f"Oz data:{len(df)}")
        X=df[FEATURE_NAMES].fillna(0).values; y=df["lb"].values
        try: Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        except: Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42)
        pw=float((ytr==0).sum()/max((ytr==1).sum(),1))
        m=xgb.XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,scale_pos_weight=pw,
            eval_metric="auc",random_state=42,n_jobs=-1)
        m.fit(Xtr,ytr,eval_set=[(Xte,yte)],verbose=False)
        yp=m.predict(Xte); ypr=m.predict_proba(Xte)[:,1]
        try: auc=float(roc_auc_score(yte,ypr))
        except: auc=0.5
        f1=float(f1_score(yte,yp,zero_division=0))
        sen=float(yp[yte==1].mean()) if yte.sum()>0 else 0.0
        spe=float((yp[yte==0]==0).mean()) if (yte==0).sum()>0 else 0.0
        if self.backbone is not None and disease_stage is None:
            old=self._eval(Xte,yte)
            if auc<old-0.02:
                return TrainResult(success=False,reason="performance_degraded",auc_roc=auc)
        nv=self._next_ver()
        if disease_stage is None:
            self.backbone=m; self.explainer=shap.TreeExplainer(m)
        else:
            self.stratum_models[disease_stage]=m; self._counts[disease_stage]=len(df)
        self.version=nv
        mp=self.model_dir/f"model_{nv}.pkl"
        joblib.dump({"backbone":self.backbone,"stratum_models":self.stratum_models,
                     "version":nv,"counts":self._counts},str(mp))
        ps={}
        if "disease_stage" in df.columns:
            for st in df["disease_stage"].unique():
                mask=df["disease_stage"]==st
                if mask.sum()>=10:
                    try: ps[st]=round(float(roc_auc_score(y[mask],m.predict_proba(X[mask])[:,1])),3)
                    except: pass
        fi=dict(zip(FEATURE_NAMES,m.feature_importances_.tolist()))
        logger.info("Model saved:%s AUC=%.3f sens=%.3f spec=%.3f",nv,auc,sen,spe)
        return TrainResult(success=True,version=nv,auc_roc=round(auc,4),f1_score=round(f1,4),
            sensitivity=round(sen,4),specificity=round(spe,4),train_samples=len(Xtr),
            test_samples=len(Xte),feature_importances=fi,per_stratum_auc=ps,model_path=str(mp))

    def _load_or_pretrain(self):
        saved=sorted(self.model_dir.glob("model_v*.pkl"),key=_ver_key,reverse=True)
        if saved:
            try:
                d=joblib.load(str(saved[0]))
                self.backbone=d["backbone"]; self.stratum_models=d.get("stratum_models",{})
                self.version=d.get("version","v1.0"); self._counts=d.get("counts",self._counts)
                self.explainer=shap.TreeExplainer(self.backbone)
                logger.info("Model loaded:%s",self.version); return
            except Exception as e: logger.warning("Load failed:%s",e)
        logger.info("Synthetic pretrain...")
        self._pretrain()

    def _pretrain(self):
        np.random.seed(42); rows=[]
        cfgs={"fibrosis_early":(150,0.0,0.15,0.05),"fibrosis_advanced":(120,0.1,0.20,0.10),
              "cirrhosis_compensated":(200,0.3,0.30,0.25),"cirrhosis_decompensated":(150,0.5,0.45,0.40),
              "post_transplant":(100,0.2,0.25,0.20),"chronic_hepatitis":(100,0.1,0.15,0.10)}
        for stage,(n,hep,jo,hn) in cfgs.items():
            nh=int(n*hep)+1; nn=n-nh
            for lbl,cnt,jm,sm,hm in [(0,nn,0.55+jo,2.2+jo*2,23.0-hn*5),(1,nh,1.30+jo,4.5+jo*2,17.0-hn*5)]:
                for _ in range(max(cnt,1)):
                    r={"jitter_local":max(0,np.random.normal(jm,0.3)),
                       "jitter_ppq5":max(0,np.random.normal(jm*0.9,0.25)),
                       "shimmer_local":max(0,np.random.normal(sm,0.7)),
                       "shimmer_apq11":max(0,np.random.normal(sm*0.85,0.6)),
                       "hnr":np.random.normal(hm,2.5),
                       "nhr":max(0,1/(hm+1)+np.random.normal(0,0.02)),
                       "dvb":max(0,min(1,np.random.normal(lbl*0.08,0.03))),
                       "spi":max(0,np.random.normal(2.0+lbl*0.5,0.5)),
                       "speech_rate":max(1,np.random.normal(5.0-lbl*0.6,0.5)),
                       "articulation_rate":max(1,np.random.normal(5.5-lbl*0.7,0.5)),
                       "pause_rate":max(0,np.random.normal(0.3+lbl*0.15,0.1)),
                       "mean_pause_duration":max(0,np.random.normal(0.3+lbl*0.2,0.1)),
                       "pause_to_speech_ratio":max(0,np.random.normal(0.1+lbl*0.1,0.05)),
                       "f0_mean":np.random.normal(130,15),"f0_std":max(0,np.random.normal(8+lbl*2,2)),
                       "f0_range":max(0,np.random.normal(60+lbl*10,10)),"f0_slope":np.random.normal(lbl*0.02,0.01),
                       "tremor_index":max(0,np.random.normal(lbl*0.15,0.05)),
                       "voiced_fraction":max(0,min(1,np.random.normal(0.85-lbl*0.05,0.05))),
                       "jitter_deviation":np.random.normal(lbl*40,15),
                       "shimmer_deviation":np.random.normal(lbl*35,12),
                       "hnr_deviation":np.random.normal(lbl*-20,8),
                       "composite_deviation":max(0,np.random.normal(lbl*30,10)),
                       "label":lbl,"disease_stage":stage}
                    rows.append(r)
        df=pd.DataFrame(rows)
        res=self.retrain(df)
        if res.success:
            self.version="v1.0-pretrain"
            logger.info("Pretrain OK AUC=%.3f",res.auc_roc)
            for st in cfgs:
                sdf=df[df["disease_stage"]==st]
                if len(sdf)>=MIN_SAMPLES_STRATUM and sdf["label"].sum()>=3:
                    sr=self.retrain(sdf,disease_stage=st)
                    if sr.success: logger.info("  strata %s AUC=%.3f",st,sr.auc_roc)

    def _arr(self,f:dict)->np.ndarray:
        return np.array([[float(f.get(n,0)) for n in FEATURE_NAMES]],dtype=np.float32)

    def _explain(self,X,features,rs):
        try:
            sv=self.explainer.shap_values(X)
            if isinstance(sv,list): sv=sv[1]
            sv=sv[0]
            sd={n:round(float(v),4) for n,v in zip(FEATURE_NAMES,sv)}
            sf=sorted(sd.items(),key=lambda x:abs(x[1]),reverse=True)[:4]
            lbs={"jitter_local":"Jitter","jitter_ppq5":"Jitter (PPQ5)","shimmer_local":"Shimmer",
                 "shimmer_apq11":"Shimmer (APQ11)","hnr":"HNR","nhr":"NHR",
                 "speech_rate":"Nutq tezligi","tremor_index":"Ovoz tremori",
                 "jitter_deviation":"Jitter og'ishi","shimmer_deviation":"Shimmer og'ishi",
                 "hnr_deviation":"HNR og'ishi","composite_deviation":"Umumiy og'ish",
                 "dvb":"Ovoz uzilishlari","spi":"Yumshoq fonatsiya (SPI)","mean_pause_duration":"Pauza davomiyligi",
                 "pause_rate":"Pauzalar soni","pause_to_speech_ratio":"Pauza/nutq nisbati",
                 "articulation_rate":"Artikulyatsiya","f0_mean":"F0 o'rtacha","f0_std":"F0 tebranishi",
                 "f0_range":"F0 diapazoni","f0_slope":"F0 trend","voiced_fraction":"Ovozli ulush"}
            # Ulush = barcha SHAP mutlaq qiymatlari ichidagi foiz (ilgari |shap|*100 — 146% kabi chiqardi)
            tot=sum(abs(v) for v in sd.values())+1e-10
            tf=[{"feature":n,"label":lbs.get(n,n),"contribution":round(abs(v)/tot*100,1),
                 "direction":"oshirdi" if v>0 else "kamaytirdi",
                 "raw_value":round(float(features.get(n,0)),3)} for n,v in sf]
            pts=[f"{f['label']} ({f['contribution']:.0f}%)" for f in tf[:3]]
            exp=f"Asosiy omillar: {', '.join(pts)}. Xavf: {rs*100:.0f}%."
            return sd,tf,exp
        except Exception as e:
            logger.warning("SHAP:%s",e); return {},[],"Xavf: %.0f%%."%(rs*100)

    @staticmethod
    def _trend(s,h):
        if len(h)<3: return "stable",0.0
        r=[float(x) for x in h[-3:]]
        sl=(r[-1]-r[0])/len(r)
        if sl>0.05: return "deteriorating",abs(sl)
        if sl<-0.05: return "improving",abs(sl)
        return "stable",abs(sl)

    def _eval(self,Xte,yte):
        try: return float(roc_auc_score(yte,self.backbone.predict_proba(Xte)[:,1]))
        except: return 0.0

    def _next_ver(self):
        try:
            b=self.version.split("-")[0].lstrip("v").split(".")
            return f"v{b[0]}.{int(b[1])+1}"
        except: return "v1.1"

_model:Optional[StratifiedHEModel]=None
_model_file:Optional[str]=None
def _newest_model_file()->Optional[str]:
    fs=sorted(Path("models").glob("model_v*.pkl"),key=_ver_key)
    return str(fs[-1]) if fs else None
def get_model()->StratifiedHEModel:
    """Celery worker qayta o'qitgan yangi model faylini API ham avtomatik oladi
    (ilgari API restart qilinmaguncha eski model ishlardi)."""
    global _model,_model_file
    newest=_newest_model_file()
    if _model is None or (newest and newest!=_model_file):
        _model=StratifiedHEModel(); _model_file=newest
    return _model

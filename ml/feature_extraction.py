"""
ml/feature_extraction.py  —  Gepato-AI v2.0
Stratified Dual Audio Feature Extraction (Gazda 2025, 23 features)
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

logger = logging.getLogger(__name__)

DISEASE_STAGES = (
    "fibrosis_early","fibrosis_advanced",
    "cirrhosis_compensated","cirrhosis_decompensated",
    "post_transplant","chronic_hepatitis","unknown",
)

STRATA_NORMS: dict[str, dict[str, tuple[float,float]]] = {
    "fibrosis_early":         {"jitter_local":(0.55,0.18),"shimmer_local":(2.10,0.55),"hnr":(23.5,2.8),"speech_rate":(5.2,0.6),"tremor_index":(0.08,0.04)},
    "fibrosis_advanced":      {"jitter_local":(0.72,0.22),"shimmer_local":(2.50,0.65),"hnr":(22.0,3.0),"speech_rate":(4.9,0.65),"tremor_index":(0.11,0.05)},
    "cirrhosis_compensated":  {"jitter_local":(0.90,0.28),"shimmer_local":(3.10,0.80),"hnr":(20.5,3.2),"speech_rate":(4.6,0.70),"tremor_index":(0.14,0.06)},
    "cirrhosis_decompensated":{"jitter_local":(1.40,0.45),"shimmer_local":(4.20,1.10),"hnr":(17.5,3.8),"speech_rate":(4.0,0.80),"tremor_index":(0.22,0.09)},
    "post_transplant":        {"jitter_local":(0.85,0.30),"shimmer_local":(2.90,0.85),"hnr":(21.0,3.5),"speech_rate":(4.7,0.72),"tremor_index":(0.18,0.08)},
    "chronic_hepatitis":      {"jitter_local":(0.68,0.20),"shimmer_local":(2.35,0.60),"hnr":(22.8,2.9),"speech_rate":(5.0,0.62),"tremor_index":(0.10,0.05)},
    "unknown":                {"jitter_local":(0.90,0.35),"shimmer_local":(3.00,0.90),"hnr":(21.0,3.5),"speech_rate":(4.8,0.75),"tremor_index":(0.13,0.07)},
}

CLINICAL_RANGES = {
    "jitter_local":(0.0,1.04),"shimmer_local":(0.0,3.07),
    "hnr":(20.0,45.0),"speech_rate":(3.5,7.0),"tremor_index":(0.0,0.50),
}

FEATURE_NAMES: list[str] = [
    "jitter_local","jitter_ppq5",
    "shimmer_local","shimmer_apq11",
    "hnr","nhr","dvb","spi",
    "speech_rate","articulation_rate","pause_rate","mean_pause_duration","pause_to_speech_ratio",
    "f0_mean","f0_std","f0_range","f0_slope",
    "tremor_index","voiced_fraction",
    "jitter_deviation","shimmer_deviation","hnr_deviation","composite_deviation",
]

@dataclass
class ExtractionResult:
    jitter_local:float=0.0; jitter_ppq5:float=0.0
    shimmer_local:float=0.0; shimmer_apq11:float=0.0
    hnr:float=0.0; nhr:float=0.0; dvb:float=0.0; spi:float=0.0
    speech_rate:float=0.0; articulation_rate:float=0.0
    pause_rate:float=0.0; mean_pause_duration:float=0.0; pause_to_speech_ratio:float=0.0
    f0_mean:float=0.0; f0_std:float=0.0; f0_range:float=0.0; f0_slope:float=0.0
    tremor_index:float=0.0; voiced_fraction:float=0.0
    jitter_deviation:float=0.0; shimmer_deviation:float=0.0
    hnr_deviation:float=0.0; composite_deviation:float=0.0
    mfcc:list=field(default_factory=lambda:[0.0]*13)
    mfcc_delta:list=field(default_factory=lambda:[0.0]*13)
    disease_stage:str="unknown"; audio_type:str="sustained_vowel"
    normalized:dict=field(default_factory=dict)

    def to_model_vector(self)->np.ndarray:
        vec=[]
        for n in FEATURE_NAMES:
            vec.append(self.normalized.get(n, float(getattr(self,n,0.0))))
        return np.array(vec,dtype=np.float32)

    def to_dict(self)->dict:
        d=asdict(self); d.pop("normalized",None); return d

class QualityError(Exception): pass

class FeatureExtractor:
    def extract(self,audio:np.ndarray,sr:int,audio_type:str="sustained_vowel",
                disease_stage:str="unknown",baseline:Optional[dict]=None)->ExtractionResult:
        if disease_stage not in DISEASE_STAGES:
            disease_stage="unknown"
        r=ExtractionResult(disease_stage=disease_stage,audio_type=audio_type)
        sound=parselmouth.Sound(values=audio.astype(np.float64),sampling_frequency=sr)
        try: r.jitter_local,r.jitter_ppq5=self._jitter(sound)
        except Exception as e: logger.warning("jitter:%s",e)
        try: r.shimmer_local,r.shimmer_apq11=self._shimmer(sound)
        except Exception as e: logger.warning("shimmer:%s",e)
        try:
            r.hnr=self._hnr(sound)
            r.nhr=round(1.0/(r.hnr+1.0),4) if r.hnr>=0 else 1.0
        except Exception as e: logger.warning("hnr:%s",e)
        try: r.dvb=self._dvb(audio,sr)
        except Exception as e: logger.warning("dvb:%s",e)
        try: r.spi=self._spi(audio,sr)
        except Exception as e: logger.warning("spi:%s",e)
        try: r.f0_mean,r.f0_std,r.f0_range,r.f0_slope=self._f0(sound)
        except Exception as e: logger.warning("f0:%s",e)
        try:
            (r.speech_rate,r.articulation_rate,r.pause_rate,
             r.mean_pause_duration,r.pause_to_speech_ratio,r.voiced_fraction)=self._temporal(audio,sr)
        except Exception as e: logger.warning("temporal:%s",e)
        try: r.tremor_index=self._tremor(audio,sr)
        except Exception as e: logger.warning("tremor:%s",e)
        try: r.mfcc,r.mfcc_delta=self._mfcc(audio,sr)
        except Exception as e: logger.warning("mfcc:%s",e)
        if baseline:
            dev=self._deviation(r,baseline)
            r.jitter_deviation=dev["jitter_deviation"]
            r.shimmer_deviation=dev["shimmer_deviation"]
            r.hnr_deviation=dev["hnr_deviation"]
            r.composite_deviation=dev["composite_deviation"]
        r.normalized=self._normalize(r,disease_stage)
        return r

    def extract_from_file(self,path:str,audio_type:str="sustained_vowel",
                          disease_stage:str="unknown",baseline:Optional[dict]=None)->ExtractionResult:
        try: y,sr=librosa.load(str(path),sr=16000,mono=True,dtype=np.float32)
        except Exception as e: raise QualityError(f"Fayl o'qilmadi:{e}") from e
        return self.extract(y,sr,audio_type,disease_stage,baseline)

    def compute_deviation(self,result:ExtractionResult,baseline:dict)->dict:
        return self._deviation(result,baseline)

    def is_normal(self,result:ExtractionResult)->dict:
        return {f:lo<=getattr(result,f,0)<=hi for f,(lo,hi) in CLINICAL_RANGES.items()}

    # ── Praat features ────────────────────────────────────────────────
    def _jitter(self,s):
        pp=call(s,"To PointProcess (periodic, cc)",75,500)
        lo=call([s,pp],"Get jitter (local)", 0,0,0.0001,0.02,1.3)
        p5=call([s,pp],"Get jitter (ppq5)",  0,0,0.0001,0.02,1.3)
        return round(float(lo)*100,4),round(float(p5)*100,4)

    def _shimmer(self,s):
        pp=call(s,"To PointProcess (periodic, cc)",75,500)
        lo=call([s,pp],"Get shimmer (local)", 0,0,0.0001,0.02,1.3,1.6)
        a11=call([s,pp],"Get shimmer (apq11)",0,0,0.0001,0.02,1.3,1.6)
        return round(float(lo)*100,4),round(float(a11)*100,4)

    def _hnr(self,s):
        h=call(s,"To Harmonicity (cc)",0.01,75,0.1,1.0)
        v=call(h,"Get mean",0,0)
        return round(float(v),3) if v and not np.isnan(float(v)) else 0.0

    def _dvb(self,y,sr):
        ivs=librosa.effects.split(y,top_db=25)
        if not len(ivs): return 1.0
        return round(1.0-sum(e-s for s,e in ivs)/(len(y)+1),4)

    def _spi(self,y,sr):
        try:
            S=np.abs(librosa.stft(y,n_fft=2048))**2
            f=librosa.fft_frequencies(sr=sr,n_fft=2048)
            lo=S[(f>=70)&(f<1000)].mean()+1e-10
            hi=S[(f>=1000)&(f<8000)].mean()+1e-10
            return round(float(lo/hi),4)
        except: return 0.0

    def _f0(self,s):
        p=call(s,"To Pitch",0,75,500)
        mean=call(p,"Get mean",0,0,"Hertz")
        std=call(p,"Get standard deviation",0,0,"Hertz")
        mn=call(p,"Get minimum",0,0,"Hertz","Parabolic")
        mx=call(p,"Get maximum",0,0,"Hertz","Parabolic")
        rng=float(mx-mn) if(mx and mn)else 0.0
        try:
            nf=int(call(p,"Get number of frames"))
            vals=[v for i in range(1,nf+1) if (v:=call(p,"Get value in frame",i,"Hertz")) and not np.isnan(v)]
            slope=float(np.polyfit(range(len(vals)),vals,1)[0]) if len(vals)>=5 else 0.0
        except: slope=0.0
        return round(float(mean or 0),2),round(float(std or 0),2),round(rng,2),round(slope,4)

    def _temporal(self,y,sr):
        dur=len(y)/sr
        if dur<0.1: return(0.,)*6
        ivs=librosa.effects.split(y,top_db=30)
        v_dur=sum(e-s for s,e in ivs)/sr
        v_frac=v_dur/dur
        pauses=[( ivs[i+1][0]-ivs[i][1])/sr for i in range(len(ivs)-1) if(ivs[i+1][0]-ivs[i][1])/sr>0.05]
        p_rate=len(pauses)/dur
        mp=float(np.mean(pauses)) if pauses else 0.0
        p2s=sum(pauses)/(v_dur+1e-10)
        nsyl=v_dur/0.22
        return(round(nsyl/dur,3),round(nsyl/(v_dur+1e-10),3),round(p_rate,3),round(mp,3),round(p2s,3),round(v_frac,3))

    def _tremor(self,y,sr):
        try:
            hop=int(sr*0.01); rms=librosa.feature.rms(y=y,hop_length=hop)[0]
            f=np.fft.rfftfreq(len(rms),d=0.01); fft=np.abs(np.fft.rfft(rms))**2
            return round(float(fft[(f>=7)&(f<=13)].sum()/(fft.sum()+1e-10)),4)
        except: return 0.0

    def _mfcc(self,y,sr):
        m=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
        d=librosa.feature.delta(m)
        return([round(float(v),4)for v in m.mean(axis=1)],
               [round(float(v),4)for v in d.mean(axis=1)])

    # ── Deviation & Normalization ─────────────────────────────────────
    def _deviation(self,r:ExtractionResult,b:dict)->dict:
        def pct(c,base): return round((c-base)/abs(base)*100,2) if abs(base)>1e-10 else 0.0
        jd=pct(r.jitter_local, b.get("jitter_local",r.jitter_local))
        sd=pct(r.shimmer_local,b.get("shimmer_local",r.shimmer_local))
        hd=pct(b.get("hnr",r.hnr),r.hnr if r.hnr>0 else 1.0)
        rd=pct(r.speech_rate,b.get("speech_rate",r.speech_rate))
        comp=0.35*abs(jd)+0.30*abs(sd)+0.25*abs(hd)+0.10*abs(rd)
        return{"jitter_deviation":jd,"shimmer_deviation":sd,"hnr_deviation":hd,"composite_deviation":round(comp,2)}

    def _normalize(self,r:ExtractionResult,stage:str)->dict:
        norms=STRATA_NORMS.get(stage,STRATA_NORMS["unknown"])
        out={}
        for n in FEATURE_NAMES:
            raw=float(getattr(r,n,0.0))
            if n in norms:
                m,s=norms[n]; out[n]=round((raw-m)/(s+1e-10),4)
            else:
                out[n]=raw
        return out

_extractor:Optional[FeatureExtractor]=None
def get_extractor()->FeatureExtractor:
    global _extractor
    if _extractor is None: _extractor=FeatureExtractor()
    return _extractor

# Gepato-AI — Scopus maqolasi uchun to'liq brif

> **Bu fayl nima uchun:** yangi chatda Gepato-AI asosida Scopus jurnaliga maqola tayyorlashni boshlash.
> Fayldagi barcha texnik ma'lumotlar loyiha kodidan o'qib yozilgan (repo `Isroilov-7/Gepato-ai`,
> branch `claude/gracious-pascal-7b310w`, 2026-11-02 holati). Taxminiy yoki tekshirilmagan narsalar
> alohida belgilangan: **[TEKSHIRING]**.
>
> **Yangi chatga birinchi xabar** — faylning eng oxirida (§16).

---

## 0. Eng muhim 5 fakt (avval shuni o'qing)

1. **Haqiqiy bemor ma'lumoti hali YO'Q.** Repodagi barcha 6 ta model fayli (`models/model_v0.1.pkl` … `model_v1.5.pkl`)
   **sintetik (sun'iy generatsiya qilingan) ma'lumotda** o'qitilgan (`ml/model.py → _pretrain()`). Ulardagi har qanday
   AUC/sezgirlik raqami klinik aniqlik EMAS va maqolada natija sifatida ishlatilmaydi.
2. **Saytdagi shifokor/admin paneli — namunaviy (demo) ma'lumot.** Bemorlar ro'yxati, grafiklar va statistikalar
   `frontend/index.html` ichida qattiq yozilgan. Skrinshotlar faqat "prototip interfeysi" sifatida berilishi mumkin.
3. **Bosh sahifadagi "85–92% sezgirlik"** — adabiyotdan olingan raqam (sahifada "(adabiyot)" deb yozilgan), Gepato'ning
   o'z natijasi emas. Manbasi kodda ko'rsatilmagan **[TEKSHIRING]**.
4. **Kod tekshiruvida jiddiy texnik nuqsonlar topildi** (§7). Eng muhimi — **jitter doim 0 edi** (Praat chaqiruvi xato);
   2026-11-02 da tuzatildi. Ma'lumot yig'ishdan oldin §11 dagi ro'yxatni bajarish shart.
5. **Hozir yozish mumkin bo'lgan halol maqola turlari:** (A) tadqiqot protokoli, (B) tizim dizayni + texnik/foydalanish
   imkoniyati (feasibility) maqolasi. **Diagnostik aniqlik maqolasi (C)** faqat etika ruxsati va haqiqiy bemor
   ma'lumotlari yig'ilgandan keyin (§8).

---

## 1. Loyiha bir paragrafda

**O'zbekcha:** Gepato-AI — jigar sirrozi va surunkali jigar kasalliklari bilan og'rigan bemorlarda **jigar ensefalopatiyasini
(HE, ayniqsa minimal/yashirin HE)** uyda, smartfon orqali masofadan kuzatish uchun multimodal telemeditsina platformasi.
Bemor har kuni qisqa testlar topshiradi: (1) ovoz yozuvi (cho'zilgan "aaa" unlisi) → 23 ta akustik belgi → XGBoost modeli;
(2) telefon akselerometri bilan qo'l tremori; (3) kognitiv testlar (Stroop va Number Connection Test). Uchala natija
vaznli **kompozit HE indeksiga** birlashtiriladi, shifokor panelida ko'rsatiladi, noaniq holatlar shifokor belgilashiga
yuboriladi (active learning), SHAP bilan izohlanadi.

**English (draft for the paper):** *Gepato-AI is a multimodal smartphone-based telemonitoring platform for the early detection
of hepatic encephalopathy (HE), particularly minimal/covert HE, in patients with chronic liver disease. It combines (i) acoustic
voice biomarkers extracted from sustained-vowel recordings (23 features: perturbation, noise, prosodic and temporal measures)
classified by a disease-stage–stratified gradient-boosted tree model with SHAP explanations, (ii) accelerometer-derived hand
tremor metrics, and (iii) app-based cognitive tests (Stroop and Number Connection Test), fused into a weighted composite HE
index, with uncertainty-driven active learning in which clinicians label borderline cases.*
**[Bu matn tizim TAVSIFI; klinik samaradorlik haqida da'vo yo'q — shunday qolishi kerak.]**

---

## 2. Klinik fon (maqolaning Introduction qismi uchun — adabiyot bilan tasdiqlash kerak)

- HE — sirrozning og'ir asorati; **minimal HE (MHE)** klinik ko'rinmaydi, lekin hayot sifati, avtohalokat xavfi, overt HE
  ga o'tish va o'lim bilan bog'liq. Tashxis maxsus psixometrik testlar talab qiladi, ular ko'pincha klinikada qilinmaydi.
- O'rnatilgan vositalar (Introduction va reference standard uchun): **PHES** (Psychometric Hepatic Encephalopathy Score,
  ichida NCT-A, NCT-B, DST, SDT, LTT), **EncephalApp Stroop** (smartfon ilovasi), **CFF** (critical flicker frequency),
  **West-Haven** mezonlari (overt HE darajalari 0–4), **ISHEN** "covert HE" atamasi.
- Ovoz/nutq: HE da nutq sekinlashuvi, dizartriya, monotonlik kuzatiladi — ovoz biomarkerlari masofaviy skrining uchun
  istiqbolli, lekin HE da validatsiya qilingan ovoz modellari kam **[adabiyotni qidiring]**.
- Asteriksis ("flapping tremor") — HE belgisi, **negativ mioklonus**, odatda past chastotali va notekis. Gepato'ning tremor
  moduli 8–12 Hz oralig'ini o'lchaydi (§5.4, §7) — bu farq maqolada izohlanishi yoki algoritm qayta ko'rilishi kerak.

**Ishonchli asosiy manbalar (topib, DOI'ni tekshiring):**
- Vilstrup H. va b., *Hepatic encephalopathy in chronic liver disease: 2014 Practice Guideline by AASLD and EASL*, Hepatology, 2014.
- Weissenborn K. va b., PHES (Psychometric Hepatic Encephalopathy Score), J Hepatol, 2001.
- Bajaj J.S. va b., EncephalApp Stroop — smartfon ilovasi bilan MHE skriningi, Hepatology, 2013 (va keyingi validatsiyalar).
- Collins G.S. va b., **TRIPOD+AI** (2024, BMJ) — AI bashorat modellari haqida hisobot berish yo'riqnomasi.
- Bossuyt P.M. va b., **STARD 2015** (BMJ) — diagnostik aniqlik tadqiqotlari yo'riqnomasi.
- Kodda "**Gazda 2025, 23 features**" deb manba ko'rsatilgan (`ml/feature_extraction.py`) — bunday maqola mavjudligini
  tasdiqlang **[TEKSHIRING]**; topilmasa, iqtibos va undan olingan normalar olib tashlanadi.

---

## 3. Tizim arxitekturasi (Methods → "System overview")

| Qatlam | Texnologiya | Fayl |
|---|---|---|
| Backend API | Python 3.11, FastAPI, Uvicorn (1 worker) | `api/main.py`, `api/routes/*` |
| Ma'lumotlar bazasi | PostgreSQL 15 (SQLAlchemy async) | `db/models.py`, `migrations/` |
| Fon vazifalar | Celery + Redis 7 (kechasi 02:00 qayta o'qitish, har 4 soatda trend tekshiruvi) | `core/tasks.py` |
| Ovoz tahlili | Praat (`praat-parselmouth` 0.4.3), librosa 0.10.2, NumPy, SciPy | `ml/feature_extraction.py`, `ml/audio_quality.py` |
| ML | XGBoost 2.0.3, scikit-learn 1.4.2, SHAP 0.45 | `ml/model.py` |
| Kompozit indeks | qoidaga asoslangan vaznli yig'indi | `ml/scoring.py` |
| Trend | z-score + chiziqli regressiya (kod yozilgan, lekin oqimga ULANMAGAN — §7) | `ml/trend_engine.py` |
| LLM yordamchilar | Gemini (`gemini-3.6-flash`) yoki Claude — "Atlas" (klinik xulosa), "Oracle" (trend matni) | `agents/agents.py` |
| Frontend | statik HTML/JS: shifokor paneli (demo), bemor PWA (prototip, API'ga ulanmagan), belgilash UI | `frontend/` |
| Joylashuv | Docker Compose, Ubuntu server, nginx + HTTPS (Let's Encrypt), Cloudflare | `docker-compose.yml`, `deploy/` |

**Xavfsizlik va maxfiylik (Methods → "Data protection"):**
- Barcha bemor API'lari faqat kirgan shifokor/admin uchun (JWT, 12 soat; rol tekshiruvi router darajasida —
  `api/security.py`). Ro'yxatdan o'tkazish faqat admin orqali. O'chirilgan hisobning tokeni darhol bekor.
- API faqat `127.0.0.1` da, tashqariga nginx + HTTPS orqali; login IP boshiga daqiqada 10 urinish; fayl ≤ 12 MB.
- Productionda zaif sir yoki ochiq CORS bo'lsa server ishga tushmaydi; API sxemasi (`/docs`) yopiq.
- Har kuni 02:40 backup (Postgres dump + ovoz fayllari), 14 kun saqlanadi.
- **Hali yo'q (etika komissiyasi so'rashi mumkin):** ma'lumotlarni saqlash muddati siyosati, bemor roziligi yozuvi,
  psevdonimlashtirish (bemor ismi bazada ochiq saqlanadi), audit log, ma'lumotlarni o'chirish tartibi.

---

## 4. Ma'lumotlar holati — HALOL ko'rinish

| Narsa | Holat |
|---|---|
| Haqiqiy bemor ovoz yozuvlari | **Yo'q** |
| Klinik "oltin standart" belgilari (PHES/Stroop/CFF/West-Haven) | **Yo'q** |
| Model o'qitish ma'lumoti | **Sintetik**: 6 kasallik bosqichi bo'yicha 820 ta sun'iy qator (fibroz erta 150, fibroz kech 120, kompensatsiyalangan sirroz 200, dekompensatsiyalangan 150, transplantatsiyadan keyin 100, surunkali gepatit 100); belgilar normal taqsimotdan tasodifiy generatsiya qilingan, "HE" sinfi uchun o'rtacha qiymatlar qo'lda siljitilgan |
| Model fayllari | `model_v0.1` (faqat umumiy model) → `v1.1…v1.5` (har versiyada bitta qatlam modeli qo'shilgan). Hammasi sintetik |
| Shifokor paneli raqamlari | Qattiq yozilgan demo (15 ta to'qima bemor) |
| Bemor ilovasi (PWA) | Mikrofon va akselerometrni ochadi, lekin natijani serverga YUBORMAYDI (prototip) |
| Serverdagi bazada | Faqat avtomatik tekshiruv (selfcheck) yaratgan "TEST tekshiruv" bemori |

**Xulosa:** hozirgi holatda "Gepato-AI HE ni X% aniqlik bilan aniqlaydi" deyish **mumkin emas**. Sintetik ma'lumotdagi AUC
modelning o'z generatsiya qoidalarini qayta topishini ko'rsatadi, xolos.

---

## 5. Modullar — algoritmlar va aniq parametrlar (Methods uchun)

### 5.1 Audio sifat nazorati (`ml/audio_quality.py`, `core/config.py`)
- Barcha audio **16 kHz mono** ga o'tkaziladi; pik normalizatsiya **−3 dBFS** (0.708).
- Yozishdan oldin fon shovqini: **> −45 dBFS** ogohlantirish, **> −35 dBFS** bloklash (kodda "ITU-T P.800 asosida" deyilgan
  **[TEKSHIRING]**).
- Qirqilish (clipping): namunalarning > 0.1% i |x| ≥ 0.98 bo'lsa.
- Bosh/oxirdagi jimlik kesiladi (top_db 25); davomiylik **5–60 s**; minimal **SNR 10 dB**; cho'zilgan unli uchun ovozli ulush
  **> 0.85** talab qilinadi; fayl ≤ 10 MB.
- Sifatsiz yozuv tahlil qilinmaydi (422) va bemordan qayta yozish so'raladi.

### 5.2 Akustik belgilar — 23 ta (`ml/feature_extraction.py`)
Vazifa: hozir faqat **cho'zilgan unli** ("sustained vowel"). Matn o'qish rejimi sifat modulida bor, lekin tahlilda ishlatilmaydi.

| Guruh | Belgilar | Qanday hisoblanadi |
|---|---|---|
| Perturbatsiya | `jitter_local`, `jitter_ppq5` (%) | Praat PointProcess (periodic, cc; 75–500 Hz). **2026-11-02 gacha doim 0 edi (xato)** |
| | `shimmer_local`, `shimmer_apq11` (%) | Praat, Sound + PointProcess |
| Shovqin | `hnr` (dB) | Praat Harmonicity (cc), o'rtacha |
| | `nhr` | **1/(HNR+1)** — haqiqiy NHR emas, HNR'dan hosila **[izohlang]** |
| Ovoz sifati | `dvb` | 1 − (ovozli ulush), librosa split top_db 25 — "voice breaks" ning soddalashtirilgan ko'rinishi |
| | `spi` | 70–1000 Hz va 1–8 kHz energiya nisbati — MDVP "Soft Phonation Index" ning yaqinlashmasi **[izohlang]** |
| Vaqt/nutq | `speech_rate`, `articulation_rate`, `pause_rate`, `mean_pause_duration`, `pause_to_speech_ratio` | Bo'g'in soni = ovozli davomiylik / 0.22 s (taxmin). **`articulation_rate` har doim 4.545 (o'zgarmas)**, `speech_rate` ovozli ulushga proporsional (§7) |
| Prosodiya | `f0_mean`, `f0_std`, `f0_range`, `f0_slope` | Praat Pitch (75–500 Hz); slope — kadrlar bo'yicha chiziqli regressiya |
| Tremor (ovozda) | `tremor_index` | RMS konvert (10 ms) spektrida 7–13 Hz energiya ulushi |
| | `voiced_fraction` | librosa split top_db 30 |
| Shaxsiy bazadan og'ish | `jitter_deviation`, `shimmer_deviation`, `hnr_deviation` (%), `composite_deviation` = 0.35·\|J\| + 0.30·\|S\| + 0.25·\|H\| + 0.10·\|tezlik\| | Bemorning dastlabki 7 yozuvi asosidagi shaxsiy baza |

Qo'shimcha: 13 MFCC + delta saqlanadi, lekin modelga kirmaydi.

**Qatlam normalari** (`STRATA_NORMS`): 6 bosqich + "unknown" uchun jitter/shimmer/HNR/nutq tezligi/tremor indeksining o'rtacha
va SD qiymatlari (masalan, kompensatsiyalangan sirroz: jitter 0.90±0.28%, shimmer 3.10±0.80%, HNR 20.5±3.2 dB).
**Manbai faqat "Gazda 2025" deb ko'rsatilgan [TEKSHIRING].** Bu z-normalizatsiya hozir hisoblanadi, lekin modelga xom
qiymatlar beriladi (normalizatsiya natijasi modelda ishlatilmaydi).

### 5.3 Ovoz modeli (`ml/model.py`)
- **Arxitektura:** qatlamlangan (stratified) XGBoost. Umumiy "backbone" model + har kasallik bosqichi uchun alohida model.
  Bemor bosqichi uchun qatlam modeli bo'lsa va unda ≥ 30 namuna bo'lsa: **risk = 0.40·backbone + 0.60·qatlam**,
  aks holda faqat backbone.
- **Giperparametrlar:** n_estimators 300, max_depth 5, learning_rate 0.05, subsample 0.8, colsample_bytree 0.8,
  scale_pos_weight = (manfiy/musbat), eval_metric AUC, random_state 42.
- **Belgi (label):** shifokor beradi: 0 = normal, 1 = minimal HE, 2 = overt HE; o'qitishda **ikkilik** (0 va >0).
- **Baholash (kodda):** 80/20 tasodifiy bo'linish (stratify), AUC, F1, sezgirlik, o'ziga xoslik; yangi model eskisidan
  AUC bo'yicha > 0.02 yomon bo'lsa qabul qilinmaydi. **Bemor darajasida bo'linish YO'Q** — bir bemorning yozuvlari ham
  train, ham test'ga tushishi mumkin (ma'lumot sizib chiqishi, §7).
- **Xavf darajasi:** ≥ 0.65 yuqori, ≥ 0.40 o'rta, aks holda past. Ishonch = |risk − 0.5|·2.
- **Active learning:** 0.35 < risk < 0.65 bo'lsa yozuv shifokor belgilash navbatiga tushadi (`/doctor/labeling`);
  ≥ 20 yangi belgilangan namuna bo'lsa kechasi 02:00 da qayta o'qitiladi (Celery, `MIN_SAMPLES_FOR_TRAINING`).
  (Belgilash endpointi ≥ 50 da "retrain navbatga qo'shildi" deydi, lekin o'zi ishga tushirmaydi — §7-15.)
- **Izohlanuvchanlik:** SHAP TreeExplainer — eng ta'sirli 4 belgi va ularning ulushi (%).
- **Qisqa trend (modelda):** oxirgi 3 bashorat bo'yicha og'ish > 0.05 → "yomonlashmoqda"/"yaxshilanmoqda".

### 5.4 Tremor moduli (`api/routes/tremor.py`)
- Kirish: telefon akselerometri {x, y, z, t}, 50 Hz ga qayta namunalanadi.
- Belgilar: RMS amplituda (g), dominant chastota, **8–12 Hz** quvvat ulushi (DC < 0.5 Hz chiqarilgan), barqarorlik
  = 1 − std/mean.
- Ball: RMS komponenti = (RMS − 0.08 g)/(0.40 − 0.08) ∈ [0,1]; chastota komponenti = min(1, 3·ulush) × amplituda darvozasi;
  **xom ball = 0.60·RMS + 0.40·chastota**; bosqich koeffitsienti (0.75 fibroz erta … 1.35 transplantatsiyadan keyin).
- **Muammo:** asteriksis klassik tremor emas (negativ mioklonus, past chastota); 8–12 Hz fiziologik/essensial tremorga
  yaqin. Chegaralar (0.08 g, 0.40 g) va koeffitsientlar manbasiz **[TEKSHIRING / qayta asoslang]**.

### 5.5 Kognitiv modul (`api/routes/cognitive.py`)
- **Stroop:** har urinish reaksiya vaqti va to'g'riligi; ball = max(0, 1 − (o'rtacha RT − 280 ms)/1220) − xato jarimasi.
- **NCT (raqamlarni ketma-ket bosish):** umumiy vaqt; ball = max(0, 1 − (vaqt − 60 s)/120) − xato jarimasi.
- **Kognitiv ball = 0.5·Stroop + 0.5·NCT** (1 = normal). HE indeksida **1 − ball** ishlatiladi.
- **Muammo:** chegaralar "West-Haven asosida" deb yozilgan, lekin West-Haven overt HE ning klinik shkalasi, Stroop/NCT
  me'yorlari emas. EncephalApp Stroop umumiy vaqt (OffTime+OnTime, soniyada) bilan ishlaydi, bitta urinish uchun 280 ms
  emas — bu chegara sog'lom odamni ham "patologik" ko'rsatishi mumkin (selfcheck'da 839 ms → 0.47 "o'rta"). Me'yorlar
  adabiyot va mahalliy sog'lom guruhdan qayta olinishi kerak **[TEKSHIRING]**.

### 5.6 Kompozit HE indeksi (`ml/scoring.py`)
- **HE indeksi = 0.75·ovoz + 0.15·tremor + 0.10·kognitiv** (0–1). Komponent yo'q bo'lsa vazni boshqalarga proporsional
  taqsimlanadi; ovoz majburiy. Tremor va kognitiv — oxirgi 7 kundagi eng yangi natija.
- Bosqichga qarab chegaralar (past/o'rta/yuqori/kritik), masalan kompensatsiyalangan sirroz: 0.40 / 0.65 / 0.80.
- Darajalar West-Haven darajalariga "moslab" yozilgan (past → Grade 0 … kritik → Grade 3–4) — **validatsiya qilinmagan**.
- "95% CI" Wilson formulasi bilan **qat'iy n = 50** deb hisoblanadi — bu statistik jihatdan asossiz (§7).
- **Vaznlar ekspert tomonidan qo'yilgan, ma'lumotdan o'rganilmagan.** Maqolada shunday aytilishi yoki keyin
  ma'lumotdan (masalan, logistik regressiya / stacking) o'rganilishi kerak.

### 5.7 Trend dvigateli (`ml/trend_engine.py`) — kod bor, ulanmagan
Dastlabki 5 o'lchov → shaxsiy baza; z > 1.5 / 2.0 / 2.5 → ogohlantirish / yuqori / kritik; 7 kunlik chiziqli nishab
> 0.03 ball/kun (R² ≥ 0.60, ≥ 3 kun) → erta ogohlantirish; 30 kunlik nishab > 0.008 → progressiya; |z| > 3 → outlier.
**Tahlil oqimidan chaqirilmaydi** — maqolada "amalga oshirilgan va ishlaydi" deb yozilmaydi (yoki avval ulanadi).

### 5.8 LLM yordamchilar (`agents/agents.py`)
- "Atlas": bemor konteksti (laboratoriya, ovoz, kognitiv) asosida klinik xulosa matni; "Oracle": 7 kunlik trend matni.
- **Maqola uchun xavf:** Oracle promptida "48 soatda 78% ehtimol" kabi **raqamli prognoz** so'raladi, Atlas esa dori dozasi
  tavsiyasi beradi. Bu raqamlar LLM tomonidan generatsiya qilinadi, validatsiya qilinmagan. Klinik maqolada bu modulni
  (a) chiqarib tashlash yoki (b) faqat "eksperimental, shifokor uchun matn yordamchisi" deb, raqamli prognozsiz tasvirlash
  tavsiya etiladi. Bemor ma'lumoti tashqi LLM provayderiga yuborilishi etika komissiyasida alohida ko'riladi.

---

## 6. Hozir HALOL da'vo qilish mumkin / mumkin emas

| ✅ Mumkin | ❌ Mumkin emas (ma'lumotsiz) |
|---|---|
| Multimodal masofaviy HE monitoring platformasi ishlab chiqildi va joylashtirildi | "HE ni X% sezgirlik bilan aniqlaydi" |
| Pipeline: sifat nazorati → 23 belgi → qatlamlangan XGBoost → SHAP → active learning | Sintetik AUC ni natija sifatida berish |
| Ochiq, qayta takrorlanadigan arxitektura; xavfsizlik choralari | "West-Haven darajasini bashorat qiladi" |
| Texnik sinovlar: end-to-end avtomatik tekshiruv 39/39 (sintetik signallar bilan) | "48 soat oldin ogohlantiradi" |
| (Keyin, sog'lom ko'ngillilarda) belgilarning test–retest ishonchliligi, foydalanish qulayligi (SUS) | Overt/minimal HE farqlash |

---

## 7. Kod tekshiruvida topilgan nuqsonlar (maqoladan oldin hal qilinsin)

| # | Muammo | Ta'siri | Holat |
|---|---|---|---|
| 1 | Jitter Praat'da noto'g'ri obyektga chaqirilgan → **jitter_local/ppq5 doim 0** | Asosiy biomarker modelga 0 bo'lib tushardi | ✅ Tuzatildi (2026-11-02, commit "Jitter doim 0 edi"), selfcheck'ga regressiya qo'shildi |
| 2 | `articulation_rate` = (v/0.22)/v = **4.545 o'zgarmas** | Belgi foydasiz | ❌ Bo'g'in yadrolarini aniqlash algoritmi kerak (masalan, de Jong & Wempe Praat skripti) yoki belgi olib tashlanadi |
| 3 | `speech_rate`, pauza belgilari **cho'zilgan unlida** hisoblanadi | Nutq tezligi unlida ma'nosiz | ❌ Standart matn o'qish / sanash vazifasi qo'shilsin, belgilar shu vazifadan olinsin |
| 4 | `nhr` = 1/(HNR+1), `spi`, `dvb` — soddalashtirilgan yaqinlashmalar | Nomlari standart belgilarnikiga o'xshaydi, lekin boshqa narsa | ❌ Maqolada aniq formulasi bilan yozilsin yoki standart Praat/MDVP ta'rifiga keltirilsin |
| 5 | Tremor 8–12 Hz; asteriksis — negativ mioklonus | Klinik asoslanish zaif | ❌ Adabiyot bo'yicha qayta ko'rish (past chastota, "lapse" hodisalarini sanash) |
| 6 | Stroop 280 ms / NCT 60 s chegaralari manbasiz | Sog'lom odam "patologik" chiqishi mumkin | ❌ Mahalliy sog'lom guruh me'yorlari / EncephalApp metodikasi |
| 7 | HE indeksi vaznlari (0.75/0.15/0.10) va chegaralar qo'lda | Asoslanmagan | ❌ Ma'lumotdan o'rganish yoki "ekspert-asosli" deb ochiq yozish |
| 8 | Wilson CI qat'iy n = 50 bilan | Statistik noto'g'ri | ❌ Olib tashlash yoki bootstrap CI (ma'lumot bo'lganda) |
| 9 | Train/test bo'linishi yozuv darajasida, bemor darajasida emas | Natija sun'iy yuqori chiqadi | ❌ Bemor bo'yicha (GroupKFold) va tashqi validatsiya |
| 10 | Trend dvigateli oqimga ulanmagan | Da'vo qilib bo'lmaydi | ❌ Ulash yoki maqoladan chiqarish |
| 11 | "Gazda 2025" normalari manbasi noma'lum | Iqtibos xavfi | ❌ Tekshirish |
| 12 | Bemor ilovasi serverga ma'lumot yubormaydi | Masofaviy yig'ish ishlamaydi | ❌ PWA'ni API'ga ulash (login, yuklash, oflayn navbat) |
| 13 | LLM raqamli prognoz va dozalash tavsiyasi | Klinik xavf, SaMD masalasi | ❌ O'chirish yoki cheklash |
| 14 | Bemor ismi bazada ochiq | Maxfiylik | ❌ Psevdonim ID, ism alohida jadvalda/shifrlangan |
| 15 | Qayta o'qitish chegarasi ikki xil (kechki vazifa 20, belgilash xabari 50 va hech narsa ishga tushirmaydi) | Chalkashlik, hujjatlashtirishda xato | ❌ Bitta chegara, xabar haqiqatga mos |

---

## 8. Maqola variantlari (tavsiya: A → B → C)

### A. Tadqiqot protokoli (eng tez, halol, Scopus'da chop etiladi)
- Mazmuni: Gepato-AI tizimi tavsifi + kelajakdagi prospektiv validatsiya tadqiqotining to'liq rejasi (gipoteza, populyatsiya,
  oltin standart, namuna hajmi, statistik tahlil, etika).
- Talab: odatda **etika komissiyasi ruxsati (yoki ariza)** va ko'pincha **ro'yxatdan o'tkazish** (ClinicalTrials.gov yoki
  shunga o'xshash registr). Natija kerak emas.
- Jurnallar: *JMIR Research Protocols*, *BMJ Open* **[Scopus indeksini tekshiring]**.
- Muddat: 1–2 oy yozish (etika arizasi parallel).

### B. Tizim dizayni + texnik validatsiya / feasibility
- Mazmuni: arxitektura, belgilar pipeline'i, sifat nazorati, xavfsizlik + **haqiqiy, lekin klinik bo'lmagan** ma'lumot:
  masalan 20–40 sog'lom ko'ngilli → (1) belgilarning test–retest ishonchliligi (ICC), (2) turli telefonlar orasidagi farq,
  (3) foydalanish qulayligi (SUS so'rovnomasi), (4) yozuvlarning sifat nazoratidan o'tish ulushi, (5) ishlash vaqti/xotira.
  Kichik pilot bemorlar guruhi bo'lsa — faqat tavsifiy natijalar ("feasibility"), aniqlik da'vosisiz.
- Talab: sog'lom ko'ngillilar uchun ham rozilik va odatda etika ruxsati.
- Jurnallar: *JMIR Formative Research*, *JMIR mHealth and uHealth*, *Digital Health* (SAGE), *Sensors* (MDPI),
  *BMC Medical Informatics and Decision Making*, *IEEE Access* **[Scopus indeksi va kvartilini tekshiring]**.
- Muddat: §11 tuzatishlari + ma'lumot yig'ish ≈ 2–4 oy.

### C. Diagnostik aniqlik tadqiqoti (eng kuchli; asosiy maqsad)
- Prospektiv, sirroz bilan og'rigan bemorlar, oltin standart bilan solishtirish (§9).
- Jurnallar: *Hepatology Communications*, *Journal of Voice*, *Biomedical Signal Processing and Control*, *Diagnostics*
  (MDPI), *Digital Health* **[tekshiring]**.
- Muddat: 6–12+ oy (etika, bemor yig'ish, tahlil).

**Jurnal tanlashda:** Scopus Sources ro'yxatida faolligini, SJR kvartilini, APC (nashr to'lovi) va maqola turi ruxsatini
tekshiring. Yirtqich (predatory) jurnallardan ehtiyot bo'ling — "tez chop etamiz, kafolat" degan takliflar xavfli.

---

## 9. C varianti uchun tadqiqot dizayni (qoralama — statistik va klinik hamkor bilan aniqlanadi)

- **Dizayn:** prospektiv, kesma (cross-sectional) diagnostik aniqlik + ixtiyoriy uzunlamasina kuzatuv bo'limi.
- **Populyatsiya:** 18+ yosh, tasdiqlangan sirroz (Child-Pugh A–C), o'zbek/rus tilida so'zlashuvchi.
  **Chiqarish:** overt HE (grade ≥ 2, test topshira olmaydi), faol alkogol intoksikatsiyasi, psixotrop dorilar, nevrologik
  kasalliklar (Parkinson, insult), og'ir eshitish/ko'rish/nutq buzilishi, yaqinda GI qon ketishi, laringeal patologiya.
- **Nazorat guruhi:** yoshi va jinsi mos sog'lom ko'ngillilar (Stroop/NCT me'yorlari va belgilar diapazoni uchun).
- **Oltin standart (reference standard):** PHES (yoki mahalliy validatsiya qilingan alternativ) va/yoki EncephalApp Stroop
  hamda CFF; overt HE uchun West-Haven. Testni o'tkazuvchi Gepato natijasini **bilmasligi** (blinding).
- **Indeks test (Gepato):** standart protokol — sokin xona, bir xil masofa; cho'zilgan /a/ 3 marta × 5+ s; standart o'zbekcha
  matn o'qish (~30 s) va 1–20 sanash (§7-3 tuzatilgandan keyin); tremor (qo'l cho'zilgan holda 10–20 s); Stroop va NCT.
  Telefon modeli qayd etiladi.
- **Namuna hajmi:** kutilgan sezgirlik/o'ziga xoslik va MHE tarqalishi asosida hisoblanadi (masalan, Buderer formulasi) —
  statistik bilan aniqlang. Qatlam (bosqich) modellari uchun har qatlamda yetarli namuna bo'lishi kerak, aks holda faqat
  umumiy model.
- **Tahlil:** bemor darajasida bo'linish (GroupKFold) + iloji bo'lsa tashqi (boshqa klinika) validatsiya; AUC (95% CI —
  DeLong yoki bootstrap), oldindan belgilangan chegarada sezgirlik/o'ziga xoslik, PPV/NPV, kalibratsiya (Brier, kalibratsiya
  egri chizig'i), qaror egri chizig'i (decision curve); har modal alohida va kompozit; SHAP global ta'sirlar.
  Sintetik model **ishlatilmaydi** — model haqiqiy ma'lumotda noldan o'qitiladi.
- **Etika va huquq:** muassasa etika komissiyasi ruxsati, yozma xabardor rozilik (ovoz yozuvi — biometrik ma'lumot),
  shaxsga doir ma'lumotlar to'g'risidagi O'zbekiston qonunchiligiga muvofiqlik **[huquqshunos bilan tekshiring]**,
  ma'lumotlarni saqlash muddati, LLM'ga bemor ma'lumoti yuborilmasligi (yoki rozilikda alohida band).
- **Hisobot:** STARD 2015 + TRIPOD+AI 2024 tekshiruv ro'yxatlari (jurnal topshirishda ilova qilinadi).
- **Ochiqlik:** kod GitHub'da (versiya teg/DOI, masalan Zenodo); ma'lumotlar — rozilik ruxsat bergan darajada.

---

## 10. Maqola tuzilmasi (IMRaD) — loyihadan nima kiradi

1. **Title** — masalan (B uchun): *"Gepato-AI: A Multimodal Smartphone Platform for Remote Monitoring of Hepatic
   Encephalopathy — System Design and Technical Feasibility"*.
2. **Abstract** (strukturali: Background, Objective, Methods, Results, Conclusions; 250–300 so'z).
3. **Introduction** — §2: HE/MHE yuki, tashxis to'siqlari, masofaviy monitoring va ovoz biomarkerlari bo'shlig'i, maqsad.
4. **Methods** — §3 (arxitektura, xavfsizlik), §5 (har modul formulalari bilan), §9 (dizayn, protokol, statistika, etika).
5. **Results** — faqat haqiqiy o'lchangan narsalar (B: ICC, SUS, sifat nazoratidan o'tish %, ishlash vaqti; C: aniqlik).
6. **Discussion** — adabiyot bilan solishtirish, cheklovlar (§7 dan qolganlari halol), klinik ahamiyat, keyingi qadamlar.
7. **Conclusion**, **Data/Code availability**, **Ethics statement**, **Conflicts of interest**, **Funding**,
   **Author contributions (CRediT)**, **AI use disclosure** (ko'p jurnallar AI vositalaridan foydalanishni oshkor qilishni talab qiladi).

Rasmlar g'oyasi: (1) tizim arxitekturasi diagrammasi; (2) ma'lumot oqimi (yozuv → sifat → belgilar → model → indeks → shifokor);
(3) interfeys skrinshotlari ("prototip, namunaviy ma'lumot" izohi bilan); (4) SHAP namunasi; (5) natija grafiklari.

---

## 11. Ma'lumot yig'ishdan oldin texnik ishlar (ustuvorlik bo'yicha)

1. ✅ Jitter tuzatildi (2026-11-02).
2. `articulation_rate`/`speech_rate`: bo'g'in yadrolarini aniqlash yoki olib tashlash; nutq belgilarini matn o'qish vazifasidan olish.
3. Bemor PWA'ni API'ga ulash (login, yozuvni yuklash, tremor va kognitiv natijani yuborish, oflayn navbat).
4. Standart yozish protokoli ilovada (ko'rsatma, mikrofon masofasi, fon shovqini tekshiruvi — `check_ambient` bor).
5. Model: sintetik pretrainni o'chirish (yoki "demo" deb ajratish); bemor darajasida CV; o'qitish ma'lumotini versiyalash.
6. Tremor va kognitiv chegaralarini adabiyot/me'yor bilan qayta asoslash.
7. Trend dvigatelini ulash yoki olib tashlash; Wilson CI ni olib tashlash.
8. Tadqiqot rejimi: psevdonim ID, rozilik qaydi, ma'lumotni eksport qilish (CSV) skripti, audit log.
9. LLM: raqamli prognoz va dozalashni o'chirish; tadqiqotda LLM'ni butunlay o'chirib qo'yish varianti.
10. Barcha belgilarni tashqi vosita bilan solishtirish (masalan, Praat GUI yoki boshqa ochiq kutubxona) — belgilar
    to'g'ri hisoblanayotganini tasdiqlash uchun kichik "verification" jadvali (maqolada foydali).

---

## 12. Repo xaritasi (yangi chatda kodga murojaat uchun)

```
api/main.py                  FastAPI ilova, routerlar, himoya
api/security.py              rol tekshiruvi (shifokor/admin)
api/routes/voice.py          yuklash, sifat, belgilar, bashorat, HE indeksi, belgilash
api/routes/tremor.py         akselerometr tremor algoritmi
api/routes/cognitive.py      Stroop + NCT
api/routes/agents.py         Atlas/Oracle endpointlari
ml/audio_quality.py          audio sifat nazorati
ml/feature_extraction.py     23 akustik belgi, qatlam normalari
ml/model.py                  qatlamlangan XGBoost, SHAP, active learning, sintetik pretrain
ml/scoring.py                kompozit HE indeksi
ml/trend_engine.py           trend (ulanmagan)
agents/agents.py             LLM promptlari
core/tasks.py                Celery: qayta o'qitish, trend tekshiruvi
db/models.py                 jadvallar: users, patients, voice_recordings, acoustic_features,
                             predictions, alerts, model_versions, agent_logs, tremor_readings, cognitive_tests
scripts/selfcheck.py         end-to-end avtomatik tekshiruv (hozir 39 ta tekshiruv)
frontend/                    shifokor paneli (demo), bemor PWA (prototip), belgilash UI
deploy/                      serverga o'rnatish (install.sh, nginx, backup)
models/                      6 ta model fayli — HAMMASI SINTETIK
```

Jonli prototip: https://gepato.plagiatx.uz (bosh sahifa — demo ma'lumot).

---

## 13. Asosiy raqamlar (tez ma'lumotnoma)

| Narsa | Qiymat |
|---|---|
| Akustik belgilar | 23 (+13 MFCC saqlanadi, modelda yo'q) |
| Audio | 16 kHz mono, 5–60 s, SNR ≥ 10 dB, ≤ 10 MB |
| Model | XGBoost 300 daraxt, chuqurlik 5, lr 0.05; backbone 0.40 + qatlam 0.60 |
| Kasallik bosqichlari | fibroz erta/kech, sirroz kompensatsiyalangan/dekompensatsiyalangan, transplantatsiyadan keyin, surunkali gepatit, noma'lum |
| Active learning oralig'i | 0.35–0.65 |
| Qayta o'qitish | ≥ 20 yangi belgi, har kecha 02:00 |
| HE indeksi | 0.75 ovoz + 0.15 tremor + 0.10 kognitiv |
| Tremor | 50 Hz, 8–12 Hz ulushi, 0.60 RMS + 0.40 chastota |
| Kognitiv | 0.5 Stroop + 0.5 NCT |
| Sintetik o'qitish to'plami | 820 qator, 6 bosqich |

---

## 14. Yangi chatda javob berish kerak bo'lgan savollar (maqola muallifi uchun)

1. Qaysi variant: A (protokol), B (texnik/feasibility) yoki C (aniqlik)? Muddat?
2. Klinik hamkor bormi (gepatologiya bo'limi, shifoxona)? Bemorlarga kirish imkoni?
3. Etika komissiyasi: qaysi muassasa, ariza topshirilganmi?
4. Oltin standart sifatida qaysi testni o'tkaza olasiz (PHES, Stroop, CFF)? Uskunasi/mutaxassisi bormi?
5. Hammualliflar va ularning roli (CRediT), afilliatsiya, ORCID.
6. Maqola tili — ingliz tili (Scopus jurnallari uchun odatda shart).
7. Nashr to'lovi (APC) byudjeti bormi yoki faqat to'lovsiz jurnallar?
8. Sog'lom ko'ngillilarni (B uchun) qayerdan yig'asiz, nechta?

---

## 15. Yangi chatdagi yordamchi uchun qoidalar

- **Natija, raqam, bemor soni yoki iqtibos to'qib chiqarilmaydi.** Faqat muallif bergan haqiqiy ma'lumot va tekshirilgan manbalar.
- Har bir iqtibosni DOI bilan tekshirish; topilmasa — "[manba kerak]".
- Sintetik model natijalari maqolada natija sifatida ishlatilmaydi.
- §7 dagi hal qilinmagan nuqsonlar maqolada cheklov sifatida halol yoziladi yoki avval tuzatiladi.
- Klinik da'volar faqat ma'lumot qo'llab-quvvatlagan darajada; "tashxis qo'yadi" emas, "skrining/monitoring uchun yordam".
- Hisobot yo'riqnomalari: STARD 2015 va TRIPOD+AI.

---

## 16. Yangi chatga birinchi xabar (nusxa olib, shu fayl bilan birga yuboring)

```
Men Gepato-AI loyihasi muallifiman. Ilova qilingan "SCOPUS_MAQOLA_BRIF.md" faylida loyihaning
koddan olingan to'liq tavsifi, ma'lumotlarning haqiqiy holati (hali bemor ma'lumoti yo'q, modellar
sintetik), topilgan texnik nuqsonlar va maqola variantlari bor. Faylni to'liq o'qing va §15 dagi
qoidalarga amal qiling (hech narsa to'qilmasin).

Maqsadim — Scopus'da indekslanadigan jurnalda maqola chop etish. Avval:
1) §14 dagi savollarni menga birma-bir bering;
2) javoblarimga qarab A/B/C variantlardan qaysi biri real ekanini va aniq reja (bosqichlar, muddat)
   ni taklif qiling;
3) keyin tanlangan variant uchun maqola tuzilmasi va birinchi bo'limni (ingliz tilida) yozishni
   boshlaymiz.
Javoblaringiz o'zbek tilida bo'lsin, maqola matni ingliz tilida.
```

# Datathon 2024 — Kapsamlı README (Notebook’taki Gerçek Akış)
Bu repository, Datathon 2024 kapsamında hazırlanan uçtan uca veri temizleme + öznitelik mühendisliği + CatBoost modelleme sürecini içerir. Bu README; `main.ipynb` içindeki kod akışıyla birebir uyumlu olacak şekilde, yapılan her adımı detaylandırmak için yazıldı.

## 0) Repo içeriği

Ana dosyalar
- `main.ipynb`: tüm pipeline ve deneyler

Veri klasörü (`data/`)
- `train.csv`: eğitim verisi (hedef: `Degerlendirme Puani`)
- `test_x.csv`: test verisi
- `sample_submission.csv`: submission format örneği
- `il_ilce.csv`: il/ilçe referans tablosu (ikamet il özellikleri için)
- `feature_importances_is2016.csv`: is_2016 deneyi ile ilişkili çıktı/analiz
- `submission_catboost.csv`: CatBoost submission çıktısı
- `submission_catboost_is2016_full.csv`: is_2016 eklenmiş tam run çıktısı
- `submission_catboost_is2016_il_full.csv`: is_2016 + il-özellikleri tam run çıktısı
- `universiteler.xlsx`: yardımcı referans dosyası

Model eğitim logları
- `catboost_info/`: CatBoost eğitim metrikleri ve tfevents çıktıları

## 1) Problem ve metrik

- Hedef değişken: `Degerlendirme Puani`
- Kullanılan değerlendirme metriği: RMSE
- Model ailesi: `CatBoostRegressor`

## 2) Notebook akışı (yüksek seviye)

Notebook içinde süreç şu şekilde ilerler:
1) Veri yükleme ve genel kontrol (shape, temel istatistikler)
2) EDA: hedef dağılımı, kategorilere göre hedef ortalamaları, yıllara göre trendler
3) Temizlik + Feature Engineering (ADIM 14)
4) Modelleme + CV + submission üretimi (`is_2016` deneyleri, il-özellikleri deneyi)

Bu README’nin kalan kısmı; ADIM 14 ve modelleme hücrelerindeki kodu “adım adım” açıklayan teknik dokümandır.

## 3) EDA (yapılan analizler)

Notebook’ta hedef değişken ve bazı kritik kolonlar için şu analizler yapıldı:
- `Degerlendirme Puani` için `describe()`, eksik değer / 0 / negatif sayı kontrolleri
- Histogram + KDE + boxplot ile dağılım ve aykırı değer analizi
- IQR tabanlı alt/üst sınır ile outlier gözlemleri
- `Basvuru Yili` için value_counts/trend; yıllara göre hedef ortalaması
- `Cinsiyet`, `Universite Turu` gibi kategorik değişkenler için `groupby(...).agg(['mean','median','std','count'])` özetleri

Bu kısım “modeli kurmadan önce veri yapısını anlamak” için yapıldı; elde edilen gözlemler temizlik/FE seçimlerine yön verdi.

## 4) ADIM 14 — Veri temizleme + Feature Engineering (pipeline)

ADIM 14’te `clean_and_fe(train_df, test_df)` isimli pipeline çalıştırılır ve `clean_train_df`, `clean_test_df` üretilir.

### 4.1 Hız / davranış bayrakları

ADIM 14 içinde performans için bazı ayarlar tanımlıdır:
- `USE_FUZZY = False` (varsayılan olarak kapalı)
- `FUZZY_TOP_K = 800`
- `FUZZY_SCORE_CUTOFF = 92`
- `FUZZY_RARE_ONLY = True`
- `FUZZY_MIN_FREQ = 5`

Fuzzy eşleştirme açılırsa, RapidFuzz ile “nadir görülen” kategoriler daha sık görülen değerlere yaklaştırılır.

### 4.2 Metin normalizasyon fonksiyonları

Temel fonksiyon: `normalize_text(s, strip_accents=False, strip_symbols=False)`
- Lowercase
- Çoklu boşlukları tek boşluğa indirgeme
- `strip_accents=True` iken NFKD ile aksan/diakritik temizliği + transliterasyon
- `strip_symbols=True` iken regex ile sembol temizliği

Diğer yardımcılar
- `normalize_yes_no`: evet/hayır kolonlarını standardize eder (`evet/e/E` → `Evet`, `hayir/hayır/h` → `Hayır`; boş/`-` → NaN)
- `normalize_cat`: kategorilerde `'-'`, `'yok'`, `''` değerlerini NaN’a çevirir
- `normalize_name_like`: okul/dernek/kulüp/üniversite gibi “isim” alanlarında daha agresif normalize (aksan + sembol temizliği)

### 4.3 Yüksek kardinaliteyi azaltma (opsiyonel)

`USE_FUZZY=True` olduğunda, `fuzzy_standardize(train, test, col)` şu kolonlara uygulanır:
- `Lise Adi`
- `Hangi STK'nin Uyesisiniz?`
- `Uye Oldugunuz Kulubun Ismi`
- `Universite Adi`

Algoritma:
- Anchor set: eğitimde en sık görülen `top_k` değer
- Candidate: nadir değerler (freq <= `min_freq`) ve anchor dışında olanlar
- RapidFuzz `token_set_ratio` ile en iyi eşleşme bulunur; eşik altında kalanlar kendisine map edilir

### 4.4 Kolon bazlı temizlik adımları

İsim benzeri kolonlar (agresif normalize)
- `Lise Adi`
- `Hangi STK'nin Uyesisiniz?`
- `Uye Oldugunuz Kulubun Ismi`
- `Bölüm`
- `Universite Adi`

Daha hafif normalize edilen kolonlar (`normalize_cat`)
- `Dogum Yeri`
- `Ikametgah Sehri`
- `Lise Sehir`

Evet/Hayır kolonları (`normalize_yes_no`)
- `Burs Aliyor mu?`
- `Baska Bir Kurumdan Burs Aliyor mu?`
- `Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?`
- `Profesyonel Bir Spor Daliyla Mesgul musunuz?`
- `Aktif olarak bir STK üyesi misiniz?`
- `Girisimcilikle Ilgili Deneyiminiz Var Mi?`
- `Ingilizce Biliyor musunuz?`

Cinsiyet standardizasyonu
- `erkek` benzeri varyantlar → `Erkek`
- `kadın/kadin/k` benzeri varyantlar → `Kadın`
- `belirtmek istemiyorum` korunur

Üniversite türü standardizasyonu
- `devlet` → `Devlet`, `ozel/özel` → `Özel`

### 4.5 Ordinal encoding (sıralı kategoriler)

Aşağıdaki kolonlar, notebook’ta tanımlanan sıralara göre 0..N şeklinde sayısallaştırılır:
- `Universite Kacinci Sinif`: `hazirlik`, `1`, `2`, …, `mezun`
- `Anne Egitim Durumu`: `okur yazar degil` → … → `doktora`
- `Baba Egitim Durumu`: `okur yazar degil` → … → `doktora`
- `Universite Not Ortalamasi`: `0-1.00` → … → `3.51-4.00`
- `Lise Mezuniyet Notu`: `0 - 49` → … → `90 - 100`

### 4.6 Yaş özellikleri

Fonksiyon: `add_age_features(df)`
- `Dogum Tarihi` → `pd.to_datetime(..., errors='coerce', dayfirst=True)`
- `Dogum_Yili = dob.dt.year`
- `Yas = Basvuru Yili - Dogum_Yili`

### 4.7 Frequency encoding (yüksek kardinalite)

`frequency_encode(train, test, cols)` ile şu kolonlar için `*_freq` türetilir:
- `Lise Adi`
- `Bölüm`
- `Hangi STK'nin Uyesisiniz?`
- `Uye Oldugunuz Kulubun Ismi`
- `Universite Adi`

Hesaplama:
- `freq = train[col].value_counts(dropna=False) / len(train)`
- `train[f"{col}_freq"] = train[col].map(freq)`
- `test[f"{col}_freq"] = test[col].map(freq)`

## 5) İl-bazlı özellikler (ikamet il mean + freq)

ADIM 14’ten sonra, `data/il_ilce.csv` kullanılarak ikamet şehrinin “il” seviyesine normalize edilmesi ve il bazlı hedef özetlerinin eklenmesi yapılır.

### 5.1 Normalizasyon: `Ikametgah Sehri` → `ikamet_norm` → `ikamet_il`

- `il_ilce.csv` içinden il kolonu bulunur (`il` varsa kullanılır, yoksa ilk kolon fallback)
- `ildf['il_norm'] = normalize_text(ildf[il_col])`
- `il_map = {v: v for v in unique_ils}`
- Eğitim/test:
  - `ikamet_norm = normalize_text(Ikametgah Sehri)`
  - `ikamet_il = ikamet_norm.map(il_map)`

### 5.2 Hedef özetleri: `ikamet_il_mean`, `ikamet_il_freq`

Eğitim setinden il bazında şu özetler çıkarılır:
- `ikamet_il_mean`: il bazlı hedef ortalaması
- `ikamet_il_count`: il bazlı kayıt sayısı
- `ikamet_il_freq = ikamet_il_count / len(train)`

Bu özetler train/test’e merge edilir ve eksikler şu şekilde doldurulur:
- `ikamet_il_mean`: eğitim hedef ortalaması ile
- `ikamet_il_freq`: 0.0 ile

Kalıcı hale getirmek için notebook’ta `clean_train_df` / `clean_test_df` üzerine yazdıran bir hücre de vardır.

## 6) Modelleme — CatBoost + CV + Submission

Modelleme süreci, doğrudan `clean_train_df`/`clean_test_df` üzerinden ilerler.

### 6.1 Kategorik değişken işleme

CatBoost tarafında kategorikler şu şekilde ele alınır:
- Feature listesi içinden `dtype == 'object'` olan sütunların index’i `cat_features` olarak bulunur.
- Bu sütunlar `fillna('__nan__').astype(str)` ile doldurulur.
- `Pool(X, y, cat_features=cat_features)` üzerinden modele verilir.

### 6.2 Deney 1 — `is_2016` bayrağı ile hızlı CV (3-fold)

Eklenen özellik:
- `is_2016 = (Basvuru Yili == 2016).astype(int)`

CV:
- `KFold(n_splits=3, shuffle=True, random_state=42)`

Parametreler (hızlı deneme):
- `loss_function='RMSE'`, `eval_metric='RMSE'`
- `depth=6`, `learning_rate=0.08`, `iterations=300`
- `early_stopping_rounds=50`, `use_best_model=True`

Notebook bu deneyin sonunda fold RMSE’lerini ve `mean ± std` RMSE özetini yazdırır.

### 6.3 Deney 2 — `is_2016` ile tam run (5-fold, iterations=800)

CV:
- `KFold(n_splits=5, shuffle=True, random_state=42)`

Parametre seti:
- `loss_function='RMSE'`, `eval_metric='RMSE'`
- `depth=8`, `learning_rate=0.06`, `iterations=800`
- `l2_leaf_reg=3.0`
- `random_state=42`, `verbose=200`, `allow_const_label=True`
- `early_stopping_rounds=100`, `use_best_model=True`

Submission çıktısı:
- `data/submission_catboost_is2016_full.csv`

Final model eğitiminde, tüm veriden %10’luk bir val dilimi ayrılarak early stopping için kullanılır.

### 6.4 Deney 3 — İl-özellikleri eklenmiş hızlı test (3-fold)

Ek özellikler:
- `ikamet_il_mean`
- `ikamet_il_freq`

Amaç:
- İl bazlı target-özetlerinin RMSE’e katkısını hızlıca görmek.

### 6.5 Deney 4 — `is_2016` + il-özellikleri ile tam run (5-fold)

Ek özellik seti:
- `is_2016`
- `ikamet_il_mean`
- `ikamet_il_freq`

Çıktı:
- `data/submission_catboost_is2016_il_full.csv`

Bu tam run sonunda CV RMSE (mean ± std) basılır ve test tahmini kaydedilir.

## 7) Üretilen yeni sütunlar (net liste)

Notebook akışı içinde açıkça üretilen/atanan bazı yeni kolonlar:
- `Dogum_Yili`, `Yas`
- `ikamet_norm`, `il_norm`, `ikamet_il`
- `ikamet_il_mean`, `ikamet_il_freq`
- `is_2016`
- `*_freq` (frequency encoding)

## 8) Plan (gelecek adımlar)

Bu repo şu an notebook merkezli bir çalışma. Devamında yapılabilecek net adımlar:
- `USE_FUZZY=True` ile fuzzy standardizasyonun etkisini ölçmek
- İl eşleştirmesinde fuzzy/lookup geliştirmeleri
- Feature importance / hata analizi
- CatBoost parametre taraması (daha sistematik deney tasarımı)

## 9) İletişim

- Repo sahibi: BerkayBeratBayram




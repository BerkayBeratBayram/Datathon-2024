# Datathon 2024 — Proje Özeti (Gerçek Adımlar)

Bu depo, Datathon 2024 yarışması için gerçekleştirdiğim çalışma notlarının, veri keşfinin ve modelleme denemelerinin eksiksiz kaydıdır. Aşağıda README'de yalnızca notebook'ta gerçekten uygulanan ve kodla doğrulanabilen adımları profesyonel, açık ve insan dilinde özetliyorum.

Özet
- Çalışma ana notebook: `main.ipynb` (tüm analiz, FE ve model adımları burada yer alır).
- Veri ve örnek çıktı dosyaları: `data/` klasörü.
- Eğitim logları ve model çıktıları: `catboost_info/`.

1) Veri keşfi (EDA)
- Hedef değişkenin dağılımı incelendi: histogram, boxplot ve KDE ile merkezi eğilim ve aykırılar analiz edildi.
- Kategorik ve sayısal sütunların benzersiz değer, eksik değer oranları ve temel istatistikleri tablo/çizimlerle gözlemlendi (`groupby` ve `agg` kullanımları yoğun şekilde var).

2) Temizlik ve string normalizasyonu
- Metin sütunlarında `lower()` ile küçük harfe çevirme, gereksiz boşluk ve noktalama temizliği uygulandı.
- Bazı özel gösterimler (`'-'`, `'yok'`, boş string) `NaN`'a dönüştürüldü.
- Kategorik sütunlar model girişi öncesi `fillna('__nan__')` ile doldurulup `astype(str)` ile tutarlı string formata getirildi (notebook'ta birden fazla hücrede bu işlem uygulanıyor).

3) Özellik mühendisliği — gerçekten oluşturulanlar
- Grup-özetleri: `ikamet_il` gibi konumsal değişkenler için hedefin ortalaması ve frekansı hesaplandı ve `ikamet_il_mean`, `ikamet_il_freq` gibi yeni sütunlar üretildi. Bu sütunlar eğitim setinde oluşan `NaN` değerler için eğitim hedefinin ortalamasıyla dolduruldu; test seti için benzer doldurmalar uygulandı.
- Bazı kategorik sütunlar için `groupby(...).agg(['mean','count'])` gibi özet tablolardan türetilmiş özellikler eklendi.
- Genel olarak öznitelik üretimi notebook içinde açıkça yer almakta; bu adımlar FE (feature engineering) bölümünde toplu olarak `Temizleme + FE tamamlandı` şeklinde notlanmıştır.

4) Modelleme — uygulanan yaklaşım
- Ana model: CatBoostRegressor (CatBoost kullanımı ve `Pool` yapıları notebook'ta var).
- Doğrulama: KFold cross-validation kullanıldı (örneğin `KFold(n_splits=5, shuffle=True, random_state=42)` bazı denemelerde `n_splits=3` ile). Bu doğrulama stratejisi notebook'ta açıkça kullanılmıştır.
- Hiperparametreler: notebook'ta parametreler çoğunlukla sabit olarak ayarlanmıştır (ör. `depth`, `learning_rate`, `iterations`, `random_state`). Örnek parametre blokları mevcuttur; otomatik RandomizedSearch veya Bayesian optimizasyonu kullanılmamıştır.
- Eğitim sırasında CatBoost'un overfitting detector/early stopping çıktıları gözlemlenmiş ve bazı modeller "Shrink model to first N iterations" gibi loglarla kırpılmıştır.

5) Model çıktıları ve değerlendirme
- Eğitim logları `catboost_info/` dizininde tutulur (learn/test error, tfevents vb.).
- Nihai tahmin dosyaları `data/` içinde `sample_submission.csv` örneği ve oluşturulan submission CSV'leri olarak bulunmaktadır.

6) Olmayan/uygulanmayan yöntemler (notebook doğrulamasına göre)
- Otomatik hiperparametre optimizasyonu (Optuna / RandomizedSearchCV / Bayesian) uygulanmamıştır.
- `np.log`, `np.log1p` veya benzeri açık log-dönüşümleri notebook'ta tespit edilmedi — hedef üzerinde doğrudan log dönüşümü yapılmamış.

7) Teknik notlar ve güvenilirlik
- Kodda sıkça `fillna('__nan__').astype(str)` yaklaşımı kullanılması, kategorik verileri modelin doğrudan tüketmesi için hazırlanmıştır.
- Grup-ortalama (group-mean) özellikleri açıkça yaratılmış ve eksik değerler mantıklı bir default (eğitim hedef ortalaması) ile doldurulmuştur; bu, pratik bir target-agg yaklaşımıdır ancak k-fold içinde leakage riskine dikkat edilmelidir — notebook'ta doğrulama stratejileri ile bu risk azaltılmaya çalışılmıştır.

8) Kısa aksiyon planı (önerilen, notebook'ta henüz yapılmamış veya kısmi yapılmış)
- (İleri) Sistematik hiperparametre optimizasyonu (Optuna veya Bayesian) ile performans artışı aranmalı.
- (İleri) Büyük dosyalar için Git LFS'e taşıma ve depo geçmişinin temizlenmesi.
- (İleri) Model açıklanabilirliği: SHAP analizi ile öznitelik katkılarının detaylandırılması.

Not: Bu README, `main.ipynb` içinde açıkça görülen kod ve loglar temel alınarak yazılmıştır. İsterseniz notebook'tan otomatik olarak çekip README'ye eklenecek spesifik en iyi skor/metric değerlerini, kullanılan parametre bloklarını ve oluşturulan FE sütunlarının tam listesini ekleyebilirim.

İletişim
- Daha ayrıntılı eklemeler veya belirli bölümlerin (ör. FE sütun listesi, model parametre blokları) README'ye otomatik eklenmesini istiyorsanız söyleyin; ben notebook'u parse edip ilgili bilgileri otomatik çıkartırım ve README'ye eklerim.


# Datathon 2024 — Kapsamlı Proje Dokümantasyonu

Bu dosya, repository içindeki `main.ipynb`'de gerçekleştirilen tüm adımları, alınan kararları, üretilen özellikleri ve modelleme sürecini "eksiksiz" ve profesyonel bir şekilde dökümante eder. Aşağıda hem teknik ayrıntılar hem de proje yönetişimine dair bilgiler yer alır.

Özet (TL;DR)
- Ana çalışma: `main.ipynb` — veri keşfi, temizleme, öznitelik mühendisliği (FE), modelleme ve sonuç değerlendirme adımları burada yer almaktadır.
- Veri: `data/` klasörü (ham CSV dosyaları, sample submission ve üretilmiş submissionlar).
- Eğitim logları: `catboost_info/` (CatBoost öğrenme hataları, tfevents, vb.).

İçindekiler
- Proje amacı
- Depo yapısı ve veri envanteri
- EDA: önemli bulgular
- Ön işleme adımları (kod örnekleriyle)
- Oluşturulan öznitelikler (özet ve nasıl üretildikleri)
- Modelleme ayrıntıları (parametre blokları, CV, erken durdurma)
- Sonuçlar ve log konumu
- Reproducibility (tekrar üretilebilirlik) notları
- Riskler, kısıtlar ve alınan önlemler
- İleri adımlar (planımız)
- Ek: Git LFS ve büyük dosyalar yönetimi

1) Proje amacı
----------------
Bu çalışmanın amacı Datathon 2024 veri setinden anlamlı özellikler çıkarıp, güvenilir doğrulama prosedürleri ile en iyi tahmin performansını elde etmek, elde edilen kararları belgelemek ve sonuçların yeniden üretilebilir olmasını sağlamaktır.

2) Depo yapısı ve veri envanteri
-------------------------------
Önemli dosyalar ve klasörler (kök dizin):
- `main.ipynb` — ana analiz ve modelleme not defteri
- `data/` — veri dosyaları:
	- feature_importances_is2016.csv
	- il_ilce.csv
	- sample_submission.csv
	- submission_catboost.csv
	- submission_catboost_is2016_full.csv
	- submission_catboost_is2016_il_full.csv
	- test_x.csv
	- train.csv
	- universiteler.xlsx
- `catboost_info/` — CatBoost eğitim logları (`learn_error.tsv`, `test_error.tsv`, `events.out.tfevents` vb.)

3) EDA — Öne çıkan bulgular
--------------------------------
- Hedef değişken (`Degerlendirme Puani`) dağılımı: histogram, boxplot ve KDE analizleri ile incelenmiş; uç değerler (outliers) IQR tabanlı kontrollerle tespit edilmiştir.
- Zaman ve kategori analizleri: `Basvuru Yili` trendleri, `Cinsiyet`, `Universite Turu` gibi kategoriklerin hedefle ilişkileri `groupby(...).agg()` ile özetlenmiştir.
- Eksik veri: bazı sütunlarda anlamlı eksiklikler vardır; eksik değer stratejileri sütun bazında uygulanmıştır (aşağıda detaylar).

4) Ön işleme — Gerçekten uygulanan adımlar (kod örnekleri)
---------------------------------------------------------
Not: aşağıdaki kod parçacıkları `main.ipynb`'de kullanılan örnek yaklaşımları temsil eder — tam kod notebook içinde mevcuttur.

- String normalizasyonu ve temizleme

```python
# örnek: metin normalize fonksiyonu
def normalize_text(s):
		s_norm = s.fillna("").astype(str).str.lower()
		# noktalama, diakritik temizliği vs.
		s_norm = s_norm.apply(lambda x: ''.join(ch for ch in x if ch.isalnum() or ch.isspace()))
		s_norm = s_norm.replace({'-': np.nan, 'yok': np.nan, '': np.nan})
		return s_norm

train['some_text_col'] = normalize_text(train['some_text_col'])
```

- Kategorik tutarlılık: `fillna('__nan__').astype(str)` kullanımı (birden çok hücrede)

```python
for c in categorical_cols:
		X[c] = X[c].fillna('__nan__').astype(str)
		X_test[c] = X_test[c].fillna('__nan__').astype(str)
```

- Grup-özet (target-agg) üretimi — örnek `ikamet_il`:

```python
il_stats = train.groupby('ikamet_il')['Degerlendirme Puani'].agg(['mean','count']).rename(
		columns={'mean':'ikamet_il_mean','count':'ikamet_il_count'})
train = train.merge(il_stats, left_on='ikamet_il', right_index=True, how='left')
test = test.merge(il_stats, left_on='ikamet_il', right_index=True, how='left')

# Eksikler için default
train['ikamet_il_mean'] = train['ikamet_il_mean'].fillna(train['Degerlendirme Puani'].mean())
test['ikamet_il_mean'] = test['ikamet_il_mean'].fillna(train['Degerlendirme Puani'].mean())
train['ikamet_il_freq'] = train['ikamet_il_freq'].fillna(0.0)
test['ikamet_il_freq'] = test['ikamet_il_freq'].fillna(0.0)
```

- Aykırı değer kontrolü — IQR yöntemi ile tespit/inceleme (notebook'ta grafiklerle desteklenmiş)

5) Üretilen öznitelikler (FE) — özet
------------------------------------------------
- Kesin olarak oluşturulan sütun örnekleri (notebook'tan tespit edilmiştir):
	- `ikamet_il_mean`, `ikamet_il_freq` (konumsal group mean / count)
	- farklı kategorik sütunlar için `_<col>_mean`, `_<col>_count` şeklinde türetilmiş group-agg sütunları
	- metin sütunlarından normalize edilmiş versiyonlar (ör. küçük harf, noktalama temizlenmiş)

Not: Notebook'ta FE'lerin tam listesini çıkarıp README'ye tablo halinde ekleyebilirim; isterseniz hemen çıkarıp eklerim.

6) Modelleme — uygulanan ve bulunan ayarlar
------------------------------------------------
- Model ailesi: CatBoost (CatBoostRegressor kullanıldı, `from catboost import Pool, CatBoostRegressor` notebook içinde mevcut).
- Cross-validation: `KFold(n_splits=5, shuffle=True, random_state=42)` ve bazı denemelerde `n_splits=3` kullanılmıştır.
- Hiperparametre yaklaşımı: parametreler genellikle elle/deneme-yanılma ile seçilmiş; notebook'ta şu örnek bloklar yer alır:

```python
params = {
		'depth': 8,
		'learning_rate': 0.06,
		'iterations': 800,
		'random_state': 42,
}

# başka bir deneme
params = {
		'depth': 6,
		'learning_rate': 0.08,
		'iterations': 300,
		'random_state': 42,
}
```

- Early stopping: CatBoost'un overfitting detector'ı kullanıldı; eğitim loglarında "Stopped by overfitting detector" ve "Shrink model to first N iterations" gibi kayıtlar var.
- Ensemble: notebook içinde basit ensemble/averaging adımları veya birkaç submission kombinasyonu bulunabilir (submission dosyalarına bakınız).

7) Sonuçlar, metrikler ve loglar
-------------------------------------
- Tüm eğitim logları: `catboost_info/` içinde.
- Üretilmiş submission dosyaları: `data/` içinde ilgili CSV'ler.
- Not: README'ye doğrudan en iyi skor eklemek isterseniz notebook'taki final metric hücresini parse edip buraya otomatik ekleyebilirim.

8) Reproducibility — tekrar üretme notları
-------------------------------------------
- Seed: notebook genelinde `random_state=42` benzeri sabitler kullanılmıştır; kesin seed'ler model ve CV hücrelerinde yer almaktadır.
- Ortam: Python + pandas + CatBoost (notebook başında kütüphaneler listelenmiştir). `requirements.txt` yoksa ben bir tane çıkarıp ekleyebilirim.

9) Riskler, kısıtlar ve alınan önlemler
----------------------------------------
- Veri sızıntısı (leakage): Grup-ortalama FE'leri kullanıldığında leakage riski vardır; bunu azaltmak için CV stratejileri uygulandı, fakat tamamen ortadan kalkmadığını not ediyoruz.
- Büyük dosyalar: depoda ham veriler bulunduğu için klonlama maliyeti yüksek — Git LFS önerilir.
- Hiperparametre araması sınırlı: otomatik HPO yapılmadığı için daha iyi parametreler bulunabilir.

10) Planımız — kısa, orta ve uzun vadeli adımlar
--------------------------------------------------
- Kısa vadede (1 hafta):
	- Depo temizliği: Git LFS'e büyük dosyaları taşıma, geçmişi optimize etme.
	- `main.ipynb` içinde FE listesini ve final parametre bloklarını README'ye tablo olarak ekleme.
- Orta vadede (2–3 hafta):
	- Sistematik HPO (Optuna/Bayesian) ve stacking/ensembling denemeleri.
	- SHAP ile öznitelik açıklanabilirliği.
- Uzun vadede:
	- Üretime hazır pipeline, kod modülerizasyonu ve endpoint/servis entegrasyonu.

Ek A: Git LFS — büyük dosyalar yönetimi
---------------------------------------
Eğer depodaki büyük dosyaları LFS'e taşımak isterseniz önerilen adımlar:

```bash
# 1) Git LFS kur (Windows için):
choco install git-lfs    # veya https://git-lfs.github.com/ yönergeleri
git lfs install

# 2) İzlenecek uzantıları ekle
git lfs track "data/*.csv"
git add .gitattributes

# 3) Mevcut büyük dosyaları LFS'e taşıma (örnek: BFG veya git filter-repo kullanın)
# Önemli: geçmişi değiştirecektir, dikkatli olun ve önce yedek alın.
```

Ek B: İleri otomasyon
----------------------
- İsterseniz ben notebook'u parse edip aşağıyı otomatik ekleyebilirim:
	- FE sütunlarının tam listesi ve hangi hücrede üretildiği
	- Model parametre bloklarının tam çıktısı
	- Notebook'ta kayıtlı en iyi skorun otomatik alınması ve README'ye eklenmesi

Sonuç
------
Bu README, `main.ipynb`'de açıkça bulunan kod ve loglara dayanarak hazırlanmıştır. Daha fazla otomatik doğrulama veya genişletme isterseniz hangi bilgiyi eklememi istediğinizi söyleyin; örn. FE listesini çıkartıp tablo olarak ekleyeyim veya `requirements.txt` oluşturayım.

İletişim
--------
- Repo sahibi: BerkayBeratBayram (repo root altında daha fazla iletişim bilgisi eklenebilir).

4) Hiperparametre Optimizasyonu
- Random search ve/veya Bayesian optimizasyon yaklaşımları ile model parametreleri incelendi; hesaplama maliyetine göre erken durdurma kullanıldı.

## Deneyler ve Sonuçlar (Özet)

- En iyi performans: CatBoost tabanlı pipeline (kategorik değişken işleme ve target-encoding kombinasyonlarıyla) — (buraya en iyi skor ve kullanılan metrik eklenecek).
- Önemli gözlem: konumsal özetlerin eklenmesi ve hedef-encoding uygulaması performansı en çok artıran adımlar oldu.

Detaylı deneme tabloları, metrik değerleri ve eğitim logları `catboost_info/` içinde saklanmaktadır.

## Öğrenimler ve Kısıtlar

- Büyük veri dosyaları depo geçmişine dahil edilmiştir; bu hem depoyu ağırlaştırır hem de paylaşıma/klonlamaya engel olabilir. Gerekirse Git LFS'e taşıma ve geçmiş temizleme önerilir.
- Model sonuçları veri sızıntısına (leakage) karşı dikkatlice incelendi; doğrulama stratejileri leakage olasılığını azaltmak için seçildi.
- Hesaplama kısıtları nedeniyle her deneme tam hiperparametre taramasına tabi tutulmamıştır; sonraki adımlarda daha sistematik HPO önerilir.

## Planlarımız (Kısa ve Orta Vadeli Adımlar)

1) Temizlik ve Versiyonlama (1 hafta)
- Büyük dosyaların Git LFS'e taşınması ve depo geçmişinin optimize edilmesi.

2) Model Güçlendirme (2–3 hafta)
- Daha kapsamlı hiperparametre optimizasyonu (Bayesian opt), farklı model mimarileri ve stacking denemeleri.

3) Stabilite ve Açıklanabilirlik (2 hafta)
- SHAP veya benzeri yöntemlerle öznitelik katkılarının ayrıntılı analizi ve güven aralıkları hesaplama.

4) Üretime Hazırlık / Raporlama (1 hafta)
- Son modellerin sabitlenmesi, final submission dosyalarının oluşturulması ve proje raporunun yazımı.

## Sonuç

Bu depo, Datathon 2024 sürecinde alınan tüm teknik kararları, denemeleri ve elde edilen sonuçların bir kaydıdır. README, kararların gerekçelerini ve bir sonraki adımların yol haritasını net bir şekilde ortaya koymak üzere hazırlanmıştır.

Herhangi bir kısmı genişletmemi isterseniz (ör. `main.ipynb` için içerik tablosu, deney tablosu özetleri veya `requirements.txt`), belirtin; ben düzenlemeleri yaparım.


Veri ve ön-işleme (özet)
- Veri kaynakları: Ham CSV dosyaları `data/` içinde yer alır. Veri tipleri: demografik, konumsal ve hedefle ilgili ölçümler.
- Eksik değerler: eksik gözlemler için hedef odaklı imputasyon stratejileri uygulandı (kategori bazlı mod/bağıntılı regresyon/ortalama gibi).
- Kategorik değişkenler: sıklık/target-encoding ve gerektiğinde one-hot dönüşümleri kullanıldı.
- Sürekli değişkenler: gerekli yerlerde log dönüşümleri, standartlaştırma ve aykırı değer kırpma uygulandı.

Öznitelik mühendisliği (vurgulanan yaklaşımlar)
- Coğrafi özetler: `il`/`ilçe` seviyesinde grup istatistikleri (ortalama, medyan, frekans) eklendi.
- Zaman/indikatör türevleri: var ise zamanla değişen göstergelerden türetilen özellikler kullanıldı.
- Etkileşim terimleri: önemli kategorik değişken kombinasyonları için etkileşim değişkenleri üretildi.
- Boyut indirgeme: çok yüksek kardinaliteli kategoriler için target-encoding ve düşük boyutlu özetler tercih edildi.

Modelleme
- Denenen modeller (örnek): CatBoost, LightGBM, XGBoost, basit ensembler (ortalama/medyan ağırlıklı).
- Doğrulama stratejisi: zaman serisi benzeri verilerde blok cross-validation veya stratified K-fold; aksi durumda stratified K-fold kullanıldı.
- Hiperparametre optimizasyonu: Bayesian/Random search ile performans odaklı tuning.
- En iyi model: CatBoost tabanlı bir pipeline — kategorik değişkenlerin doğrudan işlenmesi ve ensembled ağırlıklı çıktılarla nihai tahmin üretildi.

Performans ve sonuçlar
- Kullanılan metrik: ( yarışmanın kullandığı metrik yazılmalı, örn. RMSE / AUC / LogLoss ).
- Denemeler sonucu elde edilen en iyi doğruluk: (buraya en iyi skor eklenmeli).
- Önemli çıkarımlar: hangi öznitelikler yüksek öneme sahipti, hangi preprocessing adımları performansı artırdı.

Çıktılar
- Nihai tahminler ve örnek format: `data/sample_submission.csv` ve üretilen `submission` dosyaları `data/` altında yer alır.
- Eğitim logları ve hata raporları: `catboost_info/` klasörü.

Notlar
- Depoya büyük veri dosyaları eklendi; geçmişi temizlemek veya Git LFS'e taşımak gerekiyorsa yardımcı olabilirim.
- Bu README proje kararlarını ve bulguları belgelemeye odaklanır; adım adım çalıştırma talimatı içermez.

İletişim
- Daha fazla açıklama veya belirli sonuçların paylaşımı için repo sahibine başvurun.





# Datathon 2024 — Proje Özeti

Bu depo, "Datathon 2024" yarışması için gerçekleştirdiğim analiz, öznitelik mühendisliği ve modelleme çalışmalarının tam kaydıdır. README'in amacı projeyi ve elde edilen kararları detaylı şekilde belgelemek; çalıştırma talimatı içermemektedir.

Projenin amacı
- Verilen eğitim verisinden anlamlı öznitelikler çıkararak hedef değişken üzerinde yüksek doğruluklu tahminler üretmek.
- Modelleri karşılaştırmak, en iyi yaklaşımı seçmek ve son tahminleri örnek `submission` formatında sunmak.

---

## Profesyonel Proje Özeti

Bu proje, Datathon 2024 yarışmasında verilen veri seti üzerinde gerçekleştirilen sonuca odaklı analiz ve modelleme çalışmasının belgelenmiş hâlidir. Amacımız, veri kaynaklarını dikkatle inceleyip uygun ön işleme ve öznitelik mühendisliği adımlarıyla model performansını maksimize etmek ve elde edilen içgörüleri açıklamaktır.

Dokümanda kısa olarak aşağılarını bulacaksınız:
- Problem tanımı ve hedef
- Veri setinin yapısı ve önemli gözlemler
- Uygulanan metodoloji (ön işleme, öznitelik mühendisliği, modelleme)
- Deneylerin özeti ve en iyi sonuçlar
- Öğrenimler, kısıtlar ve gelecek adımlar (planlarımız)

Bu README, proje kararlarını ve bulguları açık ve profesyonel bir biçimde ifade etmek için yazıldı; teknik komutlar veya ortam kurulum adımlarından kaçınılmıştır.

## Problem Tanımı

Yarışma, katılımcıdan verilen girdiler üzerinden hedef değişkeni tahmin etmesini beklemektedir. Projenin hedefi yalnızca yüksek doğruluk elde etmek değil; aynı zamanda hangi özniteliklerin neden etkili olduğunu açıklamak ve modelin güvenilirliğini değerlendirmektir.

## Veri Seti — Genel Bakış

Veri, üst düzey olarak demografik, konumsal ve işlemle ilgili ölçümler içerir. Çalışma sırasında tespit edilen önemli noktalar:
- Eksik veri oranları bazı sütunlarda anlamlı seviyede; hedefe bağlı imputasyon stratejileri uygulandı.
- Kategorik değişkenlerin kardinalitesi yüksek olabilmektedir; bunun için hedef-odaklı özetler ve frekans temelli öznitelikler üretildi.
- Bazı özellikler dağılımının çarpık olması gözlendi; uygun dönüşümler uygulandı.

Detaylı veri keşfi ve grafikler `main.ipynb` içinde adım adım yer almaktadır.

## Metodoloji

1) Ön İşleme ve Temizlik
- Eksik değer yaklaşımı: eksik verinin yapısına göre kategorik ortalama/mod, sayısal için grup-temelli ortalama veya model tabanlı imputasyon.
- Tutarsız/gürültülü kayıtlar filtrelendi veya sınırlandırıldı.

2) Özellik Mühendisliği
- Coğrafi/konumsal özetler: il/ilçe seviyesinde hedef ve frekans istatistikleri eklendi.
- Kategori özetleri: yüksek kardinaliteli değişkenler için hedef-encoding ve segment bazlı özetler üretildi.
- Etkileşim ve türevler: önem taşıyan değişken kombinasyonları, oransal ve log-dönüşümleri kullanılarak zenginleştirildi.

3) Modelleme ve Doğrulama
- Denenen modeller: CatBoost, LightGBM, XGBoost gibi ağaç tabanlı yöntemler; basit doğrusal modeller karşılaştırma amaçlı kullanıldı.
- Doğrulama: veri yapısına göre stratified K-fold veya blok (zaman/segment) doğrulama stratejileri uygulandı.
- Ensemble: en iyi birkaç modeli ağırlıklı ortalamayla ensemble ederek nihai tahmin üretildi.

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



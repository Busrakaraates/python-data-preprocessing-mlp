# El Yazısı Rakam Tanıma: Veri Ön İşleme ve Yapay Sinir Ağı (MLP) Modeli 🧠

Bu projede, Scikit-learn kütüphanesinde yer alan `load_digits` veri seti (8x8 piksel boyutunda 1797 el yazısı rakam görüntüsü) kullanılarak uçtan uca bir makine öğrenmesi ve veri ön işleme süreci gerçekleştirilmiştir.

## 🛠️ Veri Ön İşleme Adımları (Data Preprocessing)
* **Eksik Veri Yönetimi:** `SimpleImputer` kullanılarak olası eksik değerler ortalama (mean) yöntemiyle doldurulmuştur.
* **Aykırı Değer Tespiti (Outlier Detection):** `LocalOutlierFactor` (LOF) algoritması ile yoğunluk tabanlı anomali tespiti yapılmış ve veriler temizlenmiştir.
* **Z-Score Normalizasyonu:** `StandardScaler` kullanılarak tüm piksel nitelikleri standartlaştırılmıştır.
* **Dengesiz Veri Yönetimi (Imbalanced Data):** `SMOTE` (Synthetic Minority Over-sampling Technique) uygulanarak sınıfların eşit ağırlıkta öğrenmesi sağlanmıştır.

## 🧠 Yapay Sinir Ağı (MLP) Mimarisi
* **Model:** `MLPClassifier` (Çok Katmanlı Algılayıcı)
* **Gizli Katmanlar:** (100, 50) nöronlu 2 katmanlı ileri beslemeli yapı
* **Aktivasyon Fonksiyonu:** ReLU
* **Optimizasyon:** Adam solver
* **Eğitim Süreci:** Early stopping mekanizması sayesinde yaklaşık 80. epoch civarında minimum kayıp (loss) değerine ulaşılmıştır.

## 📊 Kullanılan Teknolojiler
* **Dil:** Python
* **Kütüphaneler:** Scikit-learn, Pandas, NumPy, Matplotlib, Imbalanced-learn (SMOTE)

## 📄 Proje Raporu
Çalışmanın tüm teorik arka planını, kod çıktılarını ve veri tablolarını içeren rapora repo içerisinden erişebilirsiniz.

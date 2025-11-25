# Finansal Piyasalar ve Yatırım Kitabı  
## Bölüm 4: Türev Araçlar ve Opsiyon Fiyatlama

### 4.1 Giriş
Türev ürünler, değeri dayanak varlığa (hisse, tahvil, endeks, döviz, emtia) bağlı olan finansal araçlardır.  
Başlıca türleri: **Forward, Futures, Swap, Opsiyon**.  
Riskten korunma (hedging), spekülasyon ve arbitraj amaçlı kullanılırlar.

---

### 4.2 Forward ve Futures Sözleşmeleri
- **Forward:** İki taraf arasında tezgâhüstü (OTC) sözleşme, özelleştirilebilir.  
- **Futures:** Standartlaştırılmış, organize borsalarda işlem görür (örn. VİOP, CME).  
- Fiyatlama:  
\[
F_0 = S_0 (1+r)^T
\]  
- S0: Spot fiyat, r: risksiz faiz, T: vade.

---

### 4.3 Swaplar
- **Faiz Swapı:** Sabit faiz ↔ değişken faiz ödemesi takası.  
- **Döviz Swapı:** Farklı para cinslerinde anapara ve faiz takası.  
- Kullanım: Bankalar fonlama yapısını ve kur riskini yönetir.

---

### 4.4 Opsiyonlar
Opsiyon, sahibine dayanak varlığı belirli bir fiyattan alma (call) veya satma (put) hakkı verir, fakat zorunluluk getirmez.

- **Call Opsiyon:** Alım hakkı.  
- **Put Opsiyon:** Satım hakkı.  
- **Strike Price (K):** Kullanım fiyatı.  
- **Prim:** Opsiyon için ödenen fiyat.  
- **Amerikan Tipi:** Vade sonuna kadar herhangi bir zamanda kullanılabilir.  
- **Avrupa Tipi:** Sadece vade sonunda kullanılabilir.

#### Opsiyonun Getirisi
- Call: max(0, S_T − K) − Prim  
- Put: max(0, K − S_T) − Prim  

---

### 4.5 Black–Scholes Modeli (1973)
Avrupa tipi opsiyonların fiyatlanmasında kullanılır.

\[
C = S_0 N(d_1) - K e^{-rT} N(d_2)
\]

\[
d_1 = \frac{\ln(S_0/K) + (r+\sigma^2/2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
\]

- **S0:** Spot fiyat  
- **K:** Kullanım fiyatı  
- **r:** Risksiz faiz oranı  
- **T:** Vade  
- **σ:** Volatilite  
- **N(.):** Standart normal dağılım fonksiyonu

**Örnek:** S0=100, K=100, r=0.05, σ=0.2, T=1 → C ≈ 10.45

---

### 4.6 Binom Modeli
Opsiyon değerini adım adım yukarı/aşağı hareketlerle hesaplar.  
- U (yukarı faktörü), D (aşağı faktörü), p (risk‑nötr olasılık) ile değerlenir.  
- Esnekliği sayesinde Amerikan tipi opsiyonlara da uygulanır.

---

### 4.7 Opsiyonun Yunanları (The Greeks)
Opsiyon fiyatının farklı parametrelere duyarlılıklarını ölçer:  
- **Delta (Δ):** Opsiyon fiyatının dayanak fiyatına duyarlılığı.  
- **Gamma (Γ):** Delta’nın değişim hızı.  
- **Vega (ν):** Volatiliteye duyarlılık.  
- **Theta (θ):** Zaman değer kaybı.  
- **Rho (ρ):** Faiz oranına duyarlılık.

---

### 4.8 Opsiyon Stratejileri
- **Covered Call, Protective Put** → riskten korunma.  
- **Straddle, Strangle** → volatilite üzerine bahis.  
- **Butterfly Spread, Iron Condor** → dar bant beklentisi.  

---

### 4.9 Türkiye ve Dünyadan Örnekler
- **Türkiye:** VİOP’ta BIST30 endeks vadeli işlemler ve opsiyonlar yoğun işlem görür.  
- **ABD:** CME ve CBOE dünyadaki en büyük türev piyasalarıdır. Black–Scholes modeli burada doğmuştur.  
- **Şirketler:** Döviz opsiyonlarıyla kur riskinden korunur.  

---

### 4.10 Genel Değerlendirme
Türev ürünler, modern finansın en önemli risk yönetim araçlarıdır.  
Doğru kullanıldığında korunma sağlar; aşırı spekülasyon ise krizleri tetikleyebilir.  
Opsiyon fiyatlama modelleri, risk yönetiminde ve piyasa etkinliğinde temel araçtır.

---

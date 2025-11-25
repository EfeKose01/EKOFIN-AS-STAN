# Finansal Piyasalar ve Yatırım Kitabı  
## Bölüm 2: Hisse Senedi Değerleme

### 2.1 Giriş
Hisse senedi değerleme, yatırımcıların bir şirketin gerçek (içsel) değerini hesaplayarak alım-satım kararlarını vermesine yardımcı olur.  
Piyasa fiyatı ile içsel değer arasındaki fark, yatırım fırsatlarını ortaya çıkarır.  

---

### 2.2 Hisse Senedinin Temel Özellikleri
- **Mülkiyet Hakkı:** Hisse, şirkette ortaklık hakkı verir.  
- **Temettü Hakkı:** Kar dağıtımına katılım hakkı.  
- **Oy Hakkı:** Genel kurulda karar alma süreçlerine katılım.  
- **Artık Haklar:** Şirket tasfiye edilirse varlıklardan pay alma.  

---

### 2.3 Değerleme Yaklaşımları
Üç ana yöntem vardır:  

1. **Mutlak Değerleme:** Gelecekteki nakit akımlarının bugünkü değeri.  
   - Temettü İskonto Modeli (DDM)  
   - Serbest Nakit Akımı (FCFF/FCFE)  
2. **Göreli Değerleme:** Benzer şirketlerin çarpanları (P/E, P/B, EV/EBITDA).  
3. **Varlık Tabanlı Değerleme:** Şirketin varlıklarının net değerine dayanır.  

---

### 2.4 Temettü İskonto Modeli (DDM)
#### Formül
\[
P_0 = \sum_{t=1}^{\infty} \frac{D_t}{(1+k_e)^t}
\]

- \(P_0\): Hisse senedinin bugünkü değeri  
- \(D_t\): t dönemindeki temettü  
- \(k_e\): Özsermaye maliyeti  

#### Gordon Büyüme Modeli (Sabit Büyüme DDM)
\[
P_0 = \frac{D_1}{k_e - g}
\]

- \(D_1\): Gelecek yıl temettüsü  
- \(g\): Temettü büyüme oranı  

**Örnek:** D1 = 2 TL, ke = %12, g = %5 → P0 = 2 / (0.12 – 0.05) = 28.6 TL  

---

### 2.5 Serbest Nakit Akımı Modelleri (FCFF ve FCFE)
#### FCFF (Firmaya Serbest Nakit Akımı)
\[
V_0 = \sum_{t=1}^{\infty} \frac{FCFF_t}{(WACC)^t}
\]

- FCFF = Faaliyet Kârı – Vergiler + Amortisman – Yatırımlar – İşletme Sermayesi Artışı  
- WACC = Sermaye Ağırlıklı Ortalama Maliyeti  

#### FCFE (Özsermayeye Serbest Nakit Akımı)
\[
E_0 = \sum_{t=1}^{\infty} \frac{FCFE_t}{(k_e)^t}
\]

- FCFE = Net Kâr + Amortisman – Yatırımlar – İşletme Sermayesi Artışı + Net Borçlanma  

**Örnek:** Bir firmanın 1. yıl FCFE’si 5 TL, büyüme oranı g = %4, ke = %10 ise:  
P0 = 5 × (1+g) / (ke – g) = 5.2 / 0.06 = 86.7 TL  

---

### 2.6 Göreli Değerleme
En sık kullanılan çarpanlar:  
- **F/K (P/E):** Hisse fiyatı / Hisse başı kâr.  
- **PD/DD (P/B):** Piyasa değeri / Defter değeri.  
- **EV/EBITDA:** Şirketin işletme değeri / Faiz, vergi, amortisman öncesi kâr.  

Örnek: Aynı sektördeki firmaların ortalama F/K oranı 12 ise, bir firmanın hisse başı kârı 3 TL ise → Değer = 12 × 3 = 36 TL.  

---

### 2.7 Özsermaye Maliyeti ve CAPM
Özsermaye maliyeti (ke), genellikle **CAPM** ile hesaplanır:  

\[
k_e = R_f + \beta (R_m - R_f)
\]

- \(R_f\): Risksiz faiz oranı  
- \(R_m\): Piyasa portföyü getirisi  
- \(\beta\): Hissenin piyasa riskine duyarlılığı  

---

### 2.8 Türkiye ve Dünyadan Örnekler
- **Türkiye:** BIST 30 şirketleri için analistler genellikle hem DDM hem de çarpan analizini birlikte kullanır.  
- **ABD:** Apple, Microsoft gibi teknoloji devleri için FCFF daha uygundur çünkü nakit akımı temettüden daha belirleyicidir.  
- **Bankalar:** Genelde P/B çarpanı kullanılır.  

---

### 2.9 Genel Değerlendirme
Hisse senedi değerlemesi, yatırımcıların doğru yatırım kararları alması için kritik bir süreçtir.  
DDM, FCFF/FCFE ve çarpan analizleri farklı perspektifler sağlar.  
Her yöntem, şirketin özelliklerine ve sektörüne göre uygun şekilde seçilmelidir.

---

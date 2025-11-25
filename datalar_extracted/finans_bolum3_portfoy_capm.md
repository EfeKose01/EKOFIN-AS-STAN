# Finansal Piyasalar ve Yatırım Kitabı  
## Bölüm 3: Portföy Teorisi ve CAPM

### 3.1 Giriş
Modern portföy teorisi (Harry Markowitz, 1952), yatırımcıların portföylerini yalnızca beklenen getiriye değil, aynı zamanda risk düzeyine göre de optimize etmeleri gerektiğini vurgular.  
Amaç, belirli bir risk düzeyinde en yüksek getiriyi veya belirli bir getiri düzeyinde en düşük riski sağlayan portföyü seçmektir.

---

### 3.2 Beklenen Getiri ve Risk
- **Beklenen Getiri (E[R]):**  
\[
E[R_p] = \sum_{i=1}^n w_i E[R_i]
\]  
Burada \(w_i\) portföydeki ağırlık, \(E[R_i]\) ise i varlığının beklenen getirisi.

- **Varyans ve Standart Sapma (Risk):**  
\[
\sigma_p^2 = \sum_i w_i^2 \sigma_i^2 + \sum_{i\neq j} w_i w_j Cov(R_i, R_j)
\]  
Varlıklar arası kovaryans/korrelasyon, çeşitlendirme etkisini belirler.

---

### 3.3 Çeşitlendirme (Diversification)
- Düşük korelasyonlu varlıkların bir araya getirilmesi, toplam riski düşürür.  
- Korelasyon **ρ = +1** ise çeşitlendirme faydasızdır.  
- Korelasyon **ρ < 1** olduğunda risk azalır.  

---

### 3.4 Etkin Sınır (Efficient Frontier)
- Risk-getiri uzayında, en iyi portföylerin oluşturduğu eğri.  
- Etkin sınır üzerindeki portföyler, daha düşük riskle daha yüksek getiri sağlayamaz.  
- Yatırımcıların risk tercihine göre seçim yapılır.

#### Grafik Açıklaması
- Yatay eksen: risk (σ).  
- Dikey eksen: beklenen getiri (E[R]).  
- Etkin sınır, yukarı doğru dışbükey bir eğri şeklindedir.

---

### 3.5 Sermaye Piyasası Doğrusu (CML)
Risksiz varlık eklendiğinde portföy kombinasyonları **doğruya** dönüşür:  
\[
E[R_p] = R_f + \frac{E[R_M] - R_f}{\sigma_M} \cdot \sigma_p
\]  

- **R_f:** Risksiz getiri oranı  
- **E[R_M]:** Piyasa portföyü getirisi  
- **σ_M:** Piyasa portföyü riski  

CML, en iyi risk-getiri kombinasyonlarını gösterir.

---

### 3.6 Sermaye Varlıklarını Fiyatlama Modeli (CAPM)
CAPM, tekil varlıkların beklenen getirisini sistematik risk ölçüsü olan **beta** ile açıklar.

\[
E[R_i] = R_f + \beta_i (E[R_M] - R_f)
\]

- **β > 1:** Hisse piyasadan daha riskli, daha oynak.  
- **β < 1:** Hisse daha defansif, daha az oynak.  
- **β = 1:** Piyasa ile aynı risk düzeyi.

#### Beta Hesabı
\[
\beta_i = \frac{Cov(R_i, R_M)}{Var(R_M)}
\]  

---

### 3.7 Güvenlik Pazar Doğrusu (SML)
CAPM’in grafiksel gösterimidir.  
- Yatay eksen: Beta (β).  
- Dikey eksen: Beklenen Getiri (E[R]).  
- SML üzerindeki varlıklar dengededir.  
- Üzerinde olanlar **ucuz** (daha yüksek getiri), altında olanlar **pahalı**dır.

---

### 3.8 Türkiye ve Dünyadan Örnekler
- **BIST 100’de β>1 hisseler:** Bankacılık sektörü (yüksek oynaklık).  
- **Defansif hisseler:** Gıda, sağlık sektörü.  
- **ABD örneği:** Teknoloji hisseleri (NASDAQ) genelde β>1; utility şirketleri β<1.  

---

### 3.9 Genel Değerlendirme
Modern portföy teorisi ve CAPM, yatırımcıların risk-getiri ilişkisini sistematik şekilde anlamasını sağlar.  
- Markowitz etkin sınır → çeşitlendirme faydası.  
- CML → risksiz varlıkla optimum portföy.  
- CAPM → tekil varlıkların fiyatlanması.  
Gerçek hayatta ek faktörler (Fama–French 3/5 faktör, likidite primi) de kullanılır.

---

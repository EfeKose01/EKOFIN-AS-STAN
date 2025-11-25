# Finansal Piyasalar ve Yatırım Kitabı  
## Bölüm 6: Risk Yönetimi ve Hedge Stratejileri

### 6.1 Giriş
Finansal piyasalarda risk yönetimi, yatırımcıların ve kurumların maruz kaldıkları belirsizlikleri ölçmeleri, izlemeleri ve azaltmaları sürecidir.  
Riskten korunma (hedging), portföy değerindeki dalgalanmaları sınırlamak için kullanılan yöntemlerin bütünüdür.

---

### 6.2 Risk Türleri
1. **Piyasa Riski:** Faiz oranı, döviz kuru, hisse senedi ve emtia fiyatlarındaki değişimlerden kaynaklanır.  
2. **Kredi Riski:** Karşı tarafın yükümlülüklerini yerine getirmeme ihtimali.  
3. **Likidite Riski:** Varlıkların hızlıca ve uygun fiyattan nakde çevrilememesi.  
4. **Operasyonel Risk:** Sistem, insan hatası veya dış olaylardan kaynaklanan kayıplar.  
5. **Sistemik Risk:** Tüm finansal sistemin istikrarını tehdit eden risk.  

---

### 6.3 Risk Ölçüm Yöntemleri
- **Volatilite (σ):** Getiri serilerinin standart sapması.  
- **Value at Risk (VaR):** Belirli bir güven düzeyinde maksimum olası kayıp.  
  \[
  VaR = Z_{α} \cdot σ_p \cdot \sqrt{t}
  \]  
  (α güven düzeyi, σ_p portföy volatilitesi, t zaman ufku)  
- **Stres Testleri:** Olağandışı fakat olası senaryoların etkisini ölçer.  
- **Duyarlılık Analizi:** Tek bir faktördeki değişimin etkisini inceler.  

---

### 6.4 Hedge (Riskten Korunma) Yöntemleri
1. **Forward ve Futures ile Hedge:** Döviz kuru veya emtia fiyat riskine karşı.  
2. **Opsiyonlarla Hedge:** Koruyucu put (protective put), tavanlı maliyet (collar).  
3. **Swaplarla Hedge:** Faiz ve döviz riskini yönetmek için.  
4. **Doğal Hedge:** Gelir ve giderleri aynı para cinsinden ayarlamak.  

---

### 6.5 VaR Yöntemleri
- **Parametrik (Varyans-Kovaryans) VaR:** Normal dağılım varsayımıyla.  
- **Tarihsel Simülasyon:** Geçmiş verilerden doğrudan hesaplama.  
- **Monte Carlo Simülasyonu:** Rastgele senaryolarla.  

**Örnek:** 1 milyon TL portföy, günlük volatilite %2, %99 güven düzeyi →  
VaR = 2.33 × 0.02 × 1.000.000 = 46.600 TL  

---

### 6.6 Hedge Oranı ve Optimal Hedge
Hedge edilen pozisyon büyüklüğünün optimum seviyesini belirlemek için:  

\[
h^* = \frac{Cov(ΔS, ΔF)}{Var(ΔF)}
\]  

- ΔS: Spot fiyat değişimi  
- ΔF: Futures fiyat değişimi  

---

### 6.7 Türkiye’den ve Dünyadan Örnekler
- **Türkiye:** İhracatçı firmalar döviz vadeli işlemleriyle kur riskini hedge eder.  
- **Bankalar:** Faiz swapları ile bilanço uyumunu sağlar.  
- **Dünya:** Havayolu şirketleri jet yakıtı fiyat riskini futures ve opsiyonlarla yönetir.  

---

### 6.8 Genel Değerlendirme
Risk yönetimi, finansal istikrar ve yatırımcı güveni için kritik öneme sahiptir.  
VaR, stres testleri ve hedge stratejileri, kurumların risk iştahını yönetmesine ve olası kayıpları sınırlamasına olanak tanır.  
Ancak hedge stratejileri maliyetlidir; optimum düzeyde uygulanmalıdır.

---

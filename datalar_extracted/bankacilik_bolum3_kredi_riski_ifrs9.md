# Bankacılık ve Finans Kitabı  
## Bölüm 3: Kredi Riski Ölçümü ve IFRS 9 Beklenen Kredi Zararları

### 3.1 Kredi Riski Çerçevesi
**Kredi riski**, borçlunun yükümlülüklerini yerine getirememesinden kaynaklanan kayıp olasılığıdır. Yönetimi; ölçüm, fiyatlama, limit, teminat, erken uyarı ve tahsilat süreçlerini kapsar.

### 3.2 Temel Parametreler: PD, LGD, EAD
- **PD (Probability of Default):** Temerrüt olasılığı (12‑aylık veya ömür boyu).  
- **LGD (Loss Given Default):** Temerrütte kayıp oranı (1 − kurtarım). Teminata, tahsilat maliyetine, piyasa koşullarına bağlı.  
- **EAD (Exposure at Default):** Temerrüt anındaki maruziyet (kullanılmış kredi + taahhüt kullanımları).  

**Beklenen Zarar:**  
\[
EL = PD \times LGD \times EAD
\]
IFRS 9’da iskonto ve senaryo ağırlıklarıyla **ECL (Expected Credit Loss)** hesaplanır.

### 3.3 IFRS 9: Aşamalar ve SICR
- **Aşama 1:** İlk tanımlamada kredi riski anlamlı artmamışsa → **12‑aylık ECL**.  
- **Aşama 2:** **Önemli Kredi Riski Artışı (SICR)** varsa → **Ömür boyu ECL**. Eşikler: 30+ gün gecikme, içsel not düşüşü, erken uyarı sinyalleri.  
- **Aşama 3:** Temerrüt (genelde 90+ gün, yeniden yapılandırma, iflas) → Ömür boyu ECL ve faiz net taşıyıcı değere.  

### 3.4 ECL Bileşenleri
- **PIT PD (Point‑in‑Time):** Döngüye duyarlı, makro senaryolarla ileriye dönük.  
- **LGD:** Teminat değerleri (haircut), tahsilat süresi ve masrafları, indirgeme oranı.  
- **EAD:** Kredi kartı ve limitli ürünlerde **Kullanım Verim Katsayısı (CCF)**.  
- **Makro Senaryolar:** Temel/iyimser/kötümser; olasılık ağırlıkları.  
- **İskonto Oranı:** Etkin faiz oranı (EIR).

### 3.5 Modelleme Yaklaşımları
- **PD:** Lojit/Probit, survival (hazard), makine öğrenmesi (GBM, XGBoost). Kalibrasyon: PIT↔TTC dönüşümleri, **Brier**, **KS**, **AUC**.  
- **LGD:** Beta/Logit dönüşümlü regresyon, segment bazlı kurulum, **downturn LGD**.  
- **EAD:** Lineer modeller, panel veri, CCF tahmini.  
- **SICR:** Not geçişleri, PD eşiği, artış oranı, skor dilimleri.  

### 3.6 Geçiş Matrisleri ve Temerrüt Tanımı
Kredi not geçişleri ile ömür boyu PD elde edilir. Temerrüt tanımı: 90+ gün gecikme, iflas, yeniden yapılandırma, **unlikeliness to pay** göstergeleri.

### 3.7 Sayısal Örnek
Bir KOBİ kredisinin **EAD=1.000.000**, **12A PD=1.5%**, **LGD=40%** olsun.  
- **Aşama 1 ECL:** 0.015 × 0.40 × 1.000.000 = **6.000** (iskontosuz).  
SICR ile Aşama 2’ye geçtiğinde ömür boyu PD (diyelim **6%**):  
- **Aşama 2 ECL:** 0.06 × 0.40 × 1.000.000 = **24.000** (iskonto + senaryo ile ayarlanır).

### 3.8 Teminat ve Haircut
Nakit, devlet tahvili, ipotek, ticari alacak teminatları için saç kesimleri (**haircut**) belirlenir. Likidite ve volatiliteye göre artar. Net maruziyet **EADnet = EAD − teminat×(1−haircut)**.

### 3.9 Erken Uyarı ve İzleme
Gecikme **>30 gün**, nakit akışı bozulması, bilanço zayıflama göstergeleri (DSCR↓, kaldıraç↑), vergi/SGK borçları, sektör şokları.  
**Forbearance** ve yeniden yapılandırma politikaları açık yazılmalı.

### 3.10 Validasyon ve Yönetişim
Model bağımsız validasyon, performans izleme (backtesting), AUC/GINI, kalibrasyon sapmaları.  
Veri soykütüğü, model değişiklik yönetimi, **model risk** politikası.

---

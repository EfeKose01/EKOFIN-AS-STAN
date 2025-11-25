# Fine-tuning Veri Seti (JSONL)

Bu klasör, Markdown Q&A dosyalarınızdan otomatik türetilmiş **chat-format JSONL** veri setlerini içerir.

## Dosya Yapısı
- `*.train.jsonl` — eğitim verisi (%90)
- `*.valid.jsonl` — doğrulama verisi (%10)

Her satır bir örnek içerir:
```json
{"messages":[
  {"role":"system","content":"Uzman bir finans ve bankacılık asistanısın. Türkçe, net yanıt ver."},
  {"role":"user","content":"Phillips Eğrisi nedir?"},
  {"role":"assistant","content":"Phillips Eğrisi kısa dönemde işsizlik ile enflasyon arasında ters ilişki olduğunu söyler..."}
]}
```

## Öneriler
- Aynı sorunun **tek** yanıtı olmalı; çelişkili örnekleri birleştirin.
- Gerekirse `system` mesajını kurum dilinize göre özelleştirin.
- Toplam örnek sayınız arttıkça epoch sayısını makul tutun (örn. 2–4).

## OpenAI CLI/REST ile Fine-tuning (örnek)
> **API anahtarınızı bu dosyalara yazmayın.** Ortam değişkeni kullanın.

```bash
export OPENAI_API_KEY=YOUR_KEY

# Dosyayı yükle
openai files create --purpose fine-tune --file your_dataset.train.jsonl

# İş başlat
openai fine_tuning.jobs.create -m gpt-5-mini -t file-TRAIN_ID -v file-VALID_ID --n_epochs 3

# Takip et
openai fine_tuning.jobs.follow -i ftjob_xxx
```

> Not: Model ve uç noktalar için güncel resmi dökümanlara bakın.

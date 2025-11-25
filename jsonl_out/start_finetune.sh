#!/usr/bin/env bash
set -euo pipefail
# Kullanım: ./start_finetune.sh dataset_basename (uzantısız)
BASE=${1:-}
if [[ -z "$BASE" ]]; then
  echo "Kullanım: $0 <dataset_basename>"; exit 1
fi

TRAIN_FILE="${BASE}.train.jsonl"
VALID_FILE="${BASE}.valid.jsonl"

if [[ ! -f "$TRAIN_FILE" ]]; then echo "Bulunamadı: $TRAIN_FILE"; exit 1; fi
if [[ ! -f "$VALID_FILE" ]]; then echo "Bulunamadı: $VALID_FILE"; exit 1; fi

echo "Yükleniyor... $TRAIN_FILE"
TRAIN_ID=$(curl -s https://api.openai.com/v1/files   -H "Authorization: Bearer $OPENAI_API_KEY"   -F purpose=fine-tune   -F file=@${TRAIN_FILE} | jq -r .id)

echo "Yükleniyor... $VALID_FILE"
VALID_ID=$(curl -s https://api.openai.com/v1/files   -H "Authorization: Bearer $OPENAI_API_KEY"   -F purpose=fine-tune   -F file=@${VALID_FILE} | jq -r .id)

echo "Fine-tune başlatılıyor..."
curl -s https://api.openai.com/v1/fine_tuning/jobs   -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json"   -d "{"model":"gpt-5-mini","training_file":"${TRAIN_ID}","validation_file":"${VALID_ID}","hyperparameters":{"n_epochs":3}}" | jq

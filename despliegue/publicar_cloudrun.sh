#!/usr/bin/env bash
# Publica el servicio en Google Cloud Run con el mismo Dockerfile del repositorio.
#
# Por qué Cloud Run y no Hugging Face Spaces: desde 2026 un Space con SDK Docker en
# CPU Basic exige una cuenta PRO. El servicio necesita ~450 MB de RAM por worker
# (~900 MB con 2), así que las capas gratuitas de 512 MB no alcanzan. Cloud Run
# cobra solo mientras atiende solicitudes y su capa gratuita cubre la demo.
#
# Requisitos (una sola vez):
#   1. gcloud instalado y `gcloud auth login`.
#   2. Un proyecto con facturación vinculada y las APIs run, cloudbuild y
#      artifactregistry habilitadas.
#
# Uso:
#   ./despliegue/publicar_cloudrun.sh <proyecto-gcp> [region]
#
# Decisiones:
#   --max-instances 1  un solo registro SQLite; con más instancias el panel se partiría.
#   --min-instances 0  sin tráfico la instancia se apaga (costo cero) y el registro
#                      se borra: correr el simulador de tráfico antes de presentar.
#   --cpu 2 --memory 2Gi  igual que el Dockerfile: 2 workers de Uvicorn, uno por vCPU.
set -euo pipefail

PROYECTO="${1:?Uso: $0 <proyecto-gcp> [region]}"
REGION="${2:-us-central1}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$RAIZ"

gcloud run deploy monitoreo-transaccional \
  --project "$PROYECTO" --region "$REGION" \
  --source . \
  --port 7860 --cpu 2 --memory 2Gi \
  --min-instances 0 --max-instances 1 \
  --allow-unauthenticated \
  --quiet

URL="$(gcloud run services describe monitoreo-transaccional \
  --project "$PROYECTO" --region "$REGION" --format 'value(status.url)')"
echo
echo "Servicio: $URL"
curl -s --max-time 120 "$URL/salud" && echo

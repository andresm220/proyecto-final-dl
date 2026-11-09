#!/usr/bin/env bash
# Publica el servicio en un Hugging Face Space (SDK Docker).
#
# Requisitos (una sola vez):
#   1. Cuenta en https://huggingface.co y un token con permiso "write"
#      (Settings → Access Tokens).
#   2. Crear el Space vacío: New Space → SDK "Docker" → plantilla "Blank" → público.
#   3. git lfs instalado (los modelos son binarios).
#
# Uso:
#   ./despliegue/publicar_hf.sh <usuario>/<nombre-del-space>
#
# El Space recibe solo lo que el contenedor necesita (Dockerfile, src/, servicio/,
# artefactos/produccion/) y un README con la cabecera YAML que Spaces exige.
set -euo pipefail

SPACE="${1:?Uso: $0 <usuario>/<space>}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Clonando https://huggingface.co/spaces/$SPACE …"
git clone "https://huggingface.co/spaces/$SPACE" "$TMP/space"
cd "$TMP/space"
git lfs install --local
git lfs track "*.keras" "*.pkl"

rm -rf src servicio artefactos
cp -r "$RAIZ/src" "$RAIZ/servicio" .
mkdir -p artefactos && cp -r "$RAIZ/artefactos/produccion" artefactos/
cp "$RAIZ/Dockerfile" "$RAIZ/requirements-servicio.txt" "$RAIZ/.dockerignore" .
cp "$RAIZ/despliegue/README_space.md" README.md
find . -name "__pycache__" -prune -exec rm -rf {} +
rm -f servicio/registros.db

git add -A
git commit -m "Despliegue $(date -u +%Y-%m-%dT%H:%MZ) — $(python3 -c "import json;print(json.load(open('artefactos/produccion/metadata.json'))['version_modelo'])")" || true
git push
echo
echo "Listo. La imagen tarda ~5 min en construirse."
echo "Servicio: https://${SPACE/\//-}.hf.space   (en minúsculas)"

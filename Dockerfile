# Imagen del servicio de riesgo. Compatible con Hugging Face Spaces (SDK Docker, puerto 7860).
FROM python:3.13-slim

# LightGBM necesita la librería de OpenMP en tiempo de ejecución.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces corre el contenedor con el usuario 1000.
RUN useradd -m -u 1000 user
WORKDIR /app

# torch CPU (~200 MB) en vez del paquete con CUDA (~2.5 GB): el servicio no usa GPU.
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-servicio.txt .
RUN pip install --no-cache-dir -r requirements-servicio.txt

COPY --chown=user src/ src/
COPY --chown=user servicio/ servicio/
COPY --chown=user artefactos/produccion/ artefactos/produccion/

USER user
ENV KERAS_BACKEND=torch PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/salud')"
CMD ["uvicorn", "servicio.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]

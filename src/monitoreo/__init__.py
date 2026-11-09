"""Proyecto 1 - Monitoreo transaccional. Mazariegos / Herrera."""
import os

# Keras 3 es multi-backend. TensorFlow no publica ruedas para Python 3.14, que
# es el interprete de esta maquina, asi que el backend por defecto es torch.
# El codigo del Modelo B es API de Keras 3 pura y no depende del backend.
os.environ.setdefault("KERAS_BACKEND", "torch")

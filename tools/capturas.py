"""Capturas de pantalla del servicio para el README y la presentación.

    python tools/capturas.py [URL]     (por defecto http://localhost:7860)
"""
import sys
from playwright.sync_api import sync_playwright

B = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:7860"
D = "evidencias/"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome")
    pg = b.new_page(viewport={"width": 1360, "height": 900}, device_scale_factor=1.5)
    pg.set_default_timeout(20000)
    pg.goto(B + "/"); pg.wait_for_selector(".escenario")
    pg.click(".escenario >> nth=2"); pg.click("#btn-evaluar")
    pg.wait_for_selector("#resultado:not([hidden])"); pg.wait_for_timeout(500)
    pg.screenshot(path=D + "demo_sondeo_y_golpe.png", full_page=True)
    pg.click(".escenario >> nth=0"); pg.click("#btn-evaluar"); pg.wait_for_timeout(1000)
    pg.screenshot(path=D + "demo_compra_cotidiana.png", full_page=True)
    pg.goto(B + "/monitor"); pg.wait_for_selector("#t-ultimas tr")
    pg.wait_for_timeout(1500)
    pg.select_option("#origen", "simulador-deriva-x2.5"); pg.wait_for_timeout(2000)
    pg.screenshot(path=D + "monitor_deriva.png", full_page=True)
    pg.select_option("#origen", "simulador"); pg.select_option("#ventana", "5000"); pg.wait_for_timeout(2000)
    pg.screenshot(path=D + "monitor_trafico_normal.png", full_page=True)
    pg.goto(B + "/docs"); pg.wait_for_timeout(3000)
    pg.screenshot(path=D + "api_docs.png")
    b.close()
print("ok")

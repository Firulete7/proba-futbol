"""
Bot de probabilidades de fútbol (MVP).

Qué hace:
  1. Baja de football-data.org los partidos ya jugados y los próximos.
  2. Calcula la fuerza de ataque y de defensa de cada equipo
     (con más peso a los partidos recientes).
  3. Con un modelo de Poisson estima, para cada partido próximo:
     victoria local / empate / victoria visitante, más de 2,5 goles,
     ambos marcan, marcadores más probables y goles esperados.
  4. Guarda todo en predicciones.json (de ahí lo va a leer la página).

Uso:
  export FD_TOKEN="tu_token"
  python3 bot.py
"""
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

API = "https://api.football-data.org/v4"
LIGAS = {"PL": "Premier League", "PD": "LaLiga"}

DIAS_ADELANTE = 10      # cuántos días hacia adelante mostrar
MEDIA_VIDA_DIAS = 120   # un partido de hace 120 días pesa la mitad
PESO_PRIOR = 6          # cuánto tira hacia el promedio con pocos partidos
MAX_GOLES = 8           # goles máximos que considera el modelo por equipo
SALIDA = "predicciones.json"

TOKEN = os.environ.get("FD_TOKEN")
if not TOKEN:
    sys.exit('Falta el token. Corré primero: export FD_TOKEN="tu_token"')


def pedir(ruta, **params):
    """Hace una consulta a la API, esperando si nos pasamos del límite."""
    for _ in range(3):
        r = requests.get(
            API + ruta,
            headers={"X-Auth-Token": TOKEN},
            params=params,
            timeout=30,
        )
        if r.status_code == 429:
            print("  Límite de consultas alcanzado, espero 60 segundos...")
            time.sleep(60)
            continue
        if r.status_code != 200:
            sys.exit(f"Error {r.status_code} en {ruta}: {r.text[:200]}")
        return r.json()
    sys.exit("Demasiados reintentos, probá de nuevo en unos minutos.")


def poisson(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)


def ajustar_modelo(jugados, ahora):
    """Calcula ataque y defensa de cada equipo a partir de los resultados."""
    equipos = {}
    peso_total = goles_local = goles_visita = 0.0

    for m in jugados:
        h = m["score"]["fullTime"]["home"]
        a = m["score"]["fullTime"]["away"]
        if h is None or a is None:
            continue
        fecha = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        dias = max((ahora - fecha).days, 0)
        w = 0.5 ** (dias / MEDIA_VIDA_DIAS)

        peso_total += w
        goles_local += w * h
        goles_visita += w * a
        for eq, gf, gc in (
            (m["homeTeam"]["id"], h, a),
            (m["awayTeam"]["id"], a, h),
        ):
            e = equipos.setdefault(eq, {"gf": 0.0, "gc": 0.0, "w": 0.0})
            e["gf"] += w * gf
            e["gc"] += w * gc
            e["w"] += w

    if peso_total == 0:
        return None

    media_local = goles_local / peso_total
    media_visita = goles_visita / peso_total
    media = (media_local + media_visita) / 2

    fuerza = {}
    for eq, e in equipos.items():
        # Con pocos partidos, el valor se acerca al promedio de la liga.
        ataque = (e["gf"] + PESO_PRIOR * media) / (e["w"] + PESO_PRIOR) / media
        defensa = (e["gc"] + PESO_PRIOR * media) / (e["w"] + PESO_PRIOR) / media
        fuerza[eq] = (ataque, defensa)

    return {"fuerza": fuerza, "media_local": media_local, "media_visita": media_visita}


def predecir(modelo, id_local, id_visita):
    ataque_l, defensa_l = modelo["fuerza"].get(id_local, (1.0, 1.0))
    ataque_v, defensa_v = modelo["fuerza"].get(id_visita, (1.0, 1.0))
    lam_l = ataque_l * defensa_v * modelo["media_local"]
    lam_v = ataque_v * defensa_l * modelo["media_visita"]

    p_local = p_empate = p_visita = p_mas25 = p_ambos = total = 0.0
    marcadores = []
    for i in range(MAX_GOLES + 1):
        for j in range(MAX_GOLES + 1):
            p = poisson(i, lam_l) * poisson(j, lam_v)
            total += p
            marcadores.append((f"{i}-{j}", p))
            if i > j:
                p_local += p
            elif i == j:
                p_empate += p
            else:
                p_visita += p
            if i + j > 2:
                p_mas25 += p
            if i > 0 and j > 0:
                p_ambos += p

    marcadores.sort(key=lambda x: x[1], reverse=True)
    local = round(p_local / total * 100)
    visita = round(p_visita / total * 100)
    return {
        "local": local,
        "empate": 100 - local - visita,
        "visitante": visita,
        "mas_2_5": round(p_mas25 / total * 100),
        "ambos_marcan": round(p_ambos / total * 100),
        "goles_esperados": {"local": round(lam_l, 2), "visitante": round(lam_v, 2)},
        "marcadores": [
            {"resultado": r, "prob": round(p / total * 100, 1)} for r, p in marcadores[:3]
        ],
    }


def main():
    ahora = datetime.now(timezone.utc)
    hasta = ahora + timedelta(days=DIAS_ADELANTE)
    salida = {"actualizado": ahora.isoformat(timespec="seconds"), "ligas": []}

    for codigo, nombre in LIGAS.items():
        print(f"\n=== {nombre} ===")
        jugados = pedir(f"/competitions/{codigo}/matches", status="FINISHED")["matches"]
        time.sleep(7)  # el plan gratis permite 10 consultas por minuto
        proximos = pedir(
            f"/competitions/{codigo}/matches",
            status="SCHEDULED,TIMED",
            dateFrom=ahora.strftime("%Y-%m-%d"),
            dateTo=hasta.strftime("%Y-%m-%d"),
        )["matches"]
        time.sleep(7)

        print(f"Partidos jugados: {len(jugados)} | próximos {DIAS_ADELANTE} días: {len(proximos)}")
        modelo = ajustar_modelo(jugados, ahora)
        if modelo is None:
            print("Todavía no hay partidos jugados para calcular.")
            continue

        partidos = []
        for m in sorted(proximos, key=lambda x: x["utcDate"]):
            local, visita = m["homeTeam"], m["awayTeam"]
            prob = predecir(modelo, local["id"], visita["id"])
            partidos.append(
                {
                    "id": m["id"],
                    "fecha_utc": m["utcDate"],
                    "local": local.get("shortName") or local["name"],
                    "visitante": visita.get("shortName") or visita["name"],
                    "escudo_local": local.get("crest"),
                    "escudo_visitante": visita.get("crest"),
                    "prob": prob,
                }
            )
            print(
                f"{m['utcDate'][:10]}  {partidos[-1]['local']} vs {partidos[-1]['visitante']}: "
                f"{prob['local']}% / {prob['empate']}% / {prob['visitante']}%  "
                f"| +2,5: {prob['mas_2_5']}%  | ambos marcan: {prob['ambos_marcan']}%"
            )

        salida["ligas"].append({"codigo": codigo, "nombre": nombre, "partidos": partidos})

    with open(SALIDA, "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    print(f"\nListo. Resultados guardados en {SALIDA}")


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import norm

st.set_page_config(page_title="QuantBet Live: NFL, NBA, MLB", page_icon="📊", layout="wide")
st.title("⚡ Motor Estadístico en Vivo (NFL | NBA | MLB)")
st.caption("Conectado en tiempo real a TeamRankings. Se actualiza automáticamente cada vez que abres el link.")

if st.sidebar.button("🔄 Forzar Actualización Ahora"):
    st.cache_data.clear()

deporte = st.sidebar.selectbox("Selecciona la Liga", ["🏈 NFL", "🏀 NBA", "⚾ MLB"])
bankroll = st.sidebar.number_input("Tu Banca / Bankroll ($)", value=5000, step=500)

# Función que entra a la web y absorbe las estadísticas actuales automáticamente
@st.cache_data(ttl=3600)
def cargar_datos_en_vivo(liga):
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = {
        "🏈 NFL": (
            "https://www.teamrankings.com/nfl/stat/points-per-play",
            "https://www.teamrankings.com/nfl/stat/opponent-points-per-play",
            63, 13.5, 2.0 # Jugadas promedio, Desv. Estándar, Ventaja Local
        ),
        "🏀 NBA": (
            "https://www.teamrankings.com/nba/stat/offensive-efficiency",
            "https://www.teamrankings.com/nba/stat/defensive-efficiency",
            100, 11.5, 2.8
        ),
        "⚾ MLB": (
            "https://www.teamrankings.com/mlb/stat/runs-per-game",
            "https://www.teamrankings.com/mlb/stat/opponent-runs-per-game",
            1, 4.2, 0.25
        )
    }
    url_off, url_def, factor, sigma, hfa = urls[liga]
    
    df_off = pd.read_html(requests.get(url_off, headers=headers).text)[0]
    df_def = pd.read_html(requests.get(url_def, headers=headers).text)[0]
    
    # Tomamos el nombre del equipo, su promedio de la temporada actual (columna 2) y últimos 3 juegos (columna 3)
    off_clean = df_off.iloc[:, [1, 2, 3]].copy()
    off_clean.columns = ['Equipo', 'Off_Temp', 'Off_Racha3']
    
    def_clean = df_def.iloc[:, [1, 2, 3]].copy()
    def_clean.columns = ['Equipo', 'Def_Temp', 'Def_Racha3']
    
    df = pd.merge(off_clean, def_clean, on='Equipo')
    for col in ['Off_Temp', 'Off_Racha3', 'Def_Temp', 'Def_Racha3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    # Poder Neto = (60% Temporada + 40% Racha de últimos 3 juegos)
    df['Ofensiva_Ponderada'] = (df['Off_Temp'] * 0.6) + (df['Off_Racha3'] * 0.4)
    df['Defensiva_Ponderada'] = (df['Def_Temp'] * 0.6) + (df['Def_Racha3'] * 0.4)
    df['Rating_Neto'] = df['Ofensiva_Ponderada'] - df['Defensiva_Ponderada']
    return df.sort_values(by='Rating_Neto', ascending=False), factor, sigma, hfa

try:
    df_stats, factor, sigma, hfa = cargar_datos_en_vivo(deporte)
    equipos = df_stats['Equipo'].tolist()

    st.subheader(f"🎯 Simulador de Partido en Vivo ({deporte})")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        local = st.selectbox("Equipo Local", equipos, index=0)
    with c2:
        visita = st.selectbox("Equipo Visitante", equipos, index=1)
    with c3:
        linea_spread = st.number_input("Línea del Casino (Spread Local)", value=-2.5, step=0.5)
    with c4:
        momio = st.number_input("Momio del Casino (Americano)", value=-110, step=5)

    # Cálculo con las estadísticas recién extraídas de la web
    net_local = df_stats[df_stats['Equipo'] == local]['Rating_Neto'].values[0]
    net_visita = df_stats[df_stats['Equipo'] == visita]['Rating_Neto'].values[0]
    
    margen_proyectado = ((net_local - net_visita) * factor) + hfa
    prob_real = 1 - norm.cdf(-linea_spread, loc=margen_proyectado, scale=sigma)
    prob_casino = abs(momio) / (abs(momio) + 100) if momio < 0 else 100 / (momio + 100)
    dec_odds = 1 + (100 / abs(momio)) if momio < 0 else 1 + (momio / 100)
    ev = ((prob_real * (dec_odds - 1)) - (1 - prob_real)) * 100
    kelly = max(0, (((dec_odds - 1) * prob_real - (1 - prob_real)) / (dec_odds - 1)) * 0.25)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Margen Proyectado", f"{local} por {margen_proyectado:+.1f} pts")
    m2.metric("Prob. Real vs Casino", f"{prob_real*100:.1f}% vs {prob_casino*100:.1f}%")
    m3.metric("Valor Esperado (EV)", f"{ev:+.2f}%")
    m4.metric("Apuesta Sugerida (1/4 Kelly)", f"${bankroll * kelly:,.0f}")

    if ev >= 3.0:
        st.success(f"✅ **APUESTA FUERTE (VALUE BET):** Tomar **{local} {linea_spread}** tiene una ventaja matemática de {ev:+.2f}%.")
    elif ev > 0:
        st.warning("⚠️ **VENTAJA LIGERA:** Hay valor positivo pero es menor al 3% recomendado.")
    else:
        st.error(f"❌ **NO APOSTAR A {local}:** La línea está cara. El valor está del lado de **{visita} ({-linea_spread:+.1f})**.")

    st.subheader(f"📋 Tabla de Estadísticas en Vivo ({deporte} - Extraída Hoy)")
    st.dataframe(df_stats, use_container_width=True)

except Exception as e:
    st.error(f"Error al conectar con la fuente de datos: {e}")

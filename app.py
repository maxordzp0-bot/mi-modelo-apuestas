import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from scipy.stats import norm

st.set_page_config(page_title="QuantBet Pro: Motor Integral", page_icon="🧠", layout="wide")
st.title("🧠 Motor Predictivo Integral: Estadísticas + Lesiones + Clima + Localía")
st.caption("Conectado en vivo a TeamRankings (Splits Local/Visita), CBS Sports (Lesiones Oficiales) y ESPN API (Clima y Líneas).")

if st.sidebar.button("🔄 Forzar Actualización en Vivo"):
    st.cache_data.clear()

deporte = st.sidebar.selectbox("Selecciona la Liga", ["🏈 NFL", "🏀 NBA", "⚾ MLB"])
bankroll = st.sidebar.number_input("Tu Banca / Bankroll ($)", value=5000, step=500)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 1. ESTADÍSTICAS CON SPLITS REALES DE LOCAL (HOME) Y VISITA (AWAY)
@st.cache_data(ttl=3600)
def cargar_estadisticas_completas(liga):
    urls = {
        "🏈 NFL": (
            "https://www.teamrankings.com/nfl/stat/points-per-play",
            "https://www.teamrankings.com/nfl/stat/opponent-points-per-play",
            "football/nfl", "nfl", 63, 13.5, 44.5
        ),
        "🏀 NBA": (
            "https://www.teamrankings.com/nba/stat/offensive-efficiency",
            "https://www.teamrankings.com/nba/stat/defensive-efficiency",
            "basketball/nba", "nba", 1, 11.5, 224.5
        ),
        "⚾ MLB": (
            "https://www.teamrankings.com/mlb/stat/runs-per-game",
            "https://www.teamrankings.com/mlb/stat/opponent-runs-per-game",
            "baseball/mlb", "mlb", 1, 4.2, 8.5
        )
    }
    url_off, url_def, espn_path, cbs_sport, factor, sigma, base_total = urls[liga]
    
    html_off = requests.get(url_off, headers=HEADERS, timeout=15).text
    html_def = requests.get(url_def, headers=HEADERS, timeout=15).text
    
    df_off = pd.read_html(StringIO(html_off))[0]
    df_def = pd.read_html(StringIO(html_def))[0]
    
    # Extraemos: Equipo, Temporada, Racha 3 juegos, Rendimiento en Casa (Home) y de Visita (Away)
    off_clean = df_off.iloc[:, [1, 2, 3, 4, 5]].copy()
    off_clean.columns = ['Equipo', 'Off_Temp', 'Off_Racha3', 'Off_Home', 'Off_Away']
    
    def_clean = df_def.iloc[:, [1, 2, 3, 4, 5]].copy()
    def_clean.columns = ['Equipo', 'Def_Temp', 'Def_Racha3', 'Def_Home', 'Def_Away']
    
    df = pd.merge(off_clean, def_clean, on='Equipo')
    cols_num = ['Off_Temp', 'Off_Racha3', 'Off_Home', 'Off_Away', 'Def_Temp', 'Def_Racha3', 'Def_Home', 'Def_Away']
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Rellenar datos faltantes con el promedio de la temporada
    for prefix in ['Off', 'Def']:
        df[f'{prefix}_Racha3'] = df[f'{prefix}_Racha3'].fillna(df[f'{prefix}_Temp'])
        df[f'{prefix}_Home'] = df[f'{prefix}_Home'].fillna(df[f'{prefix}_Temp'])
        df[f'{prefix}_Away'] = df[f'{prefix}_Away'].fillna(df[f'{prefix}_Temp'])
        
    df = df.dropna(subset=['Off_Temp', 'Def_Temp'])
    
    # Poder Específico: Cómo juega el Local EN CASA vs Cómo juega el Visitante DE VISITA
    # (40% Temporada + 30% Racha Reciente + 30% Desempeño Específico en Casa/Visita)
    df['Poder_Como_Local'] = (
        (df['Off_Temp']*0.4 + df['Off_Racha3']*0.3 + df['Off_Home']*0.3) -
        (df['Def_Temp']*0.4 + df['Def_Racha3']*0.3 + df['Def_Home']*0.3)
    )
    df['Poder_Como_Visita'] = (
        (df['Off_Temp']*0.4 + df['Off_Racha3']*0.3 + df['Off_Away']*0.3) -
        (df['Def_Temp']*0.4 + df['Def_Racha3']*0.3 + df['Def_Away']*0.3)
    )
    return df.sort_values(by='Poder_Como_Local', ascending=False), espn_path, cbs_sport, factor, sigma, base_total

# 2. EXTRACTOR EN VIVO DE LESIONADOS OFICIALES (CBS SPORTS)
@st.cache_data(ttl=1800)
def obtener_lesionados_en_vivo(cbs_sport):
    url = f"https://www.cbssports.com/{cbs_sport}/injuries/"
    try:
        html = requests.get(url, headers=HEADERS, timeout=15).text
        tablas = pd.read_html(StringIO(html))
        return tablas
    except Exception:
        return []

# 3. CARTELERA, CLIMA, ESTADIO Y MOMIOS EN VIVO (ESPN API)
@st.cache_data(ttl=1800)
def obtener_partidos_y_clima(espn_path):
    url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/scoreboard"
    res = requests.get(url, timeout=10).json()
    partidos = []
    for ev in res.get("events", []):
        comp = ev["competitions"][0]
        equipos = comp["competitors"]
        home = next(t for t in equipos if t["homeAway"] == "home")["team"]["location"]
        away = next(t for t in equipos if t["homeAway"] == "away")["team"]["location"]
        
        venue = comp.get("venue", {}).get("fullName", "Estadio Estándar")
        indoor = comp.get("venue", {}).get("indoor", False)
        
        clima_txt = "Techado / Domo (Sin impacto climático)" if indoor else "Aire Libre"
        if "weather" in ev and not indoor:
            temp = ev["weather"].get("temperature", "")
            cond = ev["weather"].get("displayValue", "")
            clima_txt = f"{cond} ({temp}°F)"
            
        spread_txt, total_txt = "N/A", "N/A"
        if "odds" in comp and len(comp["odds"]) > 0 and comp["odds"][0]:
            spread_txt = comp["odds"][0].get("details", "N/A")
            total_txt = comp["odds"][0].get("overUnder", "N/A")
            
        partidos.append({
            "Partido": ev["shortName"], "Local": home, "Visita": away,
            "Estadio": venue, "Clima Reportado": clima_txt,
            "Spread Casino": spread_txt, "Over/Under": total_txt
        })
    return pd.DataFrame(partidos)

try:
    df_stats, espn_path, cbs_sport, factor, sigma, base_total = cargar_estadisticas_completas(deporte)
    equipos = df_stats['Equipo'].tolist()

    # SECCIÓN 1: CARTELERA Y CLIMA EN VIVO
    st.subheader(f"📅 Cartelera en Vivo, Estadio y Clima ({deporte})")
    df_espn = obtener_partidos_y_clima(espn_path)
    if not df_espn.empty:
        st.dataframe(df_espn, use_container_width=True)

    st.divider()
    
    # SECCIÓN 2: SELECCIÓN DEL ENFRENTAMIENTO Y FACTORES EXTERNOS
    st.subheader("🎯 Configurador Completo del Partido")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        local = st.selectbox("🏠 Equipo Local", equipos, index=0)
    with col2:
        visita = st.selectbox("✈️ Equipo Visitante", equipos, index=1)
    with col3:
        linea_spread = st.number_input("Línea Spread Local (Casino)", value=-2.5, step=0.5)
    with col4:
        momio = st.number_input("Momio del Casino (Americano)", value=-110, step=5)

    # PANEL DE FACTORES SITUACIONALES (CLIMA, LESIONES Y DESCANSO)
    st.markdown("#### 🧩 Ajuste de Variables de Contexto (Lesiones, Clima y Descanso)")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        bajas_local = st.slider(f"🚑 Impacto Bajas en {local} (Pts)", -7.0, 0.0, 0.0, 0.5,
                                help="Resta puntos si falta el QB1/Estrella (-4 a -6 pts) o titulares clave (-1 a -2.5 pts).")
    with f2:
        bajas_visita = st.slider(f"🚑 Impacto Bajas en {visita} (Pts)", -7.0, 0.0, 0.0, 0.5,
                                 help="Resta puntos si el visitante tiene bajas clave.")
    with f3:
        condicion_clima = st.selectbox("🌦️ Clima / Viento en el Estadio", [
            "☀️ Despejado o Domo Techado (0 pts)",
            "💨 Viento Moderado / Lluvia Ligera (-1.5 pts al aéreo)",
            "🌪️ Viento Fuerte >25 km/h o Tormenta (-3.5 pts y baja el Total)",
            "❄️ Nieve / Frío Extremo (Favorece al local +1.5 pts)"
        ])
    with f4:
        ventaja_descanso = st.selectbox("⏱️ Descanso y Viaje", [
            "Igualdad de descanso (0 pts)",
            f"{local} viene de Bye Week / más descanso (+1.5 pts)",
            f"{visita} viene de Bye Week / más descanso (-1.5 pts)",
            f"{visita} juega en Semana Corta / Back-to-Back (+2.5 pts Local)"
        ])

    # Traducir selectores situacionales a puntos matemáticos
    ajuste_clima = 0.0
    if "Moderado" in condicion_clima: ajuste_clima = -0.5
    elif "Fuerte" in condicion_clima: ajuste_clima = -1.2
    elif "Nieve" in condicion_clima: ajuste_clima = 1.5

    ajuste_descanso = 0.0
    if f"{local} viene" in ventaja_descanso: ajuste_descanso = 1.5
    elif f"{visita} viene" in ventaja_descanso: ajuste_descanso = -1.5
    elif "Semana Corta" in ventaja_descanso: ajuste_descanso = 2.5

    # CÁLCULO MATEMÁTICO INTEGRAL
    # Usa directamente el rendimiento real de Local en Casa vs Visitante Fuera
    net_local_casa = df_stats[df_stats['Equipo'] == local]['Poder_Como_Local'].values[0]
    net_visita_fuera = df_stats[df_stats['Equipo'] == visita]['Poder_Como_Visita'].values[0]
    
    # Diferencial estadístico base + Localía estructural + Lesiones + Clima + Descanso
    hfa_base = 1.8 if deporte == "🏈 NFL" else (2.5 if deporte == "🏀 NBA" else 0.25)
    diferencial_lesiones = bajas_local - bajas_visita # Si visita tiene más bajas, suma a favor del local
    
    margen_final = ((net_local_casa - net_visita_fuera) * factor) + hfa_base + diferencial_lesiones + ajuste_clima + ajuste_descanso
    
    prob_real = 1 - norm.cdf(-linea_spread, loc=margen_final, scale=sigma)
    prob_casino = abs(momio) / (abs(momio) + 100) if momio < 0 else 100 / (momio + 100)
    dec_odds = 1 + (100 / abs(momio)) if momio < 0 else 1 + (momio / 100)
    ev = ((prob_real * (dec_odds - 1)) - (1 - prob_real)) * 100
    kelly = max(0, (((dec_odds - 1) * prob_real - (1 - prob_real)) / (dec_odds - 1)) * 0.25)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Margen Final Proyectado", f"{local} por {margen_final:+.1f} pts")
    m2.metric("Prob. Real vs Casino", f"{prob_real*100:.1f}% vs {prob_casino*100:.1f}%")
    m3.metric("Valor Esperado (EV)", f"{ev:+.2f}%")
    m4.metric("Stake Sugerido (1/4 Kelly)", f"${bankroll * kelly:,.0f}")

    if ev >= 3.0:
        st.success(f"✅ **APUESTA MUY SÓLIDA (VALUE BET):** Considerando estadísticas de local/visita, lesiones, descanso y clima, **{local} ({linea_spread:+.1f})** tiene un valor de **{ev:+.2f}%**.")
    elif ev > 0:
        st.warning("⚠️ **VENTAJA LIGERA:** Hay valor positivo, pero menor al 3% de margen de seguridad.")
    else:
        st.error(f"❌ **VALOR EN CONTRA DE {local}:** Con los ajustes actuales, la mejor jugada es ir con **{visita} ({-linea_spread:+.1f})**.")

    # SECCIÓN 3: MONITOR DE LESIONES OFICIALES EN VIVO
    with st.expander("🚑 Ver Reporte Oficial de Lesionados en Vivo (Extraído hoy de CBS Sports)"):
        tablas_lesiones = obtener_lesionados_en_vivo(cbs_sport)
        if tablas_lesiones:
            st.write(f"Se encontraron **{len(tablas_lesiones)} plantillas con reporte de lesiones activo hoy**.")
            df_todas_lesiones = pd.concat(tablas_lesiones, ignore_index=True)
            busqueda = st.text_input("🔍 Filtrar jugador o posición en el reporte de lesionados:", "")
            if busqueda:
                df_todas_lesiones = df_todas_lesiones[df_todas_lesiones.astype(str).apply(lambda x: x.str.contains(busqueda, case=False)).any(axis=1)]
            st.dataframe(df_todas_lesiones, use_container_width=True)
        else:
            st.info("No hay reportes de lesiones publicados en este momento.")

    # SECCIÓN 4: TABLA COMPLETA CON SPLITS DE LOCAL Y VISITA
    with st.expander(f"📊 Ver Tabla Estadística Completa (Con Rendimiento en Casa y de Visita - {deporte})"):
        st.dataframe(df_stats, use_container_width=True)

except Exception as e:
    st.error(f"Error al procesar datos en vivo: {e}")

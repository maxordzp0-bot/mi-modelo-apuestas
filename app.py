import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from scipy.stats import norm

st.set_page_config(page_title="QuantBet Pro: Selector de Partido", page_icon="🎯", layout="wide")
st.title("🎯 Predictor por Partido: Top Apuestas (Spread, Ganador y Altas/Bajas)")
st.caption("Selecciona cualquier partido de la cartelera en vivo y obtén automáticamente las mejores apuestas calculadas con estadísticas, localía, clima y lesiones.")

if st.sidebar.button("🔄 Forzar Actualización en Vivo"):
    st.cache_data.clear()

deporte = st.sidebar.selectbox("Selecciona la Liga", ["🏈 NFL", "⚾ MLB", "🏀 NBA"])
bankroll = st.sidebar.number_input("Tu Banca / Bankroll ($)", value=500, step=100)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Diccionario inteligente para empatar nombres de ESPN con TeamRankings
TEAM_ALIASES = {
    "LA": ["LA Rams", "LA Chargers", "LA Dodgers", "LA Angels", "LA Lakers", "LA Clippers"],
    "NY": ["NY Giants", "NY Jets", "NY Yankees", "NY Mets", "NY Knicks", "Brooklyn"],
    "CHI": ["Chicago", "Chi Cubs", "Chi Sox", "Chi Bears", "Chi Bulls"]
}

def buscar_equipo_en_tabla(nombre_espn, nombre_corto, abrev, lista_equipos):
    # 1. Búsqueda exacta o contenida
    for eq in lista_equipos:
        if eq.lower() in nombre_espn.lower() or nombre_corto.lower() in eq.lower():
            return eq
    # 2. Búsqueda por ciudad primera palabra
    primera = nombre_espn.split()[0].lower()
    for eq in lista_equipos:
        if primera in eq.lower():
            return eq
    return lista_equipos[0]

# 1. DESCARGAR ESTADÍSTICAS REALES (OFENSIVA Y DEFENSIVA + LOCAL/VISITA)
@st.cache_data(ttl=3600)
def cargar_estadisticas(liga):
    urls = {
        "🏈 NFL": (
            "https://www.teamrankings.com/nfl/stat/points-per-game",
            "https://www.teamrankings.com/nfl/stat/opponent-points-per-game",
            "football/nfl", "nfl", 13.5, 2.0
        ),
        "🏀 NBA": (
            "https://www.teamrankings.com/nba/stat/points-per-game",
            "https://www.teamrankings.com/nba/stat/opponent-points-per-game",
            "basketball/nba", "nba", 11.5, 2.5
        ),
        "⚾ MLB": (
            "https://www.teamrankings.com/mlb/stat/runs-per-game",
            "https://www.teamrankings.com/mlb/stat/opponent-runs-per-game",
            "baseball/mlb", "mlb", 4.0, 0.25
        )
    }
    url_off, url_def, espn_path, cbs_sport, sigma, hfa = urls[liga]
    
    html_off = requests.get(url_off, headers=HEADERS, timeout=15).text
    html_def = requests.get(url_def, headers=HEADERS, timeout=15).text
    
    df_off = pd.read_html(StringIO(html_off))[0].iloc[:, [1, 2, 3, 4, 5]].copy()
    df_off.columns = ['Equipo', 'Off_Temp', 'Off_Racha3', 'Off_Home', 'Off_Away']
    
    df_def = pd.read_html(StringIO(html_def))[0].iloc[:, [1, 2, 3, 4, 5]].copy()
    df_def.columns = ['Equipo', 'Def_Temp', 'Def_Racha3', 'Def_Home', 'Def_Away']
    
    df = pd.merge(df_off, df_def, on='Equipo')
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    for prefix in ['Off', 'Def']:
        df[f'{prefix}_Racha3'] = df[f'{prefix}_Racha3'].fillna(df[f'{prefix}_Temp'])
        df[f'{prefix}_Home'] = df[f'{prefix}_Home'].fillna(df[f'{prefix}_Temp'])
        df[f'{prefix}_Away'] = df[f'{prefix}_Away'].fillna(df[f'{prefix}_Temp'])
        
    df = df.dropna(subset=['Off_Temp', 'Def_Temp'])
    
    # Puntos anotados y permitidos proyectados en Casa vs de Visita
    df['Pts_Anotados_Casa'] = df['Off_Temp']*0.4 + df['Off_Racha3']*0.3 + df['Off_Home']*0.3
    df['Pts_Permitidos_Casa'] = df['Def_Temp']*0.4 + df['Def_Racha3']*0.3 + df['Def_Home']*0.3
    
    df['Pts_Anotados_Visita'] = df['Off_Temp']*0.4 + df['Off_Racha3']*0.3 + df['Off_Away']*0.3
    df['Pts_Permitidos_Visita'] = df['Def_Temp']*0.4 + df['Def_Racha3']*0.3 + df['Def_Away']*0.3
    
    return df, espn_path, cbs_sport, sigma, hfa

# 2. DESCARGAR PARTIDOS, CLIMA Y LÍNEAS DE CASINO EN VIVO (ESPN API)
@st.cache_data(ttl=900)
def obtener_cartelera_completa(espn_path):
    url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/scoreboard"
    res = requests.get(url, timeout=10).json()
    partidos = []
    for ev in res.get("events", []):
        comp = ev["competitions"][0]
        equipos = comp["competitors"]
        home_obj = next(t for t in equipos if t["homeAway"] == "home")["team"]
        away_obj = next(t for t in equipos if t["homeAway"] == "away")["team"]
        
        venue = comp.get("venue", {}).get("fullName", "Estadio Oficial")
        indoor = comp.get("venue", {}).get("indoor", False)
        
        temp_val = 72
        clima_txt = "Techado / Domo" if indoor else "Despejado"
        if "weather" in ev and not indoor:
            temp_val = ev["weather"].get("temperature", 72)
            clima_txt = f"{temp_val}°F (Aire Libre)"
            
        details = "PK"
        over_under = 0.0
        home_ml, away_ml = -110, -110
        
        if "odds" in comp and len(comp["odds"]) > 0 and comp["odds"][0]:
            odd_obj = comp["odds"][0]
            details = odd_obj.get("details", "PK")
            over_under = float(odd_obj.get("overUnder", 0.0) or 0.0)
            
        partidos.append({
            "Etiqueta": f"{ev['shortName']} | {away_obj['displayName']} @ {home_obj['displayName']} ({venue})",
            "Partido": ev["shortName"],
            "Home_Full": home_obj["displayName"],
            "Home_Short": home_obj.get("name", home_obj["location"]),
            "Home_Abbr": home_obj.get("abbreviation", ""),
            "Away_Full": away_obj["displayName"],
            "Away_Short": away_obj.get("name", away_obj["location"]),
            "Away_Abbr": away_obj.get("abbreviation", ""),
            "Estadio": venue,
            "Indoor": indoor,
            "Clima": clima_txt,
            "Linea_Texto": details,
            "Over_Under": over_under
        })
    return partidos

def calcular_ev_y_kelly(prob_real, momio, bankroll):
    prob_imp = abs(momio) / (abs(momio) + 100) if momio < 0 else 100 / (momio + 100)
    dec = 1 + (100 / abs(momio)) if momio < 0 else 1 + (momio / 100)
    ev = ((prob_real * (dec - 1)) - (1 - prob_real)) * 100
    kelly = max(0, (((dec - 1) * prob_real - (1 - prob_real)) / (dec - 1)) * 0.25)
    return round(prob_imp * 100, 1), round(ev, 2), round(bankroll * kelly, 0)

try:
    df_stats, espn_path, cbs_sport, sigma, hfa = cargar_estadisticas(deporte)
    lista_equipos_tr = df_stats['Equipo'].tolist()
    partidos_hoy = obtener_cartelera_completa(espn_path)

    # PASO 1: SELECCIONAR EL PARTIDO DIRECTAMENTE
    st.subheader(f"1️⃣ Elige el Partido de {deporte} que quieres analizar")
    
    if partidos_hoy:
        opciones_partidos = [p["Etiqueta"] for p in partidos_hoy]
        seleccion = st.selectbox("🎯 Selecciona un partido de la cartelera oficial en vivo:", opciones_partidos)
        partido_actual = next(p for p in partidos_hoy if p["Etiqueta"] == seleccion)
        
        # Emparejar automáticamente los equipos con la base estadística
        eq_local_default = buscar_equipo_en_tabla(partido_actual["Home_Full"], partido_actual["Home_Short"], partido_actual["Home_Abbr"], lista_equipos_tr)
        eq_visita_default = buscar_equipo_en_tabla(partido_actual["Away_Full"], partido_actual["Away_Short"], partido_actual["Away_Abbr"], lista_equipos_tr)
        
        # Detectar el número de línea o momio desde ESPN
        num_extraido = -1.5 if deporte == "⚾ MLB" else -2.5
        match_num = re.findall(r"[-+]?\d*\.\d+|\d+", partido_actual["Linea_Texto"])
        if match_num:
            val_num = float(match_num[-1])
            if val_num < 50: # Es un spread
                num_extraido = -abs(val_num) if partido_actual["Home_Abbr"] in partido_actual["Linea_Texto"] else abs(val_num)
                
        ou_default = partido_actual["Over_Under"] if partido_actual["Over_Under"] > 0 else (44.5 if deporte == "🏈 NFL" else (8.0 if deporte == "⚾ MLB" else 224.5))
        
        st.info(f"🏟️ **Estadio:** {partido_actual['Estadio']} | 🌦️ **Clima:** {partido_actual['Clima']} | 📈 **Línea Oficial Casino:** `{partido_actual['Linea_Texto']}` | **Total (O/U):** `{ou_default}`")
    else:
        st.warning("No hay partidos en vivo hoy en esta liga, pero puedes simular cualquier cruce abajo.")
        eq_local_default, eq_visita_default = lista_equipos_tr[0], lista_equipos_tr[1]
        num_extraido, ou_default = -2.5, 44.5

    # PASO 2: VERIFICACIÓN Y AJUSTE FINO DEL PARTIDO SELECCIONADO
    with st.expander("⚙️ Ajustar Líneas del Casino, Lesiones o Clima de este Partido", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            local = st.selectbox("🏠 Equipo Local (Estadísticas)", lista_equipos_tr, index=lista_equipos_tr.index(eq_local_default))
        with c2:
            visita = st.selectbox("✈️ Equipo Visitante (Estadísticas)", lista_equipos_tr, index=lista_equipos_tr.index(eq_visita_default))
        with c3:
            spread_local = st.number_input("Spread / RunLine del Local", value=float(num_extraido), step=0.5)
        with c4:
            linea_total = st.number_input("Línea de Altas/Bajas (Over/Under)", value=float(ou_default), step=0.5)

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            momio_ml_local = st.number_input(f"Momio A Ganar ({local})", value=-130, step=5)
        with f2:
            momio_ml_visita = st.number_input(f"Momio A Ganar ({visita})", value=110, step=5)
        with f3:
            bajas_netas = st.slider("🚑 Ventaja por Lesiones (Hacia Local + / Visita -)", -6.0, 6.0, 0.0, 0.5)
        with f4:
            impacto_clima = st.slider("🌦️ Ajuste de Clima al Total de Puntos/Carreras", -5.0, 5.0, 0.0, 0.5)

    # PASO 3: MOTOR DE PROYECCIÓN DEL MARCADOR EXACTO DEL PARTIDO
    row_loc = df_stats[df_stats['Equipo'] == local].iloc[0]
    row_vis = df_stats[df_stats['Equipo'] == visita].iloc[0]

    # Puntos proyectados de Local = Promedio entre lo que anota el Local en casa y lo que permite la Visita fuera
    pts_local = ((row_loc['Pts_Anotados_Casa'] + row_vis['Pts_Permitidos_Visita']) / 2) + (hfa / 2) + (bajas_netas / 2) + (impacto_clima / 2)
    pts_visita = ((row_vis['Pts_Anotados_Visita'] + row_loc['Pts_Permitidos_Casa']) / 2) - (hfa / 2) - (bajas_netas / 2) + (impacto_clima / 2)
    
    margen_proyectado = pts_local - pts_visita
    total_proyectado = pts_local + pts_visita

    st.divider()
    st.subheader(f"🔮 Marcador Proyectado por el Modelo: **{local} {pts_local:.1f} — {visita} {pts_visita:.1f}**")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Diferencial Proyectado", f"{local} {margen_proyectado:+.1f} pts/carreras")
    m2.metric("Total Proyectado (Over/Under)", f"{total_proyectado:.1f} (Línea: {linea_total})")
    m3.metric("Probabilidad de Victoria Directa", f"{local}: {(1 - norm.cdf(0, loc=margen_proyectado, scale=sigma))*100:.1f}%")

    # PASO 4: GENERAR TODAS LAS APUESTAS RECOMENDADAS DE ESTE PARTIDO
    st.subheader(f"🏆 Top Apuestas Recomendadas para **{visita} @ {local}**")

    apuestas_partido = []

    # 1. Evaluar Ganador Directo (Moneyline Local vs Visita)
    prob_ganar_local = 1 - norm.cdf(0, loc=margen_proyectado, scale=sigma)
    prob_ganar_visita = 1 - prob_ganar_local
    
    imp_ml_loc, ev_ml_loc, stake_ml_loc = calcular_ev_y_kelly(prob_ganar_local, momio_ml_local, bankroll)
    imp_ml_vis, ev_ml_vis, stake_ml_vis = calcular_ev_y_kelly(prob_ganar_visita, momio_ml_visita, bankroll)
    
    if ev_ml_loc >= ev_ml_vis:
        apuestas_partido.append({
            "Mercado": "🥇 Ganador Directo (Moneyline)",
            "Jugada Recomendada": f"{local} a Ganar ({momio_ml_local:+d})",
            "Prob. Modelo": f"{prob_ganar_local*100:.1f}%",
            "Prob. Casino": f"{imp_ml_loc}%",
            "Valor Esperado (EV)": ev_ml_loc,
            "Cuánto Apostar (1/4 Kelly)": f"${stake_ml_loc:,.0f}"
        })
    else:
        apuestas_partido.append({
            "Mercado": "🥇 Ganador Directo (Moneyline)",
            "Jugada Recomendada": f"{visita} a Ganar ({momio_ml_visita:+d})",
            "Prob. Modelo": f"{prob_ganar_visita*100:.1f}%",
            "Prob. Casino": f"{imp_ml_vis}%",
            "Valor Esperado (EV)": ev_ml_vis,
            "Cuánto Apostar (1/4 Kelly)": f"${stake_ml_vis:,.0f}"
        })

    # 2. Evaluar Spread / Run Line (-110 estándar)
    prob_cubrir_local = 1 - norm.cdf(-spread_local, loc=margen_proyectado, scale=sigma)
    prob_cubrir_visita = 1 - prob_cubrir_local
    
    if prob_cubrir_local >= 0.5:
        imp_sp, ev_sp, stake_sp = calcular_ev_y_kelly(prob_cubrir_local, -110, bankroll)
        jugada_sp = f"{local} {spread_local:+.1f} (-110)"
        prob_sp_txt = f"{prob_cubrir_local*100:.1f}%"
    else:
        imp_sp, ev_sp, stake_sp = calcular_ev_y_kelly(prob_cubrir_visita, -110, bankroll)
        jugada_sp = f"{visita} {-spread_local:+.1f} (-110)"
        prob_sp_txt = f"{prob_cubrir_visita*100:.1f}%"

    apuestas_partido.append({
        "Mercado": "📏 Hándicap / Spread",
        "Jugada Recomendada": jugada_sp,
        "Prob. Modelo": prob_sp_txt,
        "Prob. Casino": f"{imp_sp}%",
        "Valor Esperado (EV)": ev_sp,
        "Cuánto Apostar (1/4 Kelly)": f"${stake_sp:,.0f}"
    })

    # 3. Evaluar Altas / Bajas (Over / Under -110 estándar)
    prob_over = 1 - norm.cdf(linea_total, loc=total_proyectado, scale=sigma)
    prob_under = 1 - prob_over
    
    if prob_over >= 0.5:
        imp_ou, ev_ou, stake_ou = calcular_ev_y_kelly(prob_over, -110, bankroll)
        jugada_ou = f"ALTAS / OVER {linea_total} (-110)"
        prob_ou_txt = f"{prob_over*100:.1f}%"
    else:
        imp_ou, ev_ou, stake_ou = calcular_ev_y_kelly(prob_under, -110, bankroll)
        jugada_ou = f"BAJAS / UNDER {linea_total} (-110)"
        prob_ou_txt = f"{prob_under*100:.1f}%"

    apuestas_partido.append({
        "Mercado": "🔢 Total de Puntos/Carreras (O/U)",
        "Jugada Recomendada": jugada_ou,
        "Prob. Modelo": prob_ou_txt,
        "Prob. Casino": f"{imp_ou}%",
        "Valor Esperado (EV)": ev_ou,
        "Cuánto Apostar (1/4 Kelly)": f"${stake_ou:,.0f}"
    })

    df_apuestas = pd.DataFrame(apuestas_partido).sort_values(by="Valor Esperado (EV)", ascending=False)
    st.dataframe(df_apuestas, use_container_width=True)

    mejor = df_apuestas.iloc[0]
    if mejor["Valor Esperado (EV)"] > 0:
        st.success(f"🔥 **LA APUESTA MÁS SEGURA DEL PARTIDO:** **{mejor['Jugada Recomendada']}** ({mejor['Mercado']}) con **{mejor['Prob. Modelo']}** de probabilidad y **+{mejor['Valor Esperado (EV)']}%** de ventaja matemática. Apuesta sugerida: **{mejor['Cuánto Apostar (1/4 Kelly)']}**.")
    else:
        st.warning("⚠️ Ningún mercado en este partido supera el margen del casino con los momios actuales. Ajusta el momio a lo que pague tu casino.")

except Exception as e:
    st.error(f"Error al procesar el partido: {e}")

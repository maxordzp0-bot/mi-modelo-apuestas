import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
from io import StringIO
from scipy.stats import norm, poisson

st.set_page_config(page_title="QuantBet Pro: Abridores & Player Props", page_icon="🎯", layout="wide")
st.title("🎯 Predictor Integral: Partido + Pitchers Abridores + Player Props")
st.caption("Extrae en vivo estadísticas de local/visita, pitchers abridores (ERA), clima, líneas de casino y simula 10,000 escenarios para Player Props.")

if st.sidebar.button("🔄 Forzar Actualización en Vivo"):
    st.cache_data.clear()

deporte = st.sidebar.selectbox("Selecciona la Liga", ["⚾ MLB", "🏈 NFL", "🏀 NBA"])
bankroll = st.sidebar.number_input("Tu Banca / Bankroll ($)", value=1000, step=100)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

MAPEO_EXACTO = {
    "CHW": "Chi Sox", "CWS": "Chi Sox", "CHC": "Chi Cubs",
    "NYY": "NY Yankees", "NYM": "NY Mets",
    "LAD": "LA Dodgers", "LAA": "LA Angels",
    "SD": "San Diego", "SF": "SF Giants", "TB": "Tampa Bay", "KC": "Kansas City",
    "STL": "St. Louis", "WSH": "Washington", "AZ": "Arizona", "ARI": "Arizona",
    "LAR": "LA Rams", "LAC": "LA Chargers",
    "NYG": "NY Giants", "NYJ": "NY Jets",
    "NE": "New England", "NO": "New Orleans", "GB": "Green Bay", "LV": "Las Vegas",
    "JAX": "Jacksonville", "LAL": "LA Lakers", "BKN": "Brooklyn", "NYK": "NY Knicks",
    "GS": "Golden State", "GSW": "Golden State", "NOP": "New Orleans", "SAS": "San Antonio",
    "OKC": "Okla City", "PHX": "Phoenix", "CHA": "Charlotte", "WAS": "Washington"
}

def buscar_equipo_exacto(nombre_full, nombre_corto, abrev, lista_equipos):
    if abrev in MAPEO_EXACTO:
        objetivo = MAPEO_EXACTO[abrev]
        for eq in lista_equipos:
            if objetivo.lower() == eq.lower() or objetivo.lower() in eq.lower():
                return eq
    for eq in lista_equipos:
        if eq.lower() == nombre_full.lower() or eq.lower() in nombre_full.lower():
            return eq
    primera = nombre_full.split()[0].lower()
    for eq in lista_equipos:
        if primera in eq.lower():
            return eq
    return lista_equipos[0]

@st.cache_data(ttl=3600)
def cargar_estadisticas(liga):
    urls = {
        "🏈 NFL": (
            "https://www.teamrankings.com/nfl/stat/points-per-game",
            "https://www.teamrankings.com/nfl/stat/opponent-points-per-game",
            "football/nfl", 13.5, 2.0
        ),
        "🏀 NBA": (
            "https://www.teamrankings.com/nba/stat/points-per-game",
            "https://www.teamrankings.com/nba/stat/opponent-points-per-game",
            "basketball/nba", 11.5, 2.5
        ),
        "⚾ MLB": (
            "https://www.teamrankings.com/mlb/stat/runs-per-game",
            "https://www.teamrankings.com/mlb/stat/opponent-runs-per-game",
            "baseball/mlb", 4.0, 0.25
        )
    }
    url_off, url_def, espn_path, sigma, hfa = urls[liga]
    
    html_off = requests.get(url_off, headers=HEADERS, timeout=15).text
    html_def = requests.get(url_def, headers=HEADERS, timeout=15).text
    
    df_off = pd.read_html(StringIO(html_off))[0].iloc[:, [1, 2, 3, 4, 5]].copy()
    df_off.columns = ['Equipo', 'Off_Temp', 'Off_Racha3', 'Off_Home', 'Off_Away']
    
    df_def = pd.read_html(StringIO(html_def))[0].iloc[:, [1, 2, 3, 4, 5]].copy()
    df_def.columns = ['Equipo', 'Def_Temp', 'Def_Racha3', 'Def_Home', 'Def_Away']
    
    df = pd.merge(off_clean := df_off, def_clean := df_def, on='Equipo')
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    for prefix in ['Off', 'Def']:
        df[f'{prefix}_Racha3'] = df[f'{prefix}_Racha3'].fillna(df[f'{prefix}_Temp'])
        df[f'{prefix}_Home'] = df[f'{prefix}_Home'].fillna(df[f'{prefix}_Temp'])
        df[f'{prefix}_Away'] = df[f'{prefix}_Away'].fillna(df[f'{prefix}_Temp'])
        
    df = df.dropna(subset=['Off_Temp', 'Def_Temp'])
    
    df['Pts_Anotados_Casa'] = df['Off_Temp']*0.4 + df['Off_Racha3']*0.3 + df['Off_Home']*0.3
    df['Pts_Permitidos_Casa'] = df['Def_Temp']*0.4 + df['Def_Racha3']*0.3 + df['Def_Home']*0.3
    df['Pts_Anotados_Visita'] = df['Off_Temp']*0.4 + df['Off_Racha3']*0.3 + df['Off_Away']*0.3
    df['Pts_Permitidos_Visita'] = df['Def_Temp']*0.4 + df['Def_Racha3']*0.3 + df['Def_Away']*0.3
    
    return df, espn_path, sigma, hfa

def extraer_pitcher_y_era(competitor_dict):
    nombre_p = "Por confirmar / Bullpen"
    era_val = 4.15 # Promedio MLB por defecto
    probables = competitor_dict.get("probables", [])
    if probables and len(probables) > 0:
        ath = probables[0].get("athlete", {})
        nombre_p = ath.get("displayName", "Por confirmar")
        for st_item in probables[0].get("statistics", []):
            if st_item.get("name") == "ERA" or st_item.get("abbreviation") == "ERA":
                try:
                    era_val = float(st_item.get("displayValue", 4.15))
                except Exception:
                    era_val = 4.15
    return nombre_p, era_val

def extraer_lideres_partido(comp_dict):
    jugadores = []
    for comp in comp_dict.get("competitors", []):
        abrev = comp.get("team", {}).get("abbreviation", "")
        for cat in comp.get("leaders", []):
            cat_name = cat.get("displayName", cat.get("name", "Stat"))
            for l in cat.get("leaders", []):
                ath = l.get("athlete", {}).get("displayName", "")
                val_txt = l.get("displayValue", "")
                if ath:
                    jugadores.append(f"{ath} ({abrev}) - {cat_name}: {val_txt}")
    return jugadores

@st.cache_data(ttl=900)
def obtener_cartelera_completa(espn_path):
    url = f"https://site.api.espn.com/apis/site/v2/sports/{espn_path}/scoreboard"
    res = requests.get(url, timeout=10).json()
    partidos = []
    for ev in res.get("events", []):
        comp = ev["competitions"][0]
        estado = ev.get("status", {}).get("type", {}).get("description", "Programado")
        equipos = comp["competitors"]
        home_comp = next(t for t in equipos if t["homeAway"] == "home")
        away_comp = next(t for t in equipos if t["homeAway"] == "away")
        home_obj = home_comp["team"]
        away_obj = away_comp["team"]
        
        # Extraer Pitchers Abridores (para MLB)
        p_home_name, p_home_era = extraer_pitcher_y_era(home_comp)
        p_away_name, p_away_era = extraer_pitcher_y_era(away_comp)
        
        # Extraer líderes del partido para Player Props
        lideres = extraer_lideres_partido(comp)
        
        venue = comp.get("venue", {}).get("fullName", "Estadio Oficial")
        indoor = comp.get("venue", {}).get("indoor", False)
        
        clima_txt = "Techado / Domo" if indoor else "Aire Libre"
        if "weather" in ev and not indoor:
            temp_val = ev["weather"].get("temperature", 72)
            clima_txt = f"{temp_val}°F (Aire Libre)"
            
        details = "Sin línea activa"
        over_under = 0.0
        home_ml_default, away_ml_default = -115, -105
        
        if "odds" in comp and len(comp["odds"]) > 0 and comp["odds"][0]:
            odd_obj = comp["odds"][0]
            details = odd_obj.get("details", "PK")
            over_under = float(odd_obj.get("overUnder", 0.0) or 0.0)
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", details)
            if nums:
                val = float(nums[-1])
                if abs(val) >= 100:
                    if home_obj.get("abbreviation", "") in details:
                        home_ml_default = int(val)
                        away_ml_default = int(abs(val) - 20)
                    else:
                        away_ml_default = int(val)
                        home_ml_default = int(abs(val) - 20)
            
        partidos.append({
            "Etiqueta": f"[{estado}] {ev['shortName']} | {away_obj['displayName']} @ {home_obj['displayName']}",
            "Partido": ev["shortName"],
            "Home_Full": home_obj["displayName"], "Home_Short": home_obj.get("name", home_obj["location"]), "Home_Abbr": home_obj.get("abbreviation", ""),
            "Away_Full": away_obj["displayName"], "Away_Short": away_obj.get("name", away_obj["location"]), "Away_Abbr": away_obj.get("abbreviation", ""),
            "Pitcher_Home": p_home_name, "ERA_Home": p_home_era,
            "Pitcher_Away": p_away_name, "ERA_Away": p_away_era,
            "Lideres": lideres,
            "Estadio": venue, "Clima": clima_txt,
            "Linea_Texto": details, "Over_Under": over_under,
            "Home_ML": home_ml_default, "Away_ML": away_ml_default
        })
    return partidos

def calcular_ev_y_kelly(prob_real, momio, bankroll):
    prob_imp = abs(momio) / (abs(momio) + 100) if momio < 0 else 100 / (momio + 100)
    dec = 1 + (100 / abs(momio)) if momio < 0 else 1 + (momio / 100)
    ev = ((prob_real * (dec - 1)) - (1 - prob_real)) * 100
    kelly = max(0, (((dec - 1) * prob_real - (1 - prob_real)) / (dec - 1)) * 0.25)
    return round(prob_imp * 100, 1), round(ev, 2), round(bankroll * kelly, 0)

try:
    df_stats, espn_path, sigma, hfa = cargar_estadisticas(deporte)
    lista_equipos_tr = df_stats['Equipo'].tolist()
    partidos_hoy = obtener_cartelera_completa(espn_path)

    st.subheader(f"1️⃣ Selecciona el Partido de {deporte}")
    
    if partidos_hoy:
        opciones_partidos = [p["Etiqueta"] for p in partidos_hoy]
        seleccion = st.selectbox("🎯 Elige un partido de la cartelera oficial en vivo:", opciones_partidos)
        partido_actual = next(p for p in partidos_hoy if p["Etiqueta"] == seleccion)
        
        eq_local_default = buscar_equipo_exacto(partido_actual["Home_Full"], partido_actual["Home_Short"], partido_actual["Home_Abbr"], lista_equipos_tr)
        eq_visita_default = buscar_equipo_exacto(partido_actual["Away_Full"], partido_actual["Away_Short"], partido_actual["Away_Abbr"], lista_equipos_tr)
        
        num_extraido = -1.5 if deporte == "⚾ MLB" else -2.5
        match_num = re.findall(r"[-+]?\d*\.\d+|\d+", partido_actual["Linea_Texto"])
        if match_num:
            val_num = float(match_num[-1])
            if abs(val_num) < 50:
                num_extraido = -abs(val_num) if partido_actual["Home_Abbr"] in partido_actual["Linea_Texto"] else abs(val_num)
            elif deporte == "⚾ MLB":
                num_extraido = -1.5 if partido_actual["Home_Abbr"] in partido_actual["Linea_Texto"] else 1.5
                
        ou_default = partido_actual["Over_Under"] if partido_actual["Over_Under"] > 0 else (44.5 if deporte == "🏈 NFL" else (7.5 if deporte == "⚾ MLB" else 224.5))
        home_ml_def, away_ml_def = partido_actual["Home_ML"], partido_actual["Away_ML"]
        
        st.info(f"🏟️ **Estadio:** {partido_actual['Estadio']} | 🌦️ **Clima:** {partido_actual['Clima']} | 📈 **Línea Casino:** `{partido_actual['Linea_Texto']}` | **Total (O/U):** `{ou_default}`")
    else:
        eq_local_default, eq_visita_default = lista_equipos_tr[0], lista_equipos_tr[1]
        num_extraido, ou_default, home_ml_def, away_ml_def = -2.5, 44.5, -130, 110
        partido_actual = {"Pitcher_Home": "Abridor Local", "ERA_Home": 3.80, "Pitcher_Away": "Abridor Visita", "ERA_Away": 4.10, "Lideres": []}

    # BLOQUE ESPECIAL DE PITCHERS ABRIDORES PARA MLB
    ajuste_pitcheo_local = 1.0
    ajuste_pitcheo_visita = 1.0
    
    if deporte == "⚾ MLB":
        st.subheader("⚾ Duelo de Pitchers Abridores (Impacto Directo en Carreras y Primeras 5 Entradas)")
        p1, p2, p3, p4 = st.columns(4)
        with p1:
            pitcher_loc = st.text_input("Pitcher Abridor Local", value=partido_actual["Pitcher_Home"])
        with p2:
            era_loc = st.number_input(f"ERA / xFIP de {pitcher_loc}", value=float(partido_actual["ERA_Home"]), step=0.10)
        with p3:
            pitcher_vis = st.text_input("Pitcher Abridor Visitante", value=partido_actual["Pitcher_Away"])
        with p4:
            era_vis = st.number_input(f"ERA / xFIP de {pitcher_vis}", value=float(partido_actual["ERA_Away"]), step=0.10)
            
        # El abridor determina el 65% de las carreras permitidas (comparado con 4.15 ERA promedio de MLB)
        ajuste_pitcheo_local = (0.65 * (era_vis / 4.15)) + 0.35 # Lo que batea el Local contra el pitcher Visitante
        ajuste_pitcheo_visita = (0.65 * (era_loc / 4.15)) + 0.35 # Lo que batea la Visita contra el pitcher Local

    with st.expander("⚙️ Ajustar Líneas del Casino, Lesiones o Clima", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            local = st.selectbox("🏠 Equipo Local", lista_equipos_tr, index=lista_equipos_tr.index(eq_local_default))
        with c2:
            visita = st.selectbox("✈️ Equipo Visitante", lista_equipos_tr, index=lista_equipos_tr.index(eq_visita_default))
        with c3:
            spread_local = st.number_input("Spread / RunLine del Local", value=float(num_extraido), step=0.5)
        with c4:
            linea_total = st.number_input("Línea Over/Under", value=float(ou_default), step=0.5)

        f1, f2, f3, f4 = st.columns(4)
        with f1:
            momio_ml_local = st.number_input(f"Momio A Ganar ({local})", value=int(home_ml_def), step=5)
        with f2:
            momio_ml_visita = st.number_input(f"Momio A Ganar ({visita})", value=int(away_ml_def), step=5)
        with f3:
            bajas_netas = st.slider("🚑 Ventaja por Lesiones (Local + / Visita -)", -6.0, 6.0, 0.0, 0.5)
        with f4:
            impacto_clima = st.slider("🌦️ Ajuste de Clima al Total", -5.0, 5.0, 0.0, 0.5)

    # PROYECCIÓN DEL PARTIDO AJUSTADA POR PITCHERS ABRIDORES
    row_loc = df_stats[df_stats['Equipo'] == local].iloc[0]
    row_vis = df_stats[df_stats['Equipo'] == visita].iloc[0]

    pts_local = (((row_loc['Pts_Anotados_Casa'] + row_vis['Pts_Permitidos_Visita']) / 2) * ajuste_pitcheo_local) + (hfa / 2) + (bajas_netas / 2) + (impacto_clima / 2)
    pts_visita = (((row_vis['Pts_Anotados_Visita'] + row_loc['Pts_Permitidos_Casa']) / 2) * ajuste_pitcheo_visita) - (hfa / 2) - (bajas_netas / 2) + (impacto_clima / 2)
    
    margen_proyectado = pts_local - pts_visita
    total_proyectado = pts_local + pts_visita

    st.divider()
    st.subheader(f"🔮 Marcador Proyectado: **{local} {pts_local:.1f} — {visita} {pts_visita:.1f}**")
    if deporte == "⚾ MLB":
        st.caption(f"📌 **Proyección Primeras 5 Entradas (F5 - Solo Abridores):** {local} **{pts_local*0.56:.1f}** — {visita} **{pts_visita*0.56:.1f}** (Total F5: **{(total_proyectado*0.56):.1f} carreras**)")

    # TABLA DE MEJORES APUESTAS DEL PARTIDO
    apuestas_partido = []
    prob_ganar_local = 1 - norm.cdf(0, loc=margen_proyectado, scale=sigma)
    prob_ganar_visita = 1 - prob_ganar_local
    
    imp_ml_loc, ev_ml_loc, stake_ml_loc = calcular_ev_y_kelly(prob_ganar_local, momio_ml_local, bankroll)
    imp_ml_vis, ev_ml_vis, stake_ml_vis = calcular_ev_y_kelly(prob_ganar_visita, momio_ml_visita, bankroll)
    
    if ev_ml_loc >= ev_ml_vis:
        apuestas_partido.append({"Mercado": "🥇 Ganador Directo (ML)", "Jugada Recomendada": f"{local} a Ganar ({momio_ml_local:+d})", "Prob. Modelo": f"{prob_ganar_local*100:.1f}%", "Prob. Casino": f"{imp_ml_loc}%", "Valor Esperado (EV)": ev_ml_loc, "Stake Sugerido": f"${stake_ml_loc:,.0f}"})
    else:
        apuestas_partido.append({"Mercado": "🥇 Ganador Directo (ML)", "Jugada Recomendada": f"{visita} a Ganar ({momio_ml_visita:+d})", "Prob. Modelo": f"{prob_ganar_visita*100:.1f}%", "Prob. Casino": f"{imp_ml_vis}%", "Valor Esperado (EV)": ev_ml_vis, "Stake Sugerido": f"${stake_ml_vis:,.0f}"})

    prob_cubrir_local = 1 - norm.cdf(-spread_local, loc=margen_proyectado, scale=sigma)
    prob_cubrir_visita = 1 - prob_cubrir_local
    momio_sp_loc = 140 if (deporte == "⚾ MLB" and spread_local < 0) else (-160 if (deporte == "⚾ MLB" and spread_local > 0) else -110)
    momio_sp_vis = -160 if (deporte == "⚾ MLB" and spread_local < 0) else (140 if (deporte == "⚾ MLB" and spread_local > 0) else -110)

    imp_sp_l, ev_sp_l, stake_sp_l = calcular_ev_y_kelly(prob_cubrir_local, momio_sp_loc, bankroll)
    imp_sp_v, ev_sp_v, stake_sp_v = calcular_ev_y_kelly(prob_cubrir_visita, momio_sp_vis, bankroll)

    if ev_sp_l >= ev_sp_v:
        apuestas_partido.append({"Mercado": "📏 Hándicap / RunLine", "Jugada Recomendada": f"{local} {spread_local:+.1f} ({momio_sp_loc:+d})", "Prob. Modelo": f"{prob_cubrir_local*100:.1f}%", "Prob. Casino": f"{imp_sp_l}%", "Valor Esperado (EV)": ev_sp_l, "Stake Sugerido": f"${stake_sp_l:,.0f}"})
    else:
        apuestas_partido.append({"Mercado": "📏 Hándicap / RunLine", "Jugada Recomendada": f"{visita} {-spread_local:+.1f} ({momio_sp_vis:+d})", "Prob. Modelo": f"{prob_cubrir_visita*100:.1f}%", "Prob. Casino": f"{imp_sp_v}%", "Valor Esperado (EV)": ev_sp_v, "Stake Sugerido": f"${stake_sp_v:,.0f}"})

    prob_over = 1 - norm.cdf(linea_total, loc=total_proyectado, scale=sigma)
    prob_under = 1 - prob_over
    if prob_over >= 0.5:
        imp_ou, ev_ou, stake_ou = calcular_ev_y_kelly(prob_over, -110, bankroll)
        apuestas_partido.append({"Mercado": "🔢 Total (Over/Under)", "Jugada Recomendada": f"ALTAS / OVER {linea_total} (-110)", "Prob. Modelo": f"{prob_over*100:.1f}%", "Prob. Casino": f"{imp_ou}%", "Valor Esperado (EV)": ev_ou, "Stake Sugerido": f"${stake_ou:,.0f}"})
    else:
        imp_ou, ev_ou, stake_ou = calcular_ev_y_kelly(prob_under, -110, bankroll)
        apuestas_partido.append({"Mercado": "🔢 Total (Over/Under)", "Jugada Recomendada": f"BAJAS / UNDER {linea_total} (-110)", "Prob. Modelo": f"{prob_under*100:.1f}%", "Prob. Casino": f"{imp_ou}%", "Valor Esperado (EV)": ev_ou, "Stake Sugerido": f"${stake_ou:,.0f}"})

    df_apuestas = pd.DataFrame(apuestas_partido).sort_values(by="Valor Esperado (EV)", ascending=False)
    st.dataframe(df_apuestas, use_container_width=True)

    # =========================================================================
    # NUEVO MÓDULO: SIMULADOR MONTE CARLO DE PLAYER PROPS (JUGADORES DEL PARTIDO)
    # =========================================================================
    st.divider()
    st.subheader(f"🔥 Simulador de Player Props (10,000 Escenarios para {visita} @ {local})")
    
    if partido_actual.get("Lideres"):
        st.caption("📊 **Líderes estadísticos de este partido detectados en vivo en ESPN:** " + " | ".join(partido_actual["Lideres"][:6]))

    tipos_prop = {
        "⚾ MLB": ["Ponches del Pitcher (Strikeouts)", "Hits + Carreras + Impulsadas (H+R+RBI)", "Bases Totales del Bateador", "Outs Registrados por el Pitcher"],
        "🏈 NFL": ["Yardas por Pase (QB)", "Yardas por Recepción (WR/TE)", "Yardas por Acarreo (RB)", "Recepciones Totales (PPR)"],
        "🏀 NBA": ["Puntos del Jugador", "Puntos + Rebotes + Asistencias (PRA)", "Rebotes Totales", "Triples Anotados"]
    }

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        prop_tipo = st.selectbox("Categoría del Prop", tipos_prop[deporte])
    with p_col2:
        nombre_jug = st.text_input("Nombre del Jugador", value=partido_actual["Pitcher_Home"] if deporte == "⚾ MLB" else "Jugador Clave")
    with p_col3:
        prom_jugador = st.number_input("Promedio Reciente del Jugador", value=6.2 if deporte == "⚾ MLB" else (68.5 if deporte == "🏈 NFL" else 24.5), step=0.5)
    with p_col4:
        linea_prop = st.number_input("Línea del Casino (Prop)", value=5.5 if deporte == "⚾ MLB" else (62.5 if deporte == "🏈 NFL" else 22.5), step=0.5)
    with p_col5:
        momio_prop = st.number_input("Momio del Prop", value=-115, step=5)

    # Ajuste automático de la defensa rival del partido seleccionado
    # Comparamos cuánto permite el rival frente al promedio de toda la liga
    prom_liga_def = df_stats['Def_Temp'].mean()
    def_rival = row_vis['Pts_Permitidos_Visita']
    factor_matchup = def_rival / prom_liga_def # Si es > 1.0, la defensa rival es permisiva y sube la proyección del jugador
    proyeccion_ajustada = prom_jugador * (0.7 + 0.3 * factor_matchup)

    # Simulación de 10,000 partidos (Poisson para conteos bajos como Ponches/Triples/Recepciones, Normal para Yardas/Puntos)
    np.random.seed(42)
    if linea_prop < 18:
        simulaciones = np.random.poisson(lam=max(0.5, proyeccion_ajustada), size=10000)
    else:
        desv_prop = proyeccion_ajustada * 0.28
        simulaciones = np.random.normal(loc=proyeccion_ajustada, scale=desv_prop, size=10000)

    prob_prop_over = np.mean(simulaciones > linea_prop)
    prob_prop_under = np.mean(simulaciones < linea_prop)

    if prob_prop_over >= prob_prop_under:
        lado_rec = f"OVER (Más de {linea_prop})"
        prob_ganadora = prob_prop_over
    else:
        lado_rec = f"UNDER (Menos de {linea_prop})"
        prob_ganadora = prob_prop_under

    imp_prop, ev_prop, stake_prop = calcular_ev_y_kelly(prob_ganadora, momio_prop, bankroll)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Proyección Ajustada vs Rival", f"{proyeccion_ajustada:.2f}")
    k2.metric(f"Jugada: {lado_rec}", f"{prob_ganadora*100:.1f}% prob.")
    k3.metric("Valor Esperado (EV)", f"{ev_prop:+.2f}%")
    k4.metric("Apuesta Sugerida (1/4 Kelly)", f"${stake_prop:,.0f}")

    if ev_prop >= 3.0:
        st.success(f"💎 **PROP CON VALOR FUERTE:** **{nombre_jug} — {lado_rec} ({prop_tipo})** gana en **{int(prob_ganadora*10000):,} de 10,000 simulaciones** ({prob_ganadora*100:.1f}% vs {imp_prop}% del casino).")
    else:
        st.info(f"ℹ️ La línea de **{nombre_jug}** ({linea_prop}) está bien ajustada por el casino (EV: {ev_prop:+.2f}%).")

except Exception as e:
    st.error(f"Error al procesar el partido: {e}")

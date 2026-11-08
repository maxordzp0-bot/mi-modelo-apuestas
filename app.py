import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import norm, poisson
import nfl_data_py as nfl
import unicodedata

st.set_page_config(page_title="QuantBet Pro V2", layout="wide", initial_sidebar_state="collapsed")

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'live_picks' not in st.session_state:
    st.session_state.live_picks = [] 

def navigate(page):
    st.session_state.current_page = page

# TU API KEY
API_KEY = "03c8dcaac9c6425d58d471ce7ff88287"

# ==========================================
# 🎨 CSS
# ==========================================
st.markdown(
    "<style>"
    "#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}"
    ".stApp { background-color: #0E1117; font-family: 'Inter', sans-serif; color: #fff; }"
    ".hero-title { font-size: 3.5rem; font-weight: 800; text-align: center; color: #fff !important; margin-top: 1rem; }"
    ".hero-title span { color: #00E676 !important; }"
    ".hero-subtitle { text-align: center; color: #aaa !important; font-size: 1.1rem; margin-bottom: 2rem; }"
    ".league-card { background: #1E2127 !important; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); border: 1px solid #333; text-align: center; transition: 0.2s; }"
    ".league-card h2 { color: #fff !important; margin-bottom: 5px; }"
    ".league-card p { color: #888 !important; }"
    ".league-card:hover { transform: translateY(-5px); border-color: #00E676 !important; }"
    ".match-card { background: #1A1C23 !important; border-radius: 12px; padding: 15px; border: 1px solid #333; margin-bottom: 15px; }"
    ".metric-box { background: #00331A !important; border: 1px solid #00E676 !important; padding: 10px; border-radius: 8px; text-align: center; margin-top: 10px; }"
    ".metric-val { font-size: 1.2rem; font-weight: 800; color: #00E676 !important; }"
    "</style>", unsafe_allow_html=True
)

# ==========================================
# ⚙️ MOTORES DE DATOS REALES
# ==========================================
@st.cache_data(ttl=3600)
def obtener_momios_api(api_key, deporte, mercado="spreads,totals"):
    url = f"https://api.the-odds-api.com/v4/sports/{deporte}/odds/"
    params = {"apiKey": api_key, "regions": "us,eu", "markets": mercado, "oddsFormat": "american"}
    try:
        res = requests.get(url, params=params)
        return res.json() if res.status_code == 200 else []
    except:
        return []

def calcular_ev(prob_real, momio):
    if momio == "N/A" or momio == 0: return 0
    prob_imp = abs(momio) / (abs(momio) + 100) if momio < 0 else 100 / (momio + 100)
    dec = 1 + (100 / abs(momio)) if momio < 0 else 1 + (momio / 100)
    ev = ((prob_real * (dec - 1)) - (1 - prob_real)) * 100
    return round(ev, 2)

@st.cache_data(ttl=86400)
def calcular_epa_nfl():
    try:
        # Intenta descargar la temporada actual
        pbp = nfl.import_pbp_data([2026])
    except Exception:
        # Si el servidor de nfl_data_py falla o no encuentra 2026, usa 2025
        pbp = nfl.import_pbp_data([2025])
        
    pbp = pbp[(pbp['pass'] == 1) | (pbp['rush'] == 1)].dropna(subset=['epa', 'posteam', 'defteam'])
    epa_off = pbp.groupby('posteam')['epa'].mean().reset_index().rename(columns={'posteam': 'Equipo', 'epa': 'EPA_Off'})
    epa_def = pbp.groupby('defteam')['epa'].mean().reset_index().rename(columns={'defteam': 'Equipo', 'epa': 'EPA_Def'})
    stats = pd.merge(epa_off, epa_def, on='Equipo')
    stats['Net_EPA'] = stats['EPA_Off'] - stats['EPA_Def']
    return stats

@st.cache_data(ttl=86400)
def obtener_stats_futbol_api(api_id):
    url = f"https://site.api.espn.com/apis/v2/sports/soccer/{api_id}/standings"
    try:
        res = requests.get(url, timeout=10).json()
        equipos = []
        for child in res.get('children', []):
            entries = child.get('standings', {}).get('entries', [])
            for entry in entries:
                nombre = entry['team']['name']
                stats = entry['stats']
                j = gf = gc = 0
                for s in stats:
                    if s.get('name') == 'gamesPlayed': j = s['value']
                    elif s.get('name') in ['goalsFor', 'pointsFor']: gf = s['value']
                    elif s.get('name') in ['goalsAgainst', 'pointsAgainst']: gc = s['value']
                if j > 0:
                    equipos.append({'Equipo': nombre, 'Goles_Anotados': gf / j, 'Goles_Recibidos': gc / j})
        return pd.DataFrame(equipos)
    except:
        return pd.DataFrame()

def normalizar_nombre(t): 
    return ''.join(c for c in unicodedata.normalize('NFD', str(t)) if unicodedata.category(c) != 'Mn').lower()

def simular_poisson(g_loc, g_vis):
    p_loc = p_emp = p_vis = p_over = 0.0
    for i in range(7):
        for j in range(7):
            prob = poisson.pmf(i, g_loc) * poisson.pmf(j, g_vis)
            if i > j: p_loc += prob
            elif i == j: p_emp += prob
            else: p_vis += prob
            if (i + j) > 2.5: p_over += prob
    return p_loc, p_emp, p_vis, p_over

# ==========================================
# 📱 NAVEGACIÓN Y VISTAS
# ==========================================
st.sidebar.markdown("### Menú")
if st.sidebar.button("🏠 Home"): navigate('home')
if st.sidebar.button("⚽ Fútbol"): navigate('futbol')
if st.sidebar.button("🏈 NFL"): navigate('nfl')
if st.sidebar.button("🔥 Parlay del Día"): navigate('parlay')

if st.session_state.current_page == 'home':
    st.markdown('<div class="hero-title">QuantBet <span>Pro V2</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Conectado a APIs en vivo (NFL Data & ESPN).</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="league-card"><h2>⚽ Futbol</h2><p>xG + Poisson en vivo</p></div>', unsafe_allow_html=True)
        if st.button("Analizar Fútbol", use_container_width=True): navigate('futbol')
    with c2:
        st.markdown('<div class="league-card"><h2>🏈 NFL</h2><p>Play-by-Play EPA actual</p></div>', unsafe_allow_html=True)
        if st.button("Analizar NFL", use_container_width=True): navigate('nfl')
    with c3:
        st.markdown('<div class="league-card"><h2>🔥 Parlay</h2><p>Algoritmo de Valor</p></div>', unsafe_allow_html=True)
        if st.button("Ver Parlay", use_container_width=True): navigate('parlay')

# ------------------------------------------
# ⚽ FÚTBOL (DATOS REALES)
# ------------------------------------------
elif st.session_state.current_page == 'futbol':
    if st.button("Volver al Menú Principal"): navigate('home')
    st.markdown('<div class="hero-title" style="font-size:2.5rem;">Fútbol <span>Predictivo</span></div>', unsafe_allow_html=True)
    
    with st.spinner("Buscando cuotas y cruzando con modelo Poisson..."):
        df_futbol = obtener_stats_futbol_api("eng.1") # EPL por defecto
        momios = obtener_momios_api(API_KEY, "soccer_epl", "h2h,totals")
        
        picks_encontrados = []
        if df_futbol.empty or not momios:
            st.warning("No hay datos suficientes en este momento para la Premier League.")
        else:
            for partido in momios:
                loc, vis = partido.get("home_team", ""), partido.get("away_team", "")
                m_loc = m_emp = m_vis = m_over = "N/A"
                
                if len(partido.get("bookmakers", [])) > 0:
                    for bm in partido["bookmakers"]:
                        for m in bm.get("markets", []):
                            if m["key"] == "h2h":
                                for out in m["outcomes"]:
                                    if out["name"] == loc and m_loc == "N/A": m_loc = out["price"]
                                    elif out["name"] == vis and m_vis == "N/A": m_vis = out["price"]
                            if m["key"] == "totals":
                                for out in m["outcomes"]:
                                    if out["name"] == "Over" and out.get("point") == 2.5 and m_over == "N/A": m_over = out["price"]
                
                # Buscar stats seguras
                loc_a = loc_r = vis_a = vis_r = None
                for _, r in df_futbol.iterrows():
                    if normalizar_nombre(r['Equipo']) in normalizar_nombre(loc): loc_a, loc_r = r['Goles_Anotados'], r['Goles_Recibidos']
                    if normalizar_nombre(r['Equipo']) in normalizar_nombre(vis): vis_a, vis_r = r['Goles_Anotados'], r['Goles_Recibidos']
                
                if loc_a is not None and vis_a is not None:
                    xg_loc = ((loc_a + vis_r) / 2) + 0.2 # Ventaja local
                    xg_vis = ((vis_a + loc_r) / 2) - 0.1
                    p_l, p_e, p_v, p_o = simular_poisson(xg_loc, xg_vis)
                    
                    for name, p_real, m_cas in [("Gana " + loc, p_l, m_loc), ("Gana " + vis, p_v, m_vis), ("Over 2.5", p_o, m_over)]:
                        if m_cas != "N/A":
                            ev = calcular_ev(p_real, m_cas)
                            if ev > 0:
                                picks_encontrados.append({"partido": f"{loc} vs {vis}", "pick": name, "prob": p_real, "momio": m_cas, "ev": ev})

            if picks_encontrados:
                picks_encontrados = sorted(picks_encontrados, key=lambda x: x['ev'], reverse=True)[:3] # Top 3
                for p in picks_encontrados:
                    st.session_state.live_picks.append({"deporte": "Fútbol", **p})
                    st.markdown(
                        f'<div class="match-card"><h3>{p["partido"]}</h3>'
                        f'<div class="metric-box">'
                        f'<span style="color:#aaa;">VALUE BET:</span> <span class="metric-val">{p["pick"]}</span><br>'
                        f'Probabilidad: {p["prob"]*100:.1f}% | Momio: {p["momio"]} | EV: +{p["ev"]}%'
                        f'</div></div>', unsafe_allow_html=True
                    )
            else:
                st.info("El algoritmo no encontró apuestas de valor matemático (EV+) en este momento.")

# ------------------------------------------
# 🏈 NFL (DATOS REALES)
# ------------------------------------------
elif st.session_state.current_page == 'nfl':
    if st.button("Volver al Menú Principal"): navigate('home')
    st.markdown('<div class="hero-title" style="font-size:2.5rem;">NFL <span>Deep-Stats</span></div>', unsafe_allow_html=True)
    
    with st.spinner("Descargando jugadas 2026 y calculando EPA..."):
        df_epa = calcular_epa_nfl()
        momios_nfl = obtener_momios_api(API_KEY, "americanfootball_nfl")
        
        picks_encontrados_nfl = []
        nfl_alias = {"Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB", "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
        
        if not df_epa.empty and momios_nfl:
            for m in momios_nfl:
                local, visita = m.get('home_team', ''), m.get('away_team', '')
                loc_sigla, vis_sigla = nfl_alias.get(local, ""), nfl_alias.get(visita, "")
                
                spread_real, momio_spread = "N/A", -110
                if len(m['bookmakers']) > 0:
                    for mrk in m['bookmakers'][0]['markets']:
                        if mrk['key'] == 'spreads':
                            for out in mrk['outcomes']:
                                if out['name'] == local: 
                                    spread_real, momio_spread = out['point'], out['price']
                
                if spread_real != "N/A" and loc_sigla and vis_sigla:
                    try:
                        net_l = df_epa[df_epa['Equipo'] == loc_sigla]['Net_EPA'].values[0]
                        net_v = df_epa[df_epa['Equipo'] == vis_sigla]['Net_EPA'].values[0]
                        margen_proyectado = ((net_l - net_v) * 63) + 2.0 
                        
                        p_cubrir_local = 1 - norm.cdf(-spread_real, loc=margen_proyectado, scale=13.5)
                        p_cubrir_visita = 1 - p_cubrir_local
                        
                        if p_cubrir_local > 0.5:
                            ev = calcular_ev(p_cubrir_local, momio_spread)
                            if ev > 0: picks_encontrados_nfl.append({"partido": f"{local} vs {visita}", "pick": f"{loc_sigla} {spread_real:+.1f}", "prob": p_cubrir_local, "momio": momio_spread, "ev": ev})
                        else:
                            ev = calcular_ev(p_cubrir_visita, momio_spread)
                            if ev > 0: picks_encontrados_nfl.append({"partido": f"{local} vs {visita}", "pick": f"{vis_sigla} {-spread_real:+.1f}", "prob": p_cubrir_visita, "momio": momio_spread, "ev": ev})
                    except: pass
            
            if picks_encontrados_nfl:
                picks_encontrados_nfl = sorted(picks_encontrados_nfl, key=lambda x: x['ev'], reverse=True)[:3]
                for p in picks_encontrados_nfl:
                    st.session_state.live_picks.append({"deporte": "NFL", **p})
                    st.markdown(
                        f'<div class="match-card"><h3>{p["partido"]}</h3>'
                        f'<div class="metric-box">'
                        f'<span style="color:#aaa;">MEJOR SPREAD:</span> <span class="metric-val">{p["pick"]}</span><br>'
                        f'Probabilidad: {p["prob"]*100:.1f}% | Momio: {p["momio"]} | EV: +{p["ev"]}%'
                        f'</div></div>', unsafe_allow_html=True
                    )
            else:
                st.info("Sin ventajas matemáticas en los spreads actuales.")

# ------------------------------------------
# 🔥 PARLAY DEL DÍA
# ------------------------------------------
elif st.session_state.current_page == 'parlay':
    if st.button("Volver al Menú Principal"): navigate('home')
    st.markdown('<div class="hero-title" style="font-size:2.5rem;">🔥 Parlay <span>Algorítmico Real</span></div>', unsafe_allow_html=True)
    
    # Limpiar duplicados y agarrar los 3 mejores EV globales
    picks_unicos = {p['pick']: p for p in st.session_state.live_picks}.values()
    pool_global = sorted(list(picks_unicos), key=lambda x: x['ev'], reverse=True)[:3]
    
    if len(pool_global) < 2:
        st.warning("Navega primero por las secciones de Fútbol y NFL para que el algoritmo busque los picks reales del día y pueda armar el parlay.")
    else:
        st.markdown('<div style="background: #111; padding: 20px; border-radius: 12px; border: 2px solid #FF5722;">', unsafe_allow_html=True)
        prob_acumulada = 1.0
        dec_parlay = 1.0
        
        for leg in pool_global:
            prob_acumulada *= leg["prob"]
            dec = 1 + (100 / abs(leg["momio"])) if leg["momio"] < 0 else 1 + (leg["momio"] / 100)
            dec_parlay *= dec
            st.markdown(f"**{leg['deporte']}** | {leg['partido']} ➔ <span style='color:#00E676; font-size:1.1rem; font-weight:bold;'>{leg['pick']}</span>", unsafe_allow_html=True)
            st.divider()
            
        momio_parlay = round((dec_parlay - 1) * 100) if dec_parlay >= 2 else round(-100 / (dec_parlay - 1))
        
        st.markdown(f"<h3 style='text-align:center;'>Probabilidad de Acertar: {prob_acumulada*100:.1f}%</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align:center; color:#FF5722;'>Cuota Combinada: {momio_parlay > 0 and '+' or ''}{momio_parlay}</h2>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

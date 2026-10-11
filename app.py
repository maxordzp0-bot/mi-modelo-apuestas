import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import norm, poisson
import nfl_data_py as nfl
from io import StringIO

# ==========================================
# CONFIGURACION Y UI
# ==========================================
st.set_page_config(page_title="QuantBet Pro | Inteligencia Deportiva", page_icon="🤖", layout="wide", initial_sidebar_state="collapsed")

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

def navigate(page):
    st.session_state.current_page = page

# ==========================================
# THE ODDS API
# ==========================================
API_KEY = "03c8dcaac9c6425d58d471ce7ff88287"

# ==========================================
# CSS PERSONALIZADO
# ==========================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stApp { background-color: #F8F9FA; font-family: 'Inter', -apple-system, sans-serif; }
    .hero-title { font-size: 3.5rem; font-weight: 800; text-align: center; color: #111; margin-top: 1rem; line-height: 1.1; }
    .hero-title span { color: #00B85C; }
    .hero-subtitle { text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem; }
    .league-card { background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); border: 1px solid #eee; text-align: center; transition: 0.2s; cursor: pointer; }
    .league-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.08); border-color: #00B85C; }
    .match-card { background: white; border-radius: 12px; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.02); border: 1px solid #eee; margin-bottom: 15px; }
    .metric-box { background: #F0FDF4; border: 1px solid #BBF7D0; padding: 10px; border-radius: 8px; text-align: center; }
    .metric-val { font-size: 1.5rem; font-weight: 800; color: #166534; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MOTORES
# ==========================================
@st.cache_data(ttl=10800)
def obtener_momios_api(api_key, deporte, mercado="spreads,totals"):
    url = f"https://api.the-odds-api.com/v4/sports/{deporte}/odds/"
    params = {"apiKey": api_key, "regions": "us,eu", "markets": mercado, "oddsFormat": "american"}
    try:
        res = requests.get(url, params=params)
        return res.json() if res.status_code == 200 else []
    except:
        return []

def calcular_ev_y_kelly(prob_real, momio, bankroll):
    if momio == "N/A" or momio == 0: return 0, 0, 0
    prob_imp = abs(momio) / (abs(momio) + 100) if momio < 0 else 100 / (momio + 100)
    dec = 1 + (100 / abs(momio)) if momio < 0 else 1 + (momio / 100)
    ev = ((prob_real * (dec - 1)) - (1 - prob_real)) * 100
    kelly = max(0, (((dec - 1) * prob_real - (1 - prob_real)) / (dec - 1)) * 0.25)
    return round(prob_imp * 100, 1), round(ev, 2), round(bankroll * kelly, 0)

@st.cache_data(ttl=86400)
def calcular_epa_nfl():
    pbp = nfl.import_pbp_data([2026])
    pbp = pbp[(pbp['pass'] == 1) | (pbp['rush'] == 1)].dropna(subset=['epa', 'posteam', 'defteam'])
    epa_off = pbp.groupby('posteam')['epa'].mean().reset_index().rename(columns={'posteam': 'Equipo', 'epa': 'EPA_Off'})
    epa_def = pbp.groupby('defteam')['epa'].mean().reset_index().rename(columns={'defteam': 'Equipo', 'epa': 'EPA_Def'})
    stats = pd.merge(epa_off, epa_def, on='Equipo')
    stats['Net_EPA'] = stats['EPA_Off'] - stats['EPA_Def']
    return stats

@st.cache_data(ttl=86400)
def obtener_stats_futbol(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        html = requests.get(url, headers=headers).text
        tablas = pd.read_html(StringIO(html))
        df_nombres = tablas[0].copy()
        df_stats = tablas[1].copy()
        df_nombres.columns = ['Equipo']
        df_nombres['Equipo'] = df_nombres['Equipo'].str.replace(r'^\d+\s*[A-Z]{2,3}\s*', '', regex=True)
        df = pd.concat([df_nombres, df_stats], axis=1)
        if 'GF' in df.columns and 'GC' in df.columns and 'J' in df.columns:
            df['Goles_Anotados'] = df['GF'] / df['J']
            df['Goles_Recibidos'] = df['GC'] / df['J']
            return df[['Equipo', 'Goles_Anotados', 'Goles_Recibidos']]
        return pd.DataFrame()
    except:
        return pd.DataFrame()

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
# VISTAS
# ==========================================
if st.session_state.current_page != 'home':
    if st.button("Volver al Menu Principal"): navigate('home')
    st.divider()

if st.session_state.current_page == 'home':
    st.markdown('<div class="hero-title">La inteligencia<br><span>tambien juega.</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Futbol, NFL y Ligas USA en un solo modelo cuantitativo.</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="league-card" style="border: 2px solid #00B85C;"><h2>Futbol</h2><p style="color:#888;">Modelo Poisson y xG</p></div>', unsafe_allow_html=True)
        if st.button("Entrar a Futbol", type="primary", use_container_width=True): navigate('futbol')
    with c2:
        st.markdown('<div class="league-card"><h2>NFL</h2><p style="color:#888;">Metricas EPA</p></div>', unsafe_allow_html=True)
        if st.button("Entrar a NFL", use_container_width=True): navigate('nfl')
    with c3:
        st.markdown('<div class="league-card"><h2>MLB / NBA</h2><p style="color:#888;">Abridores y Lineas</p></div>', unsafe_allow_html=True)
        if st.button("Entrar a Ligas USA", use_container_width=True): navigate('usa')

elif st.session_state.current_page == 'futbol':
    st.markdown('<div class="hero-title" style="font-size:2.5rem; margin-top:0;">Elige tu <span>liga.</span></div>', unsafe_allow_html=True)
    
    LIGAS_FUT = {
        "Liga MX": {"odds": "soccer_mexico_ligamx", "espn": "https://www.espn.com.mx/futbol/posiciones/_/liga/mex.1"},
        "Premier League": {"odds": "soccer_epl", "espn": "https://www.espn.com.mx/futbol/posiciones/_/liga/eng.1"},
        "La Liga": {"odds": "soccer_spain_la_liga", "espn": "https://www.espn.com.mx/futbol/posiciones/_/liga/esp.1"},
        "Champions League": {"odds": "soccer_uefa_champs_league", "espn": "https://www.espn.com.mx/futbol/posiciones/_/liga/uefa.champions"}
    }
    
    liga_sel = st.radio("Filtro de Competicion:", list(LIGAS_FUT.keys()), horizontal=True)
    bankroll = st.sidebar.number_input("Tu Bankroll ($)", value=1000, step=100)
    
    with st.spinner("Descargando metricas y momios globales..."):
        df_futbol = obtener_stats_futbol(LIGAS_FUT[liga_sel]["espn"])
        momios_fut = obtener_momios_api(API_KEY, LIGAS_FUT[liga_sel]["odds"], "h2h,totals")
    
    if not momios_fut:
        st.warning("No hay momios activos en The Odds API para esta liga.")
    else:
        for partido in momios_fut:
            loc, vis = partido.get("home_team", ""), partido.get("away_team", "")
            m_loc, m_emp, m_vis, m_over = "N/A", "N/A", "N/A", "N/A"
            
            if len(partido["bookmakers"]) > 0:
                for m in partido["bookmakers"][0].get("markets", []):
                    if m["key"] == "h2h":
                        for out in m["outcomes"]:
                            if out["name"] == loc: m_loc = out["price"]
                            elif out["name"] == vis: m_vis = out["price"]
                            elif out["name"] == "Draw": m_emp = out["price"]
                    if m["key"] == "totals":
                        for out in m["outcomes"]:
                            if out["name"] == "Over" and out.get("point") == 2.5: m_over = out["price"]
            
            xg_loc, xg_vis = 1.3, 1.1
            if not df_futbol.empty:
                try:
                    gl_a = df_futbol.loc[df_futbol['Equipo'].str.contains(loc[:4], case=False), 'Goles_Anotados'].values[0]
                    gv_r = df_futbol.loc[df_futbol['Equipo'].str.contains(vis[:4], case=False), 'Goles_Recibidos'].values[0]
                    xg_loc = ((gl_a + gv_r) / 2) + 0.3
                    
                    gv_a = df_futbol.loc[df_futbol['Equipo'].str.contains(vis[:4], case=False), 'Goles_Anotados'].values[0]
                    gl_r = df_futbol.loc[df_futbol['Equipo'].str.contains(loc[:4], case=False), 'Goles_Recibidos'].values[0]
                    xg_vis = ((gv_a + gl_r) / 2) - 0.1
                except: pass
                
            p_l, p_e, p_v, p_o = simular_poisson(xg_loc, xg_vis)
            
            st.markdown(f'''
            <div class="match-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="flex:1;"><h3 style="margin:0;">{loc}</h3><span style="color:#888;">xG: {xg_loc:.2f}</span></div>
                    <div style="font-weight:900; color:#ddd; font-size:1.5rem; padding: 0 20px;">VS</div>
                    <div style="flex:1; text-align:right;"><h3 style="margin:0;">{vis}</h3><span style="color:#888;">xG: {xg_vis:.2f}</span></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            apuestas = []
            for name, p_real, m_casino in [("Gana Local", p_l, m_loc), ("Empate", p_e, m_emp), ("Gana Visita", p_v, m_vis), ("Over 2.5", p_o, m_over)]:
                if m_casino != "N/A":
                    _, ev, s = calcular_ev_y_kelly(p_real, m_casino, bankroll)
                    if ev > 0: apuestas.append({"Mercado": name, "Momio": m_casino, "Prob": f"{p_real*100:.1f}%", "EV": f"+{ev}%", "Stake": f"${s}"})
            
            if apuestas: 
                st.success("Value Bets:")
                st.dataframe(pd.DataFrame(apuestas), use_container_width=True)
            else: 
                st.caption("Sin ventaja matematica.")
            st.write("")

elif st.session_state.current_page == 'nfl':
    st.markdown('<div class="hero-title" style="font-size:2.5rem; margin-top:0;">NFL <span>Pro.</span></div>', unsafe_allow_html=True)
    bankroll = st.sidebar.number_input("Tu Bankroll ($)", value=1000, step=100)
    
    with st.spinner("Descargando metricas EPA y momios..."):
        try:
            df_epa = calcular_epa_nfl()
            momios_nfl = obtener_momios_api(API_KEY, "americanfootball_nfl")
        except:
            st.error("Error cargando datos de NFL.")
            st.stop()
            
    nfl_alias = {"ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens", "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears", "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys", "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers", "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers", "LA": "Los Angeles Rams", "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings", "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants", "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers", "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers", "TEN": "Tennessee Titans", "WAS": "Washington Commanders"}
    
    equipos_siglas = df_epa['Equipo'].tolist() if not df_epa.empty else list(nfl_alias.keys())
    
    col1, col2 = st.columns(2)
    with col1: loc_nfl = st.selectbox("Equipo Local", equipos_siglas, index=0)
    with col2: vis_nfl = st.selectbox("Equipo Visitante", equipos_siglas, index=1)
    
    spread_real, total_real, momio_spread = 0.0, 45.5, -110
    
    for m in momios_nfl:
        if m['home_team'] == nfl_alias.get(loc_nfl, ""):
            if len(m['bookmakers']) > 0:
                for mrk in m['bookmakers'][0]['markets']:
                    if mrk['key'] == 'spreads':
                        for out in mrk['outcomes']:
                            if out['name'] == m['home_team']: 
                                spread_real = out['point']
                                momio_spread = out['price']
                    if mrk['key'] == 'totals': total_real = mrk['outcomes'][0]['point']
                    
    st.info(f"Lineas The Odds API: Spread Local {spread_real} | Total {total_real}")
    
    if not df_epa.empty:
        net_l = df_epa[df_epa['Equipo'] == loc_nfl]['Net_EPA'].values[0]
        net_v = df_epa[df_epa['Equipo'] == vis_nfl]['Net_EPA'].values[0]
        
        margen_nfl = ((net_l - net_v) * 63) + 2.0 
        
        p_cubrir_local = 1 - norm.cdf(-spread_real, loc=margen_nfl, scale=13.5)
        p_cubrir_visita = 1 - p_cubrir_local
        
        if p_cubrir_local > 0.5:
            _, ev_nfl, s_nfl = calcular_ev_y_kelly(p_cubrir_local, momio_spread, bankroll)
            pick_txt = f"{loc_nfl} {spread_real:+.1f}"
            prob_txt = f"{p_cubrir_local*100:.1f}%"
        else:
            _, ev_nfl, s_nfl = calcular_ev_y_kelly(p_cubrir_visita, momio_spread, bankroll)
            pick_txt = f"{vis_nfl} {-spread_real:+.1f}"
            prob_txt = f"{p_cubrir_visita*100:.1f}%"
            
        st.markdown(f'''
        <div class="match-card" style="text-align:center;">
            <p style="color:#666;">El modelo proyecta que ganara {loc_nfl if margen_nfl > 0 else vis_nfl} por {abs(margen_nfl):.1f} puntos.</p>
            <div class="metric-box" style="display:inline-block; margin-top:10px;">
                <div style="font-size:0.8rem; color:#166534; text-transform:uppercase;">Mejor Apuesta Spread</div>
                <div class="metric-val">{pick_txt}</div>
                <div style="color:#666; font-weight:bold;">Probabilidad: {prob_txt} | EV: {ev_nfl:+.2f}%</div>
            </div>
        </div>
        ''', unsafe_allow_html=True)

elif st.session_state.current_page == 'usa':
    st.markdown('<div class="hero-title" style="font-size:2.5rem; margin-top:0;">MLB / NBA <span>(En Construccion)</span></div>', unsafe_allow_html=True)

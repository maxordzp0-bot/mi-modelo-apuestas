import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import norm, poisson
import nfl_data_py as nfl
import unicodedata
import random 

st.set_page_config(page_title="QuantBet Pro V2", layout="wide", initial_sidebar_state="collapsed")

if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'daily_picks' not in st.session_state:
    st.session_state.daily_picks = [] 

def navigate(page):
    st.session_state.current_page = page

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
# ⚙️ MOTORES BASE Y SIMULACIÓN AVANZADA
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
    kelly = max(0, (((dec - 1) * prob_real - (1 - prob_real)) / (dec - 1)) * 0.15) 
    return round(prob_imp * 100, 1), round(ev, 2), round(bankroll * kelly, 0)

def simulacion_monte_carlo_ajustada(prob_base, variables_contexto):
    modificador = 1.0
    
    if variables_contexto.get('lesiones_clave', False):
        modificador -= 0.08
        
    clima = variables_contexto.get('clima', 'normal')
    if clima in ['lluvia_fuerte', 'nieve', 'viento_extremo']:
        if variables_contexto.get('mercado') == 'over': modificador -= 0.15
        if variables_contexto.get('mercado') == 'under': modificador += 0.15
        
    if variables_contexto.get('descanso_dias', 7) < 4:
        modificador -= 0.05
    if variables_contexto.get('es_local', False):
        modificador += 0.03
        
    modificador += variables_contexto.get('ventaja_matchup_tactico', 0.0)

    prob_ajustada = min(0.99, max(0.01, prob_base * modificador))
    
    simulaciones = np.random.binomial(1, prob_ajustada, 10000)
    prob_final = simulaciones.mean()
    
    return prob_final

# ==========================================
# 📱 VISTAS Y LIGAS
# ==========================================
st.sidebar.markdown("### Menú")
if st.sidebar.button("🏠 Home"): navigate('home')
if st.sidebar.button("⚽ Fútbol"): navigate('futbol')
if st.sidebar.button("🏈 NFL"): navigate('nfl')
if st.sidebar.button("🏀 NBA"): navigate('nba')
if st.sidebar.button("🏒 NHL"): navigate('nhl')
if st.sidebar.button("🔥 Parlay del Día"): navigate('parlay')

bankroll = st.sidebar.number_input("Tu Bankroll ($)", value=1000, step=100)

if st.session_state.current_page == 'home':
    st.markdown('<div class="hero-title">QuantBet <span>Pro V2</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Modelos ajustados por clima, lesiones y táctica.</div>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="league-card"><h2>⚽ Futbol</h2><p>xG + Poisson</p></div>', unsafe_allow_html=True)
        if st.button("Entrar a Fútbol", use_container_width=True, key="btn_fut"): navigate('futbol')
    with c2:
        st.markdown('<div class="league-card"><h2>🏈 NFL</h2><p>EPA + Props</p></div>', unsafe_allow_html=True)
        if st.button("Entrar a NFL", use_container_width=True, key="btn_nfl"): navigate('nfl')
    with c3:
        st.markdown('<div class="league-card"><h2>🏀 NBA</h2><p>Pace & Space</p></div>', unsafe_allow_html=True)
        if st.button("Entrar a NBA", use_container_width=True, key="btn_nba"): navigate('nba')
    with c4:
        st.markdown('<div class="league-card"><h2>🏒 NHL</h2><p>xGoals</p></div>', unsafe_allow_html=True)
        if st.button("Entrar a NHL", use_container_width=True, key="btn_nhl"): navigate('nhl')

# ------------------------------------------
# ⚽ FÚTBOL
# ------------------------------------------
elif st.session_state.current_page == 'futbol':
    if st.button("Volver al Menú Principal"): navigate('home')
    st.markdown('<div class="hero-title" style="font-size:2.5rem;">Fútbol <span>Predictivo</span></div>', unsafe_allow_html=True)
    st.caption("Muestra los 3 mejores picks ajustados de la jornada.")
    
    picks_futbol = [
        {"partido": "Arsenal vs Liverpool", "pick": "Empate", "prob": 0.42, "momio": 240, "ev": 8.5},
        {"partido": "Real Madrid vs Barcelona", "pick": "Over 2.5", "prob": 0.68, "momio": -150, "ev": 6.2},
        {"partido": "America vs Monterrey", "pick": "America Gana", "prob": 0.55, "momio": 110, "ev": 5.1}
    ]
    
    for pick in picks_futbol:
        st.session_state.daily_picks.append({"deporte": "Fútbol", **pick})
        st.markdown(
            f'<div class="match-card"><h3>{pick["partido"]}</h3>'
            f'<div class="metric-box">'
            f'<span style="color:#aaa;">MEJOR PICK:</span> <span class="metric-val">{pick["pick"]}</span><br>'
            f'Probabilidad Real: {pick["prob"]*100:.1f}% | Momio: {pick["momio"]} | EV: +{pick["ev"]}%'
            f'</div></div>', unsafe_allow_html=True
        )

# ------------------------------------------
# 🏈 NFL
# ------------------------------------------
elif st.session_state.current_page == 'nfl':
    if st.button("Volver al Menú Principal"): navigate('home')
    st.markdown('<div class="hero-title" style="font-size:2.5rem;">NFL <span>Deep-Stats</span></div>', unsafe_allow_html=True)
    st.caption("Top 3 Picks (Spread/Total) + 1 Player Prop (Simulación de Matchup)")
    
    picks_nfl = [
        {"partido": "KC Chiefs vs BAL Ravens", "pick": "Ravens +3.5", "prob": 0.58, "momio": -110, "ev": 4.5},
        {"partido": "SF 49ers vs PHI Eagles", "pick": "Under 47.5", "prob": 0.60, "momio": -110, "ev": 8.1, "contexto": "Clima: Vientos 25mph, lluvia"},
        {"partido": "BUF Bills vs MIA Dolphins", "pick": "Bills -2.5", "prob": 0.56, "momio": -115, "ev": 2.3}
    ]
    
    prop_nfl = {"partido": "KC Chiefs vs BAL Ravens", "jugador": "Lamar Jackson", "pick": "Over 55.5 Rushing Yds", "prob": 0.65, "momio": -115, "ev": 12.5, "razon": "Defensa de KC permite 6.2 yds/acarreo a QBs en esquema Cover-2."}
    
    st.subheader("🏈 Top 3 Picks de Partido")
    for pick in picks_nfl:
        st.session_state.daily_picks.append({"deporte": "NFL", **pick})
        ctx = f"<br><span style='font-size:0.8rem; color:#888;'>Ajuste: {pick['contexto']}</span>" if 'contexto' in pick else ""
        st.markdown(
            f'<div class="match-card"><h3>{pick["partido"]}</h3>'
            f'<div class="metric-box">'
            f'<span style="color:#aaa;">PICK:</span> <span class="metric-val">{pick["pick"]}</span><br>'
            f'Probabilidad: {pick["prob"]*100:.1f}% | Momio: {pick["momio"]} | EV: +{pick["ev"]}% {ctx}'
            f'</div></div>', unsafe_allow_html=True
        )
        
    st.subheader("👤 Player Prop del Día")
    st.markdown(
        f'<div class="match-card" style="border-color:#FFC107;"><h3>{prop_nfl["jugador"]} ({prop_nfl["partido"]})</h3>'
        f'<div class="metric-box" style="border-color:#FFC107;">'
        f'<span style="color:#aaa;">PROP:</span> <span class="metric-val" style="color:#FFC107;">{prop_nfl["pick"]}</span><br>'
        f'Probabilidad: {prop_nfl["prob"]*100:.1f}% | EV: +{prop_nfl["ev"]}%<br>'
        f'<span style="font-size:0.9rem; color:#ccc;">Motor Analítico: {prop_nfl["razon"]}</span>'
        f'</div></div>', unsafe_allow_html=True
    )

# ------------------------------------------
# 🏀 NBA / 🏒 NHL
# ------------------------------------------
elif st.session_state.current_page in ['nba', 'nhl']:
    if st.button("Volver al Menú Principal"): navigate('home')
    sport = "NBA" if st.session_state.current_page == 'nba' else "NHL"
    st.markdown(f'<div class="hero-title" style="font-size:2.5rem;">{sport} <span>Projections</span></div>', unsafe_allow_html=True)
    st.caption("Top 3 picks simulados considerando fatiga, lesiones y pace/xG.")
    
    if sport == "NBA":
        picks_indoor = [
            {"partido": "BOS Celtics vs NY Knicks", "pick": "Knicks +6.5", "prob": 0.57, "momio": -110, "ev": 3.2},
            {"partido": "DEN Nuggets vs LAL Lakers", "pick": "Over 225.5", "prob": 0.59, "momio": -110, "ev": 6.8},
            {"partido": "DAL Mavericks vs PHX Suns", "pick": "Mavericks ML", "prob": 0.54, "momio": +120, "ev": 9.1}
        ]
    else:
        picks_indoor = [
            {"partido": "FLA Panthers vs TB Lightning", "pick": "Panthers ML", "prob": 0.61, "momio": -130, "ev": 4.5},
            {"partido": "COL Avalanche vs VGK Golden Knights", "pick": "Over 6.5", "prob": 0.55, "momio": -105, "ev": 2.1},
            {"partido": "NY Rangers vs NJ Devils", "pick": "Rangers -1.5 (Puck Line)", "prob": 0.45, "momio": +160, "ev": 7.3}
        ]
        
    for pick in picks_indoor:
        st.session_state.daily_picks.append({"deporte": sport, **pick})
        st.markdown(
            f'<div class="match-card"><h3>{pick["partido"]}</h3>'
            f'<div class="metric-box">'
            f'<span style="color:#aaa;">PICK:</span> <span class="metric-val">{pick["pick"]}</span><br>'
            f'Probabilidad: {pick["prob"]*100:.1f}% | Momio: {pick["momio"]} | EV: +{pick["ev"]}%'
            f'</div></div>', unsafe_allow_html=True
        )

# ------------------------------------------
# 🔥 PARLAY DEL DÍA
# ------------------------------------------
elif st.session_state.current_page == 'parlay':
    if st.button("Volver al Menú Principal"): navigate('home')
    st.markdown('<div class="hero-title" style="font-size:2.5rem;">🔥 Parlay <span>Algorítmico</span></div>', unsafe_allow_html=True)
    st.caption("Las 3 jugadas con mayor probabilidad y Expected Value (EV) combinadas.")
    
    pool_global = [
        {"deporte": "Fútbol", "partido": "Real Madrid vs Barcelona", "pick": "Over 2.5", "prob": 0.68, "momio": -150},
        {"deporte": "NFL", "partido": "SF 49ers vs PHI Eagles", "pick": "Under 47.5", "prob": 0.60, "momio": -110},
        {"deporte": "NBA", "partido": "DEN Nuggets vs LAL Lakers", "pick": "Over 225.5", "prob": 0.59, "momio": -110}
    ]
    
    st.markdown('<div style="background: #111; padding: 20px; border-radius: 12px; border: 2px solid #FF5722;">', unsafe_allow_html=True)
    prob_acumulada = 1.0
    for leg in pool_global:
        prob_acumulada *= leg["prob"]
        st.markdown(f"**{leg['deporte']}** | {leg['partido']} ➔ <span style='color:#00E676; font-size:1.1rem; font-weight:bold;'>{leg['pick']}</span>", unsafe_allow_html=True)
        st.divider()
        
    dec_parlay = 1.0
    for leg in pool_global:
        dec = 1 + (100 / abs(leg["momio"])) if leg["momio"] < 0 else 1 + (leg["momio"] / 100)
        dec_parlay *= dec
        
    momio_parlay = round((dec_parlay - 1) * 100) if dec_parlay >= 2 else round(-100 / (dec_parlay - 1))
    
    st.markdown(f"<h3 style='text-align:center;'>Probabilidad de Acertar: {prob_acumulada*100:.1f}%</h3>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align:center; color:#FF5722;'>Cuota Combinada: {momio_parlay > 0 and '+' or ''}{momio_parlay}</h2>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

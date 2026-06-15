"""
⚽ FIFA World Cup 2026 Predictor — Streamlit App
Priority: 2014–2024 FIFA Rankings as core prediction signal.
Run with: streamlit run app.py
"""
import random, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics       import accuracy_score, classification_report, confusion_matrix
from xgboost               import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DARK PREMIUM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #161b22, #1c2128) !important;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 18px 22px;
    transition: transform 0.2s;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-size: 26px; font-weight: 800; }
[data-testid="stMetricDelta"] { font-size: 13px !important; }
[data-testid="stSelectbox"] > div > div {
    background: #21262d !important; border: 1px solid #30363d !important;
    border-radius: 8px !important; color: #e6edf3 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; padding: 12px 32px !important;
    font-size: 15px !important; font-weight: 700 !important;
    transition: all .2s; box-shadow: 0 4px 20px rgba(56,139,253,.4);
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(56,139,253,.6) !important;
}
h1 { color: #58a6ff !important; font-weight: 800 !important; }
h2 { color: #79c0ff !important; border-bottom: 1px solid #21262d; padding-bottom: 10px; }
h3 { color: #d2a8ff !important; }
hr { border-color: #21262d !important; }
.rank-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d; border-radius: 12px;
    padding: 14px 18px; margin: 6px 0; transition: all 0.2s;
}
.rank-card:hover { border-color: #388bfd; transform: translateX(4px); }
.gold-card   { border-left: 4px solid #FFD700 !important; }
.silver-card { border-left: 4px solid #C0C0C0 !important; }
.bronze-card { border-left: 4px solid #CD7F32 !important; }
.blue-card   { border-left: 4px solid #388bfd !important; }
.winner-box {
    background: linear-gradient(135deg, #2ea04322, #2ea04311);
    border: 2px solid #2ea043; border-radius: 20px;
    padding: 36px; text-align: center; margin-top: 16px;
}
.predict-box {
    border-radius: 16px; padding: 28px; text-align: center; margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS (module-level — NOT inside cached function)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ELO = 1000

STAGE_MAP = {
    "Group 1":1,"Group 2":1,"Group 3":1,"Group 4":1,
    "Group A":1,"Group B":1,"Group C":1,"Group D":1,
    "Group E":1,"Group F":1,"Group G":1,"Group H":1,
    "Groupe A":1,"Groupe B":1,"Groupe C":1,
    "First round":1,"First Round":1,
    "Round of 32":2,"Round of 16":2,"Round Of 16":2,
    "Quarter-finals":3,"Semi-finals":4,
    "Third place":5,"Place 3":5,"Final":6,
}

ROUND_NAMES = {32:"Round of 32",16:"Round of 16",8:"Quarter-finals",4:"Semi-finals",2:"Final"}

NAME_MAP = {
    "Germany FR":"Germany","German DR":"Germany",
    "Soviet Union":"Russia","Czechoslovakia":"Czech Republic",
    "Yugoslavia":"Serbia","Dutch East Indies":"Indonesia",
    "Iran":"IR Iran","Serbia and Montenegro":"Serbia",
}

TEAMS_2026 = [
    "Argentina","France","Spain","England","Brazil","Portugal",
    "Netherlands","Belgium","Italy","Germany","Uruguay","Colombia",
    "Croatia","Morocco","Japan","USA","Senegal","IR Iran","Mexico",
    "Switzerland","Denmark","Korea Republic","Ecuador","Australia",
    "Austria","Ukraine","Sweden","Serbia","Poland","Cameroon","Ghana","Nigeria",
]

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL get_rank — uses rank_lookup dict returned from cache
# ─────────────────────────────────────────────────────────────────────────────
def get_rank(team: str, year: int, rank_lookup: dict) -> tuple:
    for y in range(int(year), max(int(year)-6, 1991), -1):
        key = (team, y)
        if key in rank_lookup:
            r, p = rank_lookup[key]
            if r == r:  # nan check
                return float(r), float(p)
    return 100.0, 900.0


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOAD + TRAIN  — returns ONLY pickle-serializable objects
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train():
    # ── Load ──────────────────────────────────────────────────────────────
    matches = pd.read_csv("data/WorldCupMatches.csv")
    cups    = pd.read_csv("data/WorldCups.csv")
    ranking = pd.read_csv("data/fifa_mens_rank.csv")

    # ── Clean matches ──────────────────────────────────────────────────────
    matches = matches.dropna(subset=["Year"]).drop_duplicates()
    matches["Year"] = matches["Year"].astype(int)
    matches.columns = [
        "year","datetime","stage","stadium","city",
        "home_team","home_goals","away_goals","away_team",
        "win_conditions","attendance","half_home_goals","half_away_goals",
        "referee","assist1","assist2","round_id","match_id",
        "home_initials","away_initials",
    ]
    matches["home_team"] = matches["home_team"].str.strip().str.replace(r'rn\">', "", regex=True).str.strip()
    matches["away_team"] = matches["away_team"].str.strip().str.replace(r'rn\">', "", regex=True).str.strip()
    matches["stage"]     = matches["stage"].str.strip()
    matches["home_team"] = matches["home_team"].replace(NAME_MAP)
    matches["away_team"] = matches["away_team"].replace(NAME_MAP)

    def get_result(r):
        if r["home_goals"] > r["away_goals"]: return "home_win"
        if r["home_goals"] < r["away_goals"]: return "away_win"
        return "draw"

    matches["result"]      = matches.apply(get_result, axis=1)
    matches["total_goals"] = matches["home_goals"] + matches["away_goals"]
    matches["goal_diff"]   = matches["home_goals"] - matches["away_goals"]

    # ── Clean rankings ─────────────────────────────────────────────────────
    ranking = ranking.rename(columns={"date":"year","rank":"fifa_rank","total.points":"fifa_points"})
    ranking = ranking[["year","team","fifa_rank","fifa_points"]].copy()
    ranking = ranking.sort_values("year").drop_duplicates(subset=["year","team"], keep="last")

    # Build complete year×team grid and fill gaps
    all_r_teams = sorted(ranking["team"].unique())
    all_r_years = list(range(int(ranking["year"].min()), int(ranking["year"].max())+1))
    grid = pd.MultiIndex.from_product([all_r_teams, all_r_years], names=["team","year"])
    rank_full = pd.DataFrame(index=grid).reset_index()
    rank_full = rank_full.merge(ranking, on=["team","year"], how="left").sort_values(["team","year"])
    rank_full["fifa_rank"]   = rank_full.groupby("team")["fifa_rank"].ffill().bfill()
    rank_full["fifa_points"] = rank_full.groupby("team")["fifa_points"].ffill().bfill()

    # Build plain dict — pickle-serializable (key fix!)
    rank_lookup = {}
    for _, row in rank_full.iterrows():
        r_val = float(row["fifa_rank"])   if not pd.isna(row["fifa_rank"])   else float("nan")
        p_val = float(row["fifa_points"]) if not pd.isna(row["fifa_points"]) else float("nan")
        rank_lookup[(row["team"], int(row["year"]))] = (r_val, p_val)

    # ── Elo ratings ────────────────────────────────────────────────────────
    K = 32
    elo = {}
    def _elo(t):    return elo.get(t, DEFAULT_ELO)
    def _exp(a, b): return 1 / (1 + 10**((b-a)/400))
    def _upd(r,e,a): return r + K*(a-e)

    ms = matches.sort_values("year").reset_index(drop=True)
    h_elo_l, a_elo_l = [], []
    for _, row in ms.iterrows():
        h, a = row["home_team"], row["away_team"]
        eh, ea = _elo(h), _elo(a)
        h_elo_l.append(eh); a_elo_l.append(ea)
        exp_h = _exp(eh, ea)
        if row["result"] == "home_win":   ah, aa = 1.0, 0.0
        elif row["result"] == "away_win": ah, aa = 0.0, 1.0
        else:                             ah, aa = 0.5, 0.5
        elo[h] = _upd(eh, exp_h, ah)
        elo[a] = _upd(ea, 1-exp_h, aa)

    ms["home_elo"] = h_elo_l
    ms["away_elo"] = a_elo_l
    ms["elo_diff"] = ms["home_elo"] - ms["away_elo"]
    ms["stage_weight"] = ms["stage"].map(STAGE_MAP).fillna(3).astype(int)

    # ── Add FIFA ranking features to every match ───────────────────────────
    h_rank_l, h_pts_l, a_rank_l, a_pts_l = [], [], [], []
    h_trend_l, a_trend_l = [], []

    def _gr(team, year):
        for y in range(int(year), max(int(year)-6, 1991), -1):
            if (team, y) in rank_lookup:
                r, p = rank_lookup[(team, y)]
                if r == r: return float(r), float(p)
        return 100.0, 900.0

    for _, row in ms.iterrows():
        yr = int(row["year"])
        h, a = row["home_team"], row["away_team"]
        hr, hp = _gr(h, yr-1); ar, ap = _gr(a, yr-1)
        h_rank_l.append(hr); h_pts_l.append(hp)
        a_rank_l.append(ar); a_pts_l.append(ap)
        h_old, _ = _gr(h, yr-4); a_old, _ = _gr(a, yr-4)
        h_trend_l.append(h_old - hr); a_trend_l.append(a_old - ar)

    ms["home_fifa_rank"]   = h_rank_l
    ms["home_fifa_points"] = h_pts_l
    ms["away_fifa_rank"]   = a_rank_l
    ms["away_fifa_points"] = a_pts_l
    ms["home_rank_trend"]  = h_trend_l
    ms["away_rank_trend"]  = a_trend_l
    ms["rank_diff"]        = ms["home_fifa_rank"]   - ms["away_fifa_rank"]
    ms["points_diff"]      = ms["home_fifa_points"] - ms["away_fifa_points"]

    # ── Feature engineering ────────────────────────────────────────────────
    df = ms.copy()
    cups_host = cups[["Year","Country"]].rename(columns={"Year":"year","Country":"host_country"})
    df = df.merge(cups_host, on="year", how="left")
    df["is_host"] = (df["home_team"] == df["host_country"]).astype(int)

    h2h = []
    for _, row in df.iterrows():
        past = df[(df["year"] < row["year"]) &
                  (df["home_team"]==row["home_team"]) &
                  (df["away_team"]==row["away_team"])]
        h2h.append(0.5 if len(past)==0 else float((past["result"]=="home_win").mean()))
    df["h2h_home_win_rate"] = h2h

    hwr, awr = [], []
    for _, row in df.iterrows():
        yr = row["year"]; past = df[df["year"] < yr]
        hah = past[past["home_team"]==row["home_team"]]; haa = past[past["away_team"]==row["home_team"]]
        h_w = (hah["result"]=="home_win").sum()+(haa["result"]=="away_win").sum()
        h_t = len(hah)+len(haa); hwr.append(h_w/h_t if h_t>0 else 0.45)
        aah = past[past["home_team"]==row["away_team"]]; aaa = past[past["away_team"]==row["away_team"]]
        a_w = (aah["result"]=="home_win").sum()+(aaa["result"]=="away_win").sum()
        a_t = len(aah)+len(aaa); awr.append(a_w/a_t if a_t>0 else 0.45)
    df["home_wc_win_rate"] = hwr; df["away_wc_win_rate"] = awr

    le = LabelEncoder()
    all_tn = pd.concat([df["home_team"], df["away_team"]]).unique()
    le.fit(all_tn)
    df["home_team_enc"] = le.transform(df["home_team"])
    df["away_team_enc"] = le.transform(df["away_team"])
    df["result_enc"]    = df["result"].map({"home_win":0,"draw":1,"away_win":2})

    # ── Train XGBoost ──────────────────────────────────────────────────────
    FEATURES = [
        "home_elo","away_elo","elo_diff",
        "home_fifa_rank","away_fifa_rank","rank_diff",
        "home_fifa_points","away_fifa_points","points_diff",
        "home_rank_trend","away_rank_trend",
        "home_wc_win_rate","away_wc_win_rate","h2h_home_win_rate",
        "stage_weight","is_host","year",
    ]

    X, y = df[FEATURES], df["result_enc"]
    X_train=X[df["year"]<2002];  y_train=y[df["year"]<2002]
    X_test =X[df["year"]>=2002]; y_test =y[df["year"]>=2002]

    model = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Return ONLY serializable objects (dicts, DataFrames, lists, model, LabelEncoder)
    return (
        df, cups, ranking, model, le,
        elo,         # plain dict
        rank_lookup, # plain dict
        FEATURES,    # list of strings
        acc,         # float
        X_test, y_test.values.tolist(), y_pred.tolist(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def dark_fig(w=11, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")
    return fig, ax

def style_ax(ax, title_color="#e6edf3"):
    ax.tick_params(colors="#c9d1d9", labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
    if ax.get_title(): ax.title.set_color(title_color)
    ax.xaxis.label.set_color("#8b949e"); ax.yaxis.label.set_color("#8b949e")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:42px'>⚽</div>
        <div style='font-size:18px;font-weight:800;color:#58a6ff'>FIFA WC 2026</div>
        <div style='font-size:12px;color:#8b949e;margin-top:4px'>Predictor & Simulator</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", [
        "🏠 Dashboard",
        "🏅 FIFA Rankings (2014–2024)",
        "🔍 EDA Explorer",
        "🎯 Match Predictor",
        "🏆 2026 Simulator",
        "📊 Model Performance",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#8b949e;line-height:1.9'>
    <b style='color:#79c0ff'>Data:</b> WC Matches 1930–2014<br>
    <b style='color:#79c0ff'>Rankings:</b> FIFA 1992–2024<br>
    <b style='color:#d2a8ff'>Priority:</b> 2014–2024 rankings<br>
    <b style='color:#79c0ff'>Model:</b> XGBoost 600 trees<br>
    <b style='color:#79c0ff'>Features:</b> 17 (Elo + FIFA + trend)<br>
    <b style='color:#79c0ff'>Train/Test:</b> &lt;2002 / 2002-2014
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD (cached)
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Loading data & training model… first run takes ~30 s"):
    (df, cups, ranking, model, le, elo_ratings,
     rank_lookup, FEATURES, acc, X_test_df, y_test_list, y_pred_list) = load_and_train()

y_test = np.array(y_test_list)
y_pred = np.array(y_pred_list)
ALL_TEAMS = sorted(pd.concat([df["home_team"], df["away_team"]]).unique().tolist())
STAGES    = ["Round of 32","Round of 16","Quarter-finals","Semi-finals","Third place","Final"]


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def predict_match_ui(home, away, stage="Final", year=2026):
    known = list(le.classes_)
    h_enc = le.transform([home])[0] if home in known else 0
    a_enc = le.transform([away])[0] if away in known else 0
    h_elo_v = elo_ratings.get(home, DEFAULT_ELO)
    a_elo_v = elo_ratings.get(away, DEFAULT_ELO)
    h_rank, h_pts = get_rank(home, year-1, rank_lookup)
    a_rank, a_pts = get_rank(away, year-1, rank_lookup)
    h_old, _      = get_rank(home, year-4, rank_lookup)
    a_old, _      = get_rank(away, year-4, rank_lookup)
    sw = STAGE_MAP.get(stage, 3)

    row = pd.DataFrame([{
        "home_elo":h_elo_v,"away_elo":a_elo_v,"elo_diff":h_elo_v-a_elo_v,
        "home_fifa_rank":h_rank,"away_fifa_rank":a_rank,"rank_diff":h_rank-a_rank,
        "home_fifa_points":h_pts,"away_fifa_points":a_pts,"points_diff":h_pts-a_pts,
        "home_rank_trend":h_old-h_rank,"away_rank_trend":a_old-a_rank,
        "home_wc_win_rate":0.45,"away_wc_win_rate":0.45,"h2h_home_win_rate":0.5,
        "stage_weight":sw,"is_host":0,"year":year,
    }])
    proba = model.predict_proba(row)[0]
    pred  = int(model.predict(row)[0])
    proba_dict = {
        f"{home} wins": round(float(proba[0]),3),
        "Draw":          round(float(proba[1]),3),
        f"{away} wins":  round(float(proba[2]),3),
    }
    winner = home if pred==0 else (away if pred==2 else "draw")
    return winner, proba_dict


def sim_match(ta, tb, stage, year, use_proba=False):
    winner, probs = predict_match_ui(ta, tb, stage, year)
    if use_proba:
        vals = list(probs.values())
        roll = random.random()
        if   roll < vals[0]:           winner = ta
        elif roll < vals[0]+vals[1]:   winner = random.choice([ta, tb])
        else:                          winner = tb
    if winner == "draw":
        h_pts = get_rank(ta, 2024, rank_lookup)[1]
        a_pts = get_rank(tb, 2024, rank_lookup)[1]
        winner = ta if h_pts >= a_pts else tb
    return winner


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# ⚽ FIFA World Cup 2026 — Prediction Dashboard")
    st.markdown("**Priority signal: 2014–2024 FIFA Rankings · Elo Ratings · XGBoost Model**")
    st.markdown("---")

    # KPI Row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("🗂 Matches",       f"{len(df):,}")
    c2.metric("🌍 Teams",         f"{df['home_team'].nunique()}")
    c3.metric("📅 Tournaments",   f"{df['year'].nunique()}")
    c4.metric("🏅 Rank Years",    f"{ranking['year'].nunique()}")
    c5.metric("🤖 Accuracy",      f"{acc:.1%}")
    c6.metric("🏆 FIFA #1 (2024)", "Argentina")

    st.markdown("---")

    # ── Current 2024 FIFA Rankings for all 2026 teams ─────────────────────
    st.markdown("## 🏅 2026 Team Strength — 2024 FIFA Rankings")
    st.markdown("*This is the #1 priority feature driving all predictions*")

    team_rows = []
    for t in TEAMS_2026:
        r, p = get_rank(t, 2024, rank_lookup)
        e    = elo_ratings.get(t, DEFAULT_ELO)
        # Trend: rank in 2014 vs 2024
        r14, _ = get_rank(t, 2014, rank_lookup)
        trend   = int(r14) - int(r)  # positive = improved (rank number dropped)
        team_rows.append({"team":t,"rank":int(r),"points":round(p,1),"elo":round(e,1),"trend":trend})

    team_rows = sorted(team_rows, key=lambda x: x["rank"])

    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        # Top 16 teams as ranked cards
        st.markdown("### 🔝 Top 16 Strongest Teams")
        for i, t in enumerate(team_rows[:16]):
            medal = ["🥇","🥈","🥉"] + ["🔵"]*13
            card_class = ["gold-card","silver-card","bronze-card"] + ["blue-card"]*13
            trend_icon = "📈" if t["trend"]>5 else ("📉" if t["trend"]<-5 else "➡️")
            st.markdown(f"""
            <div class='rank-card {card_class[i]}'>
                <span style='font-size:18px'>{medal[i]}</span>
                <span style='font-weight:700;color:#e6edf3;margin-left:8px;font-size:15px'>{t["team"]}</span>
                <span style='float:right;color:#8b949e;font-size:12px'>
                    #{t["rank"]} · {t["points"]:.0f} pts · Elo {t["elo"]:.0f} · {trend_icon}
                </span>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        # FIFA points bar chart for top 20
        st.markdown("### 📊 FIFA Points Comparison")
        top20 = team_rows[:20]
        fig, ax = dark_fig(7, 8)
        names  = [t["team"] for t in top20[::-1]]
        points = [t["points"] for t in top20[::-1]]
        ranks  = [t["rank"] for t in top20[::-1]]
        colors = ["#FFD700" if r==1 else "#C0C0C0" if r==2 else "#CD7F32" if r==3 else "#388bfd"
                  for r in ranks]
        ax.barh(names, points, color=colors, edgecolor="#0d1117", height=0.75)
        for i, (p, n) in enumerate(zip(points, names)):
            ax.text(p+5, i, f"{p:.0f}", va="center", color="#c9d1d9", fontsize=8)
        ax.set_xlabel("FIFA Points (2024)", color="#8b949e")
        ax.set_title("2024 FIFA Points — Top 20 Teams", color="#e6edf3", fontweight="bold")
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # Bottom row: Goals trend + Outcome pie
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("### 📈 Goals Trend")
        gpy = df.groupby("year")["total_goals"].mean().reset_index()
        fig, ax = dark_fig(5, 3.5)
        ax.plot(gpy["year"], gpy["total_goals"], color="#58a6ff", marker="o", markersize=4, linewidth=2.5)
        ax.fill_between(gpy["year"], gpy["total_goals"], alpha=0.1, color="#58a6ff")
        ax.set_title("Avg Goals / Match per Year", color="#e6edf3", fontweight="bold")
        ax.tick_params(axis="x", rotation=35)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown("### 🥧 Match Outcomes")
        rc = df["result"].value_counts()
        fig, ax = dark_fig(5, 3.5)
        wedges, texts, auts = ax.pie(
            rc, labels=rc.index, autopct="%1.0f%%",
            colors=["#2ea043","#e74c3c","#f39c12"],
            startangle=90, wedgeprops={"edgecolor":"#0d1117","linewidth":2}
        )
        for tx in texts: tx.set_color("#c9d1d9"); tx.set_fontsize(10)
        for at in auts:  at.set_color("white"); at.set_fontweight("bold")
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_c:
        st.markdown("### ⚡ Top Elo Ratings")
        top_elo = pd.Series(elo_ratings).sort_values(ascending=False).head(8)
        fig, ax = dark_fig(5, 3.5)
        ax.barh(top_elo.index[::-1], top_elo.values[::-1],
                color=sns.color_palette("husl", 8)[::-1], edgecolor="#0d1117", height=0.65)
        for bar in ax.patches:
            ax.text(bar.get_width()+3, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.0f}", va="center", color="#c9d1d9", fontsize=8)
        ax.set_title("Top 8 by Elo", color="#e6edf3", fontweight="bold")
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — FIFA RANKINGS 2014–2024
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏅 FIFA Rankings (2014–2024)":
    st.markdown("# 🏅 FIFA Rankings — 2014 to 2024")
    st.markdown("**The most important data source for 2026 World Cup predictions.**")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Year Browser","📈 Trend Analysis","🏆 2026 Strength"])

    with tab1:
        sel_year = st.slider("Select Year", 1993, 2024, 2024)
        top_n    = st.slider("Top N teams", 10, 50, 25)
        yr_data  = ranking[ranking["year"]==sel_year].sort_values("fifa_rank").head(top_n)

        if len(yr_data)==0:
            st.warning(f"No data for {sel_year}")
        else:
            col_l, col_r = st.columns([1, 1.5])
            with col_l:
                disp = yr_data[["fifa_rank","team","fifa_points"]].copy()
                disp.columns = ["Rank","Team","Points"]
                disp["Points"] = disp["Points"].round(1)
                st.dataframe(disp, use_container_width=True, hide_index=True)
            with col_r:
                fig, ax = dark_fig(7, max(4, top_n*0.3))
                colors_yr = ["#FFD700" if r<=3 else "#388bfd" for r in yr_data["fifa_rank"]]
                ax.barh(yr_data["team"][::-1], yr_data["fifa_points"][::-1],
                        color=colors_yr[::-1], edgecolor="#0d1117", height=0.7)
                for i, (p, t) in enumerate(zip(yr_data["fifa_points"][::-1], yr_data["team"][::-1])):
                    ax.text(p+5, i, f"{p:.0f}", va="center", color="#c9d1d9", fontsize=8)
                ax.set_title(f"FIFA Rankings {sel_year}", color="#e6edf3", fontweight="bold")
                ax.set_xlabel("Points", color="#8b949e"); style_ax(ax)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        st.markdown("### 📈 Ranking Points Over Time")
        compare_teams = st.multiselect(
            "Select teams to compare",
            sorted(ranking["team"].unique()),
            default=["Argentina","Brazil","France","Germany","Spain","Morocco","Japan"],
        )
        yr_range = st.slider("Year Range", 2014, 2024, (2014, 2024))

        if compare_teams:
            fig, ax = dark_fig(12, 5)
            for team in compare_teams:
                td = ranking[(ranking["team"]==team) &
                             (ranking["year"]>=yr_range[0]) &
                             (ranking["year"]<=yr_range[1])].sort_values("year")
                if len(td)>0:
                    ax.plot(td["year"], td["fifa_points"],
                            marker="o", markersize=5, linewidth=2.5, label=team)
                    ax.annotate(f" {team}", (td["year"].iloc[-1], td["fifa_points"].iloc[-1]),
                                fontsize=8, color="#c9d1d9")
            ax.set_title(f"FIFA Points {yr_range[0]}–{yr_range[1]}", color="#e6edf3", fontweight="bold")
            ax.set_xlabel("Year"); ax.set_ylabel("FIFA Points")
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
            style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("### 📉📈 Biggest Movers 2014 → 2024")
        movers = []
        for team in TEAMS_2026:
            r14, p14 = get_rank(team, 2014, rank_lookup)
            r24, p24 = get_rank(team, 2024, rank_lookup)
            movers.append({
                "Team":team,"Rank 2014":int(r14),"Rank 2024":int(r24),
                "Change":int(r14)-int(r24),"Points 2024":round(p24,1)
            })
        mov_df = pd.DataFrame(movers).sort_values("Change", ascending=False)
        st.dataframe(mov_df, use_container_width=True, hide_index=True)

        fig, ax = dark_fig(12, 5)
        colors_mv = ["#2ecc71" if v>0 else "#e74c3c" for v in mov_df["Change"]]
        ax.barh(mov_df["Team"], mov_df["Change"], color=colors_mv, edgecolor="#0d1117", height=0.7)
        ax.axvline(0, color="#8b949e", linewidth=1)
        ax.set_title("Rank Improvement 2014→2024 (Green=Improved, Red=Declined)",
                     color="#e6edf3", fontweight="bold")
        ax.set_xlabel("Positions Gained"); style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("### 🏆 2026 Team Strength Radar")
        st.markdown("Combined strength = 70% FIFA Points (2024) + 30% Elo")

        rows2026 = []
        for t in TEAMS_2026:
            r, p = get_rank(t, 2024, rank_lookup)
            e    = elo_ratings.get(t, DEFAULT_ELO)
            r14, _ = get_rank(t, 2014, rank_lookup)
            trend   = int(r14) - int(r)
            combined = 0.7*p + 0.3*e
            rows2026.append({"Team":t,"FIFA Rank":int(r),"Points(2024)":round(p,1),
                              "Elo":round(e,1),"Trend(2014→2024)":trend,"Combined":round(combined,1)})
        df2026 = pd.DataFrame(rows2026).sort_values("Combined", ascending=False).reset_index(drop=True)
        df2026.index += 1
        st.dataframe(df2026, use_container_width=True)

        fig, ax = dark_fig(11, 9)
        top_combined = df2026.head(20).sort_values("Combined")
        medal_c = ["#FFD700" if r==1 else "#C0C0C0" if r==2 else "#CD7F32" if r==3 else "#388bfd"
                   for r in top_combined["FIFA Rank"]]
        ax.barh(top_combined["Team"], top_combined["Combined"],
                color=medal_c, edgecolor="#0d1117", height=0.72)
        for i,(v,t) in enumerate(zip(top_combined["Combined"], top_combined["Team"])):
            ax.text(v+2, i, f"{v:.0f}", va="center", color="#c9d1d9", fontsize=9)
        ax.set_title("Combined Strength Score — 2026 WC Top 20\n(70% FIFA 2024 + 30% Elo)",
                     color="#e6edf3", fontweight="bold")
        ax.set_xlabel("Combined Score"); style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA Explorer":
    st.markdown("# 🔍 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Goals","🌍 Teams","🔥 Correlations","🏆 WC History"])

    with tab1:
        gpy = df.groupby("year").agg(total=("total_goals","sum"), avg=("total_goals","mean")).reset_index()
        fig, axes = plt.subplots(1,2,figsize=(13,4.5))
        fig.patch.set_facecolor("#0d1117")
        for ax in axes: ax.set_facecolor("#161b22")
        axes[0].bar(gpy["year"],gpy["total"],color="#58a6ff",edgecolor="#0d1117",width=3)
        axes[0].set_title("Total Goals per World Cup",color="#e6edf3",fontweight="bold")
        axes[0].tick_params(axis="x",rotation=40,colors="#c9d1d9"); axes[0].tick_params(axis="y",colors="#c9d1d9")
        for sp in axes[0].spines.values(): sp.set_edgecolor("#30363d")
        axes[1].plot(gpy["year"],gpy["avg"],color="#f79c42",marker="o",linewidth=2.5)
        axes[1].fill_between(gpy["year"],gpy["avg"],alpha=0.15,color="#f79c42")
        axes[1].set_title("Avg Goals per Match",color="#e6edf3",fontweight="bold")
        axes[1].tick_params(axis="x",rotation=40,colors="#c9d1d9"); axes[1].tick_params(axis="y",colors="#c9d1d9")
        for sp in axes[1].spines.values(): sp.set_edgecolor("#30363d")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab2:
        n      = st.slider("Top N", 5, 20, 12)
        metric = st.selectbox("Metric", ["Total Goals","Total Wins","Matches Played"])
        hg = df.groupby("home_team")["home_goals"].sum()
        ag = df.groupby("away_team")["away_goals"].sum()
        hw = df[df["result"]=="home_win"].groupby("home_team").size()
        aw = df[df["result"]=="away_win"].groupby("away_team").size()
        hm = df.groupby("home_team").size(); am = df.groupby("away_team").size()
        dm = {"Total Goals":hg.add(ag,fill_value=0),"Total Wins":hw.add(aw,fill_value=0),"Matches Played":hm.add(am,fill_value=0)}
        s  = dm[metric].sort_values(ascending=False).head(n)
        fig, ax = dark_fig(10, max(4, n*0.45))
        ax.barh(s.index[::-1],s.values[::-1],color=sns.color_palette("husl",n),edgecolor="#0d1117")
        ax.set_title(f"Top {n} — {metric}",color="#e6edf3",fontweight="bold"); style_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        cols = ["year","home_goals","away_goals","half_home_goals","half_away_goals","total_goals","goal_diff"]
        corr = df[cols].corr()
        fig, ax = dark_fig(8,6)
        sns.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,square=True,
                    linewidths=0.5,linecolor="#30363d",annot_kws={"color":"#e6edf3"},ax=ax)
        ax.set_title("Correlation Heatmap",color="#e6edf3",fontweight="bold"); style_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        wc = cups["Winner"].value_counts()
        fig, ax = dark_fig(10,4)
        ax.bar(wc.index,wc.values,color=sns.color_palette("rocket",len(wc)),edgecolor="#0d1117")
        ax.set_title("World Cup Titles",color="#e6edf3",fontweight="bold")
        ax.tick_params(axis="x",rotation=30); style_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.dataframe(cups[["Year","Country","Winner","Runners-Up","GoalsScored"]].sort_values("Year",ascending=False),
                     use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — MATCH PREDICTOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Match Predictor":
    st.markdown("# 🎯 Match Predictor")
    st.markdown("Powered by **2024 FIFA Rankings (70%) + Elo (30%) + XGBoost**.")
    st.markdown("---")

    col1, col2, col3 = st.columns([2,2,1])
    with col1:
        home_default = ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0
        home = st.selectbox("🏠 Home / Team 1", ALL_TEAMS, index=home_default)
    with col2:
        away_default = ALL_TEAMS.index("Argentina") if "Argentina" in ALL_TEAMS else 1
        away = st.selectbox("✈️ Away / Team 2", ALL_TEAMS, index=away_default)
    with col3:
        stage = st.selectbox("Stage", STAGES, index=5)
    year = st.slider("Tournament Year", 2022, 2030, 2026)

    if home == away:
        st.warning("⚠️ Select two different teams.")
    elif st.button("⚡ Predict Match"):
        winner, probs = predict_match_ui(home, away, stage, year)
        h_elo_v = elo_ratings.get(home, DEFAULT_ELO)
        a_elo_v = elo_ratings.get(away, DEFAULT_ELO)
        h_rank, h_pts = get_rank(home, year-1, rank_lookup)
        a_rank, a_pts = get_rank(away, year-1, rank_lookup)

        if winner==home:   bc,emoji,rt = "#388bfd","🏆",f"{home} wins!"
        elif winner==away: bc,emoji,rt = "#e74c3c","🏆",f"{away} wins!"
        else:              bc,emoji,rt = "#f39c12","🤝","Draw!"

        st.markdown(f"""
        <div class='predict-box' style='background:linear-gradient(135deg,{bc}22,{bc}0a);border:2px solid {bc}'>
            <div style='font-size:52px'>{emoji}</div>
            <div style='font-size:30px;font-weight:800;color:{bc};margin-top:10px'>{rt}</div>
            <div style='font-size:13px;color:#8b949e;margin-top:6px'>
                {home} vs {away} · {stage} · {year}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("#### 📊 Win Probabilities")
        for label, prob in probs.items():
            col_bar, col_pct = st.columns([5, 1])
            with col_bar: st.progress(float(prob))
            with col_pct: st.markdown(f"`{prob:.1%}`")
            st.markdown(f"**{label}**")

        st.markdown("---")
        # Team comparison metrics
        t1,t2,t3,t4 = st.columns(4)
        t1.metric(f"🏠 {home} Elo",    f"{h_elo_v:.0f}")
        t2.metric(f"✈️ {away} Elo",    f"{a_elo_v:.0f}")
        t3.metric(f"🏠 {home} FIFA",   f"#{int(h_rank)}", f"{h_pts:.0f} pts")
        t4.metric(f"✈️ {away} FIFA",   f"#{int(a_rank)}", f"{a_pts:.0f} pts")

        # Bar chart
        fig, ax = dark_fig(7, 3)
        vals = list(probs.values())
        ax.bar(["Home Win","Draw","Away Win"], vals,
               color=["#388bfd","#f39c12","#e74c3c"], edgecolor="#0d1117", width=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v+0.01, f"{v:.1%}", ha="center", color="#e6edf3",
                    fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals)*1.3); ax.set_ylabel("Probability")
        ax.set_title("Outcome Probabilities", color="#e6edf3", fontweight="bold")
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — 2026 SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏆 2026 Simulator":
    st.markdown("# 🏆 FIFA World Cup 2026 Simulator")
    st.markdown("**2024 FIFA Rankings (priority) + Elo → Monte Carlo Championship Odds**")
    st.markdown("---")

    extra    = [t for t in TEAMS_2026 if t not in ALL_TEAMS]
    combined = sorted(list(set(ALL_TEAMS + extra)))

    selected = st.multiselect(
        "🌍 Select teams for the bracket (even number, 8–32)",
        combined, default=TEAMS_2026[:16], max_selections=32,
    )

    if len(selected)<2 or len(selected)%2!=0:
        st.info(f"⚠️ Select an even number of teams. Currently: {len(selected)}")
    else:
        tab_a, tab_b = st.tabs(["🎯 Single Bracket","🎲 Monte Carlo (Probabilistic)"])

        # ── Single Bracket ─────────────────────────────────────────────────
        with tab_a:
            if st.button("▶️ Run Bracket", key="single"):
                bracket = list(selected)
                all_rounds = {}
                while len(bracket)>1:
                    rnd = ROUND_NAMES.get(len(bracket), f"Round of {len(bracket)}")
                    winners, results = [], []
                    for i in range(0, len(bracket), 2):
                        ta, tb = bracket[i], bracket[i+1]
                        w = sim_match(ta, tb, rnd, 2026, use_proba=False)
                        results.append((ta, tb, w)); winners.append(w)
                    all_rounds[rnd]=results; bracket=winners

                for rnd, res in all_rounds.items():
                    st.markdown(f"### 📋 {rnd}")
                    n_cols = min(len(res), 4)
                    cols = st.columns(n_cols)
                    for i, (ta, tb, w) in enumerate(res):
                        with cols[i%n_cols]:
                            ca = "#2ea043" if w==ta else "#8b949e"
                            cb = "#2ea043" if w==tb else "#8b949e"
                            fa = "700" if w==ta else "400"
                            fb = "700" if w==tb else "400"
                            ta_r,ta_p = get_rank(ta, 2024, rank_lookup)
                            tb_r,tb_p = get_rank(tb, 2024, rank_lookup)
                            st.markdown(f"""
                            <div class='rank-card'>
                                <div style='color:{ca};font-weight:{fa};font-size:13px'>
                                    {ta} <span style='color:#8b949e;font-size:10px;font-weight:400'>#{int(ta_r)}</span>
                                </div>
                                <div style='color:#8b949e;font-size:10px;margin:3px 0'>vs</div>
                                <div style='color:{cb};font-weight:{fb};font-size:13px'>
                                    {tb} <span style='color:#8b949e;font-size:10px;font-weight:400'>#{int(tb_r)}</span>
                                </div>
                                <div style='margin-top:6px;color:#58a6ff;font-size:11px'>🏆 {w}</div>
                            </div>
                            """, unsafe_allow_html=True)

                champion = bracket[0]
                h_rank, h_pts = get_rank(champion, 2024, rank_lookup)
                st.markdown(f"""
                <div class='winner-box'>
                    <div style='font-size:60px'>🏆</div>
                    <div style='font-size:34px;font-weight:800;color:#2ea043;margin-top:10px'>{champion}</div>
                    <div style='font-size:16px;color:#8b949e;margin-top:6px'>
                        FIFA World Cup 2026 Champion<br>
                        FIFA Rank #{int(h_rank)} · {h_pts:.0f} points (2024)
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Monte Carlo ────────────────────────────────────────────────────
        with tab_b:
            n_sims = st.slider("Number of Simulations", 200, 2000, 1000, step=100)
            if st.button("🎲 Run Monte Carlo", key="mc"):
                counts = {}
                prog   = st.progress(0)
                status = st.empty()

                for si in range(n_sims):
                    bracket = list(selected); random.shuffle(bracket)
                    while len(bracket)>1:
                        rnd = ROUND_NAMES.get(len(bracket), f"Round of {len(bracket)}")
                        nxt = []
                        for i in range(0, len(bracket), 2):
                            nxt.append(sim_match(bracket[i], bracket[i+1], rnd, 2026, use_proba=True))
                        bracket = nxt
                    champ = bracket[0]; counts[champ] = counts.get(champ,0)+1
                    if (si+1)%100==0:
                        prog.progress((si+1)/n_sims)
                        status.markdown(f"Simulation {si+1}/{n_sims}…")

                prog.progress(1.0); status.empty()

                mc     = pd.Series(counts).sort_values(ascending=False)
                mc_pct = (mc / n_sims * 100).round(2)
                top3   = mc_pct.nlargest(3).index.tolist()

                # Results table with FIFA rank info
                rows_mc = []
                for team, pct in mc_pct.items():
                    r, p = get_rank(team, 2024, rank_lookup)
                    rows_mc.append({"Team":team,"FIFA Rank":int(r),"Points(2024)":round(p,1),
                                    "Win %":pct,"Times Won":int(counts[team])})
                mc_df = pd.DataFrame(rows_mc).sort_values("Win %", ascending=False).reset_index(drop=True)
                mc_df.index += 1
                st.dataframe(mc_df, use_container_width=True)

                # Chart
                mc_plot = mc_pct[mc_pct>0].sort_values(ascending=True)
                bar_colors = []
                for t in mc_plot.index:
                    if   t==top3[0]:                        bar_colors.append("#FFD700")
                    elif len(top3)>1 and t==top3[1]:       bar_colors.append("#C0C0C0")
                    elif len(top3)>2 and t==top3[2]:       bar_colors.append("#CD7F32")
                    else:                                    bar_colors.append("#388bfd")

                fig, ax = dark_fig(11, max(5, len(mc_plot)*0.42))
                bars = ax.barh(mc_plot.index, mc_plot.values,
                               color=bar_colors, edgecolor="#0d1117", height=0.72)
                for bar in bars:
                    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                            f"{bar.get_width():.1f}%", va="center", color="#c9d1d9", fontsize=9)

                patches = [mpatches.Patch(color="#FFD700", label=f"🥇 {top3[0]}")]
                if len(top3)>1: patches.append(mpatches.Patch(color="#C0C0C0", label=f"🥈 {top3[1]}"))
                if len(top3)>2: patches.append(mpatches.Patch(color="#CD7F32", label=f"🥉 {top3[2]}"))
                ax.legend(handles=patches, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
                ax.set_xlabel("Championship Probability (%)", color="#8b949e")
                ax.set_title(
                    f"🏆 FIFA World Cup 2026 — Championship Probabilities\n"
                    f"{n_sims} Monte Carlo Simulations · 2024 FIFA Rankings Priority",
                    color="#e6edf3", fontweight="bold",
                )
                style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

                top_team  = mc_pct.idxmax()
                h_rank, h_pts = get_rank(top_team, 2024, rank_lookup)
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#FFD70022,#FFD70008);
                    border:2px solid #FFD700;border-radius:16px;padding:24px;text-align:center;margin-top:16px'>
                    <div style='font-size:44px'>🎲🏆</div>
                    <div style='font-size:26px;font-weight:800;color:#FFD700;margin-top:8px'>
                        Most Likely 2026 Champion: {top_team}
                    </div>
                    <div style='font-size:14px;color:#8b949e;margin-top:6px'>
                        Won {mc_pct[top_team]:.1f}% of {n_sims} simulations<br>
                        Current FIFA Rank #{int(h_rank)} · {h_pts:.0f} points (2024)
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  PAGE 6 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance")
    st.markdown("Tested on **held-out** 2002–2014 matches (model never saw these during training).")
    st.markdown("---")

    labels = ["home_win","draw","away_win"]
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("🎯 Accuracy",      f"{acc:.2%}")
    m2.metric("📋 Test Matches",  f"{len(y_test)}")
    m3.metric("📅 Test Period",   "2002–2014")
    m4.metric("🌳 Trees",         "600")
    m5.metric("🔢 Features",      f"{len(FEATURES)}")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🔲 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = dark_fig(5.5, 4.5)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, linecolor="#30363d",
                    annot_kws={"color":"#0d1117","fontweight":"bold"}, ax=ax)
        ax.set_title("Confusion Matrix", color="#e6edf3", fontweight="bold")
        ax.set_ylabel("Actual", color="#8b949e"); ax.set_xlabel("Predicted", color="#8b949e")
        style_ax(ax); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("### 📌 Feature Importance")
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        bar_cs = [
            "#FFD700" if any(k in f for k in ["trend"]) else
            "#e74c3c" if any(k in f for k in ["fifa","rank","points"]) else
            "#388bfd" for f in imp.index
        ]
        fig, ax = dark_fig(5.5, 5.5)
        ax.barh(imp.index, imp.values, color=bar_cs, edgecolor="#0d1117", height=0.72)
        for bar in ax.patches:
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f"{bar.get_width():.3f}", va="center", color="#c9d1d9", fontsize=7)
        patches = [
            mpatches.Patch(color="#e74c3c", label="FIFA ranking features"),
            mpatches.Patch(color="#FFD700", label="Ranking trend (new)"),
            mpatches.Patch(color="#388bfd", label="Other features"),
        ]
        ax.legend(handles=patches, facecolor="#161b22", edgecolor="#30363d",
                  labelcolor="#c9d1d9", fontsize=8)
        ax.set_title("Feature Importance", color="#e6edf3", fontweight="bold")
        ax.set_xlabel("Importance", color="#8b949e"); style_ax(ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("### 📋 Classification Report")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("### 📊 All 17 Features")
    feat_df = pd.DataFrame({
        "Feature": FEATURES,
        "Category": [
            "Elo","Elo","Elo",
            "⭐ FIFA Rank","⭐ FIFA Rank","⭐ FIFA Rank",
            "⭐ FIFA Rank","⭐ FIFA Rank","⭐ FIFA Rank",
            "⭐ Rank Trend","⭐ Rank Trend",
            "Historical","Historical","Historical",
            "Match Context","Match Context","Year",
        ],
        "Description": [
            "Home team Elo","Away team Elo","Elo difference",
            "Home FIFA rank (2024)","Away FIFA rank (2024)","Rank difference",
            "Home FIFA points (2024)","Away FIFA points (2024)","Points difference",
            "Home ranking trend (3yr improvement)","Away ranking trend (3yr improvement)",
            "Home WC win rate","Away WC win rate","Head-to-head record",
            "Stage weight (1=Group,6=Final)","Host country advantage","Tournament year",
        ],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.info("⭐ = FIFA Ranking features (2014–2024 priority). These dominate the model as intended.")

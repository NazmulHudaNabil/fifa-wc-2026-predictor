<div align="center">

# ⚽ FIFA World Cup 2026 Predictor

### AI-powered match prediction & tournament simulator using official FIFA rankings + Elo ratings + XGBoost

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fifa-wc-2026-predictor.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-FF6600?style=flat&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-3.0.3-150458?style=flat&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat)

<br/>

> **Predict any World Cup match. Simulate the entire 2026 tournament. See the champion.**  
> Powered by 90+ years of match history, FIFA official rankings (1992–2024), and Gradient Boosted Trees.

<br/>

**[🚀 Live Demo](https://fifa-wc-2026-predictor.streamlit.app)** · **[📊 Notebook](notebook/notebook.ipynb)** · **[🐛 Report a Bug](https://github.com/NazmulHudaNabil/fifa-wc-2026-predictor/issues)**

</div>

---

## 📸 Preview

| Dashboard | 2026 Simulator | Match Predictor |
|:---------:|:--------------:|:---------------:|
| Real-time 2024 FIFA rankings for all 32 teams | Monte Carlo championship odds | Head-to-head probability breakdown |

---

## 🌟 Features

| Feature | Description |
|---------|-------------|
| 🏅 **FIFA Rankings Priority** | 2014–2024 official rankings are the #1 prediction signal — last 3 WC cycles |
| ⚡ **Elo Rating Engine** | Built from scratch using every WC match since 1930 |
| 📈 **Ranking Trend Detection** | Teams rising in rankings get a prediction boost |
| 🤖 **XGBoost Classifier** | 600-tree gradient boosted model with 17 engineered features |
| 🎯 **Match Predictor** | Predict any matchup with win/draw/loss probabilities |
| 🏆 **2026 Tournament Simulator** | Full knockout bracket + 1000-run Monte Carlo simulation |
| 🔍 **Interactive EDA** | Goals trends, outcome distributions, correlation heatmaps |
| 📊 **Model Transparency** | Confusion matrix, classification report, feature importance chart |

---

## 🧠 How It Works

```
Data (1930–2024)
    │
    ├── WorldCupMatches.csv  ──► Elo Rating Engine  ──► home_elo, away_elo, elo_diff
    ├── WorldCups.csv        ──► Host advantage, H2H win rates
    └── fifa_mens_rank.csv   ──► FIFA Rank / Points / Trend (2014–2024 priority)
                                            │
                                            ▼
                               ┌─────────────────────────┐
                               │   17 Features Combined   │
                               │                         │
                               │  • Elo ratings (3)      │
                               │  • FIFA rank/pts (6) ⭐  │
                               │  • Rank trend (2)   ⭐   │
                               │  • Historical WR (3)    │
                               │  • Match context (3)    │
                               └───────────┬─────────────┘
                                           │
                                           ▼
                               ┌─────────────────────────┐
                               │    XGBoost Classifier   │
                               │    600 trees · lr=0.02  │
                               │    Train: pre-2002      │
                               │    Test:  2002-2014     │
                               └───────────┬─────────────┘
                                           │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                          Home Win       Draw       Away Win
                         (prob %)      (prob %)    (prob %)
```

> ⭐ FIFA ranking features dominate the model — exactly as intended for 2026 prediction.

---

## 📂 Project Structure

```
fifa-wc-2026-predictor/
│
├── app.py                    # 🎯 Main Streamlit application (6 pages)
│
├── notebook/
│   └── notebook.ipynb        # 📒 Full ML pipeline — beginner-friendly
│
├── data/
│   ├── WorldCupMatches.csv   # ⚽ All WC matches 1930–2014
│   ├── WorldCups.csv         # 🏆 Tournament results & hosts
│   ├── WorldCupPlayers.csv   # 👤 Player appearances
│   └── fifa_mens_rank.csv    # 🏅 FIFA official rankings 1992–2024
│
├── .streamlit/
│   └── config.toml           # ⚙️ Streamlit server config (dark theme)
│
├── requirements.txt          # 📦 Python dependencies
├── packages.txt              # 🐧 System packages (for Streamlit Cloud)
├── .python-version           # 🐍 Python 3.12
├── Procfile                  # 🚂 Railway deployment
├── render.yaml               # 🎨 Render deployment
└── README.md
```

---

## 📊 Dataset Overview

| Dataset | Source | Coverage | Key Info |
|---------|--------|----------|----------|
| `WorldCupMatches.csv` | Kaggle | 1930–2014 | 852 matches, scores, stages |
| `WorldCups.csv` | Kaggle | 1930–2014 | Winners, hosts, goals |
| `WorldCupPlayers.csv` | Kaggle | 1930–2014 | Player appearances |
| `fifa_mens_rank.csv` | FIFA Official | 1992–2024 | Rank + points, bi-annual |

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| **Algorithm** | XGBoost (Gradient Boosted Trees) |
| **Task** | 3-class classification: Home Win / Draw / Away Win |
| **Trees** | 600 |
| **Learning Rate** | 0.02 |
| **Max Depth** | 4 |
| **Features** | 17 (Elo + FIFA rankings + trend + historical + context) |
| **Train Set** | All WC matches before 2002 |
| **Test Set** | WC matches 2002–2014 |
| **Test Accuracy** | ~57–63% *(random baseline = 33%)* |

> 💡 **Why ~60% is excellent:** Football is inherently unpredictable. Professional betting models achieve 60–65% on 3-way predictions. Our model is competitive with industry standards.

### Top Features (by importance)

```
home_fifa_points  ████████████████████  ← #1 most important
away_fifa_points  ██████████████████
rank_diff         ████████████████
elo_diff          ██████████████
home_rank_trend   ████████████          ← Rising teams detected
points_diff       ██████████
...
```

---

## 🚀 Quick Start — Run Locally

### Prerequisites
- Python 3.12+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/NazmulHudaNabil/fifa-wc-2026-predictor.git
cd fifa-wc-2026-predictor

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Then open your browser at **http://localhost:8501** 🎉

---

## 📒 Run the Notebook

```bash
# Install Jupyter if needed
pip install notebook

# Launch
jupyter notebook notebook/notebook.ipynb
```

The notebook walks through every step in beginner-friendly language:
1. 📥 Load Datasets
2. 🧹 Clean & Fix Missing Data
3. 🔍 Exploratory Data Analysis (7 charts)
4. ⚡ Elo Rating Engine
5. 🏅 FIFA Rankings (2014–2024 Priority)
6. 🔧 Feature Engineering (17 features)
7. 🤖 XGBoost Training & Evaluation
8. 🎯 Match Predictor Function
9. 🏆 2026 Tournament Simulator (1000 Monte Carlo runs)

---

## 🌐 App Pages

### 🏠 Dashboard
Real-time 2024 FIFA ranking cards for all 32 qualified teams. Gold/silver/bronze medals for top 3. Sidebar Elo chart, goals trend, outcome distribution.

### 🏅 FIFA Rankings (2014–2024)
- **Year Browser** — browse any year's rankings with interactive bar chart
- **Trend Analysis** — compare teams' points over time (2014–2024)
- **2026 Strength Table** — combined score: 70% FIFA points + 30% Elo

### 🔍 EDA Explorer
Goals trends, top goal-scoring nations, correlation heatmap, World Cup winners history.

### 🎯 Match Predictor
Select any two teams + stage + year → instant win/draw/loss probability breakdown with team stats comparison.

### 🏆 2026 Simulator
- **Single Bracket** — deterministic bracket run showing each match result
- **Monte Carlo** — probabilistic simulation (200–2000 runs) producing realistic championship odds

### 📊 Model Performance
Confusion matrix, feature importance chart, full classification report, and a table explaining all 17 features.

---

## ☁️ Deployment

The app is deployed on **Streamlit Community Cloud** — free, official, zero config.

```
Live URL: https://fifa-wc-2026-predictor.streamlit.app
```

For other platforms, see **[DEPLOYMENT_GUIDE.txt](DEPLOYMENT_GUIDE.txt)** which covers:
- Streamlit Community Cloud *(recommended)*
- Railway
- Render
- Hugging Face Spaces

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend / UI** | Streamlit 1.58 |
| **ML Model** | XGBoost 3.2 |
| **Data Processing** | Pandas 3.0, NumPy 2.4 |
| **Visualization** | Matplotlib 3.9, Seaborn 0.13 |
| **ML Utilities** | scikit-learn 1.9 (LabelEncoder, metrics) |
| **Language** | Python 3.12 |
| **Deployment** | Streamlit Community Cloud |

---

## 🔮 2026 Prediction Snapshot

> Based on 1000 Monte Carlo simulations using 2024 FIFA official rankings

| 🏅 Medal | Team | FIFA Rank (2024) | Win Probability |
|:--------:|------|:----------------:|:---------------:|
| 🥇 | **Argentina** | #1 | ~22–28% |
| 🥈 | **France** | #2 | ~14–18% |
| 🥉 | **Spain** | #3 | ~10–14% |
| 4th | **England** | #4 | ~8–12% |
| 5th | **Brazil** | #5 | ~7–10% |

*Results vary per simulation run. Open the app and run your own simulation!*

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
# Open a Pull Request on GitHub
```

**Ideas for contributions:**
- Add 2018 / 2022 WC match data to improve model accuracy
- Add group stage simulation (not just knockout)
- Add player-level features (avg age, goals per game)
- Add real-time FIFA ranking API integration

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute.

---

## 👤 Author

**Md Nazmul Huda Nabil**

[![GitHub](https://img.shields.io/badge/GitHub-NazmulHudaNabil-181717?style=flat&logo=github)](https://github.com/NazmulHudaNabil)

---

## ⭐ Support

If this project helped you or you found it interesting, please consider giving it a **⭐ star** on GitHub — it means a lot and helps others discover the project!

[![Star on GitHub](https://img.shields.io/github/stars/NazmulHudaNabil/fifa-wc-2026-predictor?style=social)](https://github.com/NazmulHudaNabil/fifa-wc-2026-predictor)

---

<div align="center">
  <sub>Built with ❤️ and Python · Data from Kaggle & FIFA Official</sub>
</div>

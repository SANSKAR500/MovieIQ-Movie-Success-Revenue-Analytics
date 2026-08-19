<div align="center">

# 🎬 MovieIQ
live streamlit link :- https://movieiq-movie-success-revenue-analytics-pxptbtcpqwekos6ae6lrde.streamlit.app/
### *Predictive Analytics on Film Success*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An interactive ML-powered dashboard that predicts whether a movie will be financially successful.**

[📊 Live Demo](#) • [🚀 Quick Start](#quick-start) • [📖 Documentation](#project-methodology) • [🔮 Try Predictions](#prediction-engine)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📈 **Interactive EDA** | Explore budget vs. revenue, genre trends, and feature distributions with dynamic filtering |
| 🧪 **Statistical Testing** | Built-in T-Tests and Chi-Square results with plain-language interpretations |
| 🤖 **Random Forest Model** | Trained classifier with 78.5% accuracy predicting movie success |
| 🎛️ **Smart Filters** | Sidebar controls for genre and minimum vote average |
| 🔮 **Live Predictions** | Input any movie's details and get an instant success/failure prediction with confidence score |
| 🌙 **Dark Premium UI** | Custom-styled Streamlit interface with gradient cards and smooth animations |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/movieiq.git
cd movieiq

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run MovieIQ.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## 📁 Project Structure

```
movieiq/
│
├── 📄 MovieIQ.py                 # Main Streamlit application
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # You are here!
│
├── 📂 assets/                    # Charts & trained model
│   ├── budget_vs_revenue.png
│   ├── genre_trends.png
│   ├── features_vs_success.png
│   ├── correlation_heatmap.png
│   ├── model_evaluation.png
│   └── movieiq_model.pkl         # Serialized Random Forest
│
└── 📄 movies.csv                 # Dataset (2,000 movies)
```

---

## 📊 Dataset

| Column | Type | Description |
|--------|------|-------------|
| `budget` | int | Production budget in USD |
| `revenue` | int | Worldwide box office revenue in USD |
| `popularity` | float | TMDB popularity score |
| `runtime` | int | Duration in minutes |
| `vote_average` | float | Average user rating (0–10) |
| `title` | str | Movie title |
| `genres` | str | JSON-like list of genre objects |

**Success Rule:** A movie is labeled **successful** when `revenue > budget`.

---

## 🔬 Project Methodology

### Stage 1 — Data Preparation
- Loaded 2,000 rows × 7 columns
- No missing values or zero anomalies detected
- Created binary `success` target column
- Extracted **primary genre** from JSON-like `genres` field
- **Class distribution:** 80.7% Success | 19.3% Failure *(imbalanced)*

### Stage 2 — Exploratory Data Analysis
- **Budget vs Revenue:** Strong positive correlation (r = 0.76)
- **Top Genres:** Romance, Adventure, Science Fiction (most common)
- **Most Successful Genres:** Drama (~82%), Horror (~82%), Animation (~81%)
- **Correlation Concern:** Budget ↔ Revenue is strong — revenue was excluded from model features to prevent data leakage

### Stage 3 — Statistical Testing

| Test | P-Value | Result |
|------|---------|--------|
| T-Test (Popularity) | **0.039** | ✅ Significant — successful movies are more popular |
| T-Test (Vote Average) | 0.301 | ❌ Not significant |
| Chi-Square (Genre) | 0.995 | ❌ Not significant — genre alone is not predictive |

*Threshold: α = 0.05*

### Stage 4 — Predictive Modeling

**Algorithm:** Random Forest Classifier (200 trees, max_depth=10, balanced class weights)

| Metric | Score |
|--------|-------|
| **Accuracy** | 78.5% |
| **Precision** | 80.8% |
| **Recall** | 96.3% |

**Feature Importance:**
1. 🥇 Popularity — 23.1%
2. 🥈 Vote Average — 22.9%
3. 🥉 Budget — 22.3%
4. Runtime — 20.4%
5. Genre (combined) — ~11%

**Known Limitation:** The model struggles to identify true failures (74 false positives) due to the imbalanced dataset. It tends to predict "success" by default.

---

## 🎮 Usage Guide

### Dashboard Tabs

#### 📈 Exploratory Analysis
- View pre-computed charts filtered by your sidebar selections
- Compare budget vs. revenue across genres
- Analyze feature distributions by success/failure

#### 🧪 Statistics & Model
- Review correlation heatmap and confusion matrix
- Read statistical test summaries
- Understand model performance at a glance

#### 🔮 Predict Success *(Prediction Engine)*
Enter a hypothetical movie:
- **Budget** — production cost in USD
- **Popularity** — pre-release buzz score
- **Runtime** — film length in minutes
- **Vote Average** — expected critic rating
- **Genre** — primary genre category

Click **"🎯 Predict Success"** to receive:
- Binary prediction (Success / Likely Failure)
- Confidence probability
- Visual progress bar

---

## 🌐 Deployment

### Streamlit Community Cloud *(Bonus)*
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path to `MovieIQ.py`
5. Deploy!

**No changes required** — the app is deployment-ready with relative paths and no hardcoded secrets.

---

## 🛠️ Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white" />
</p>

---

## 📝 Reflection

> **How confident is MovieIQ?**
>
> I would be **moderately confident** for typical films but would flag predictions with caveats. The 81% success rate in the dataset creates a bias toward predicting success. 
>
> **Limitation:** The dataset is imbalanced and appears synthetic — relationships like vote average vs. success are counter-intuitive.
>
> **Improvement:** With more time, I would add real-world features (cast star power, director track record, release season, marketing spend), apply SMOTE for class balancing, and use cross-validation for robust metrics.

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

<div align="center">

**Made with 🎬 + 🤖 + ☕**

*[MovieIQ — Turning box office guesswork into data-driven decisions]*

</div>

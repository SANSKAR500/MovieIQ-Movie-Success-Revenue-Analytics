import streamlit as st
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Page Config ───
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #2d2d44;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-4px); border-color: #667eea; }
    .metric-label { color: #a0a0b0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ffffff; font-size: 2rem; font-weight: 700; margin-top: 0.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e1e2e;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        color: #a0a0b0;
        border: none;
    }
    .stTabs [aria-selected="true"] { background: #667eea !important; color: white !important; }
    div[data-testid="stSidebarContent"] { background: #16162a; }
    .prediction-success {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 2rem; border-radius: 16px; text-align: center;
        color: white; font-size: 1.5rem; font-weight: 700;
    }
    .prediction-failure {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        padding: 2rem; border-radius: 16px; text-align: center;
        color: white; font-size: 1.5rem; font-weight: 700;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; padding: 0.75rem 2rem;
        border-radius: 8px; font-weight: 600; width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# ─── Helper: resolve path relative to this script ───
HERE = Path(__file__).parent.resolve()

def load_and_train():
    """Load CSV, engineer features, and train Random Forest. Returns (df, model, feature_cols, metrics)."""
    csv_path = HERE / "movies.csv"
    if not csv_path.exists():
        st.error("movies.csv not found. Please upload it to the same folder as MovieIQ.py")
        st.stop()

    df = pd.read_csv(csv_path)
    df['success'] = (df['revenue'] > df['budget']).astype(int)

    def extract_genre(g):
        if pd.isna(g) or g == '[]':
            return 'Unknown'
        try:
            l = ast.literal_eval(g)
            return l[0]['name'] if isinstance(l, list) and len(l) > 0 else 'Unknown'
        except Exception:
            return 'Unknown'

    df['primary_genre'] = df['genres'].apply(extract_genre)

    # One-hot encode genre for modelling
    df_model = pd.get_dummies(df, columns=['primary_genre'], prefix='genre')
    feature_cols = [c for c in df_model.columns if c.startswith('genre_')] + \
                   ['budget', 'popularity', 'runtime', 'vote_average']
    X = df_model[feature_cols]
    y = df_model['success']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'cm': confusion_matrix(y_test, y_pred)
    }

    return df, model, feature_cols, metrics

# ─── Load & Train (cached so it only runs once) ───
@st.cache_resource(show_spinner=False)
def _cached_load():
    return load_and_train()

with st.spinner("🎬 Loading dataset & training model..."):
    df, model, feature_cols, metrics = _cached_load()

all_genres = sorted(df['primary_genre'].unique().tolist())

# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:3rem;">🎬 MovieIQ</h1>
    <p style="margin:0.5rem 0 0 0; opacity:0.9; font-size:1.2rem;">
        Predictive Analytics on Film Success • Powered by Random Forest
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar Filters ───
st.sidebar.markdown("## 🎛️ Filters")
selected_genre = st.sidebar.selectbox(
    "Filter by Genre", ["All"] + all_genres
)
min_vote = st.sidebar.slider("Minimum Vote Average", 3.0, 9.0, 3.0, 0.1)

filtered_df = df.copy()
if selected_genre != "All":
    filtered_df = filtered_df[filtered_df['primary_genre'] == selected_genre]
filtered_df = filtered_df[filtered_df['vote_average'] >= min_vote]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(filtered_df)} movies")

# ─── KPI Cards ───
col1, col2, col3, col4 = st.columns(4)
success_rate = filtered_df['success'].mean() if len(filtered_df) > 0 else 0

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Movies</div>
        <div class="metric-value">{len(filtered_df):,}</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Success Rate</div>
        <div class="metric-value">{success_rate:.1%}</div>
    </div>""", unsafe_allow_html=True)
with col3:
    ab = filtered_df['budget'].mean() / 1e6 if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Budget</div>
        <div class="metric-value">${ab:.1f}M</div>
    </div>""", unsafe_allow_html=True)
with col4:
    ar = filtered_df['revenue'].mean() / 1e6 if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Avg Revenue</div>
        <div class="metric-value">${ar:.1f}M</div>
    </div>""", unsafe_allow_html=True)

# ─── Tabs ───
tab1, tab2, tab3 = st.tabs(["📈 Exploratory Analysis", "🧪 Statistics & Model", "🔮 Predict Success"])

# ─── TAB 1: EDA ───
with tab1:
    st.markdown("### Budget vs. Revenue")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['#e74c3c' if s == 0 else '#2ecc71' for s in filtered_df['success']]
    ax1.scatter(filtered_df['budget'] / 1e6, filtered_df['revenue'] / 1e6,
                c=colors, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    ax1.plot([0, filtered_df['budget'].max() / 1e6],
             [0, filtered_df['budget'].max() / 1e6],
             'k--', alpha=0.5, label='Break-even line')
    ax1.set_xlabel('Budget ($ Millions)')
    ax1.set_ylabel('Revenue ($ Millions)')
    ax1.set_title('Budget vs Revenue')
    ax1.legend()
    st.pyplot(fig1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Genre Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        genre_counts = filtered_df['primary_genre'].value_counts().head(10)
        ax2.barh(genre_counts.index[::-1], genre_counts.values[::-1], color='#3498db')
        ax2.set_xlabel('Count')
        st.pyplot(fig2)
    with c2:
        st.markdown("### Success by Genre")
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        gs = filtered_df.groupby('primary_genre')['success'].agg(['mean', 'count']).reset_index()
        gs = gs[gs['count'] >= 5].sort_values('mean')
        ax3.barh(gs['primary_genre'], gs['mean'], color='#9b59b6')
        ax3.axvline(x=success_rate, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Success Rate')
        st.pyplot(fig3)

# ─── TAB 2: Stats & Model ───
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Correlation Heatmap")
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        numeric_df = filtered_df[['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'success']]
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=1, cbar_kws={"shrink": .8}, ax=ax4)
        st.pyplot(fig4)
    with c2:
        st.markdown("### Confusion Matrix")
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Failure', 'Success'],
                    yticklabels=['Failure', 'Success'], ax=ax5)
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        st.pyplot(fig5)

    st.markdown("---")
    st.markdown(f"""
    ### Model Performance
    | Metric | Score |
    |---|---|
    | **Accuracy** | {metrics['accuracy']:.1%} |
    | **Precision** | {metrics['precision']:.1%} |
    | **Recall** | {metrics['recall']:.1%} |

    ### Key Notes
    - **Features used:** `budget`, `popularity`, `runtime`, `vote_average`, `primary_genre` (one-hot)
    - **Excluded:** `title` (non-predictive), `revenue` (target leakage), `genres` (raw string)
    - **Algorithm:** Random Forest (200 trees, max_depth=10, balanced class weights)
    - **Weakness:** Class imbalance causes many false positives (model leans toward predicting success).
    """)

# ─── TAB 3: Prediction ───
with tab3:
    st.markdown("### Enter Movie Details")
    pc1, pc2 = st.columns(2)
    with pc1:
        pred_budget = st.number_input("Budget ($)", min_value=1_000_000,
                                      max_value=300_000_000, value=100_000_000, step=5_000_000)
        pred_popularity = st.slider("Popularity Score", 0.0, 100.0, 50.0, 1.0)
    with pc2:
        pred_runtime = st.slider("Runtime (minutes)", 80, 180, 120, 1)
        pred_vote = st.slider("Vote Average", 3.0, 9.0, 6.0, 0.1)

    pred_genre = st.selectbox("Primary Genre", all_genres)

    if st.button("🎯 Predict Success", type="primary"):
        input_dict = {col: 0 for col in feature_cols}
        input_dict['budget'] = pred_budget
        input_dict['popularity'] = pred_popularity
        input_dict['runtime'] = pred_runtime
        input_dict['vote_average'] = pred_vote

        genre_col = f"genre_{pred_genre}"
        if genre_col in input_dict:
            input_dict[genre_col] = 1

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-success">
                ✅ PREDICTION: SUCCESS<br>
                <span style="font-size:1rem; opacity:0.9;">Confidence: {proba[1]:.1%}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-failure">
                ❌ PREDICTION: LIKELY FAILURE<br>
                <span style="font-size:1rem; opacity:0.9;">Confidence: {proba[0]:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

        st.progress(int(proba[1] * 100), text=f"Success Probability: {proba[1]:.1%}")

# ─── Footer ───
st.markdown("---")
st.caption("MovieIQ v1.0 • Built with Streamlit, scikit-learn & Python")

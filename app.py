# app.py - Premium Bitcoin Price Prediction Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.graph_objects as go
import yfinance as yf

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Bitcoin Predictor",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded" 
)

# ============================================================================
# PREMIUM STYLING
# ============================================================================
st.markdown("""
<style>
    /* Global Styles */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #f7931a 0%, #ffb347 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e293b;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Premium Card */
    .premium-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .info-card-title {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .info-card-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Success Card */
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .success-card h3 {
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
    }
    
    /* Compact Info Box */
    .compact-info {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    
    /* Data Table Styling */
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Metric Enhancement */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Compact Bullet Points */
    .compact-list {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #475569;
    }
    
    .compact-list li {
        margin-bottom: 0.3rem;
    }
    
    /* Confidence Badge */
    .confidence-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    
    /* Prediction Price */
    .prediction-price {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #10b981 0%, #059669 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 1rem 0;
    }
    
    .current-price {
        font-size: 2rem;
        font-weight: 700;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model_and_info():
    try:
        model = joblib.load('bitcoin_model.pkl')
        model_info = joblib.load('model_info.pkl')
        return model, model_info
    except:
        return None, None

model, model_info = load_model_and_info()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def create_features(df, lookback=7):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Close']].copy()
    
    for i in range(1, lookback + 1):
        df[f'close_lag_{i}'] = df['Close'].shift(i)
    
    df['ma_7'] = df['Close'].rolling(window=7).mean()
    df['ma_14'] = df['Close'].rolling(window=14).mean()
    df['ma_30'] = df['Close'].rolling(window=30).mean()
    df['volatility_7'] = df['Close'].rolling(window=7).std()
    df['price_change_1d'] = df['Close'].pct_change(1)
    df['price_change_7d'] = df['Close'].pct_change(7)
    
    df = df.dropna()
    return df

# ============================================================================
# SIDEBAR - MINIMAL & CLEAN
# ============================================================================
with st.sidebar:
    st.markdown("### Model Info")
    
    if model_info:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Features", len(model_info['feature_names']), label_visibility="visible")
        with col2:
            st.metric("Accuracy", f"{100-model_info['metrics']['test_mape']:.1f}%", label_visibility="visible")
        
        st.markdown(f"""
        <div style='background:#f1f5f9; padding:0.8rem; border-radius:8px; margin:1rem 0; font-size:0.85rem;'>
        <strong>Algorithm:</strong> Random Forest<br>
        <strong>Updated:</strong> {model_info['train_date']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("###  How It Works")
    st.markdown("""
    <ul class='compact-list'>
    <li>Analyzes 7-day price patterns</li>
    <li>Calculates moving averages</li>
    <li>Measures volatility trends</li>
    <li>Predicts next-day closing price</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='text-align:center; color:#94a3b8; font-size:0.8rem; margin-top:2rem;'>
    <strong>Educational Use Only</strong><br>
    Not financial advice
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown('<h1 class="main-header">₿ Bitcoin Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Next-Day Price Forecasting</p>', unsafe_allow_html=True)

# ============================================================================
# MAIN CONTENT
# ============================================================================
tab1, tab2 = st.tabs([" Predict", " Live Data"])

with tab1:
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    st.markdown('<h2 class="section-header"> Input Data</h2>', unsafe_allow_html=True)
    
    input_method = st.radio(
        "Choose method:",
        [" Auto-Fetch", " Manual Entry"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if input_method == " Auto-Fetch":
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col2:
            fetch_btn = st.button(" Fetch Latest Data", type="primary", use_container_width=True)
        
        if fetch_btn:
            with st.spinner(" Fetching..."):
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=90)
                    btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
                    
                    if isinstance(btc_data.columns, pd.MultiIndex):
                        btc_data.columns = btc_data.columns.get_level_values(0)
                    
                    # Ensure we have enough data
                    if len(btc_data) < 7:
                        st.error("Not enough historical data. Need at least 7 days.")
                        st.stop()

                    last_7_days = btc_data['Close'].tail(7).values
                    current_price = float(btc_data['Close'].iloc[-1])

                    # Validate data
                    if len(last_7_days) != 7:
                        st.error(f"Expected 7 days of data, got {len(last_7_days)}")
                        st.stop()
                    
                    # ========================================================
                    # DATA DISPLAY - CLEAN TABLE
                    # ========================================================
                    st.markdown('<h2 class="section-header"> Fetched Data</h2>', unsafe_allow_html=True)
                    
                    display_df = pd.DataFrame({
                        'Day': [f'Day {i+1}' for i in range(7)],
                        'Date': btc_data.index[-7:].strftime('%b %d').tolist(),
                        'Price': [f"${p:,.0f}" for p in last_7_days]
                    })
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Day": st.column_config.TextColumn("Day", width="small"),
                            "Date": st.column_config.TextColumn("Date", width="medium"),
                            "Price": st.column_config.TextColumn("Price", width="medium")
                        }
                    )
                    
                    # ========================================================
                    # PREDICTION SECTION
                    # ========================================================
                    st.markdown('<h2 class="section-header"> Prediction Results</h2>', unsafe_allow_html=True)
                    
                    if model and model_info:
                        df_features = create_features(btc_data, lookback=7)
                        X_pred = df_features[model_info['feature_names']].iloc[-1:].values
                        prediction = model.predict(X_pred)[0]
                        price_change = prediction - current_price
                        pct_change = (price_change / current_price) * 100
                        
                        # Prediction Card
                        st.markdown("""
                        <div class='success-card'>
                            <h3> Prediction Complete</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics Row
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Current Price",
                                f"${current_price:,.0f}",
                                help="Latest closing price"
                            )
                        
                        with col2:
                            delta_color = "normal" if pct_change >= 0 else "inverse"
                            st.metric(
                                "Predicted Price",
                                f"${prediction:,.0f}",
                                f"{pct_change:+.1f}%",
                                delta_color=delta_color
                            )
                        
                        with col3:
                            st.metric(
                                "Expected Change",
                                f"${abs(price_change):,.0f}",
                                "Up" if price_change > 0 else "Down"
                            )
                        
                        with col4:
                            margin = model_info['metrics']['test_mae']
                            st.metric(
                                "Confidence",
                                f"±${margin:,.0f}",
                                help="Average prediction error"
                            )
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Confidence Interval
                        st.markdown(f"""
                        <div class='compact-info'>
                        <strong> Confidence Range:</strong> ${prediction - margin:,.0f} – ${prediction + margin:,.0f} 
                        &nbsp;&nbsp;|&nbsp;&nbsp; <strong> For:</strong> {(datetime.now() + timedelta(days=1)).strftime('%b %d, %Y')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatter(
                            x=list(range(1, 8)),
                            y=last_7_days,
                            mode='lines+markers',
                            name='Historical',
                            line=dict(color='#3b82f6', width=3),
                            marker=dict(size=8),
                            hovertemplate='Day %{x}<br>$%{y:,.0f}<extra></extra>'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=[7, 8],
                            y=[last_7_days[-1], prediction],
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(color='#10b981', width=4, dash='dash'),
                            marker=dict(size=12, symbol='star'),
                            hovertemplate='Day %{x}<br>$%{y:,.0f}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title='7-Day Trend + Prediction',
                            xaxis_title='Day',
                            yaxis_title='Price (USD)',
                            hovermode='x unified',
                            height=450,
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            showlegend=True,
                            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
                            yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f" Error: {str(e)}")
    
    else:
        # ====================================================================
        # MANUAL INPUT
        # ====================================================================
        st.markdown('<h2 class="section-header">Enter 7-Day Prices</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            price_1 = st.number_input("Day 1", value=80000.0, step=100.0, format="%.0f")
            price_2 = st.number_input("Day 2", value=81000.0, step=100.0, format="%.0f")
            price_3 = st.number_input("Day 3", value=82000.0, step=100.0, format="%.0f")
        
        with col2:
            price_4 = st.number_input("Day 4", value=83000.0, step=100.0, format="%.0f")
            price_5 = st.number_input("Day 5", value=84000.0, step=100.0, format="%.0f")
            price_6 = st.number_input("Day 6", value=85000.0, step=100.0, format="%.0f")
        
        price_7 = st.number_input("Day 7 (Most Recent)", value=85090.0, step=100.0, format="%.0f")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            predict_btn = st.button(" Predict", type="primary", use_container_width=True)
        
        if predict_btn:
            if model and model_info:
                try:
                    prices = [price_1, price_2, price_3, price_4, price_5, price_6, price_7]
                    dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
                    df_manual = pd.DataFrame({'Close': prices}, index=dates)
                    
                    df_features = create_features(df_manual, lookback=7)
                    X_pred = df_features[model_info['feature_names']].iloc[-1:].values
                    prediction = model.predict(X_pred)[0]
                    price_change = prediction - price_7
                    pct_change = (price_change / price_7) * 100
                    
                    st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current", f"${price_7:,.0f}")
                    with col2:
                        st.metric("Predicted", f"${prediction:,.0f}", f"{pct_change:+.1f}%")
                    with col3:
                        st.metric("Change", f"${abs(price_change):,.0f}")
                    
                    # Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(1, 8)),
                        y=prices,
                        mode='lines+markers',
                        name='Input',
                        line=dict(color='#3b82f6', width=3)
                    ))

                    fig.add_trace(go.Scatter(
                        x=[7, 8],
                        y=[price_7, prediction],
                        mode='lines+markers',
                        name='Predicted',
                        line=dict(color='#10b981', width=4, dash='dash')
                    ))

                    fig.update_layout(
                        title='Prediction Forecast',
                        height=400,
                        plot_bgcolor='white',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f" Prediction error: {str(e)}")
                    st.error(f" Error: {str(e)}")
            else:
                st.error("Model not loaded!")

with tab2:
    # ========================================================================
    # LIVE DATA TAB
    # ========================================================================
    st.markdown('<h2 class="section-header">Live Market Data</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        refresh_btn = st.button(" Refresh", type="primary", use_container_width=True)
    
    if refresh_btn:
        with st.spinner("Loading..."):
            try:
                btc_live = yf.download('BTC-USD', period='30d', progress=False)
                if isinstance(btc_live.columns, pd.MultiIndex):
                    btc_live.columns = btc_live.columns.get_level_values(0)
                
                current = float(btc_live['Close'].iloc[-1])
                prev = float(btc_live['Close'].iloc[-2])
                change = current - prev
                pct = (change / prev) * 100
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Price", f"${current:,.0f}", f"{pct:+.1f}%")
                col2.metric("24h High", f"${float(btc_live['High'].iloc[-1]):,.0f}")
                col3.metric("24h Low", f"${float(btc_live['Low'].iloc[-1]):,.0f}")
                col4.metric("Volume", f"${float(btc_live['Volume'].iloc[-1])/1e9:.1f}B")
                
                # Candlestick Chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=btc_live.index,
                    open=btc_live['Open'],
                    high=btc_live['High'],
                    low=btc_live['Low'],
                    close=btc_live['Close'],
                    name='BTC'
                ))
                
                fig.update_layout(
                    title='30-Day Price Action',
                    xaxis_rangeslider_visible=False,
                    height=500,
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f" Error: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:0.85rem; padding:2rem 0;'>
<strong>Krunal Arvindbhai</strong> | Northeastern University ALY6110<br>
Machine Learning Model Deployment | Academic Project
</div>
""", unsafe_allow_html=True)
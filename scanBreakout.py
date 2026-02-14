"""
Yoda Pattern Scanner - Professional Dashboard
With Summary Cards, Target Suggestions, and Return Potential Ranking
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG & STYLING ====================

def apply_custom_css():
    """Apply custom CSS for professional styling."""
    st.markdown("""
    <style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Summary cards */
    .summary-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 10px;
    }
    
    .summary-card-green {
        background: linear-gradient(135deg, #134e5e 0%, #71b280 100%);
    }
    
    .summary-card-orange {
        background: linear-gradient(135deg, #f46b45 0%, #eea849 100%);
    }
    
    .summary-card-purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .summary-card-red {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
    }
    
    .card-title {
        font-size: 14px;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 5px;
    }
    
    .card-value {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    
    .card-subtitle {
        font-size: 12px;
        opacity: 0.8;
        margin-top: 5px;
    }
    
    /* Stock cards */
    .stock-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #2196F3;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stock-card-confirmed {
        border-left-color: #4CAF50;
    }
    
    .stock-card-potential {
        border-left-color: #FF9800;
    }
    
    .stock-symbol {
        font-size: 20px;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .stock-timeframe {
        font-size: 12px;
        color: #666;
        background: #f0f0f0;
        padding: 2px 8px;
        border-radius: 4px;
        margin-left: 8px;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    
    .metric-item {
        text-align: center;
    }
    
    .metric-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
    }
    
    .metric-value {
        font-size: 16px;
        font-weight: 600;
        color: #333;
    }
    
    .metric-value-green {
        color: #4CAF50;
    }
    
    .metric-value-red {
        color: #f44336;
    }
    
    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .badge-buy {
        background: #e8f5e9;
        color: #2e7d32;
    }
    
    .badge-sell {
        background: #ffebee;
        color: #c62828;
    }
    
    .badge-rank {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #333;
        font-size: 14px;
        padding: 5px 12px;
    }
    
    /* Table improvements */
    .dataframe {
        font-size: 13px !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Sector badges */
    .sector-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 8px;
        text-transform: uppercase;
    }
    
    .sector-tech { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    .sector-healthcare { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; }
    .sector-finance { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; }
    .sector-consumer { background: linear-gradient(135deg, #f46b45 0%, #eea849 100%); color: white; }
    .sector-energy { background: linear-gradient(135deg, #c31432 0%, #240b36 100%); color: white; }
    .sector-industrial { background: linear-gradient(135deg, #4b6cb7 0%, #182848 100%); color: white; }
    .sector-materials { background: linear-gradient(135deg, #8e9eab 0%, #eef2f3 100%); color: #333; }
    .sector-utilities { background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%); color: white; }
    .sector-realestate { background: linear-gradient(135deg, #834d9b 0%, #d04ed6 100%); color: white; }
    .sector-communication { background: linear-gradient(135deg, #00b4db 0%, #0083b0 100%); color: white; }
    .sector-unknown { background: #e0e0e0; color: #666; }
    
    /* Opportunity card - creative design */
    .opp-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e8e8e8;
        position: relative;
        overflow: hidden;
    }
    
    .opp-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
    }
    
    .opp-card-confirmed::before { background: linear-gradient(180deg, #4CAF50 0%, #81C784 100%); }
    .opp-card-potential::before { background: linear-gradient(180deg, #FF9800 0%, #FFB74D 100%); }
    .opp-card-hot::before { background: linear-gradient(180deg, #f44336 0%, #ff7043 100%); }
    
    .opp-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 15px;
    }
    
    .opp-symbol {
        font-size: 28px;
        font-weight: 800;
        color: #1a1a2e;
        letter-spacing: -0.5px;
    }
    
    .opp-name {
        font-size: 13px;
        color: #666;
        margin-top: 2px;
    }
    
    .opp-score {
        text-align: right;
    }
    
    .opp-score-value {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .opp-score-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
    }
    
    /* Signal gauge */
    .signal-gauge {
        display: flex;
        align-items: center;
        margin: 15px 0;
        padding: 12px;
        background: #f8f9fa;
        border-radius: 12px;
    }
    
    .gauge-bar {
        flex: 1;
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        margin: 0 15px;
        overflow: hidden;
    }
    
    .gauge-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .gauge-fill-green { background: linear-gradient(90deg, #4CAF50 0%, #81C784 100%); }
    .gauge-fill-orange { background: linear-gradient(90deg, #FF9800 0%, #FFB74D 100%); }
    .gauge-fill-red { background: linear-gradient(90deg, #f44336 0%, #ff7043 100%); }
    
    /* Target progress */
    .target-progress {
        display: flex;
        align-items: center;
        padding: 10px 0;
    }
    
    .target-dot {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    
    .target-met { background: #4CAF50; color: white; }
    .target-pending { background: #e0e0e0; color: #666; }
    .target-next { background: #FF9800; color: white; }
    
    /* Sector group card */
    .sector-group {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e8e8e8;
    }
    
    .sector-group-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e8e8e8;
    }
    
    .sector-group-title {
        font-size: 18px;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .sector-group-count {
        background: #e8e8e8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    
    /* Heat indicator */
    .heat-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .heat-hot { background: #ffebee; color: #c62828; }
    .heat-warm { background: #fff3e0; color: #e65100; }
    .heat-neutral { background: #e8f5e9; color: #2e7d32; }
    .heat-cold { background: #e3f2fd; color: #1565c0; }
    
    /* MACD indicator pill */
    .macd-pill {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    
    .macd-strong { background: linear-gradient(135deg, #4CAF50 0%, #81C784 100%); color: white; }
    .macd-positive { background: #e8f5e9; color: #2e7d32; }
    .macd-uptick { background: #e3f2fd; color: #1565c0; }
    
    /* Mini stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 15px;
    }
    
    .stat-box {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 18px;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .stat-label {
        font-size: 10px;
        color: #888;
        text-transform: uppercase;
        margin-top: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def summary_card(title, value, subtitle="", card_class=""):
    """Create a summary card HTML."""
    return f"""
    <div class="summary-card {card_class}">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        <div class="card-subtitle">{subtitle}</div>
    </div>
    """


def get_sector_class(sector):
    """Get CSS class for sector badge."""
    sector_map = {
        'Technology': 'sector-tech',
        'Information Technology': 'sector-tech',
        'Healthcare': 'sector-healthcare',
        'Health Care': 'sector-healthcare',
        'Financial Services': 'sector-finance',
        'Financials': 'sector-finance',
        'Consumer Cyclical': 'sector-consumer',
        'Consumer Defensive': 'sector-consumer',
        'Consumer Discretionary': 'sector-consumer',
        'Consumer Staples': 'sector-consumer',
        'Energy': 'sector-energy',
        'Industrials': 'sector-industrial',
        'Basic Materials': 'sector-materials',
        'Materials': 'sector-materials',
        'Utilities': 'sector-utilities',
        'Real Estate': 'sector-realestate',
        'Communication Services': 'sector-communication',
    }
    return sector_map.get(sector, 'sector-unknown')


def sector_badge_html(sector):
    """Create sector badge HTML."""
    sector_class = get_sector_class(sector)
    short_sector = sector[:12] + '...' if len(sector) > 12 else sector
    return f'<span class="sector-badge {sector_class}">{short_sector}</span>'


def opportunity_card_html(symbol, name, sector, industry, score, upside, rr, 
                          entry, target, stop, yoda_state, macd_status,
                          targets_met, total_targets, card_type='confirmed',
                          volume_strong=False, double_bottom=False):
    """Create creative opportunity card HTML with inline styles."""
    
    # Card border color
    if upside > 25:
        border_color = '#f44336'
    elif card_type == 'confirmed':
        border_color = '#4CAF50'
    else:
        border_color = '#FF9800'
    
    # Sector badge color
    sector_colors = {
        'Technology': '#667eea', 'Information Technology': '#667eea',
        'Healthcare': '#11998e', 'Health Care': '#11998e',
        'Financial Services': '#1e3c72', 'Financials': '#1e3c72',
        'Consumer Cyclical': '#f46b45', 'Consumer Defensive': '#f46b45',
        'Consumer Discretionary': '#f46b45', 'Consumer Staples': '#f46b45',
        'Energy': '#c31432', 'Industrials': '#4b6cb7',
        'Basic Materials': '#8e9eab', 'Materials': '#8e9eab',
        'Utilities': '#1f4037', 'Real Estate': '#834d9b',
        'Communication Services': '#00b4db'
    }
    sector_color = sector_colors.get(sector, '#888888')
    
    # MACD pill color
    if '↑' in macd_status or 'strong' in macd_status.lower():
        macd_bg = '#4CAF50'
        macd_text = 'white'
    elif '+' in macd_status or 'positive' in macd_status.lower():
        macd_bg = '#e8f5e9'
        macd_text = '#2e7d32'
    else:
        macd_bg = '#e3f2fd'
        macd_text = '#1565c0'
    
    # Indicators
    indicators = []
    if volume_strong:
        indicators.append('🔥')
    if double_bottom:
        indicators.append('W')
    indicator_str = ' '.join(indicators)
    
    # Gauge fill percentage and color
    gauge_pct = min(100, score)
    if score >= 70:
        gauge_color = '#4CAF50'
    elif score >= 50:
        gauge_color = '#FF9800'
    else:
        gauge_color = '#f44336'
    
    # Target progress dots HTML - build as simple spans
    target_dots_html = ''
    for i in range(min(total_targets, 5)):  # Limit to 5 dots
        if i < targets_met:
            target_dots_html += f'<span style="display:inline-block;width:22px;height:22px;line-height:22px;border-radius:50%;background:#4CAF50;color:white;text-align:center;font-size:11px;margin-right:4px;">✓</span>'
        elif i == targets_met:
            target_dots_html += f'<span style="display:inline-block;width:22px;height:22px;line-height:22px;border-radius:50%;background:#FF9800;color:white;text-align:center;font-size:11px;margin-right:4px;">→</span>'
        else:
            target_dots_html += f'<span style="display:inline-block;width:22px;height:22px;line-height:22px;border-radius:50%;background:#e0e0e0;color:#666;text-align:center;font-size:11px;margin-right:4px;">{i+1}</span>'
    
    # Truncate name
    display_name = name[:28] + '...' if len(name) > 28 else name
    
    # Build HTML as single concatenated string (no f-string newlines)
    html = f'<div style="background:#fff;border-radius:12px;padding:16px;margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);border-left:4px solid {border_color};">'
    
    # Header row
    html += f'<div style="display:flex;justify-content:space-between;margin-bottom:12px;">'
    html += f'<div>'
    html += f'<span style="font-size:24px;font-weight:700;color:#1a1a2e;">{symbol}</span>'
    if indicator_str:
        html += f'<span style="margin-left:8px;">{indicator_str}</span>'
    html += f'<div style="font-size:12px;color:#666;margin-top:2px;">{display_name}</div>'
    html += f'<span style="display:inline-block;padding:3px 10px;border-radius:12px;font-size:10px;font-weight:600;background:{sector_color};color:white;margin-top:6px;margin-right:6px;">{sector[:12]}</span>'
    html += f'<span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600;background:{macd_bg};color:{macd_text};">{macd_status}</span>'
    html += f'</div>'
    html += f'<div style="text-align:right;">'
    html += f'<div style="font-size:32px;font-weight:800;color:#667eea;">{score:.0f}</div>'
    html += f'<div style="font-size:10px;color:#888;">SCORE</div>'
    html += f'</div>'
    html += f'</div>'
    
    # Signal gauge
    html += f'<div style="background:#f5f5f5;padding:10px;border-radius:8px;margin-bottom:10px;">'
    html += f'<div style="display:flex;align-items:center;">'
    html += f'<span style="font-size:11px;color:#666;width:50px;">Signal</span>'
    html += f'<div style="flex:1;height:6px;background:#e0e0e0;border-radius:3px;margin:0 12px;position:relative;">'
    html += f'<div style="position:absolute;left:0;top:0;height:6px;width:{gauge_pct}%;background:{gauge_color};border-radius:3px;"></div>'
    html += f'</div>'
    html += f'<span style="font-size:13px;font-weight:600;color:#4CAF50;">+{upside:.1f}%</span>'
    html += f'</div>'
    html += f'</div>'
    
    # Targets row
    html += f'<div style="display:flex;align-items:center;margin-bottom:12px;">'
    html += f'<span style="font-size:11px;color:#666;margin-right:8px;">Targets:</span>'
    html += target_dots_html
    html += f'<span style="font-size:11px;color:#888;margin-left:auto;">{targets_met}/{total_targets} met</span>'
    html += f'</div>'
    
    # Stats row - using table for better compatibility
    html += f'<table style="width:100%;border-collapse:separate;border-spacing:8px 0;">'
    html += f'<tr>'
    html += f'<td style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center;width:25%;">'
    html += f'<div style="font-size:15px;font-weight:700;color:#1a1a2e;">${entry:.2f}</div>'
    html += f'<div style="font-size:9px;color:#888;margin-top:2px;">ENTRY</div>'
    html += f'</td>'
    html += f'<td style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center;width:25%;">'
    html += f'<div style="font-size:15px;font-weight:700;color:#4CAF50;">${target:.2f}</div>'
    html += f'<div style="font-size:9px;color:#888;margin-top:2px;">TARGET</div>'
    html += f'</td>'
    html += f'<td style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center;width:25%;">'
    html += f'<div style="font-size:15px;font-weight:700;color:#f44336;">${stop:.2f}</div>'
    html += f'<div style="font-size:9px;color:#888;margin-top:2px;">STOP</div>'
    html += f'</td>'
    html += f'<td style="background:#f8f9fa;padding:10px;border-radius:8px;text-align:center;width:25%;">'
    html += f'<div style="font-size:15px;font-weight:700;color:#1a1a2e;">{rr:.1f}:1</div>'
    html += f'<div style="font-size:9px;color:#888;margin-top:2px;">R/R</div>'
    html += f'</td>'
    html += f'</tr>'
    html += f'</table>'
    
    html += f'</div>'
    
    return html


def sector_group_header_html(sector, count, total_upside):
    """Create sector group header HTML."""
    sector_class = get_sector_class(sector)
    return f"""
    <div class="sector-group-header">
        <div>
            <span class="sector-badge {sector_class}" style="font-size: 14px; padding: 6px 16px;">{sector}</span>
            <span class="sector-group-title" style="margin-left: 10px;">Opportunities</span>
        </div>
        <div style="display: flex; gap: 15px;">
            <span class="sector-group-count">{count} stocks</span>
            <span class="heat-indicator heat-warm">+{total_upside:.1f}% avg</span>
        </div>
    </div>
    """


def stock_card_html(symbol, tf, status, score, upside, rr, entry, target, stop, yoda_state):
    """Create a stock card HTML."""
    card_class = "stock-card-confirmed" if "✅" in status else "stock-card-potential" if "⏳" in status else ""
    yoda_badge = "badge-buy" if yoda_state == "BUY" else "badge-sell" if yoda_state == "SELL" else ""
    upside_class = "metric-value-green" if upside > 0 else "metric-value-red"
    
    return f"""
    <div class="stock-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="stock-symbol">{symbol}</span>
                <span class="stock-timeframe">{tf}</span>
                <span class="badge {yoda_badge}" style="margin-left: 8px;">{yoda_state}</span>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 24px; font-weight: 700;">{score:.0f}</div>
                <div style="font-size: 11px; color: #888;">SCORE</div>
            </div>
        </div>
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-label">Entry</div>
                <div class="metric-value">${entry:.2f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Target</div>
                <div class="metric-value metric-value-green">${target:.2f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Stop</div>
                <div class="metric-value metric-value-red">${stop:.2f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Upside</div>
                <div class="metric-value {upside_class}">+{upside:.1f}%</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">R/R</div>
                <div class="metric-value">{rr:.1f}:1</div>
            </div>
        </div>
    </div>
    """


# ==================== YODA INDICATOR ====================

@st.cache_data(ttl=300)
def YodaSignal(data, fa=12, sa=26, sig=9, sma_length=50):
    """Compute Yoda indicator signals with buy volume analysis."""
    if data is None or data.empty or len(data) < 20:
        return pd.DataFrame()
    
    df = data.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series([0]*len(df)))
    
    # Volume Analysis
    df['Volume_Avg'] = volume.rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = np.where(df['Volume_Avg'] > 0, volume / df['Volume_Avg'], 1)
    df['Volume_Above_Avg'] = df['Volume_Ratio'] > 1.5
    df['Volume_Confirmed'] = df['Volume_Above_Avg']
    
    # Buy Volume Analysis (volume on up days)
    df['Is_Up_Day'] = close > close.shift(1)
    df['Buy_Volume'] = np.where(df['Is_Up_Day'], volume, 0)
    df['Sell_Volume'] = np.where(~df['Is_Up_Day'], volume, 0)
    
    # Rolling buy/sell volume averages
    df['Buy_Volume_Avg'] = df['Buy_Volume'].rolling(20, min_periods=1).mean()
    df['Sell_Volume_Avg'] = df['Sell_Volume'].rolling(20, min_periods=1).mean()
    
    # Buy volume ratio (current buy volume vs average buy volume)
    df['Buy_Volume_Ratio'] = np.where(
        df['Buy_Volume_Avg'] > 0, 
        df['Buy_Volume'] / df['Buy_Volume_Avg'], 
        1
    )
    
    # Buy volume above average on up days
    df['Buy_Volume_Above_Avg'] = (df['Is_Up_Day']) & (df['Buy_Volume_Ratio'] > 1.5)
    
    # Accumulation/Distribution indicator
    df['Vol_Pressure'] = np.where(
        df['Buy_Volume_Avg'] + df['Sell_Volume_Avg'] > 0,
        (df['Buy_Volume_Avg'] - df['Sell_Volume_Avg']) / (df['Buy_Volume_Avg'] + df['Sell_Volume_Avg']),
        0
    )
    
    # Weekly buy volume strength (rolling 5-day for daily data approximating a week)
    df['Weekly_Buy_Vol'] = df['Buy_Volume'].rolling(5, min_periods=1).sum()
    df['Weekly_Sell_Vol'] = df['Sell_Volume'].rolling(5, min_periods=1).sum()
    df['Weekly_Buy_Vol_Avg'] = df['Weekly_Buy_Vol'].rolling(4, min_periods=1).mean()  # 4-week average
    
    df['Weekly_Buy_Vol_Ratio'] = np.where(
        df['Weekly_Buy_Vol_Avg'] > 0,
        df['Weekly_Buy_Vol'] / df['Weekly_Buy_Vol_Avg'],
        1
    )
    df['Weekly_Buy_Vol_Strong'] = df['Weekly_Buy_Vol_Ratio'] > 1.2  # 20% above average
    
    # MACD
    ema_fast = close.ewm(span=fa, adjust=False).mean()
    ema_slow = close.ewm(span=sa, adjust=False).mean()
    df['MACD_Line'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=sig, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    
    # MACD colors
    df['MACD_Rising'] = df['MACD_Line'] > df['MACD_Line'].shift(1)
    df['Signal_Rising'] = df['MACD_Signal'] > df['MACD_Signal'].shift(1)
    
    # SMA
    df['SMA'] = close.rolling(sma_length, min_periods=1).mean()
    df['SMA_20'] = close.rolling(20, min_periods=1).mean()
    
    # Bollinger Bands
    bb_std = close.rolling(20, min_periods=1).std()
    df['BB_Basis'] = df['SMA_20']
    df['BB_Upper'] = df['BB_Basis'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Basis'] - 2 * bb_std
    
    # ATR
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14, min_periods=1).mean()
    
    # Keltner Channels
    df['KC_Upper'] = df['SMA_20'] + 1.5 * df['ATR']
    df['KC_Lower'] = df['SMA_20'] - 1.5 * df['ATR']
    
    # Squeeze
    df['In_Squeeze'] = (df['BB_Lower'] >= df['KC_Lower']) & (df['BB_Upper'] <= df['KC_Upper'])
    in_squeeze_prev = df['In_Squeeze'].shift(1).fillna(False).astype(bool)
    in_squeeze_curr = df['In_Squeeze'].fillna(False).astype(bool)
    df['Squeeze_Fired'] = in_squeeze_prev & (~in_squeeze_curr)
    
    # Signals - handle NaN with fillna
    close_prev = close.shift(1)
    sma_prev = df['SMA'].shift(1)
    df['Cross_Up'] = ((close_prev < sma_prev) & (close > df['SMA'])).fillna(False)
    df['Cross_Down'] = ((close_prev > sma_prev) & (close < df['SMA'])).fillna(False)
    
    # MACD buy/sell - ensure boolean types
    macd_rising_prev = df['MACD_Rising'].shift(1).fillna(False).astype(bool)
    signal_rising_prev = df['Signal_Rising'].shift(1).fillna(False).astype(bool)
    macd_rising_curr = df['MACD_Rising'].fillna(False).astype(bool)
    signal_rising_curr = df['Signal_Rising'].fillna(False).astype(bool)
    
    prev_both_green = macd_rising_prev & signal_rising_prev
    curr_both_green = macd_rising_curr & signal_rising_curr
    prev_both_red = (~macd_rising_prev) & (~signal_rising_prev)
    curr_both_red = (~macd_rising_curr) & (~signal_rising_curr)
    
    df['MACD_Buy'] = (~prev_both_green) & curr_both_green
    df['MACD_Sell'] = (~prev_both_red) & curr_both_red
    
    # Yoda signals
    df['YODA_BUY_SELL'] = 'NA'
    df.loc[df['MACD_Buy'] | df['Cross_Up'], 'YODA_BUY_SELL'] = 'BUY'
    df.loc[df['MACD_Sell'] | df['Cross_Down'], 'YODA_BUY_SELL'] = 'SELL'
    
    # Signal strength with buy volume bonus
    df['Signal_Strength'] = 0
    df.loc[df['MACD_Buy'], 'Signal_Strength'] += 20
    df.loc[df['Cross_Up'], 'Signal_Strength'] += 20
    df.loc[df['Squeeze_Fired'], 'Signal_Strength'] += 15
    df.loc[df['Volume_Above_Avg'], 'Signal_Strength'] += 10
    df.loc[df['Buy_Volume_Above_Avg'], 'Signal_Strength'] += 10  # Extra for buy volume
    df.loc[df['Weekly_Buy_Vol_Strong'], 'Signal_Strength'] += 10  # Weekly buy volume bonus
    df.loc[df['MACD_Hist'] > 0, 'Signal_Strength'] += 10
    df.loc[close > df['SMA'], 'Signal_Strength'] += 5
    df.loc[close > df['SMA_20'], 'Signal_Strength'] += 5
    
    # State tracking
    df['YODA_STATE'] = 'NA'
    state = 'NA'
    for i in range(len(df)):
        sig_val = df.loc[i, 'YODA_BUY_SELL']
        if sig_val == 'BUY':
            state = 'BUY'
        elif sig_val == 'SELL':
            state = 'SELL'
        df.loc[i, 'YODA_STATE'] = state
    
    return df


# ==================== PATTERN DETECTION ====================

def find_swing_highs(df, lookback=5):
    highs = []
    high = df['high'].values
    for i in range(lookback, len(df) - lookback):
        if all(high[i] > high[i-j] and high[i] > high[i+j] for j in range(1, lookback+1)):
            highs.append({'index': i, 'price': high[i]})
    return highs


def find_swing_lows(df, lookback=5):
    lows = []
    low = df['low'].values
    for i in range(lookback, len(df) - lookback):
        if all(low[i] < low[i-j] and low[i] < low[i+j] for j in range(1, lookback+1)):
            lows.append({'index': i, 'price': low[i]})
    return lows


def detect_double_bottom(df, swing_lows, tolerance_pct=0.03, min_bars_between=5, max_bars_between=50):
    """
    Detect double bottom pattern.
    
    A double bottom consists of:
    1. First bottom (swing low)
    2. Rally to a peak (neckline)
    3. Second bottom at approximately same level as first (within tolerance)
    4. Breakout above neckline confirms the pattern
    
    Returns dict with pattern info or None if not found.
    """
    if len(swing_lows) < 2 or len(df) < 20:
        return None
    
    patterns = []
    
    # Look for pairs of swing lows at similar levels
    for i in range(len(swing_lows) - 1):
        for j in range(i + 1, len(swing_lows)):
            bottom1 = swing_lows[i]
            bottom2 = swing_lows[j]
            
            # Check distance between bottoms
            bars_between = bottom2['index'] - bottom1['index']
            if bars_between < min_bars_between or bars_between > max_bars_between:
                continue
            
            # Check if bottoms are at similar price levels
            price_diff_pct = abs(bottom1['price'] - bottom2['price']) / bottom1['price']
            if price_diff_pct > tolerance_pct:
                continue
            
            # Find the peak (neckline) between the two bottoms
            between_start = bottom1['index'] + 1
            between_end = bottom2['index']
            
            if between_end <= between_start:
                continue
            
            between_highs = df['high'].iloc[between_start:between_end]
            if len(between_highs) == 0:
                continue
            
            neckline_idx = between_highs.idxmax()
            neckline_price = between_highs.max()
            
            # Neckline should be significantly above the bottoms
            avg_bottom = (bottom1['price'] + bottom2['price']) / 2
            neckline_height_pct = ((neckline_price - avg_bottom) / avg_bottom) * 100
            
            if neckline_height_pct < 3:  # At least 3% above bottoms
                continue
            
            # Calculate pattern target (measured move)
            pattern_height = neckline_price - avg_bottom
            target_price = neckline_price + pattern_height
            
            # Check if breakout has occurred (price closed above neckline after second bottom)
            breakout_confirmed = False
            breakout_index = None
            breakout_price = None
            
            for k in range(bottom2['index'] + 1, len(df)):
                if df['close'].iloc[k] > neckline_price:
                    breakout_confirmed = True
                    breakout_index = k
                    breakout_price = df['close'].iloc[k]
                    break
            
            # Check if pattern is still forming (potential breakout)
            current_price = df['close'].iloc[-1]
            distance_to_neckline = ((neckline_price - current_price) / current_price) * 100 if current_price < neckline_price else 0
            is_potential = not breakout_confirmed and current_price < neckline_price and distance_to_neckline <= 5
            
            # Volume analysis - ideally volume higher on second bottom or breakout
            vol_at_bottom1 = df['volume'].iloc[bottom1['index']] if 'volume' in df.columns else 0
            vol_at_bottom2 = df['volume'].iloc[bottom2['index']] if 'volume' in df.columns else 0
            volume_confirms = vol_at_bottom2 >= vol_at_bottom1 * 0.8  # Second bottom volume at least 80% of first
            
            # Calculate pattern quality score
            quality_score = 50
            if price_diff_pct < 0.01:  # Very tight bottoms
                quality_score += 15
            elif price_diff_pct < 0.02:
                quality_score += 10
            if neckline_height_pct >= 5:
                quality_score += 10
            if volume_confirms:
                quality_score += 15
            if breakout_confirmed:
                quality_score += 10
            
            patterns.append({
                'bottom1': bottom1,
                'bottom2': bottom2,
                'neckline_price': neckline_price,
                'neckline_index': neckline_idx,
                'avg_bottom_price': avg_bottom,
                'pattern_height': pattern_height,
                'target_price': target_price,
                'target_pct': ((target_price - neckline_price) / neckline_price) * 100,
                'breakout_confirmed': breakout_confirmed,
                'breakout_index': breakout_index,
                'breakout_price': breakout_price,
                'is_potential': is_potential,
                'distance_to_neckline': distance_to_neckline,
                'neckline_height_pct': neckline_height_pct,
                'bars_between': bars_between,
                'price_diff_pct': price_diff_pct * 100,
                'volume_confirms': volume_confirms,
                'quality_score': min(100, quality_score),
                'is_recent': bottom2['index'] >= len(df) - 20  # Pattern formed in last 20 bars
            })
    
    if not patterns:
        return None
    
    # Return the best quality pattern
    patterns.sort(key=lambda x: (x['breakout_confirmed'], x['is_recent'], x['quality_score']), reverse=True)
    return patterns[0]


def detect_trendlines(df, swing_highs, max_trendlines=5):
    if len(swing_highs) < 2:
        return []
    
    trendlines = []
    for i in range(len(swing_highs) - 1):
        for j in range(i + 1, len(swing_highs)):
            sh1, sh2 = swing_highs[i], swing_highs[j]
            if sh2['price'] >= sh1['price'] or sh2['index'] == sh1['index']:
                continue
            
            slope = (sh2['price'] - sh1['price']) / (sh2['index'] - sh1['index'])
            intercept = sh1['price'] - slope * sh1['index']
            
            if slope >= 0:
                continue
            
            touches, touch_idx = 0, []
            for sh in swing_highs:
                expected = slope * sh['index'] + intercept
                tol = abs(expected * 0.03)
                if abs(sh['price'] - expected) <= tol:
                    touches += 1
                    touch_idx.append(sh['index'])
            
            if touches >= 2:
                span = max(touch_idx) - min(touch_idx) if touch_idx else 0
                price_range = df['high'].max() - df['low'].min()
                angle = np.degrees(np.arctan(slope / price_range * len(df))) if price_range > 0 else 0
                quality = touches * 10 + span * 0.5 + abs(angle) * 0.2
                
                trendlines.append({
                    'slope': slope, 'intercept': intercept, 'touches': touches,
                    'touch_indices': touch_idx, 'start_index': min(touch_idx) if touch_idx else 0,
                    'end_index': max(touch_idx) if touch_idx else 0, 'angle_deg': angle,
                    'span': span, 'quality_score': quality
                })
    
    unique = []
    for tl in sorted(trendlines, key=lambda x: x['quality_score'], reverse=True):
        is_dup = any(abs(tl['slope'] - u['slope']) < abs(tl['slope'] * 0.1) for u in unique if tl['slope'] != 0)
        if not is_dup:
            unique.append(tl)
    
    return unique[:max_trendlines]


def find_breakouts(df, trendlines, yoda_df=None):
    """Find breakout points with volume analysis."""
    breakouts = []
    for tl_idx, tl in enumerate(trendlines):
        slope, intercept = tl['slope'], tl['intercept']
        start = max(tl['start_index'] + 1, 0)
        in_breakout = False
        
        for i in range(start, len(df)):
            tl_price = slope * i + intercept
            close = df['close'].iloc[i]
            
            if close > tl_price and not in_breakout:
                in_breakout = True
                pct = ((close - tl_price) / tl_price) * 100 if tl_price > 0 else 0
                
                # Volume analysis at breakout
                volume_confirmed = False
                buy_volume_strong = False
                weekly_buy_vol_strong = False
                volume_ratio = 1.0
                buy_vol_ratio = 1.0
                weekly_buy_vol_ratio = 1.0
                
                if yoda_df is not None and len(yoda_df) > i:
                    volume_confirmed = bool(yoda_df['Volume_Above_Avg'].iloc[i]) if 'Volume_Above_Avg' in yoda_df.columns else False
                    buy_volume_strong = bool(yoda_df['Buy_Volume_Above_Avg'].iloc[i]) if 'Buy_Volume_Above_Avg' in yoda_df.columns else False
                    weekly_buy_vol_strong = bool(yoda_df['Weekly_Buy_Vol_Strong'].iloc[i]) if 'Weekly_Buy_Vol_Strong' in yoda_df.columns else False
                    volume_ratio = float(yoda_df['Volume_Ratio'].iloc[i]) if 'Volume_Ratio' in yoda_df.columns else 1.0
                    buy_vol_ratio = float(yoda_df['Buy_Volume_Ratio'].iloc[i]) if 'Buy_Volume_Ratio' in yoda_df.columns else 1.0
                    weekly_buy_vol_ratio = float(yoda_df['Weekly_Buy_Vol_Ratio'].iloc[i]) if 'Weekly_Buy_Vol_Ratio' in yoda_df.columns else 1.0
                
                # Calculate volume score for breakout (0-100)
                vol_score = 0
                if volume_confirmed:
                    vol_score += 30
                if buy_volume_strong:
                    vol_score += 35
                if weekly_buy_vol_strong:
                    vol_score += 35
                
                breakouts.append({
                    'trendline_idx': tl_idx, 
                    'breakout_index': i,
                    'breakout_price': close, 
                    'trendline_price': tl_price,
                    'breakout_pct': pct, 
                    'is_recent': i >= len(df) - 10,
                    'volume_confirmed': volume_confirmed,
                    'buy_volume_strong': buy_volume_strong,
                    'weekly_buy_vol_strong': weekly_buy_vol_strong,
                    'volume_ratio': volume_ratio,
                    'buy_vol_ratio': buy_vol_ratio,
                    'weekly_buy_vol_ratio': weekly_buy_vol_ratio,
                    'vol_score': vol_score
                })
            elif close < tl_price * 0.98 and in_breakout:
                in_breakout = False
    return breakouts


# ==================== TARGETS & RESISTANCE ====================

def find_resistance_levels(df, yoda_df, swing_highs, current_price):
    levels = []
    
    for sh in swing_highs:
        if sh['price'] > current_price * 1.005:
            levels.append({'price': sh['price'], 'type': 'Swing High', 'strength': 4})
    
    if yoda_df is not None and 'BB_Upper' in yoda_df.columns and len(yoda_df) > 0:
        bb = yoda_df['BB_Upper'].iloc[-1]
        if pd.notna(bb) and bb > current_price * 1.005:
            levels.append({'price': bb, 'type': 'BB Upper', 'strength': 2})
    
    if len(df) > 0:
        period_high = df['high'].max()
        if period_high > current_price * 1.01:
            levels.append({'price': period_high, 'type': 'Period High', 'strength': 5})
    
    if len(df) >= 20:
        recent_high = df['high'].tail(20).max()
        if recent_high > current_price * 1.005:
            levels.append({'price': recent_high, 'type': '20-Bar High', 'strength': 3})
    
    if current_price > 0:
        mag = 10 ** max(0, len(str(int(current_price))) - 2)
        for mult in [1, 2, 5]:
            rnd = np.ceil(current_price / (mag * mult)) * (mag * mult)
            if current_price * 1.005 < rnd < current_price * 1.3:
                levels.append({'price': rnd, 'type': 'Round Number', 'strength': 1})
    
    if len(df) >= 50:
        recent_low = df['low'].tail(50).min()
        period_high = df['high'].max()
        rng = period_high - recent_low
        for fib, name in [(1.0, 'Fib 100%'), (1.272, 'Fib 127%'), (1.618, 'Fib 162%')]:
            fib_price = recent_low + rng * fib
            if current_price * 1.01 < fib_price < current_price * 1.5:
                levels.append({'price': fib_price, 'type': name, 'strength': 2})
    
    unique = {}
    for l in levels:
        key = round(l['price'], 2)
        if key not in unique or l['strength'] > unique[key]['strength']:
            unique[key] = l
    
    return sorted(unique.values(), key=lambda x: x['price'])[:5]


def find_support_levels(df, yoda_df, swing_lows, current_price):
    levels = []
    
    for sl in swing_lows:
        if sl['price'] < current_price * 0.995:
            levels.append({'price': sl['price'], 'type': 'Swing Low', 'strength': 3})
    
    if yoda_df is not None and len(yoda_df) > 0:
        if 'BB_Lower' in yoda_df.columns:
            bb = yoda_df['BB_Lower'].iloc[-1]
            if pd.notna(bb) and bb < current_price * 0.995:
                levels.append({'price': bb, 'type': 'BB Lower', 'strength': 2})
        
        if 'SMA' in yoda_df.columns:
            sma = yoda_df['SMA'].iloc[-1]
            if pd.notna(sma) and sma < current_price * 0.995:
                levels.append({'price': sma, 'type': 'SMA 50', 'strength': 3})
        
        if 'ATR' in yoda_df.columns:
            atr = yoda_df['ATR'].iloc[-1]
            if pd.notna(atr):
                levels.append({'price': current_price - 2 * atr, 'type': 'ATR Stop', 'strength': 2})
    
    if len(df) >= 20:
        recent_low = df['low'].tail(20).min()
        if recent_low < current_price * 0.995:
            levels.append({'price': recent_low, 'type': 'Recent Low', 'strength': 3})
    
    unique = {}
    for l in levels:
        key = round(l['price'], 2)
        if key not in unique or l['strength'] > unique[key]['strength']:
            unique[key] = l
    
    return sorted(unique.values(), key=lambda x: x['price'], reverse=True)[:3]


def calculate_targets(current_price, resistance, support, pattern_score, volume_weight=1.0):
    targets = {
        'entry': current_price, 'targets': [], 'stop_loss': None,
        'risk_reward': 0, 'max_upside_pct': 0, 'risk_pct': 5.0, 'return_score': 0,
        'volume_weight': volume_weight
    }
    
    if current_price <= 0:
        return targets
    
    for i, r in enumerate(resistance[:3]):
        pct = ((r['price'] - current_price) / current_price) * 100
        targets['targets'].append({
            'level': i + 1, 'price': r['price'], 'pct': pct,
            'type': r['type'], 'strength': r['strength']
        })
    
    if support:
        targets['stop_loss'] = support[0]['price']
        targets['risk_pct'] = ((current_price - targets['stop_loss']) / current_price) * 100
    else:
        targets['stop_loss'] = current_price * 0.95
        targets['risk_pct'] = 5.0
    
    if targets['targets'] and targets['stop_loss']:
        reward = targets['targets'][0]['price'] - current_price
        risk = current_price - targets['stop_loss']
        if risk > 0:
            targets['risk_reward'] = reward / risk
    
    if targets['targets']:
        targets['max_upside_pct'] = max(t['pct'] for t in targets['targets'])
    
    # Return score with volume weight bonus
    upside_score = min(25, targets['max_upside_pct'] * 1.5)
    rr_score = min(25, targets['risk_reward'] * 8)
    pattern_contrib = pattern_score * 0.3
    
    # Volume weight bonus (up to 20 points extra)
    volume_bonus = (volume_weight - 1.0) * 40  # 0 to 20 points
    
    targets['return_score'] = min(100, upside_score + rr_score + pattern_contrib + volume_bonus)
    
    return targets


def check_trend_health(df, yoda_df):
    if df is None or df.empty or len(df) < 20:
        return {'is_healthy': False, 'trend_score': 0}
    
    current = df['close'].iloc[-1]
    sma20 = df['close'].rolling(20).mean().iloc[-1]
    sma50 = df['close'].rolling(min(50, len(df))).mean().iloc[-1]
    
    above_sma20 = current > sma20 if pd.notna(sma20) else False
    above_sma50 = current > sma50 if pd.notna(sma50) else False
    
    recent_low = df['low'].tail(5).min()
    prior_low = df['low'].tail(20).head(15).min() if len(df) >= 20 else recent_low
    not_breaking = recent_low >= prior_low * 0.95
    
    yoda_bullish = yoda_df is not None and len(yoda_df) > 0 and yoda_df['YODA_STATE'].iloc[-1] == 'BUY'
    macd_bullish = yoda_df is not None and len(yoda_df) >= 2 and yoda_df['MACD_Hist'].iloc[-1] > yoda_df['MACD_Hist'].iloc[-2]
    
    score = (above_sma20 * 25 + above_sma50 * 25 + not_breaking * 20 + yoda_bullish * 20 + macd_bullish * 10)
    
    return {
        'is_healthy': score >= 50 and not_breaking, 'trend_score': score,
        'above_sma20': above_sma20, 'above_sma50': above_sma50,
        'not_breaking_down': not_breaking, 'yoda_bullish': yoda_bullish, 'macd_bullish': macd_bullish
    }


def detect_potential_breakout(df, trendlines, threshold=5.0):
    if not trendlines or df is None or df.empty:
        return {'is_potential': False}
    
    current = df['close'].iloc[-1]
    idx = len(df) - 1
    
    for tl_idx, tl in enumerate(trendlines):
        tl_price = tl['slope'] * idx + tl['intercept']
        if current < tl_price:
            dist = ((tl_price - current) / current) * 100
            if 0 < dist <= threshold:
                return {
                    'is_potential': True, 'trendline_price': tl_price,
                    'distance_to_breakout': dist,
                    'best_setup': {'trendline_idx': tl_idx, 'trendline': tl}
                }
    return {'is_potential': False}


def detect_pattern(df, yoda_df, timeframe='1d'):
    result = {
        'has_pattern': False, 'yoda_signal': 'NA', 'yoda_state': 'NA',
        'signal_strength': 0, 'volume_confirmed': False,
        'has_descending_trendline': False, 'trendline_touches': 0,
        'has_trendline_breakout': False, 'breakout_strength': 0,
        'pattern_score': 0, 'all_trendlines': [], 'all_breakouts': [],
        'recent_breakouts': [], 'swing_highs': [], 'swing_lows': [],
        'potential_breakout': {}, 'is_potential_breakout': False,
        'distance_to_breakout': 0, 'trend_health': {}, 'is_healthy_trend': False,
        'resistance_levels': [], 'support_levels': [], 'targets': {}, 'return_score': 0,
        # Volume metrics
        'buy_volume_strong': False,
        'weekly_buy_vol_strong': False,
        'weekly_buy_vol_ratio': 1.0,
        'buy_vol_ratio': 1.0,
        'breakout_vol_score': 0,
        'volume_weight': 1.0,
        # Double bottom pattern
        'double_bottom': None,
        'has_double_bottom': False,
        'double_bottom_confirmed': False,
        'double_bottom_potential': False
    }
    
    if df is None or df.empty or len(df) < 20:
        return result
    
    recent = df.tail(100).reset_index(drop=True)
    current_price = recent['close'].iloc[-1]
    
    # Prepare yoda_df for the recent period
    yoda_recent = None
    if yoda_df is not None and len(yoda_df) >= len(recent):
        yoda_recent = yoda_df.tail(len(recent)).reset_index(drop=True)
    
    swing_highs = find_swing_highs(recent)
    swing_lows = find_swing_lows(recent)
    result['swing_highs'] = swing_highs
    result['swing_lows'] = swing_lows
    
    # Detect double bottom pattern (especially useful for weekly)
    # Adjust parameters based on timeframe
    if timeframe == '1wk':
        db_params = {'tolerance_pct': 0.04, 'min_bars_between': 3, 'max_bars_between': 30}
    elif timeframe == '1mo':
        db_params = {'tolerance_pct': 0.05, 'min_bars_between': 2, 'max_bars_between': 20}
    else:  # daily
        db_params = {'tolerance_pct': 0.03, 'min_bars_between': 5, 'max_bars_between': 50}
    
    double_bottom = detect_double_bottom(recent, swing_lows, **db_params)
    if double_bottom:
        result['double_bottom'] = double_bottom
        result['has_double_bottom'] = True
        result['double_bottom_confirmed'] = double_bottom['breakout_confirmed']
        result['double_bottom_potential'] = double_bottom['is_potential']
    
    trendlines = detect_trendlines(recent, swing_highs)
    result['all_trendlines'] = trendlines
    
    if trendlines:
        result['has_descending_trendline'] = True
        result['trendline_touches'] = trendlines[0]['touches']
        
        # Pass yoda_df to get volume analysis at breakouts
        breakouts = find_breakouts(recent, trendlines, yoda_recent)
        result['all_breakouts'] = breakouts
        result['recent_breakouts'] = [b for b in breakouts if b['is_recent']]
        
        if result['recent_breakouts']:
            result['has_trendline_breakout'] = True
            result['breakout_strength'] = max(b['breakout_pct'] for b in result['recent_breakouts'])
            
            # Get best breakout volume metrics
            best_breakout = max(result['recent_breakouts'], key=lambda x: x.get('vol_score', 0))
            result['breakout_vol_score'] = best_breakout.get('vol_score', 0)
    
    # Current volume metrics from yoda_df
    if yoda_df is not None and len(yoda_df) > 0:
        result['yoda_signal'] = yoda_df['YODA_BUY_SELL'].iloc[-1]
        result['yoda_state'] = yoda_df['YODA_STATE'].iloc[-1]
        result['signal_strength'] = yoda_df['Signal_Strength'].iloc[-1]
        result['volume_confirmed'] = bool(yoda_df['Volume_Confirmed'].iloc[-1])
        
        # Buy volume metrics
        if 'Buy_Volume_Above_Avg' in yoda_df.columns:
            result['buy_volume_strong'] = bool(yoda_df['Buy_Volume_Above_Avg'].iloc[-1])
        if 'Weekly_Buy_Vol_Strong' in yoda_df.columns:
            result['weekly_buy_vol_strong'] = bool(yoda_df['Weekly_Buy_Vol_Strong'].iloc[-1])
        if 'Weekly_Buy_Vol_Ratio' in yoda_df.columns:
            result['weekly_buy_vol_ratio'] = float(yoda_df['Weekly_Buy_Vol_Ratio'].iloc[-1])
        if 'Buy_Volume_Ratio' in yoda_df.columns:
            result['buy_vol_ratio'] = float(yoda_df['Buy_Volume_Ratio'].iloc[-1])
        
        # Calculate volume weight multiplier (1.0 to 1.5)
        vol_weight = 1.0
        if result['volume_confirmed']:
            vol_weight += 0.1
        if result['buy_volume_strong']:
            vol_weight += 0.15
        if result['weekly_buy_vol_strong']:
            vol_weight += 0.25  # Weekly buy volume gets highest weight
        result['volume_weight'] = min(1.5, vol_weight)
    
    # Pattern score with volume weightage
    score = 0
    if result['yoda_signal'] == 'BUY': score += 25
    elif result['yoda_state'] == 'BUY': score += 15
    if result['has_descending_trendline']: score += 10
    if result['trendline_touches'] >= 3: score += 5
    if result['has_trendline_breakout']: score += 20
    if result['breakout_strength'] > 2: score += 5
    
    # Volume scoring (up to 35 points)
    if result['volume_confirmed']: score += 10
    if result['buy_volume_strong']: score += 10
    if result['weekly_buy_vol_strong']: score += 15  # Weekly buy volume above avg = strong signal
    
    # Double bottom bonus (up to 25 points)
    if result['has_double_bottom']:
        db = result['double_bottom']
        if db['breakout_confirmed']:
            score += 25  # Confirmed double bottom breakout is very bullish
        elif db['is_potential']:
            score += 15  # Potential double bottom setup
        else:
            score += 10  # Pattern exists
    
    # Apply volume weight to score
    score = int(score * result['volume_weight'])
    
    result['pattern_score'] = min(100, score)
    result['has_pattern'] = score >= 50
    
    potential = detect_potential_breakout(recent, trendlines)
    result['potential_breakout'] = potential
    result['is_potential_breakout'] = potential.get('is_potential', False)
    result['distance_to_breakout'] = potential.get('distance_to_breakout', 0)
    
    # Also consider double bottom potential breakout
    if result['double_bottom_potential'] and not result['is_potential_breakout']:
        result['is_potential_breakout'] = True
        result['distance_to_breakout'] = result['double_bottom']['distance_to_neckline']
    
    health = check_trend_health(recent, yoda_df)
    result['trend_health'] = health
    result['is_healthy_trend'] = health.get('is_healthy', False)
    
    resistance = find_resistance_levels(recent, yoda_df, swing_highs, current_price)
    support = find_support_levels(recent, yoda_df, swing_lows, current_price)
    
    # Add double bottom neckline as resistance if applicable
    if result['double_bottom_potential'] and result['double_bottom']:
        neckline = result['double_bottom']['neckline_price']
        if neckline > current_price:
            resistance.insert(0, {'price': neckline, 'type': 'DB Neckline', 'strength': 5})
    
    # Add double bottom target as resistance if breakout confirmed
    if result['double_bottom_confirmed'] and result['double_bottom']:
        target = result['double_bottom']['target_price']
        if target > current_price:
            resistance.insert(0, {'price': target, 'type': 'DB Target', 'strength': 4})
    
    result['resistance_levels'] = resistance
    result['support_levels'] = support
    
    targets = calculate_targets(current_price, resistance, support, result['pattern_score'], result['volume_weight'])
    result['targets'] = targets
    result['return_score'] = targets['return_score']
    
    return result


# ==================== DATA FETCHING ====================

@st.cache_data(ttl=3600)  # Cache for 1 hour since sector info doesn't change often
def fetch_stock_info(symbol):
    """Fetch stock info including sector and industry."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'name': info.get('shortName', symbol),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0),
            'week_52_high': info.get('fiftyTwoWeekHigh', 0),
            'week_52_low': info.get('fiftyTwoWeekLow', 0)
        }
    except:
        return {
            'sector': 'Unknown', 'industry': 'Unknown', 'name': symbol,
            'market_cap': 0, 'pe_ratio': 0, 'dividend_yield': 0, 'beta': 0,
            'week_52_high': 0, 'week_52_low': 0
        }


@st.cache_data(ttl=300)
def fetch_data(symbol, period='1y', interval='1d'):
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return None


def scan_stock(symbol, timeframes):
    # Fetch stock info for sector/industry
    stock_info = fetch_stock_info(symbol)
    
    results = {
        'symbol': symbol, 'timeframes': {}, 'best_timeframe': None,
        'max_score': 0, 'has_pattern': False, 'return_score': 0, 'max_upside': 0,
        'has_double_bottom': False, 'double_bottom_confirmed': False,
        'sector': stock_info['sector'],
        'industry': stock_info['industry'],
        'name': stock_info['name'],
        'market_cap': stock_info['market_cap'],
        'beta': stock_info['beta'],
        'week_52_high': stock_info['week_52_high'],
        'week_52_low': stock_info['week_52_low']
    }
    
    periods = {'1d': '1y', '1wk': '2y', '1mo': '5y'}
    
    for tf in timeframes:
        df = fetch_data(symbol, periods.get(tf, '1y'), tf)
        if df is None or len(df) < 30:
            results['timeframes'][tf] = {'error': 'No data'}
            continue
        
        yoda_df = YodaSignal(df.copy())
        pattern = detect_pattern(df, yoda_df, timeframe=tf)
        
        results['timeframes'][tf] = {
            'df': df, 'yoda_df': yoda_df, 'pattern': pattern,
            'pattern_score': pattern['pattern_score'],
            'has_pattern': pattern['has_pattern'],
            'current_price': df['close'].iloc[-1],
            'return_score': pattern['return_score'],
            'max_upside': pattern['targets'].get('max_upside_pct', 0),
            'has_double_bottom': pattern['has_double_bottom'],
            'double_bottom_confirmed': pattern['double_bottom_confirmed'],
            'double_bottom_potential': pattern['double_bottom_potential']
        }
        
        if pattern['pattern_score'] > results['max_score']:
            results['max_score'] = pattern['pattern_score']
            results['best_timeframe'] = tf
            results['has_pattern'] = pattern['has_pattern']
            results['return_score'] = pattern['return_score']
            results['max_upside'] = pattern['targets'].get('max_upside_pct', 0)
            results['has_double_bottom'] = pattern['has_double_bottom']
            results['double_bottom_confirmed'] = pattern['double_bottom_confirmed']
    
    return results


# ==================== CHARTING ====================

def create_tradingview_chart(df, yoda_df, pattern, symbol, tf, height=400):
    """Create a TradingView-style chart with dark theme and clean design."""
    
    # Reset index to use integer positions for easier handling
    df_plot = df.reset_index()
    date_col = df_plot.columns[0]  # First column is the date after reset
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.75, 0.25]
    )
    
    # Candlestick with TradingView colors
    fig.add_trace(go.Candlestick(
        x=df_plot[date_col], open=df_plot['open'], high=df_plot['high'],
        low=df_plot['low'], close=df_plot['close'], name='',
        increasing_fillcolor='#26a69a', increasing_line_color='#26a69a',
        decreasing_fillcolor='#ef5350', decreasing_line_color='#ef5350',
        showlegend=False
    ), row=1, col=1)
    
    # EMAs like TradingView
    if len(df_plot) >= 9:
        ema9 = df_plot['close'].ewm(span=9).mean()
        fig.add_trace(go.Scatter(x=df_plot[date_col], y=ema9, mode='lines', name='EMA 9',
                                line=dict(color='#f7931a', width=1)), row=1, col=1)
    
    if len(df_plot) >= 21:
        ema21 = df_plot['close'].ewm(span=21).mean()
        fig.add_trace(go.Scatter(x=df_plot[date_col], y=ema21, mode='lines', name='EMA 21',
                                line=dict(color='#2962ff', width=1)), row=1, col=1)
    
    if yoda_df is not None and 'SMA' in yoda_df.columns and len(yoda_df) > 0:
        yoda_plot = yoda_df.reset_index(drop=True)
        fig.add_trace(go.Scatter(x=df_plot[date_col], y=yoda_plot['SMA'], mode='lines', name='SMA 50',
                                line=dict(color='#9c27b0', width=1)), row=1, col=1)
    
    # Double Bottom Pattern Visualization
    db = pattern.get('double_bottom')
    if db and pattern.get('has_double_bottom'):
        chart_len = len(df_plot)
        
        # Get indices relative to the chart (they may need adjustment)
        b1_idx = min(db['bottom1']['index'], chart_len - 1)
        b2_idx = min(db['bottom2']['index'], chart_len - 1)
        neckline_idx = min(db['neckline_index'], chart_len - 1)
        
        if b1_idx < chart_len and b2_idx < chart_len:
            # Draw the W pattern
            # Bottom 1 marker
            fig.add_trace(go.Scatter(
                x=[df_plot[date_col].iloc[b1_idx]],
                y=[db['bottom1']['price']],
                mode='markers+text',
                text=['B1'],
                textposition='bottom center',
                marker=dict(symbol='triangle-up', size=14, color='#00e676'),
                name='Bottom 1',
                showlegend=False
            ), row=1, col=1)
            
            # Bottom 2 marker
            fig.add_trace(go.Scatter(
                x=[df_plot[date_col].iloc[b2_idx]],
                y=[db['bottom2']['price']],
                mode='markers+text',
                text=['B2'],
                textposition='bottom center',
                marker=dict(symbol='triangle-up', size=14, color='#00e676'),
                name='Bottom 2',
                showlegend=False
            ), row=1, col=1)
            
            # Neckline
            fig.add_hline(
                y=db['neckline_price'], line_dash="dash", line_color="#ff9800", line_width=2,
                annotation_text=f"Neckline: ${db['neckline_price']:.2f}",
                annotation_position="right", annotation_font_color="#ff9800",
                row=1, col=1
            )
            
            # Connect the W pattern with lines
            if neckline_idx < chart_len:
                # Draw W shape
                w_x = [
                    df_plot[date_col].iloc[b1_idx],
                    df_plot[date_col].iloc[neckline_idx],
                    df_plot[date_col].iloc[b2_idx]
                ]
                w_y = [
                    db['bottom1']['price'],
                    db['neckline_price'],
                    db['bottom2']['price']
                ]
                fig.add_trace(go.Scatter(
                    x=w_x, y=w_y, mode='lines',
                    line=dict(color='#00e676', width=2, dash='dot'),
                    name='Double Bottom',
                    showlegend=True
                ), row=1, col=1)
            
            # Target line if confirmed
            if db['breakout_confirmed']:
                fig.add_hline(
                    y=db['target_price'], line_dash="dot", line_color="#00e676", line_width=2,
                    annotation_text=f"DB Target: ${db['target_price']:.2f} (+{db['target_pct']:.1f}%)",
                    annotation_position="right", annotation_font_color="#00e676",
                    row=1, col=1
                )
                
                # Mark breakout point
                if db['breakout_index'] and db['breakout_index'] < chart_len:
                    fig.add_trace(go.Scatter(
                        x=[df_plot[date_col].iloc[db['breakout_index']]],
                        y=[db['breakout_price']],
                        mode='markers+text',
                        text=['BREAKOUT'],
                        textposition='top center',
                        marker=dict(symbol='star', size=16, color='#ffd700'),
                        name='Breakout',
                        showlegend=False
                    ), row=1, col=1)
    
    # Trendlines - draw simple diagonal lines based on price action
    chart_len = len(df_plot)
    if chart_len >= 20:
        # Find recent highs for resistance line
        recent_highs = df_plot['high'].tail(40)
        max_price = recent_highs.max()
        current_price = df_plot['close'].iloc[-1]
        
        # Draw a simple resistance line if price is below recent high
        if current_price < max_price * 0.98:
            start_idx = max(0, chart_len - 40)
            end_idx = chart_len - 1
            
            fig.add_trace(go.Scatter(
                x=[df_plot[date_col].iloc[start_idx], df_plot[date_col].iloc[end_idx]], 
                y=[max_price, max_price],
                mode='lines', name='Resistance', showlegend=True,
                line=dict(color='#ff5252', width=2, dash='dot')
            ), row=1, col=1)
    
    # Target and Stop lines
    targets = pattern.get('targets', {})
    if targets.get('targets'):
        t1 = targets['targets'][0]
        fig.add_hline(y=t1['price'], line_dash="solid", line_color="#26a69a", line_width=1,
                     annotation_text=f"TP: ${t1['price']:.2f}", annotation_position="right",
                     annotation_font_color="#26a69a", row=1, col=1)
    
    if targets.get('stop_loss'):
        fig.add_hline(y=targets['stop_loss'], line_dash="solid", line_color="#ef5350", line_width=1,
                     annotation_text=f"SL: ${targets['stop_loss']:.2f}", annotation_position="right",
                     annotation_font_color="#ef5350", row=1, col=1)
    
    # Entry line
    if targets.get('entry'):
        fig.add_hline(y=targets['entry'], line_dash="dash", line_color="#ffd700", line_width=1,
                     annotation_text=f"Entry: ${targets['entry']:.2f}", annotation_position="left",
                     annotation_font_color="#ffd700", row=1, col=1)
    
    # Volume with TradingView styling
    vol_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_plot['close'], df_plot['open'])]
    fig.add_trace(go.Bar(x=df_plot[date_col], y=df_plot['volume'], marker_color=vol_colors,
                        opacity=0.7, showlegend=False), row=2, col=1)
    
    # TradingView-style layout
    fig.update_layout(
        height=height,
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation='h', yanchor='top', y=0.99, xanchor='left', x=0.01,
            bgcolor='rgba(0,0,0,0)', font=dict(size=10, color='#b2b5be')
        ),
        margin=dict(l=50, r=50, t=30, b=30),
        hovermode='x unified'
    )
    
    # Style axes like TradingView
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#1e222d',
        showline=False, zeroline=False,
        tickfont=dict(color='#787b86', size=10)
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='#1e222d',
        showline=False, zeroline=False, side='right',
        tickfont=dict(color='#787b86', size=10)
    )
    
    # Remove x-axis labels on price chart, keep only on volume
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=True, row=2, col=1)
    
    return fig


def display_flash_cards(matches, potential):
    """Display flash cards with TradingView-style charts for best opportunities."""
    
    st.markdown('<div class="section-header">📱 Flash Cards (No SELL Signals)</div>', unsafe_allow_html=True)
    
    # Combine and sort opportunities
    opps = []
    
    for m in matches:
        tf = m.get('best_timeframe')
        if not tf:
            continue
        tf_data = m.get('timeframes', {}).get(tf, {})
        pattern = tf_data.get('pattern', {})
        targets = pattern.get('targets', {})
        
        opps.append({
            'symbol': m['symbol'], 'tf': tf, 'type': 'confirmed',
            'return_score': m.get('return_score', 0),
            'max_upside': targets.get('max_upside_pct', 0),
            'risk_reward': targets.get('risk_reward', 0),
            'entry': targets.get('entry', 0),
            'target1': targets['targets'][0]['price'] if targets.get('targets') else 0,
            'target1_pct': targets['targets'][0]['pct'] if targets.get('targets') else 0,
            'stop': targets.get('stop_loss', 0),
            'risk_pct': targets.get('risk_pct', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'signal_strength': pattern.get('signal_strength', 0),
            'volume_confirmed': pattern.get('volume_confirmed', False),
            'buy_volume_strong': pattern.get('buy_volume_strong', False),
            'weekly_buy_vol_strong': pattern.get('weekly_buy_vol_strong', False),
            'weekly_buy_vol_ratio': pattern.get('weekly_buy_vol_ratio', 1.0),
            'volume_weight': pattern.get('volume_weight', 1.0),
            'pattern_score': pattern.get('pattern_score', 0),
            'trendlines': len(pattern.get('all_trendlines', [])),
            'breakouts': len(pattern.get('recent_breakouts', [])),
            'has_double_bottom': pattern.get('has_double_bottom', False),
            'double_bottom_confirmed': pattern.get('double_bottom_confirmed', False),
            'double_bottom_potential': pattern.get('double_bottom_potential', False),
            'double_bottom': pattern.get('double_bottom'),
            'macd_positive': m.get('macd_positive', False),
            'macd_uptick': m.get('macd_uptick', False),
            'tf_data': tf_data, 'pattern': pattern
        })
    
    for p in potential:
        pattern = p.get('pattern', {})
        targets = pattern.get('targets', {})
        
        opps.append({
            'symbol': p['symbol'], 'tf': p['timeframe'], 'type': 'potential',
            'distance': p.get('distance', 0),
            'return_score': pattern.get('return_score', 0),
            'max_upside': targets.get('max_upside_pct', 0),
            'risk_reward': targets.get('risk_reward', 0),
            'entry': targets.get('entry', 0),
            'target1': targets['targets'][0]['price'] if targets.get('targets') else 0,
            'target1_pct': targets['targets'][0]['pct'] if targets.get('targets') else 0,
            'stop': targets.get('stop_loss', 0),
            'risk_pct': targets.get('risk_pct', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'signal_strength': pattern.get('signal_strength', 0),
            'volume_confirmed': pattern.get('volume_confirmed', False),
            'buy_volume_strong': pattern.get('buy_volume_strong', False),
            'weekly_buy_vol_strong': pattern.get('weekly_buy_vol_strong', False),
            'weekly_buy_vol_ratio': pattern.get('weekly_buy_vol_ratio', 1.0),
            'volume_weight': pattern.get('volume_weight', 1.0),
            'pattern_score': pattern.get('pattern_score', 0),
            'trendlines': len(pattern.get('all_trendlines', [])),
            'breakouts': 0,
            'has_double_bottom': pattern.get('has_double_bottom', False),
            'double_bottom_confirmed': pattern.get('double_bottom_confirmed', False),
            'double_bottom_potential': pattern.get('double_bottom_potential', False),
            'double_bottom': pattern.get('double_bottom'),
            'tf_data': p.get('tf_data', {}), 'pattern': pattern
        })
    
    if not opps:
        st.info("🔍 Run a scan to see flash cards for top opportunities")
        return
    
    # Sort by return score
    opps.sort(key=lambda x: (x['return_score'], x['max_upside']), reverse=True)
    
    # Display top opportunities as flash cards
    for i, opp in enumerate(opps[:8]):  # Show top 8
        symbol = opp['symbol']
        tf = opp['tf']
        tf_data = opp['tf_data']
        pattern = opp['pattern']
        
        # Determine status
        if opp['type'] == 'confirmed':
            # Add MACD status
            if opp.get('macd_positive') and opp.get('macd_uptick'):
                macd_str = "🟢↑"
            elif opp.get('macd_positive'):
                macd_str = "🟢"
            elif opp.get('macd_uptick'):
                macd_str = "📈"
            else:
                macd_str = ""
            status_text = f"✅ BUY {macd_str}"
            status_color = "green"
        else:
            status_text = f"⏳ {opp.get('distance', 0):.1f}% AWAY"
            status_color = "orange"
        
        yoda_color = "green" if opp['yoda_state'] == 'BUY' else "red"
        
        # Card container
        with st.container():
            st.markdown("---")
            
            # Header row
            header_col1, header_col2, header_col3 = st.columns([3, 2, 1])
            
            with header_col1:
                st.markdown(f"### {symbol} `{tf.upper()}`")
                st.markdown(f":{status_color}[{status_text}] · :{yoda_color}[{opp['yoda_state']}]")
            
            with header_col3:
                st.metric("Score", f"{opp['return_score']:.0f}")
            
            # Main content: Chart + Metrics
            chart_col, metrics_col = st.columns([3, 2])
            
            with chart_col:
                df = tf_data.get('df')
                yoda_df = tf_data.get('yoda_df')
                
                if df is not None and yoda_df is not None:
                    df_chart = df.tail(60).copy()
                    yoda_chart = yoda_df.tail(60).copy()
                    fig = create_tradingview_chart(df_chart, yoda_chart, pattern, symbol, tf, height=300)
                    st.plotly_chart(fig, use_container_width=True, key=f"flash_{symbol}_{tf}_{i}")
            
            with metrics_col:
                # Trade Setup
                st.markdown("**📊 Trade Setup**")
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Entry", f"${opp['entry']:.2f}")
                    st.metric("Stop Loss", f"${opp['stop']:.2f}", f"-{opp['risk_pct']:.1f}%", delta_color="inverse")
                
                with m2:
                    st.metric("Target", f"${opp['target1']:.2f}", f"+{opp['target1_pct']:.1f}%")
                    rr = opp['risk_reward']
                    st.metric("R/R Ratio", f"{rr:.1f}:1" if rr else "N/A")
                
                st.markdown("---")
                
                # Performance metrics
                perf1, perf2 = st.columns(2)
                with perf1:
                    st.markdown(f"**:green[+{opp['max_upside']:.1f}%]**")
                    st.caption("MAX UPSIDE")
                with perf2:
                    st.markdown(f"**:red[-{opp['risk_pct']:.1f}%]**")
                    st.caption("RISK")
                
                st.markdown("---")
                
                # Analysis summary
                st.markdown("**📝 Analysis**")
                
                summary_parts = []
                if opp['type'] == 'confirmed':
                    summary_parts.append(f"**{symbol}** has broken above resistance with {opp['breakouts']} recent breakout(s).")
                else:
                    summary_parts.append(f"**{symbol}** is approaching resistance, {opp.get('distance', 0):.1f}% away from breakout.")
                
                # Double bottom pattern
                if opp.get('has_double_bottom'):
                    if opp.get('double_bottom_confirmed'):
                        db = opp.get('double_bottom', {})
                        summary_parts.append(f"📊 **Double Bottom CONFIRMED** with target +{db.get('target_pct', 0):.1f}%!")
                    elif opp.get('double_bottom_potential'):
                        summary_parts.append("📊 **Double Bottom forming** - watch for neckline breakout!")
                    else:
                        summary_parts.append("📊 Double Bottom pattern detected.")
                
                if opp['yoda_state'] == 'BUY':
                    summary_parts.append(f"Yoda shows **BUY** signal (strength: {opp['signal_strength']}).")
                
                # Volume analysis
                if opp.get('weekly_buy_vol_strong'):
                    vol_ratio = opp.get('weekly_buy_vol_ratio', 1.0)
                    summary_parts.append(f"📊 **Weekly buy volume {vol_ratio:.1f}x above average** - strong accumulation!")
                elif opp.get('buy_volume_strong'):
                    summary_parts.append("Buy volume **above average**.")
                elif opp.get('volume_confirmed'):
                    summary_parts.append("Volume confirmed.")
                
                if opp.get('volume_weight', 1.0) > 1.2:
                    summary_parts.append(f"Volume weight: **{opp['volume_weight']:.2f}x**.")
                
                if opp['trendlines'] > 0:
                    summary_parts.append(f"{opp['trendlines']} trendline(s) detected.")
                
                if rr and rr > 2:
                    summary_parts.append(f"R/R of **{rr:.1f}:1** is favorable.")
                
                st.write(" ".join(summary_parts))



    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} ({tf})', 'MACD', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Bollinger
    if yoda_df is not None and 'BB_Upper' in yoda_df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['BB_Upper'], mode='lines',
                                line=dict(color='rgba(100,100,255,0.3)'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['BB_Lower'], mode='lines',
                                line=dict(color='rgba(100,100,255,0.3)'),
                                fill='tonexty', fillcolor='rgba(100,100,255,0.1)', showlegend=False), row=1, col=1)
    
    # SMA
    if yoda_df is not None and 'SMA' in yoda_df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['SMA'], mode='lines',
                                name='SMA 50', line=dict(color='orange', width=2)), row=1, col=1)
    
    # Trendlines
    colors = ['#00ff00', '#00bfff', '#ff00ff', '#ffaa00', '#00ffff']
    for i, tl in enumerate(pattern.get('all_trendlines', [])[:3]):
        x1, x2 = tl['start_index'], len(df) - 1
        y1 = tl['slope'] * x1 + tl['intercept']
        y2 = tl['slope'] * x2 + tl['intercept']
        fig.add_trace(go.Scatter(
            x=[df.index[x1], df.index[x2]], y=[y1, y2], mode='lines',
            name=f"TL{i+1} ({tl['touches']}T)",
            line=dict(color=colors[i % len(colors)], width=2, dash='dash')
        ), row=1, col=1)
    
    # Breakouts
    for bo in pattern.get('all_breakouts', []):
        if bo['breakout_index'] < len(df):
            fig.add_trace(go.Scatter(
                x=[df.index[bo['breakout_index']]], y=[bo['breakout_price']],
                mode='markers+text', text=[f"+{bo['breakout_pct']:.1f}%"], textposition='top center',
                marker=dict(symbol='diamond' if bo['is_recent'] else 'star', size=14,
                           color='lime' if bo['is_recent'] else 'gold'),
                showlegend=False
            ), row=1, col=1)
    
    # Targets
    targets = pattern.get('targets', {})
    for t in targets.get('targets', [])[:3]:
        fig.add_hline(y=t['price'], line_dash="dot", line_color="#4CAF50", line_width=1,
                     annotation_text=f"T{t['level']}: ${t['price']:.2f} (+{t['pct']:.1f}%)",
                     annotation_position="right", row=1, col=1)
    
    if targets.get('stop_loss'):
        fig.add_hline(y=targets['stop_loss'], line_dash="dash", line_color="#f44336", line_width=1,
                     annotation_text=f"Stop: ${targets['stop_loss']:.2f}",
                     annotation_position="right", row=1, col=1)
    
    # Potential breakout
    pot = pattern.get('potential_breakout', {})
    if pot.get('is_potential'):
        fig.add_hline(y=pot['trendline_price'], line_dash="solid", line_color="#ff5722", line_width=2,
                     annotation_text=f"🎯 Breakout: ${pot['trendline_price']:.2f}",
                     annotation_position="right", row=1, col=1)
    
    # Signals
    if yoda_df is not None:
        buys = yoda_df[yoda_df['YODA_BUY_SELL'] == 'BUY']
        sells = yoda_df[yoda_df['YODA_BUY_SELL'] == 'SELL']
        
        if len(buys) > 0:
            fig.add_trace(go.Scatter(
                x=df.index[buys.index], y=df['low'].iloc[buys.index] * 0.98,
                mode='markers', name='BUY',
                marker=dict(symbol='triangle-up', size=12, color='#4CAF50')
            ), row=1, col=1)
        
        if len(sells) > 0:
            fig.add_trace(go.Scatter(
                x=df.index[sells.index], y=df['high'].iloc[sells.index] * 1.02,
                mode='markers', name='SELL',
                marker=dict(symbol='triangle-down', size=12, color='#f44336')
            ), row=1, col=1)
        
        # MACD
        if 'MACD_Line' in yoda_df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=yoda_df['MACD_Line'], mode='lines',
                                    name='MACD', line=dict(color='#2196F3', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=yoda_df['MACD_Signal'], mode='lines',
                                    name='Signal', line=dict(color='#ff9800', width=1)), row=2, col=1)
            colors = ['#26a69a' if h >= 0 else '#ef5350' for h in yoda_df['MACD_Hist']]
            fig.add_trace(go.Bar(x=df.index, y=yoda_df['MACD_Hist'],
                                marker_color=colors, showlegend=False), row=2, col=1)
    
    # Volume
    vol_colors = ['#26a69a' if c > o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=vol_colors, showlegend=False), row=3, col=1)
    
    if yoda_df is not None and 'Volume_Avg' in yoda_df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['Volume_Avg'], mode='lines',
                                name='Vol Avg', line=dict(color='orange', width=2)), row=3, col=1)
    
    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right'),
        hovermode='x unified', template='plotly_dark',
        margin=dict(l=60, r=60, t=80, b=40)
    )
    
    return fig


def create_chart(df, yoda_df, pattern, symbol, tf):
    """Create standard chart for detail view with double bottom pattern."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} ({tf})', 'MACD', 'Volume')
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Price',
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # Bollinger
    if yoda_df is not None and 'BB_Upper' in yoda_df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['BB_Upper'], mode='lines',
                                line=dict(color='rgba(100,100,255,0.3)'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['BB_Lower'], mode='lines',
                                line=dict(color='rgba(100,100,255,0.3)'),
                                fill='tonexty', fillcolor='rgba(100,100,255,0.1)', showlegend=False), row=1, col=1)
    
    # SMA
    if yoda_df is not None and 'SMA' in yoda_df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['SMA'], mode='lines',
                                name='SMA 50', line=dict(color='orange', width=2)), row=1, col=1)
    
    # Double Bottom Pattern
    db = pattern.get('double_bottom')
    if db and pattern.get('has_double_bottom'):
        df_reset = df.reset_index()
        date_col = df_reset.columns[0]
        chart_len = len(df_reset)
        
        b1_idx = min(db['bottom1']['index'], chart_len - 1)
        b2_idx = min(db['bottom2']['index'], chart_len - 1)
        neckline_idx = min(db['neckline_index'], chart_len - 1)
        
        if b1_idx < chart_len and b2_idx < chart_len:
            # Bottom markers
            fig.add_trace(go.Scatter(
                x=[df.index[b1_idx]], y=[db['bottom1']['price']],
                mode='markers+text', text=['B1'], textposition='bottom center',
                marker=dict(symbol='triangle-up', size=16, color='#00e676'),
                name='Bottom 1'
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=[df.index[b2_idx]], y=[db['bottom2']['price']],
                mode='markers+text', text=['B2'], textposition='bottom center',
                marker=dict(symbol='triangle-up', size=16, color='#00e676'),
                name='Bottom 2'
            ), row=1, col=1)
            
            # Neckline
            fig.add_hline(
                y=db['neckline_price'], line_dash="dash", line_color="#ff9800", line_width=2,
                annotation_text=f"Neckline: ${db['neckline_price']:.2f}",
                annotation_position="right", row=1, col=1
            )
            
            # W pattern line
            if neckline_idx < chart_len:
                fig.add_trace(go.Scatter(
                    x=[df.index[b1_idx], df.index[neckline_idx], df.index[b2_idx]],
                    y=[db['bottom1']['price'], db['neckline_price'], db['bottom2']['price']],
                    mode='lines', line=dict(color='#00e676', width=3, dash='dot'),
                    name='Double Bottom'
                ), row=1, col=1)
            
            # Target and breakout
            if db['breakout_confirmed']:
                fig.add_hline(
                    y=db['target_price'], line_dash="dot", line_color="#00e676", line_width=2,
                    annotation_text=f"DB Target: ${db['target_price']:.2f} (+{db['target_pct']:.1f}%)",
                    annotation_position="right", row=1, col=1
                )
                
                if db['breakout_index'] and db['breakout_index'] < chart_len:
                    fig.add_trace(go.Scatter(
                        x=[df.index[db['breakout_index']]], y=[db['breakout_price']],
                        mode='markers+text', text=['✓ BREAKOUT'], textposition='top center',
                        marker=dict(symbol='star', size=18, color='#ffd700'),
                        name='DB Breakout'
                    ), row=1, col=1)
    
    # Trendlines
    colors = ['#00ff00', '#00bfff', '#ff00ff', '#ffaa00', '#00ffff']
    for i, tl in enumerate(pattern.get('all_trendlines', [])[:3]):
        x1, x2 = tl['start_index'], len(df) - 1
        if x1 < len(df) and x2 < len(df):
            y1 = tl['slope'] * x1 + tl['intercept']
            y2 = tl['slope'] * x2 + tl['intercept']
            fig.add_trace(go.Scatter(
                x=[df.index[x1], df.index[x2]], y=[y1, y2], mode='lines',
                name=f"TL{i+1} ({tl['touches']}T)",
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ), row=1, col=1)
    
    # Breakouts
    for bo in pattern.get('all_breakouts', []):
        if bo['breakout_index'] < len(df):
            fig.add_trace(go.Scatter(
                x=[df.index[bo['breakout_index']]], y=[bo['breakout_price']],
                mode='markers+text', text=[f"+{bo['breakout_pct']:.1f}%"], textposition='top center',
                marker=dict(symbol='diamond' if bo['is_recent'] else 'star', size=14,
                           color='lime' if bo['is_recent'] else 'gold'),
                showlegend=False
            ), row=1, col=1)
    
    # Targets
    targets = pattern.get('targets', {})
    for t in targets.get('targets', [])[:3]:
        fig.add_hline(y=t['price'], line_dash="dot", line_color="#4CAF50", line_width=1,
                     annotation_text=f"T{t['level']}: ${t['price']:.2f} (+{t['pct']:.1f}%)",
                     annotation_position="right", row=1, col=1)
    
    if targets.get('stop_loss'):
        fig.add_hline(y=targets['stop_loss'], line_dash="dash", line_color="#f44336", line_width=1,
                     annotation_text=f"Stop: ${targets['stop_loss']:.2f}",
                     annotation_position="right", row=1, col=1)
    
    # Potential breakout
    pot = pattern.get('potential_breakout', {})
    if pot.get('is_potential'):
        fig.add_hline(y=pot['trendline_price'], line_dash="solid", line_color="#ff5722", line_width=2,
                     annotation_text=f"🎯 Breakout: ${pot['trendline_price']:.2f}",
                     annotation_position="right", row=1, col=1)
    
    # Signals
    if yoda_df is not None:
        buys = yoda_df[yoda_df['YODA_BUY_SELL'] == 'BUY']
        sells = yoda_df[yoda_df['YODA_BUY_SELL'] == 'SELL']
        
        if len(buys) > 0:
            fig.add_trace(go.Scatter(
                x=df.index[buys.index], y=df['low'].iloc[buys.index] * 0.98,
                mode='markers', name='BUY',
                marker=dict(symbol='triangle-up', size=12, color='#4CAF50')
            ), row=1, col=1)
        
        if len(sells) > 0:
            fig.add_trace(go.Scatter(
                x=df.index[sells.index], y=df['high'].iloc[sells.index] * 1.02,
                mode='markers', name='SELL',
                marker=dict(symbol='triangle-down', size=12, color='#f44336')
            ), row=1, col=1)
        
        # MACD
        if 'MACD_Line' in yoda_df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=yoda_df['MACD_Line'], mode='lines',
                                    name='MACD', line=dict(color='#2196F3', width=1)), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=yoda_df['MACD_Signal'], mode='lines',
                                    name='Signal', line=dict(color='#ff9800', width=1)), row=2, col=1)
            colors = ['#26a69a' if h >= 0 else '#ef5350' for h in yoda_df['MACD_Hist']]
            fig.add_trace(go.Bar(x=df.index, y=yoda_df['MACD_Hist'],
                                marker_color=colors, showlegend=False), row=2, col=1)
    
    # Volume
    vol_colors = ['#26a69a' if c > o else '#ef5350' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], marker_color=vol_colors, showlegend=False), row=3, col=1)
    
    if yoda_df is not None and 'Volume_Avg' in yoda_df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=yoda_df['Volume_Avg'], mode='lines',
                                name='Vol Avg', line=dict(color='orange', width=2)), row=3, col=1)
    
    fig.update_layout(
        height=700, xaxis_rangeslider_visible=False, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=1, xanchor='right'),
        hovermode='x unified', template='plotly_dark',
        margin=dict(l=60, r=60, t=80, b=40)
    )
    
    return fig


# ==================== DASHBOARD ====================

def display_dashboard():
    all_results = st.session_state.get('all_results', [])
    matches = st.session_state.get('matches', [])
    potential = st.session_state.get('potential_breakouts', [])
    
    # Header
    st.markdown('<h1 style="text-align: center; margin-bottom: 30px;">📈 Yoda Pattern Scanner</h1>', unsafe_allow_html=True)
    
    # Summary cards row
    excluded_sell = st.session_state.get('excluded_sell', 0)
    total_scanned = st.session_state.get('total_scanned', len(all_results))
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(summary_card("SCANNED", total_scanned, f"{excluded_sell} SELL excluded"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(summary_card("PASSED", len(all_results), "No SELL signals", "summary-card-green"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(summary_card("CONFIRMED", len(matches), "BUY + MACD+", "summary-card-green"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(summary_card("POTENTIAL", len(potential), "Near breakout", "summary-card-orange"), unsafe_allow_html=True)
    
    with col5:
        best_upside = max((m.get('max_upside', 0) for m in matches), default=0) if matches else 0
        st.markdown(summary_card("BEST UPSIDE", f"+{best_upside:.1f}%", "Maximum potential", "summary-card-purple"), unsafe_allow_html=True)
    
    with col6:
        # Count stocks with strong weekly buy volume
        strong_vol_count = sum(
            1 for m in matches 
            if m.get('timeframes', {}).get(m.get('best_timeframe', ''), {}).get('pattern', {}).get('weekly_buy_vol_strong', False)
        )
        st.markdown(summary_card("STRONG VOL", strong_vol_count, "Weekly buy vol ↑", "summary-card-green"), unsafe_allow_html=True)
    
    # Second row - Target Status Summary
    if matches:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Calculate target stats
        with_targets_pending = 0
        all_targets_met = 0
        total_remaining_upside = 0
        
        for m in matches:
            tf = m.get('best_timeframe')
            if not tf:
                continue
            tf_data = m.get('timeframes', {}).get(tf, {})
            pattern = tf_data.get('pattern', {})
            targets = pattern.get('targets', {})
            current_price = tf_data.get('current_price', targets.get('entry', 0))
            target_list = targets.get('targets', [])
            
            targets_pending = sum(1 for t in target_list if current_price < t['price'])
            if targets_pending > 0:
                with_targets_pending += 1
                for t in target_list:
                    if current_price < t['price']:
                        remaining = ((t['price'] - current_price) / current_price) * 100
                        total_remaining_upside = max(total_remaining_upside, remaining)
            elif len(target_list) > 0:
                all_targets_met += 1
        
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        
        with col_t1:
            st.markdown(summary_card("🎯 TARGETS PENDING", with_targets_pending, "Still have upside", "summary-card-green"), unsafe_allow_html=True)
        
        with col_t2:
            st.markdown(summary_card("✅ TARGETS MET", all_targets_met, "Completed", "summary-card-purple"), unsafe_allow_html=True)
        
        with col_t3:
            avg_remaining = total_remaining_upside / with_targets_pending if with_targets_pending > 0 else 0
            st.markdown(summary_card("📈 MAX REMAINING", f"+{total_remaining_upside:.1f}%", "Best opportunity", "summary-card-orange"), unsafe_allow_html=True)
        
        with col_t4:
            # Active trades ratio
            active_pct = (with_targets_pending / len(matches) * 100) if matches else 0
            st.markdown(summary_card("📊 ACTIVE", f"{active_pct:.0f}%", f"{with_targets_pending} of {len(matches)}", "summary-card-red"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🎯 Creative Dashboard", "📱 Flash Cards", "🏆 Rankings", "🎯 Confirmed", "⏳ Potential", "📊 Full Analysis"])
    
    with tab1:
        display_creative_top(matches, potential, all_results)
    with tab2:
        display_flash_cards(matches, potential)
    with tab3:
        display_rankings(matches, potential)
    with tab4:
        display_confirmed(matches, all_results)
    with tab5:
        display_potential(potential, all_results)
    with tab6:
        display_all_data(all_results)


def display_creative_top(matches, potential, all_results):
    """Display creative top opportunities with sector grouping."""
    st.markdown('<div class="section-header">🎯 Top Opportunities (No SELL Signals in Daily/Weekly)</div>', unsafe_allow_html=True)
    
    # Combine all opportunities with sector info
    opps = []
    opp_id = 0  # Global unique ID for each opportunity
    
    # Build sector lookup from all_results
    sector_lookup = {}
    for r in all_results:
        sector_lookup[r['symbol']] = {
            'sector': r.get('sector', 'Unknown'),
            'industry': r.get('industry', 'Unknown'),
            'name': r.get('name', r['symbol']),
            'market_cap': r.get('market_cap', 0)
        }
    
    for m in matches:
        tf = m.get('best_timeframe')
        if not tf:
            continue
        tf_data = m.get('timeframes', {}).get(tf, {})
        pattern = tf_data.get('pattern', {})
        targets = pattern.get('targets', {})
        current_price = tf_data.get('current_price', targets.get('entry', 0))
        
        # Get sector info
        info = sector_lookup.get(m['symbol'], {'sector': 'Unknown', 'industry': 'Unknown', 'name': m['symbol']})
        
        # MACD status
        macd_positive = m.get('macd_positive', False)
        macd_uptick = m.get('macd_uptick', False)
        if macd_positive and macd_uptick:
            macd_str = "MACD+ ↑"
        elif macd_positive:
            macd_str = "MACD+"
        elif macd_uptick:
            macd_str = "MACD↑"
        else:
            macd_str = "MACD"
        
        # Target status
        target_list = targets.get('targets', [])
        targets_met = sum(1 for t in target_list if current_price >= t['price'])
        
        # Remaining upside
        remaining_upside = 0
        for t in target_list:
            if current_price < t['price']:
                remaining_pct = ((t['price'] - current_price) / current_price) * 100
                remaining_upside = max(remaining_upside, remaining_pct)
        
        opps.append({
            'id': opp_id,
            'symbol': m['symbol'],
            'name': info['name'],
            'sector': info['sector'],
            'industry': info['industry'],
            'tf': tf,
            'type': 'confirmed',
            'return_score': m.get('return_score', 0),
            'upside': remaining_upside if remaining_upside > 0 else targets.get('max_upside_pct', 0),
            'risk_reward': targets.get('risk_reward', 0),
            'entry': current_price,
            'target': targets['targets'][0]['price'] if targets.get('targets') else current_price,
            'stop': targets.get('stop_loss', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'macd_str': macd_str,
            'targets_met': targets_met,
            'total_targets': len(target_list),
            'volume_strong': pattern.get('weekly_buy_vol_strong', False),
            'double_bottom': pattern.get('has_double_bottom', False),
            'tf_data': tf_data,
            'pattern': pattern
        })
        opp_id += 1
    
    for p in potential:
        pattern = p.get('pattern', {})
        targets = pattern.get('targets', {})
        info = sector_lookup.get(p['symbol'], {'sector': 'Unknown', 'industry': 'Unknown', 'name': p['symbol']})
        
        macd_str = "MACD↑" if p.get('macd_uptick') else ("MACD+" if p.get('macd_positive') else "MACD")
        
        opps.append({
            'id': opp_id,
            'symbol': p['symbol'],
            'name': info['name'],
            'sector': info['sector'],
            'industry': info['industry'],
            'tf': p['timeframe'],
            'type': 'potential',
            'distance': p.get('distance', 0),
            'return_score': pattern.get('return_score', 0),
            'upside': targets.get('max_upside_pct', 0),
            'risk_reward': targets.get('risk_reward', 0),
            'entry': targets.get('entry', 0),
            'target': targets['targets'][0]['price'] if targets.get('targets') else 0,
            'stop': targets.get('stop_loss', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'macd_str': macd_str,
            'targets_met': 0,
            'total_targets': len(targets.get('targets', [])),
            'volume_strong': pattern.get('weekly_buy_vol_strong', False),
            'double_bottom': pattern.get('has_double_bottom', False),
            'tf_data': p.get('tf_data', {}),
            'pattern': pattern
        })
        opp_id += 1
    
    if not opps:
        st.info("🔍 Run a scan to discover opportunities")
        return
    
    # Sort by score
    opps.sort(key=lambda x: (x['return_score'], x['upside']), reverse=True)
    
    # View selector
    view_mode = st.radio("View Mode", ["🏆 Top 10", "📊 By Sector", "🔥 Hot Picks"], horizontal=True, key="creative_view")
    
    st.markdown("---")
    
    if view_mode == "🏆 Top 10":
        # Display top 10 with creative cards
        st.markdown("### 🏆 Top 10 Opportunities")
        
        for rank, opp in enumerate(opps[:10], 1):
            col1, col2 = st.columns([5, 1])
            
            with col1:
                st.markdown(
                    opportunity_card_html(
                        symbol=opp['symbol'],
                        name=opp['name'],
                        sector=opp['sector'],
                        industry=opp['industry'],
                        score=opp['return_score'],
                        upside=opp['upside'],
                        rr=opp['risk_reward'] if opp['risk_reward'] else 0,
                        entry=opp['entry'],
                        target=opp['target'],
                        stop=opp['stop'] if opp['stop'] else opp['entry'] * 0.95,
                        yoda_state=opp['yoda_state'],
                        macd_status=opp['macd_str'],
                        targets_met=opp['targets_met'],
                        total_targets=opp['total_targets'],
                        card_type=opp['type'],
                        volume_strong=opp['volume_strong'],
                        double_bottom=opp['double_bottom']
                    ),
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(f"<br><br>", unsafe_allow_html=True)
                st.markdown(f'<span class="badge badge-rank">#{rank}</span>', unsafe_allow_html=True)
                if st.button("📈", key=f"top_{opp['id']}"):
                    st.session_state['selected'] = {
                        'symbol': opp['symbol'], 'tf': opp['tf'],
                        'tf_data': opp['tf_data'], 'pattern': opp['pattern']
                    }
                    st.session_state['view'] = 'detail'
                    st.rerun()
    
    elif view_mode == "📊 By Sector":
        # Group by sector
        sectors = {}
        for opp in opps:
            sector = opp['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(opp)
        
        # Sort sectors by total opportunity count
        sorted_sectors = sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Sector summary
        st.markdown("### 📊 Sector Overview")
        cols = st.columns(min(4, len(sorted_sectors)))
        for i, (sector, sector_opps) in enumerate(sorted_sectors[:4]):
            with cols[i]:
                avg_upside = np.mean([o['upside'] for o in sector_opps])
                confirmed = sum(1 for o in sector_opps if o['type'] == 'confirmed')
                sector_class = get_sector_class(sector)
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 12px; text-align: center;">
                    <span class="sector-badge {sector_class}">{sector[:12]}</span>
                    <div style="font-size: 24px; font-weight: 700; margin: 10px 0;">{len(sector_opps)}</div>
                    <div style="font-size: 12px; color: #666;">
                        {confirmed} confirmed | +{avg_upside:.1f}% avg
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display each sector group
        for sector, sector_opps in sorted_sectors:
            if len(sector_opps) == 0:
                continue
            
            avg_upside = np.mean([o['upside'] for o in sector_opps])
            
            with st.expander(f"**{sector}** ({len(sector_opps)} stocks) — Avg +{avg_upside:.1f}% upside", expanded=len(sector_opps) <= 5):
                st.markdown(sector_group_header_html(sector, len(sector_opps), avg_upside), unsafe_allow_html=True)
                
                # Sort within sector by score
                sector_opps.sort(key=lambda x: x['return_score'], reverse=True)
                
                for idx, opp in enumerate(sector_opps[:5]):  # Show top 5 per sector
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        # Compact card for sector view
                        type_indicator = "✅" if opp['type'] == 'confirmed' else f"⏳ {opp.get('distance', 0):.1f}%"
                        vol_indicator = "🔥" if opp['volume_strong'] else ""
                        db_indicator = "W" if opp['double_bottom'] else ""
                        
                        st.markdown(f"""
                        <div style="background: #fff; padding: 12px 15px; border-radius: 10px; margin-bottom: 8px; border-left: 4px solid {'#4CAF50' if opp['type'] == 'confirmed' else '#FF9800'};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 18px; font-weight: 700;">{opp['symbol']}</span>
                                    <span style="font-size: 12px; color: #888; margin-left: 8px;">{opp['tf'].upper()}</span>
                                    <span style="font-size: 11px; margin-left: 8px;">{vol_indicator}{db_indicator}</span>
                                </div>
                                <div style="text-align: right;">
                                    <span style="font-size: 16px; font-weight: 600; color: #4CAF50;">+{opp['upside']:.1f}%</span>
                                    <span style="font-size: 12px; color: #888; margin-left: 10px;">{type_indicator}</span>
                                </div>
                            </div>
                            <div style="font-size: 11px; color: #666; margin-top: 5px;">
                                {opp['industry'][:40]} | Score: {opp['return_score']:.0f} | R/R: {opp['risk_reward']:.1f}:1
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if st.button("📈", key=f"sec_{opp['id']}"):
                            st.session_state['selected'] = {
                                'symbol': opp['symbol'], 'tf': opp['tf'],
                                'tf_data': opp['tf_data'], 'pattern': opp['pattern']
                            }
                            st.session_state['view'] = 'detail'
                            st.rerun()
    
    else:  # Hot Picks
        # Filter for hot picks: high score + high upside + volume strong
        hot_picks = [o for o in opps if o['upside'] >= 15 and o['return_score'] >= 60]
        hot_picks.sort(key=lambda x: (x['volume_strong'], x['upside'], x['return_score']), reverse=True)
        
        st.markdown("### 🔥 Hot Picks (15%+ upside, 60+ score)")
        
        if not hot_picks:
            st.info("No hot picks found. Try lowering the threshold.")
        else:
            # Display as a heat map style grid
            cols_per_row = 3
            for i in range(0, len(hot_picks[:9]), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(hot_picks):
                        opp = hot_picks[i + j]
                        grid_idx = i + j
                        with col:
                            heat_class = 'heat-hot' if opp['upside'] >= 25 else ('heat-warm' if opp['upside'] >= 20 else 'heat-neutral')
                            st.markdown(f"""
                            <div style="background: linear-gradient(145deg, #fff 0%, #f8f9fa 100%); padding: 20px; border-radius: 16px; text-align: center; border: 2px solid {'#f44336' if opp['upside'] >= 25 else '#FF9800'}; margin-bottom: 15px;">
                                <div style="font-size: 11px; color: #888; text-transform: uppercase;">{opp['sector'][:15]}</div>
                                <div style="font-size: 28px; font-weight: 800; margin: 10px 0;">{opp['symbol']}</div>
                                <div style="font-size: 12px; color: #666;">{opp['name'][:25]}</div>
                                <div style="margin: 15px 0;">
                                    <span class="heat-indicator {heat_class}">+{opp['upside']:.1f}% upside</span>
                                </div>
                                <div style="display: flex; justify-content: space-around; margin-top: 15px;">
                                    <div>
                                        <div style="font-size: 18px; font-weight: 700;">{opp['return_score']:.0f}</div>
                                        <div style="font-size: 10px; color: #888;">SCORE</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 18px; font-weight: 700;">{opp['risk_reward']:.1f}:1</div>
                                        <div style="font-size: 10px; color: #888;">R/R</div>
                                    </div>
                                    <div>
                                        <div style="font-size: 18px; font-weight: 700;">{'🔥' if opp['volume_strong'] else '📊'}</div>
                                        <div style="font-size: 10px; color: #888;">VOL</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if st.button("View Chart", key=f"hot_{opp['id']}"):
                                st.session_state['selected'] = {
                                    'symbol': opp['symbol'], 'tf': opp['tf'],
                                    'tf_data': opp['tf_data'], 'pattern': opp['pattern']
                                }
                                st.session_state['view'] = 'detail'
                                st.rerun()


def display_rankings(matches, potential):
    st.markdown('<div class="section-header">🏆 Top Opportunities (No SELL Signals)</div>', unsafe_allow_html=True)
    
    # Combine opportunities
    opps = []
    
    for m in matches:
        tf = m.get('best_timeframe')
        if not tf:
            continue
        tf_data = m.get('timeframes', {}).get(tf, {})
        pattern = tf_data.get('pattern', {})
        targets = pattern.get('targets', {})
        current_price = tf_data.get('current_price', targets.get('entry', 0))
        yoda_df = tf_data.get('yoda_df')
        
        # Calculate MACD status
        macd_positive = m.get('macd_positive', False)
        macd_uptick = m.get('macd_uptick', False)
        
        # MACD indicator for display
        if macd_positive and macd_uptick:
            macd_str = "🟢↑"
        elif macd_positive:
            macd_str = "🟢"
        elif macd_uptick:
            macd_str = "📈"
        else:
            macd_str = ""
        
        # Calculate target status
        target_list = targets.get('targets', [])
        targets_met = sum(1 for t in target_list if current_price >= t['price'])
        targets_pending = len(target_list) - targets_met
        
        # Calculate remaining upside
        remaining_upside = 0
        for t in target_list:
            if current_price < t['price']:
                remaining_pct = ((t['price'] - current_price) / current_price) * 100
                remaining_upside = max(remaining_upside, remaining_pct)
        
        opps.append({
            'symbol': m['symbol'], 'tf': tf, 'type': f'✅ BUY {macd_str}',
            'return_score': m.get('return_score', 0),
            'max_upside': targets.get('max_upside_pct', 0),
            'remaining_upside': remaining_upside,
            'risk_reward': targets.get('risk_reward', 0),
            'entry': targets.get('entry', 0),
            'current_price': current_price,
            'target1': targets['targets'][0]['price'] if targets.get('targets') else 0,
            'stop': targets.get('stop_loss', 0),
            'risk_pct': targets.get('risk_pct', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'targets_met': targets_met,
            'targets_pending': targets_pending,
            'total_targets': len(target_list),
            'macd_positive': macd_positive,
            'macd_uptick': macd_uptick,
            'tf_data': tf_data, 'pattern': pattern
        })
    
    for p in potential:
        pattern = p.get('pattern', {})
        targets = pattern.get('targets', {})
        
        opps.append({
            'symbol': p['symbol'], 'tf': p['timeframe'],
            'type': f"⏳ {p.get('distance', 0):.1f}% away",
            'return_score': pattern.get('return_score', 0),
            'max_upside': targets.get('max_upside_pct', 0),
            'remaining_upside': targets.get('max_upside_pct', 0),
            'risk_reward': targets.get('risk_reward', 0),
            'entry': targets.get('entry', 0),
            'current_price': targets.get('entry', 0),
            'target1': targets['targets'][0]['price'] if targets.get('targets') else 0,
            'stop': targets.get('stop_loss', 0),
            'risk_pct': targets.get('risk_pct', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'targets_met': 0,
            'targets_pending': len(targets.get('targets', [])),
            'total_targets': len(targets.get('targets', [])),
            'tf_data': p.get('tf_data', {}), 'pattern': pattern
        })
    
    if not opps:
        st.info("🔍 Run a scan to discover opportunities")
        return
    
    opps.sort(key=lambda x: (x['return_score'], x['remaining_upside']), reverse=True)
    
    # Display top opportunities as cards
    for rank, opp in enumerate(opps[:10], 1):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            # Add target status to card
            target_info = ""
            if opp['type'].startswith('✅'):
                if opp['targets_pending'] > 0:
                    target_info = f" | 🎯 {opp['targets_met']}/{opp['total_targets']} met"
                elif opp['targets_met'] > 0:
                    target_info = " | ✅ All targets met"
            
            st.markdown(
                stock_card_html(
                    opp['symbol'], opp['tf'], opp['type'] + target_info,
                    opp['return_score'], opp['remaining_upside'] if opp['targets_pending'] > 0 else opp['max_upside'],
                    opp['risk_reward'] if opp['risk_reward'] else 0,
                    opp['current_price'], opp['target1'] if opp['target1'] else opp['entry'],
                    opp['stop'] if opp['stop'] else opp['entry'] * 0.95,
                    opp['yoda_state']
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(f"<br>", unsafe_allow_html=True)
            if st.button(f"📈 Chart", key=f"rank_{opp['symbol']}_{opp['tf']}_{rank}"):
                st.session_state['selected'] = opp
                st.session_state['view'] = 'detail'
                st.rerun()
    
    # Summary table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Complete Rankings</div>', unsafe_allow_html=True)
    
    table_data = [{
        'Rank': i+1, 'Symbol': o['symbol'], 'TF': o['tf'], 'Status': o['type'],
        'Score': f"{o['return_score']:.0f}", 
        'Remaining': f"+{o['remaining_upside']:.1f}%" if o['targets_pending'] > 0 else "Done",
        'Targets': f"{o['targets_met']}/{o['total_targets']}" if o['total_targets'] > 0 else "-",
        'R/R': f"{o['risk_reward']:.1f}:1" if o['risk_reward'] else "-",
        'Price': f"${o['current_price']:.2f}" if o['current_price'] else "-",
        'Target': f"${o['target1']:.2f}" if o['target1'] else "-",
        'Stop': f"${o['stop']:.2f}" if o['stop'] else "-"
    } for i, o in enumerate(opps)]
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True, height=400)


def display_confirmed(matches, all_results=None):
    """Display confirmed breakouts grouped by upside buckets with target status and sector info."""
    st.markdown('<div class="section-header">🎯 Confirmed Breakouts (No SELL in Daily/Weekly)</div>', unsafe_allow_html=True)
    
    if not matches:
        st.info("No confirmed breakouts found. Stocks with **SELL signals** in daily or weekly timeframes are excluded.")
        return
    
    # Build sector lookup from all_results
    sector_lookup = {}
    if all_results:
        for r in all_results:
            sector_lookup[r['symbol']] = {
                'sector': r.get('sector', 'Unknown'),
                'industry': r.get('industry', 'Unknown'),
                'name': r.get('name', r['symbol'])
            }
    
    # View mode selector
    view_mode = st.radio("Group By", ["📊 Upside Buckets", "🏢 Sector"], horizontal=True, key="confirmed_view")
    
    st.markdown("---")
    
    # Process matches to calculate remaining upside and target status
    processed = []
    item_id = 0  # Unique ID for each item
    for m in matches:
        tf = m.get('best_timeframe')
        if not tf:
            continue
        
        tf_data = m.get('timeframes', {}).get(tf, {})
        pattern = tf_data.get('pattern', {})
        targets = pattern.get('targets', {})
        current_price = tf_data.get('current_price', targets.get('entry', 0))
        yoda_df = tf_data.get('yoda_df')
        
        # Get sector info
        info = sector_lookup.get(m['symbol'], {'sector': 'Unknown', 'industry': 'Unknown', 'name': m['symbol']})
        
        # Calculate MACD status
        macd_positive = False
        macd_uptick = False
        if yoda_df is not None and len(yoda_df) >= 2 and 'MACD_Hist' in yoda_df.columns:
            current_hist = yoda_df['MACD_Hist'].iloc[-1]
            prev_hist = yoda_df['MACD_Hist'].iloc[-2]
            macd_positive = current_hist > 0
            macd_uptick = current_hist > prev_hist
        
        # Calculate target status
        target_list = targets.get('targets', [])
        targets_met = 0
        targets_pending = 0
        next_target = None
        next_target_pct = 0
        max_remaining_upside = 0
        
        for t in target_list:
            if current_price >= t['price']:
                targets_met += 1
            else:
                targets_pending += 1
                remaining_pct = ((t['price'] - current_price) / current_price) * 100
                if next_target is None:
                    next_target = t
                    next_target_pct = remaining_pct
                max_remaining_upside = max(max_remaining_upside, remaining_pct)
        
        # Use remaining upside if targets pending, otherwise use original max_upside
        upside = max_remaining_upside if targets_pending > 0 else m.get('max_upside', 0)
        
        processed.append({
            'id': item_id,
            'symbol': m['symbol'],
            'name': info['name'],
            'sector': info['sector'],
            'industry': info['industry'],
            'tf': tf,
            'tf_data': tf_data,
            'pattern': pattern,
            'targets': targets,
            'current_price': current_price,
            'entry_price': targets.get('entry', current_price),
            'return_score': m.get('return_score', 0),
            'original_upside': m.get('max_upside', 0),
            'remaining_upside': max_remaining_upside,
            'upside': upside,
            'targets_met': targets_met,
            'targets_pending': targets_pending,
            'total_targets': len(target_list),
            'next_target': next_target,
            'next_target_pct': next_target_pct,
            'all_targets_met': targets_pending == 0 and targets_met > 0,
            'has_targets': len(target_list) > 0,
            'stop_loss': targets.get('stop_loss', 0),
            'risk_reward': targets.get('risk_reward', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'volume_confirmed': pattern.get('volume_confirmed', False),
            'weekly_buy_vol_strong': pattern.get('weekly_buy_vol_strong', False),
            'has_double_bottom': pattern.get('has_double_bottom', False),
            'double_bottom_confirmed': pattern.get('double_bottom_confirmed', False),
            'macd_positive': macd_positive,
            'macd_uptick': macd_uptick
        })
        item_id += 1
    
    # Define upside buckets (based on remaining upside)
    buckets = [
        {'name': '🚀 30%+ Remaining', 'min': 30, 'max': float('inf'), 'items': []},
        {'name': '📈 20-30% Remaining', 'min': 20, 'max': 30, 'items': []},
        {'name': '📊 10-20% Remaining', 'min': 10, 'max': 20, 'items': []},
        {'name': '📉 5-10% Remaining', 'min': 5, 'max': 10, 'items': []},
        {'name': '✅ <5% / Targets Met', 'min': -float('inf'), 'max': 5, 'items': []}
    ]
    
    # Sort into buckets
    for p in processed:
        upside = p['remaining_upside'] if p['targets_pending'] > 0 else 0
        for bucket in buckets:
            if bucket['min'] <= upside < bucket['max']:
                bucket['items'].append(p)
                break
    
    # Summary metrics
    st.markdown("### 📊 Summary")
    
    total_with_targets = sum(1 for p in processed if p['targets_pending'] > 0)
    total_met = sum(1 for p in processed if p['all_targets_met'])
    
    col_summary = st.columns(4)
    col_summary[0].metric("Total Confirmed", len(processed))
    col_summary[1].metric("Targets Pending", total_with_targets, "opportunities")
    col_summary[2].metric("All Targets Met", total_met, "completed")
    col_summary[3].metric("Avg Remaining", f"+{np.mean([p['remaining_upside'] for p in processed if p['targets_pending'] > 0]):.1f}%" if total_with_targets > 0 else "N/A")
    
    st.markdown("---")
    
    if view_mode == "📊 Upside Buckets":
        # Bucket distribution
        cols = st.columns(5)
        for i, bucket in enumerate(buckets):
            count = len(bucket['items'])
            with cols[i]:
                st.metric(bucket['name'].split(' ')[0], count)
                st.caption(bucket['name'].split(' ', 1)[1] if count > 0 else "none")
        
        st.markdown("---")
        
        # Display each bucket
        for bucket in buckets:
            if not bucket['items']:
                continue
            
            # Sort by remaining upside within bucket
            bucket['items'].sort(key=lambda x: (x['targets_pending'] > 0, x['remaining_upside']), reverse=True)
            
            st.markdown(f"### {bucket['name']} ({len(bucket['items'])} stocks)")
            
            for idx, p in enumerate(bucket['items']):
                _display_confirmed_item(p)
    
    else:  # Sector view
        # Group by sector
        sectors = {}
        for p in processed:
            sector = p['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(p)
        
        # Sort sectors by count
        sorted_sectors = sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Sector summary
        cols = st.columns(min(4, len(sorted_sectors)))
        for i, (sector, sector_items) in enumerate(sorted_sectors[:4]):
            with cols[i]:
                avg_upside = np.mean([p['remaining_upside'] for p in sector_items if p['targets_pending'] > 0]) if any(p['targets_pending'] > 0 for p in sector_items) else 0
                sector_class = get_sector_class(sector)
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 12px; border-radius: 10px; text-align: center;">
                    <span class="sector-badge {sector_class}" style="font-size: 10px;">{sector[:12]}</span>
                    <div style="font-size: 22px; font-weight: 700; margin: 8px 0;">{len(sector_items)}</div>
                    <div style="font-size: 11px; color: #666;">+{avg_upside:.1f}% avg</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display each sector
        for sector, sector_items in sorted_sectors:
            avg_upside = np.mean([p['remaining_upside'] for p in sector_items if p['targets_pending'] > 0]) if any(p['targets_pending'] > 0 for p in sector_items) else 0
            
            with st.expander(f"**{sector}** ({len(sector_items)} stocks) — Avg +{avg_upside:.1f}% remaining", expanded=len(sector_items) <= 3):
                sector_items.sort(key=lambda x: x['remaining_upside'], reverse=True)
                for idx, p in enumerate(sector_items):
                    _display_confirmed_item(p)


def _display_confirmed_item(p):
    """Helper function to display a single confirmed item."""
    # Status indicator
    if p['all_targets_met']:
        status = "✅ ALL TARGETS MET"
    elif p['targets_pending'] > 0:
        status = f"🎯 {p['targets_met']}/{p['total_targets']} targets met"
    else:
        status = "📊 No targets"
    
    # MACD status indicator
    macd_status = ""
    if p['macd_positive'] and p['macd_uptick']:
        macd_status = "🟢"
    elif p['macd_positive']:
        macd_status = "🟢"
    elif p['macd_uptick']:
        macd_status = "📈"
    
    # Volume/Pattern indicators
    indicators = [macd_status]
    if p['weekly_buy_vol_strong']:
        indicators.append("🔥")
    if p['double_bottom_confirmed']:
        indicators.append("W")
    indicator_str = " ".join(indicators)
    
    # Sector badge
    sector_class = get_sector_class(p.get('sector', 'Unknown'))
    sector_badge = f'<span class="sector-badge {sector_class}" style="font-size: 9px;">{p.get("sector", "")[:10]}</span>'
    
    # Expander title
    if p['targets_pending'] > 0 and p['next_target']:
        title = f"**{p['symbol']}** ({p['tf']}) — {status} | Next: +{p['next_target_pct']:.1f}% {indicator_str}"
    else:
        title = f"**{p['symbol']}** ({p['tf']}) — {status} | Score: {p['return_score']:.0f} {indicator_str}"
    
    with st.expander(title, expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Position**")
            st.markdown(f"{sector_badge}", unsafe_allow_html=True)
            st.write(f"Entry: ${p['entry_price']:.2f}")
            st.write(f"Current: ${p['current_price']:.2f}")
            pnl = ((p['current_price'] - p['entry_price']) / p['entry_price']) * 100 if p['entry_price'] > 0 else 0
            pnl_color = "green" if pnl >= 0 else "red"
            st.markdown(f"P&L: :{pnl_color}[{'+' if pnl >= 0 else ''}{pnl:.1f}%]")
        
        with col2:
            st.markdown("**Targets**")
            target_list = p['targets'].get('targets', [])
            for i, t in enumerate(target_list[:3]):
                if p['current_price'] >= t['price']:
                    st.markdown(f"~~T{t['level']}: ${t['price']:.2f}~~ ✅")
                else:
                    remaining = ((t['price'] - p['current_price']) / p['current_price']) * 100
                    st.write(f"T{t['level']}: ${t['price']:.2f} (+{remaining:.1f}%)")
        
        with col3:
            st.markdown("**Status**")
            st.write(f"Targets: {p['targets_met']}/{p['total_targets']}")
            st.write(f"Remaining: +{p['remaining_upside']:.1f}%")
            st.write(f"Yoda: {p['yoda_state']}")
            st.write(f"R/R: {p['risk_reward']:.1f}:1" if p['risk_reward'] else "R/R: N/A")
        
        with col4:
            st.markdown("**Signals**")
            st.write(f"Score: {p['return_score']:.0f}/100")
            if p['macd_positive'] and p['macd_uptick']:
                st.write("MACD: 🟢 Positive & ↑")
            elif p['macd_positive']:
                st.write("MACD: 🟢 Positive")
            elif p['macd_uptick']:
                st.write("MACD: 📈 Uptick")
            st.write(f"Volume: {'🔥 Strong' if p['weekly_buy_vol_strong'] else ('✅' if p['volume_confirmed'] else '❌')}")
            if p['has_double_bottom']:
                st.write(f"Dbl Btm: {'✅' if p['double_bottom_confirmed'] else '📊'}")
        
        if st.button(f"📈 View Chart", key=f"conf_{p['id']}"):
            st.session_state['selected'] = {
                'symbol': p['symbol'], 'tf': p['tf'],
                'tf_data': p['tf_data'], 'pattern': p['pattern']
            }
            st.session_state['view'] = 'detail'
            st.rerun()


def display_potential(potential, all_results=None):
    """Display potential breakouts grouped by upside buckets with MACD filter and sector info."""
    st.markdown('<div class="section-header">⏳ Potential Breakouts (No SELL in Daily/Weekly)</div>', unsafe_allow_html=True)
    
    if not potential:
        st.info("No potential setups found. Stocks with **SELL signals** in daily or weekly timeframes are excluded.")
        return
    
    # Build sector lookup from all_results
    sector_lookup = {}
    if all_results:
        for r in all_results:
            sector_lookup[r['symbol']] = {
                'sector': r.get('sector', 'Unknown'),
                'industry': r.get('industry', 'Unknown'),
                'name': r.get('name', r['symbol'])
            }
    
    # View mode selector
    view_mode = st.radio("Group By", ["📊 Upside Buckets", "🏢 Sector"], horizontal=True, key="potential_view")
    
    st.markdown("---")
    
    # Add sector info and unique ID to potential items
    for item_id, p in enumerate(potential):
        info = sector_lookup.get(p['symbol'], {'sector': 'Unknown', 'industry': 'Unknown', 'name': p['symbol']})
        p['id'] = item_id
        p['sector'] = info['sector']
        p['industry'] = info['industry']
        p['name'] = info['name']
        
        if 'upside' not in p or p.get('upside', 0) == 0:
            pattern = p.get('pattern', {})
            targets = pattern.get('targets', {})
            p['upside'] = targets.get('max_upside_pct', 0)
    
    # Define upside buckets
    buckets = [
        {'name': '🚀 30%+ Upside', 'min': 30, 'max': float('inf'), 'items': []},
        {'name': '📈 20-30% Upside', 'min': 20, 'max': 30, 'items': []},
        {'name': '📊 10-20% Upside', 'min': 10, 'max': 20, 'items': []},
        {'name': '📉 5-10% Upside', 'min': 5, 'max': 10, 'items': []},
        {'name': '⚠️ <5% Upside', 'min': 0, 'max': 5, 'items': []}
    ]
    
    # Sort potential into buckets
    for p in potential:
        upside = p.get('upside', 0)
        for bucket in buckets:
            if bucket['min'] <= upside < bucket['max']:
                bucket['items'].append(p)
                break
    
    # Display summary metrics
    st.markdown("### 📊 Summary")
    cols = st.columns(5)
    for i, bucket in enumerate(buckets):
        count = len(bucket['items'])
        label = bucket['name'].split(' ', 1)[1] if ' ' in bucket['name'] else bucket['name']
        cols[i].metric(bucket['name'].split(' ')[0], count, label if count > 0 else "none")
    
    st.markdown("---")
    
    if view_mode == "📊 Upside Buckets":
        # Display each bucket with items
        for bucket in buckets:
            if not bucket['items']:
                continue
            
            # Sort by distance within bucket (closest first)
            bucket['items'].sort(key=lambda x: x.get('distance', 100))
            
            st.markdown(f"### {bucket['name']} ({len(bucket['items'])} stocks)")
            
            for idx, p in enumerate(bucket['items']):
                _display_potential_item(p)
    
    else:  # Sector view
        # Group by sector
        sectors = {}
        for p in potential:
            sector = p.get('sector', 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(p)
        
        # Sort sectors by count
        sorted_sectors = sorted(sectors.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Sector summary
        cols = st.columns(min(4, len(sorted_sectors)))
        for i, (sector, sector_items) in enumerate(sorted_sectors[:4]):
            with cols[i]:
                avg_upside = np.mean([p.get('upside', 0) for p in sector_items])
                sector_class = get_sector_class(sector)
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 12px; border-radius: 10px; text-align: center;">
                    <span class="sector-badge {sector_class}" style="font-size: 10px;">{sector[:12]}</span>
                    <div style="font-size: 22px; font-weight: 700; margin: 8px 0;">{len(sector_items)}</div>
                    <div style="font-size: 11px; color: #666;">+{avg_upside:.1f}% avg</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display each sector
        for sector, sector_items in sorted_sectors:
            avg_upside = np.mean([p.get('upside', 0) for p in sector_items])
            
            with st.expander(f"**{sector}** ({len(sector_items)} stocks) — Avg +{avg_upside:.1f}% upside", expanded=len(sector_items) <= 3):
                sector_items.sort(key=lambda x: x.get('distance', 100))
                for idx, p in enumerate(sector_items):
                    _display_potential_item(p)


def _display_potential_item(p):
    """Helper function to display a single potential item."""
    pattern = p.get('pattern', {})
    targets = pattern.get('targets', {})
    dist = p.get('distance', 0)
    upside = p.get('upside', 0)
    
    # MACD indicators
    macd_status = ""
    if p.get('macd_positive') and p.get('macd_uptick'):
        macd_status = "🟢 MACD+ & ↑"
    elif p.get('macd_positive'):
        macd_status = "🟢 MACD+"
    elif p.get('macd_uptick'):
        macd_status = "📈 MACD↑"
    else:
        macd_status = "📊 MACD"
    
    # Pattern type indicator
    pattern_type = p.get('pattern_type', 'trendline')
    type_icon = "📊 DB" if pattern_type == 'double_bottom' else "📐 TL"
    
    # Proximity indicator
    prox = "🔥" if dist <= 2 else ("⚡" if dist <= 3 else "👀")
    
    # Sector badge
    sector_class = get_sector_class(p.get('sector', 'Unknown'))
    
    with st.expander(f"{prox} **{p['symbol']}** ({p['timeframe']}) — {dist:.1f}% away | +{upside:.1f}% upside | {macd_status}"):
        # Sector badge at top
        st.markdown(f'<span class="sector-badge {sector_class}">{p.get("sector", "Unknown")[:15]}</span> <span style="color: #888; font-size: 12px;">{p.get("industry", "")[:30]}</span>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Setup**")
            st.write(f"Distance: {dist:.1f}%")
            st.write(f"Pattern: {type_icon}")
            st.write(f"Yoda: {pattern.get('yoda_state', 'NA')}")
            st.write(f"Healthy: {'✅' if pattern.get('is_healthy_trend') else '❌'}")
        
        with col2:
            st.markdown("**MACD Status**")
            st.write(f"Positive: {'✅' if p.get('macd_positive') else '❌'}")
            st.write(f"Uptick: {'✅' if p.get('macd_uptick') else '❌'}")
            vol_strong = pattern.get('weekly_buy_vol_strong', False)
            st.write(f"Weekly Vol: {'🔥' if vol_strong else '📊'}")
        
        with col3:
            st.markdown("**Targets**")
            for t in targets.get('targets', [])[:2]:
                st.write(f"T{t['level']}: ${t['price']:.2f} (+{t['pct']:.1f}%)")
            st.write(f"Stop: ${targets.get('stop_loss', 0):.2f}")
        
        with col4:
            st.markdown("**Metrics**")
            st.write(f"Score: {pattern.get('return_score', 0):.0f}")
            rr = targets.get('risk_reward', 0)
            st.write(f"R/R: {rr:.1f}:1" if rr else "R/R: N/A")
            st.write(f"Price: ${targets.get('entry', 0):.2f}")
        
        if st.button(f"📈 View Chart", key=f"pot_{p['id']}"):
            st.session_state['selected'] = {
                'symbol': p['symbol'], 'tf': p['timeframe'],
                'tf_data': p.get('tf_data', {}), 'pattern': pattern
            }
            st.session_state['view'] = 'detail'
            st.rerun()


def display_all_data(all_results):
    st.markdown('<div class="section-header">📊 Complete Scan Results</div>', unsafe_allow_html=True)
    
    data = []
    for r in all_results:
        for tf, td in r.get('timeframes', {}).items():
            if 'error' in td:
                continue
            
            p = td.get('pattern', {})
            t = p.get('targets', {})
            current_price = td.get('current_price', t.get('entry', 0))
            
            # Calculate target status
            target_list = t.get('targets', [])
            targets_met = sum(1 for tgt in target_list if current_price >= tgt['price'])
            targets_pending = len(target_list) - targets_met
            
            # Calculate remaining upside
            remaining_upside = 0
            for tgt in target_list:
                if current_price < tgt['price']:
                    remaining_pct = ((tgt['price'] - current_price) / current_price) * 100
                    remaining_upside = max(remaining_upside, remaining_pct)
            
            status = "❌ No Signal"
            if p.get('has_trendline_breakout'):
                if targets_pending == 0 and targets_met > 0:
                    status = "✅ Done"
                else:
                    status = f"✅ {targets_met}/{len(target_list)}"
            elif p.get('is_potential_breakout') and p.get('is_healthy_trend'):
                status = f"⏳ {p.get('distance_to_breakout', 0):.1f}%"
            
            # Volume indicators
            vol_status = ""
            if p.get('weekly_buy_vol_strong'):
                vol_status = "🔥"
            elif p.get('buy_volume_strong'):
                vol_status = "✅"
            elif p.get('volume_confirmed'):
                vol_status = "📊"
            else:
                vol_status = "❌"
            
            # Double bottom indicator
            db_status = ""
            if p.get('double_bottom_confirmed'):
                db_status = "✅"
            elif p.get('double_bottom_potential'):
                db_status = "⏳"
            elif p.get('has_double_bottom'):
                db_status = "📊"
            else:
                db_status = "-"
            
            data.append({
                'Symbol': r['symbol'],
                'Sector': r.get('sector', 'Unknown')[:12],
                'TF': tf,
                'Status': status,
                'Score': p.get('return_score', 0),
                'Remaining': f"+{remaining_upside:.1f}%" if remaining_upside > 0 else "Done" if targets_met > 0 else "-",
                'Targets': f"{targets_met}/{len(target_list)}" if target_list else "-",
                'R/R': f"{t.get('risk_reward', 0):.1f}" if t.get('risk_reward') else "-",
                'Price': f"${current_price:.2f}",
                'Yoda': p.get('yoda_state', '-'),
                'Vol': vol_status,
                'DB': db_status,
                'Trend': '✅' if p.get('is_healthy_trend') else '❌'
            })
    
    if data:
        df = pd.DataFrame(data).sort_values('Score', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True, height=500)
    else:
        st.info("No data to display")


def display_detail():
    sel = st.session_state.get('selected', {})
    if not sel:
        st.session_state['view'] = 'dashboard'
        st.rerun()
        return
    
    # Back button
    if st.button("← Back to Dashboard", type="primary"):
        st.session_state['view'] = 'dashboard'
        st.rerun()
    
    symbol = sel.get('symbol', '')
    tf = sel.get('tf', '')
    tf_data = sel.get('tf_data', {})
    pattern = sel.get('pattern', {})
    targets = pattern.get('targets', {})
    
    st.markdown(f"<h1 style='text-align: center;'>📈 {symbol} ({tf})</h1>", unsafe_allow_html=True)
    
    # Key metrics
    cols = st.columns(6)
    cols[0].metric("Entry", f"${targets.get('entry', 0):.2f}")
    
    if targets.get('targets'):
        cols[1].metric("Target 1", f"${targets['targets'][0]['price']:.2f}", f"+{targets['targets'][0]['pct']:.1f}%")
    else:
        cols[1].metric("Target 1", "N/A")
    
    cols[2].metric("Stop Loss", f"${targets.get('stop_loss', 0):.2f}", f"-{targets.get('risk_pct', 0):.1f}%", delta_color="inverse")
    
    rr = targets.get('risk_reward', 0)
    cols[3].metric("Risk/Reward", f"{rr:.2f}:1" if rr else "N/A")
    cols[4].metric("Max Upside", f"+{targets.get('max_upside_pct', 0):.1f}%")
    cols[5].metric("Return Score", f"{pattern.get('return_score', 0):.0f}/100")
    
    st.markdown("---")
    
    # Chart
    df = tf_data.get('df')
    yoda_df = tf_data.get('yoda_df')
    
    if df is not None and yoda_df is not None:
        fig = create_chart(df, yoda_df, pattern, symbol, tf)
        st.plotly_chart(fig, use_container_width=True)
    
    # Details
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Target Levels")
        for t in targets.get('targets', []):
            st.write(f"**T{t['level']}**: ${t['price']:.2f} (+{t['pct']:.1f}%) — {t['type']}")
        
        st.markdown("### 🛡️ Support Levels")
        for s in pattern.get('support_levels', [])[:3]:
            st.write(f"${s['price']:.2f} ({s['type']})")
    
    with col2:
        st.markdown("### 📊 Signal Analysis")
        st.write(f"**Yoda State**: {pattern.get('yoda_state', 'NA')}")
        st.write(f"**Signal**: {pattern.get('yoda_signal', 'NA')}")
        st.write(f"**Strength**: {pattern.get('signal_strength', 0)}")
        st.write(f"**Volume**: {'✅ Confirmed' if pattern.get('volume_confirmed') else '❌ Low'}")
        st.write(f"**Pattern Score**: {pattern.get('pattern_score', 0)}/100")
        
        # Volume analysis section
        st.markdown("### 📈 Volume Analysis")
        st.write(f"**Buy Volume Strong**: {'✅' if pattern.get('buy_volume_strong') else '❌'}")
        st.write(f"**Weekly Buy Vol Strong**: {'✅ YES' if pattern.get('weekly_buy_vol_strong') else '❌ No'}")
        weekly_ratio = pattern.get('weekly_buy_vol_ratio', 1.0)
        st.write(f"**Weekly Buy Vol Ratio**: {weekly_ratio:.2f}x")
        vol_weight = pattern.get('volume_weight', 1.0)
        st.write(f"**Volume Weight**: {vol_weight:.2f}x")
    
    with col3:
        st.markdown("### 📐 Technical Analysis")
        for i, tl in enumerate(pattern.get('all_trendlines', [])[:3]):
            st.write(f"**TL{i+1}**: {tl['touches']} touches, {tl['angle_deg']:.1f}°")
        
        st.markdown("### 📈 Trend Health")
        health = pattern.get('trend_health', {})
        st.write(f"Above SMA20: {'✅' if health.get('above_sma20') else '❌'}")
        st.write(f"Above SMA50: {'✅' if health.get('above_sma50') else '❌'}")
        st.write(f"MACD Bullish: {'✅' if health.get('macd_bullish') else '❌'}")
        st.write(f"Trend Score: {health.get('trend_score', 0)}/100")
        
        # Double Bottom Pattern Section
        if pattern.get('has_double_bottom'):
            st.markdown("### 📊 Double Bottom")
            db = pattern.get('double_bottom', {})
            
            if pattern.get('double_bottom_confirmed'):
                st.success("✅ CONFIRMED BREAKOUT")
            elif pattern.get('double_bottom_potential'):
                st.warning("⏳ POTENTIAL - Watch Neckline")
            else:
                st.info("📊 Pattern Detected")
            
            st.write(f"**Bottom 1**: ${db.get('bottom1', {}).get('price', 0):.2f}")
            st.write(f"**Bottom 2**: ${db.get('bottom2', {}).get('price', 0):.2f}")
            st.write(f"**Neckline**: ${db.get('neckline_price', 0):.2f}")
            st.write(f"**Target**: ${db.get('target_price', 0):.2f} (+{db.get('target_pct', 0):.1f}%)")
            st.write(f"**Quality Score**: {db.get('quality_score', 0)}/100")
            st.write(f"**Volume Confirms**: {'✅' if db.get('volume_confirms') else '❌'}")


# ==================== MAIN ====================

def main():
    st.set_page_config(
        page_title="Yoda Pattern Scanner",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    if 'view' not in st.session_state:
        st.session_state['view'] = 'dashboard'
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Scanner Settings")
        st.markdown("---")
        
        method = st.radio("📥 Input Method", ["Manual Entry", "Default Watchlist", "CSV Upload"])
        
        if method == "Manual Entry":
            txt = st.text_area("Enter Symbols", "AAPL, MSFT, NVDA, AMD, TSLA, META, GOOGL, AMZN, NFLX, CRM",
                              help="Comma-separated stock symbols")
            symbols = [s.strip().upper() for s in txt.replace('\n', ',').split(',') if s.strip()]
        elif method == "CSV Upload":
            file = st.file_uploader("Upload CSV", type=['csv'])
            symbols = []
            if file:
                csv_df = pd.read_csv(file)
                for col in ['symbol', 'Symbol', 'ticker', 'Ticker']:
                    if col in csv_df.columns:
                        symbols = csv_df[col].tolist()
                        break
                if not symbols and len(csv_df.columns) > 0:
                    symbols = csv_df.iloc[:, 0].tolist()
        else:
            symbols = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA', 'META', 'GOOGL', 'AMZN', 'NFLX', 'CRM',
                      'ADBE', 'INTC', 'QCOM', 'AVGO', 'TXN']
        
        st.markdown("---")
        
        timeframes = st.multiselect("📅 Timeframes", ['1d', '1wk', '1mo'], default=['1d', '1wk'],
                                    help="Select analysis timeframes")
        
        min_score = st.slider("📊 Min Pattern Score", 0, 100, 40,
                             help="Minimum score to consider a pattern valid")
        
        threshold = st.slider("🎯 Breakout Proximity %", 1.0, 10.0, 5.0,
                             help="Maximum distance to consider approaching breakout")
        
        st.markdown("---")
        
        if st.button("🔍 Run Scanner", type="primary", use_container_width=True):
            if not symbols:
                st.error("Please enter at least one symbol")
                return
            
            progress = st.progress(0)
            status = st.empty()
            
            all_results, matches, potential = [], [], []
            excluded_sell = 0  # Counter for stocks excluded due to SELL signals
            
            for i, sym in enumerate(symbols):
                status.text(f"Scanning {sym}...")
                progress.progress((i+1) / len(symbols))
                
                result = scan_stock(sym, timeframes)
                
                # Check for SELL signals in daily or weekly - exclude if found
                has_sell_signal = False
                for check_tf in ['1d', '1wk']:
                    if check_tf in result.get('timeframes', {}):
                        tf_data = result['timeframes'][check_tf]
                        if 'error' not in tf_data:
                            pattern = tf_data.get('pattern', {})
                            yoda_signal = pattern.get('yoda_signal', 'NA')
                            yoda_state = pattern.get('yoda_state', 'NA')
                            if yoda_signal == 'SELL' or yoda_state == 'SELL':
                                has_sell_signal = True
                                break
                
                # Skip this stock entirely if it has a SELL signal in daily or weekly
                if has_sell_signal:
                    excluded_sell += 1
                    continue
                
                # Only add to all_results if no SELL signal
                all_results.append(result)
                
                if result['has_pattern'] and result['max_score'] >= min_score:
                    # Filter: only include if MACD is positive/uptick AND Yoda is BUY
                    tf = result.get('best_timeframe')
                    if tf:
                        td = result['timeframes'].get(tf, {})
                        yoda_df = td.get('yoda_df')
                        pattern = td.get('pattern', {})
                        
                        # Check MACD conditions
                        macd_bullish = False
                        if yoda_df is not None and len(yoda_df) >= 2:
                            if 'MACD_Hist' in yoda_df.columns:
                                current_hist = yoda_df['MACD_Hist'].iloc[-1]
                                prev_hist = yoda_df['MACD_Hist'].iloc[-2]
                                macd_positive = current_hist > 0
                                macd_uptick = current_hist > prev_hist
                                macd_bullish = macd_positive or macd_uptick
                        
                        # Check Yoda BUY signal
                        yoda_buy = pattern.get('yoda_state') == 'BUY'
                        
                        # Only add to matches if MACD bullish AND Yoda BUY
                        if macd_bullish and yoda_buy:
                            result['macd_positive'] = current_hist > 0 if yoda_df is not None and 'MACD_Hist' in yoda_df.columns else False
                            result['macd_uptick'] = macd_uptick if yoda_df is not None else False
                            matches.append(result)
                
                for tf in timeframes:
                    td = result['timeframes'].get(tf, {})
                    p = td.get('pattern', {})
                    yoda_df = td.get('yoda_df')
                    
                    # Check MACD conditions
                    macd_bullish = False
                    macd_positive = False
                    macd_uptick = False
                    
                    if yoda_df is not None and len(yoda_df) >= 2:
                        if 'MACD_Hist' in yoda_df.columns:
                            current_hist = yoda_df['MACD_Hist'].iloc[-1]
                            prev_hist = yoda_df['MACD_Hist'].iloc[-2]
                            macd_positive = current_hist > 0
                            macd_uptick = current_hist > prev_hist
                            macd_bullish = macd_positive or macd_uptick
                    
                    # Check for potential trendline breakout
                    if (p.get('is_potential_breakout') and 
                        p.get('is_healthy_trend') and 
                        p.get('yoda_state') == 'BUY' and
                        macd_bullish):  # Added MACD filter
                        dist = p.get('distance_to_breakout', 0)
                        upside = p.get('targets', {}).get('max_upside_pct', 0)
                        if dist <= threshold:
                            potential.append({
                                'symbol': sym, 'timeframe': tf,
                                'pattern': p, 'tf_data': td, 'distance': dist,
                                'pattern_type': 'trendline',
                                'upside': upside,
                                'macd_positive': macd_positive,
                                'macd_uptick': macd_uptick
                            })
                    
                    # Check for double bottom potential (especially for weekly)
                    if (p.get('double_bottom_potential') and 
                        p.get('yoda_state') == 'BUY' and
                        macd_bullish):  # Added MACD filter
                        db = p.get('double_bottom', {})
                        dist = db.get('distance_to_neckline', 0)
                        upside = p.get('targets', {}).get('max_upside_pct', 0)
                        if dist <= threshold:
                            # Avoid duplicates
                            existing = [x for x in potential if x['symbol'] == sym and x['timeframe'] == tf]
                            if not existing:
                                potential.append({
                                    'symbol': sym, 'timeframe': tf,
                                    'pattern': p, 'tf_data': td, 'distance': dist,
                                    'pattern_type': 'double_bottom',
                                    'upside': upside,
                                    'macd_positive': macd_positive,
                                    'macd_uptick': macd_uptick
                                })
            
            progress.empty()
            status.empty()
            
            st.session_state['all_results'] = all_results
            st.session_state['matches'] = matches
            st.session_state['potential_breakouts'] = potential
            st.session_state['excluded_sell'] = excluded_sell
            st.session_state['total_scanned'] = len(symbols)
            st.session_state['view'] = 'dashboard'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 Legend")
        st.markdown("""
        **Filters Applied:**
        - ❌ **SELL Excluded**: Stocks with SELL signal in Daily or Weekly are removed
        - ✅ **BUY + MACD+**: Confirmed = BUY signal + positive/rising MACD
        
        **Indicators:**
        - 🏆 **Score**: Overall opportunity rating
        - 🎯 **Target**: Price targets from resistance
        - 🛡️ **Stop**: Suggested stop loss level
        - ⚖️ **R/R**: Risk-to-Reward ratio
        - 📊 **Double Bottom**: W-pattern reversal
        - 🔥 **Strong Vol**: Weekly buy volume above average
        """)
    
    # Main content
    if st.session_state.get('view') == 'detail':
        display_detail()
    elif st.session_state.get('all_results'):
        display_dashboard()
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 50px 0;">
            <h1>📈 Yoda Pattern Scanner</h1>
            <p style="font-size: 20px; color: #666; margin: 20px 0;">
                Professional Stock Analysis with Automatic Target Detection
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="summary-card summary-card-green" style="text-align: center;">
                <div class="card-title">🎯 SMART TARGETS</div>
                <div style="font-size: 16px; margin-top: 10px;">
                    Automatic price targets from resistance levels, Fibonacci extensions, and swing highs
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="summary-card summary-card-purple" style="text-align: center;">
                <div class="card-title">⚖️ RISK ANALYSIS</div>
                <div style="font-size: 16px; margin-top: 10px;">
                    Calculated stop losses and Risk/Reward ratios for every opportunity
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="summary-card summary-card-orange" style="text-align: center;">
                <div class="card-title">🏆 OPPORTUNITY RANKING</div>
                <div style="font-size: 16px; margin-top: 10px;">
                    Stocks ranked by return potential combining upside, R/R, and signals
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center;">
            <p style="font-size: 18px;">👈 <strong>Configure settings in the sidebar and click "Run Scanner" to begin</strong></p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

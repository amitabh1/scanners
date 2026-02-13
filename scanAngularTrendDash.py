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
    """Compute Yoda indicator signals."""
    if data is None or data.empty or len(data) < 20:
        return pd.DataFrame()
    
    df = data.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series([0]*len(df)))
    
    # Volume
    df['Volume_Avg'] = volume.rolling(20, min_periods=1).mean()
    df['Volume_Ratio'] = np.where(df['Volume_Avg'] > 0, volume / df['Volume_Avg'], 1)
    df['Volume_Above_Avg'] = df['Volume_Ratio'] > 1.5
    df['Volume_Confirmed'] = df['Volume_Above_Avg']
    
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
    
    # Signal strength
    df['Signal_Strength'] = 0
    df.loc[df['MACD_Buy'], 'Signal_Strength'] += 20
    df.loc[df['Cross_Up'], 'Signal_Strength'] += 20
    df.loc[df['Squeeze_Fired'], 'Signal_Strength'] += 15
    df.loc[df['Volume_Above_Avg'], 'Signal_Strength'] += 15
    df.loc[df['MACD_Hist'] > 0, 'Signal_Strength'] += 10
    df.loc[close > df['SMA'], 'Signal_Strength'] += 10
    df.loc[close > df['SMA_20'], 'Signal_Strength'] += 10
    
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


def find_breakouts(df, trendlines):
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
                breakouts.append({
                    'trendline_idx': tl_idx, 'breakout_index': i,
                    'breakout_price': close, 'trendline_price': tl_price,
                    'breakout_pct': pct, 'is_recent': i >= len(df) - 10
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


def calculate_targets(current_price, resistance, support, pattern_score):
    targets = {
        'entry': current_price, 'targets': [], 'stop_loss': None,
        'risk_reward': 0, 'max_upside_pct': 0, 'risk_pct': 5.0, 'return_score': 0
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
    
    upside_score = min(30, targets['max_upside_pct'] * 2)
    rr_score = min(30, targets['risk_reward'] * 10)
    pattern_contrib = pattern_score * 0.4
    targets['return_score'] = min(100, upside_score + rr_score + pattern_contrib)
    
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


def detect_pattern(df, yoda_df):
    result = {
        'has_pattern': False, 'yoda_signal': 'NA', 'yoda_state': 'NA',
        'signal_strength': 0, 'volume_confirmed': False,
        'has_descending_trendline': False, 'trendline_touches': 0,
        'has_trendline_breakout': False, 'breakout_strength': 0,
        'pattern_score': 0, 'all_trendlines': [], 'all_breakouts': [],
        'recent_breakouts': [], 'swing_highs': [], 'swing_lows': [],
        'potential_breakout': {}, 'is_potential_breakout': False,
        'distance_to_breakout': 0, 'trend_health': {}, 'is_healthy_trend': False,
        'resistance_levels': [], 'support_levels': [], 'targets': {}, 'return_score': 0
    }
    
    if df is None or df.empty or len(df) < 20:
        return result
    
    recent = df.tail(100).reset_index(drop=True)
    current_price = recent['close'].iloc[-1]
    
    swing_highs = find_swing_highs(recent)
    swing_lows = find_swing_lows(recent)
    result['swing_highs'] = swing_highs
    result['swing_lows'] = swing_lows
    
    trendlines = detect_trendlines(recent, swing_highs)
    result['all_trendlines'] = trendlines
    
    if trendlines:
        result['has_descending_trendline'] = True
        result['trendline_touches'] = trendlines[0]['touches']
        
        breakouts = find_breakouts(recent, trendlines)
        result['all_breakouts'] = breakouts
        result['recent_breakouts'] = [b for b in breakouts if b['is_recent']]
        
        if result['recent_breakouts']:
            result['has_trendline_breakout'] = True
            result['breakout_strength'] = max(b['breakout_pct'] for b in result['recent_breakouts'])
    
    if yoda_df is not None and len(yoda_df) > 0:
        result['yoda_signal'] = yoda_df['YODA_BUY_SELL'].iloc[-1]
        result['yoda_state'] = yoda_df['YODA_STATE'].iloc[-1]
        result['signal_strength'] = yoda_df['Signal_Strength'].iloc[-1]
        result['volume_confirmed'] = bool(yoda_df['Volume_Confirmed'].iloc[-1])
    
    score = 0
    if result['yoda_signal'] == 'BUY': score += 30
    elif result['yoda_state'] == 'BUY': score += 20
    if result['has_descending_trendline']: score += 15
    if result['trendline_touches'] >= 3: score += 5
    if result['has_trendline_breakout']: score += 25
    if result['breakout_strength'] > 2: score += 5
    if result['volume_confirmed']: score += 10
    
    result['pattern_score'] = min(100, score)
    result['has_pattern'] = score >= 50
    
    potential = detect_potential_breakout(recent, trendlines)
    result['potential_breakout'] = potential
    result['is_potential_breakout'] = potential.get('is_potential', False)
    result['distance_to_breakout'] = potential.get('distance_to_breakout', 0)
    
    health = check_trend_health(recent, yoda_df)
    result['trend_health'] = health
    result['is_healthy_trend'] = health.get('is_healthy', False)
    
    resistance = find_resistance_levels(recent, yoda_df, swing_highs, current_price)
    support = find_support_levels(recent, yoda_df, swing_lows, current_price)
    result['resistance_levels'] = resistance
    result['support_levels'] = support
    
    targets = calculate_targets(current_price, resistance, support, result['pattern_score'])
    result['targets'] = targets
    result['return_score'] = targets['return_score']
    
    return result


# ==================== DATA FETCHING ====================

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
    results = {
        'symbol': symbol, 'timeframes': {}, 'best_timeframe': None,
        'max_score': 0, 'has_pattern': False, 'return_score': 0, 'max_upside': 0
    }
    
    periods = {'1d': '1y', '1wk': '2y', '1mo': '5y'}
    
    for tf in timeframes:
        df = fetch_data(symbol, periods.get(tf, '1y'), tf)
        if df is None or len(df) < 30:
            results['timeframes'][tf] = {'error': 'No data'}
            continue
        
        yoda_df = YodaSignal(df.copy())
        pattern = detect_pattern(df, yoda_df)
        
        results['timeframes'][tf] = {
            'df': df, 'yoda_df': yoda_df, 'pattern': pattern,
            'pattern_score': pattern['pattern_score'],
            'has_pattern': pattern['has_pattern'],
            'current_price': df['close'].iloc[-1],
            'return_score': pattern['return_score'],
            'max_upside': pattern['targets'].get('max_upside_pct', 0)
        }
        
        if pattern['pattern_score'] > results['max_score']:
            results['max_score'] = pattern['pattern_score']
            results['best_timeframe'] = tf
            results['has_pattern'] = pattern['has_pattern']
            results['return_score'] = pattern['return_score']
            results['max_upside'] = pattern['targets'].get('max_upside_pct', 0)
    
    return results


# ==================== CHARTING ====================

def create_chart(df, yoda_df, pattern, symbol, tf):
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


# ==================== DASHBOARD ====================

def display_dashboard():
    all_results = st.session_state.get('all_results', [])
    matches = st.session_state.get('matches', [])
    potential = st.session_state.get('potential_breakouts', [])
    
    # Header
    st.markdown('<h1 style="text-align: center; margin-bottom: 30px;">📈 Yoda Pattern Scanner</h1>', unsafe_allow_html=True)
    
    # Summary cards row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(summary_card("STOCKS SCANNED", len(all_results), "Total analyzed"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(summary_card("CONFIRMED", len(matches), "Breakout signals", "summary-card-green"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(summary_card("POTENTIAL", len(potential), "Approaching breakout", "summary-card-orange"), unsafe_allow_html=True)
    
    with col4:
        best_upside = max((m.get('max_upside', 0) for m in matches), default=0) if matches else 0
        st.markdown(summary_card("BEST UPSIDE", f"+{best_upside:.1f}%", "Maximum potential", "summary-card-purple"), unsafe_allow_html=True)
    
    with col5:
        avg_rr = np.mean([
            m.get('timeframes', {}).get(m.get('best_timeframe', ''), {}).get('pattern', {}).get('targets', {}).get('risk_reward', 0)
            for m in matches if m.get('timeframes', {}).get(m.get('best_timeframe', ''), {}).get('pattern', {}).get('targets', {}).get('risk_reward', 0) > 0
        ]) if matches else 0
        st.markdown(summary_card("AVG R/R", f"{avg_rr:.1f}:1", "Risk/Reward ratio", "summary-card-red"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Opportunities", "🎯 Confirmed Breakouts", "⏳ Potential Setups", "📊 Full Analysis"])
    
    with tab1:
        display_rankings(matches, potential)
    with tab2:
        display_confirmed(matches)
    with tab3:
        display_potential(potential)
    with tab4:
        display_all_data(all_results)


def display_rankings(matches, potential):
    st.markdown('<div class="section-header">🏆 Top Opportunities by Return Potential</div>', unsafe_allow_html=True)
    
    # Combine opportunities
    opps = []
    
    for m in matches:
        tf = m.get('best_timeframe')
        if not tf:
            continue
        tf_data = m.get('timeframes', {}).get(tf, {})
        pattern = tf_data.get('pattern', {})
        targets = pattern.get('targets', {})
        
        opps.append({
            'symbol': m['symbol'], 'tf': tf, 'type': '✅ Breakout',
            'return_score': m.get('return_score', 0),
            'max_upside': targets.get('max_upside_pct', 0),
            'risk_reward': targets.get('risk_reward', 0),
            'entry': targets.get('entry', 0),
            'target1': targets['targets'][0]['price'] if targets.get('targets') else 0,
            'stop': targets.get('stop_loss', 0),
            'risk_pct': targets.get('risk_pct', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
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
            'risk_reward': targets.get('risk_reward', 0),
            'entry': targets.get('entry', 0),
            'target1': targets['targets'][0]['price'] if targets.get('targets') else 0,
            'stop': targets.get('stop_loss', 0),
            'risk_pct': targets.get('risk_pct', 0),
            'yoda_state': pattern.get('yoda_state', 'NA'),
            'tf_data': p.get('tf_data', {}), 'pattern': pattern
        })
    
    if not opps:
        st.info("🔍 Run a scan to discover opportunities")
        return
    
    opps.sort(key=lambda x: (x['return_score'], x['max_upside']), reverse=True)
    
    # Display top opportunities as cards
    for rank, opp in enumerate(opps[:10], 1):
        col1, col2 = st.columns([6, 1])
        
        with col1:
            st.markdown(
                stock_card_html(
                    opp['symbol'], opp['tf'], opp['type'],
                    opp['return_score'], opp['max_upside'],
                    opp['risk_reward'] if opp['risk_reward'] else 0,
                    opp['entry'], opp['target1'] if opp['target1'] else opp['entry'],
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
        'Rank': i+1, 'Symbol': o['symbol'], 'Timeframe': o['tf'], 'Status': o['type'],
        'Score': f"{o['return_score']:.0f}", 'Upside': f"+{o['max_upside']:.1f}%",
        'Risk': f"-{o['risk_pct']:.1f}%",
        'R/R': f"{o['risk_reward']:.1f}:1" if o['risk_reward'] else "-",
        'Entry': f"${o['entry']:.2f}" if o['entry'] else "-",
        'Target': f"${o['target1']:.2f}" if o['target1'] else "-",
        'Stop': f"${o['stop']:.2f}" if o['stop'] else "-"
    } for i, o in enumerate(opps)]
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True, height=400)


def display_confirmed(matches):
    st.markdown('<div class="section-header">🎯 Confirmed Breakouts</div>', unsafe_allow_html=True)
    
    if not matches:
        st.info("No confirmed breakouts found in this scan")
        return
    
    for m in sorted(matches, key=lambda x: x.get('return_score', 0), reverse=True):
        tf = m.get('best_timeframe')
        if not tf:
            continue
        
        tf_data = m.get('timeframes', {}).get(tf, {})
        pattern = tf_data.get('pattern', {})
        targets = pattern.get('targets', {})
        
        with st.expander(f"**{m['symbol']}** ({tf}) — Score: {m.get('return_score', 0):.0f} | Upside: +{m.get('max_upside', 0):.1f}%", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Entry Price", f"${targets.get('entry', 0):.2f}")
                st.metric("Yoda State", pattern.get('yoda_state', 'NA'))
            
            with col2:
                if targets.get('targets'):
                    st.metric("Target 1", f"${targets['targets'][0]['price']:.2f}", f"+{targets['targets'][0]['pct']:.1f}%")
                st.metric("Signal Strength", f"{pattern.get('signal_strength', 0)}")
            
            with col3:
                st.metric("Stop Loss", f"${targets.get('stop_loss', 0):.2f}", f"-{targets.get('risk_pct', 0):.1f}%", delta_color="inverse")
                st.metric("Volume", "✅ Confirmed" if pattern.get('volume_confirmed') else "❌ Low")
            
            with col4:
                rr = targets.get('risk_reward', 0)
                st.metric("Risk/Reward", f"{rr:.2f}:1" if rr else "N/A")
                st.metric("Pattern Score", f"{pattern.get('pattern_score', 0)}/100")
            
            if st.button(f"📈 View Full Chart", key=f"conf_{m['symbol']}_{tf}"):
                st.session_state['selected'] = {
                    'symbol': m['symbol'], 'tf': tf,
                    'tf_data': tf_data, 'pattern': pattern
                }
                st.session_state['view'] = 'detail'
                st.rerun()


def display_potential(potential):
    st.markdown('<div class="section-header">⏳ Potential Breakout Setups</div>', unsafe_allow_html=True)
    
    if not potential:
        st.info("No potential setups found — stocks may need to approach resistance levels")
        return
    
    for p in sorted(potential, key=lambda x: x.get('distance', 100)):
        pattern = p.get('pattern', {})
        targets = pattern.get('targets', {})
        dist = p.get('distance', 0)
        
        emoji = "🔥" if dist <= 2 else ("⚡" if dist <= 3 else "👀")
        proximity = "VERY CLOSE" if dist <= 2 else ("CLOSE" if dist <= 3 else "APPROACHING")
        
        with st.expander(f"{emoji} **{p['symbol']}** ({p['timeframe']}) — {proximity} ({dist:.1f}% to breakout)", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Distance to Breakout", f"{dist:.1f}%")
                st.metric("Yoda State", pattern.get('yoda_state', 'NA'))
            
            with col2:
                if targets.get('targets'):
                    st.metric("Post-Breakout Target", f"${targets['targets'][0]['price']:.2f}")
                st.metric("Trend Health", "✅ Healthy" if pattern.get('is_healthy_trend') else "⚠️ Weak")
            
            with col3:
                st.metric("Current Price", f"${targets.get('entry', 0):.2f}")
                pot = pattern.get('potential_breakout', {})
                st.metric("Breakout Level", f"${pot.get('trendline_price', 0):.2f}")
            
            with col4:
                st.metric("Return Score", f"{pattern.get('return_score', 0):.0f}/100")
                rr = targets.get('risk_reward', 0)
                st.metric("Potential R/R", f"{rr:.1f}:1" if rr else "N/A")
            
            if st.button(f"📈 View Chart", key=f"pot_{p['symbol']}_{p['timeframe']}"):
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
            
            status = "❌ No Signal"
            if p.get('has_trendline_breakout'):
                status = "✅ Breakout"
            elif p.get('is_potential_breakout') and p.get('is_healthy_trend'):
                status = f"⏳ {p.get('distance_to_breakout', 0):.1f}%"
            
            data.append({
                'Symbol': r['symbol'], 'TF': tf, 'Status': status,
                'Pattern Score': p.get('pattern_score', 0),
                'Return Score': p.get('return_score', 0),
                'Upside': f"+{t.get('max_upside_pct', 0):.1f}%",
                'R/R': f"{t.get('risk_reward', 0):.1f}:1" if t.get('risk_reward') else "-",
                'Price': f"${td.get('current_price', 0):.2f}",
                'Yoda': p.get('yoda_state', '-'),
                'Trend': '✅' if p.get('is_healthy_trend') else '❌',
                'Volume': '✅' if p.get('volume_confirmed') else '❌'
            })
    
    if data:
        df = pd.DataFrame(data).sort_values('Return Score', ascending=False)
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
            
            for i, sym in enumerate(symbols):
                status.text(f"Scanning {sym}...")
                progress.progress((i+1) / len(symbols))
                
                result = scan_stock(sym, timeframes)
                all_results.append(result)
                
                if result['has_pattern'] and result['max_score'] >= min_score:
                    matches.append(result)
                
                for tf in timeframes:
                    td = result['timeframes'].get(tf, {})
                    p = td.get('pattern', {})
                    if p.get('is_potential_breakout') and p.get('is_healthy_trend') and p.get('yoda_state') == 'BUY':
                        dist = p.get('distance_to_breakout', 0)
                        if dist <= threshold:
                            potential.append({
                                'symbol': sym, 'timeframe': tf,
                                'pattern': p, 'tf_data': td, 'distance': dist
                            })
            
            progress.empty()
            status.empty()
            
            st.session_state['all_results'] = all_results
            st.session_state['matches'] = matches
            st.session_state['potential_breakouts'] = potential
            st.session_state['view'] = 'dashboard'
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📖 Legend")
        st.markdown("""
        - 🏆 **Score**: Overall opportunity rating
        - 🎯 **Target**: Price targets from resistance
        - 🛡️ **Stop**: Suggested stop loss level
        - ⚖️ **R/R**: Risk-to-Reward ratio
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

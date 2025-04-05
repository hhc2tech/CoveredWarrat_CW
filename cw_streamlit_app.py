import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import io

from cw_logic import bs_price, greeks, simulate_pnl

# --- App Interface ---
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

# --- Greeks calculator ---
def greeks(S, K, T, r, sigma, option_type='call'):
    price, d1, d2 = bs_price(S, K, T, r, sigma, option_type)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2)) if option_type == 'call' else (
             -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
             r * K * np.exp(-r * T) * norm.cdf(-d2))
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)

    return {
        'Price': price,
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }

# --- App Interface ---
st.title("Covered Warrant (CW) Valuation Tool - Black-Scholes")
st.markdown("Choose a sample file or upload your own CW data file (CSV or Excel).")

file_choice = st.radio("Select data source:", ["Use sample file (data.csv)", "Upload a new file"])
df = None

if file_choice == "Use sample file (data.csv)":
    try:
        df = pd.read_csv("data.csv")
        st.success("Sample file 'data.csv' loaded successfully.")
    except FileNotFoundError:
        st.error("Sample file 'data.csv' not found.")
else:
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.success("File uploaded and read successfully.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

if df is not None and not df.empty:
    st.subheader("Initial Data Preview")
    st.dataframe(df)

    results = []
    for _, row in df.iterrows():
        try:
            result = greeks(row['S'], row['K'], row['T'], row['r'], row['sigma'], row['option_type'].lower())
            result['Market Price'] = row.get('CW_market_price', None)
            results.append(result)
        except:
            results.append({'Price': None, 'Delta': None, 'Gamma': None, 'Vega': None, 'Theta': None, 'Rho': None, 'Market Price': None})

    greek_df = pd.DataFrame(results)
    final_df = pd.concat([df, greek_df], axis=1)

    st.success("Calculation completed!")
    st.dataframe(final_df)

    st.subheader("Overlay: Theoretical vs Market CW Price")
    if 'CW_market_price' in df.columns:
        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Scatter(y=final_df['Price'], x=final_df.index, mode='lines+markers', name='Black-Scholes Price'))
        fig_overlay.add_trace(go.Scatter(y=final_df['CW_market_price'], x=final_df.index, mode='lines+markers', name='Market CW Price'))
        fig_overlay.update_layout(title="CW Market Price vs Theoretical Price", xaxis_title="Data Point", yaxis_title="CW Price")
        st.plotly_chart(fig_overlay, use_container_width=True)
    else:
        st.info("CW_market_price column not found for comparison.")

    st.subheader("CW Profit/Loss Simulation at Maturity")
    selected_index = st.number_input("Select a CW row to simulate", min_value=0, max_value=len(final_df)-1, value=0)
    selected_row = final_df.iloc[selected_index]

    with st.expander("Input Simulation Parameters"):
        buy_price = st.number_input("CW Purchase Price (VND)", value=3000.0, step=100.0)
        ratio = st.number_input("Conversion Ratio (e.g. enter 5 for 5:1)", value=5.0, min_value=0.1)
        fee = st.number_input("Transaction Fee One-Way (VND)", value=0.0, step=100.0)
        final_T = st.number_input("Time to Maturity (Years)", value=selected_row['T'], min_value=0.01)
        K = selected_row['K']
        r = selected_row['r']
        sigma = selected_row['sigma']
        option_type = selected_row['option_type'].lower()

    S_range = np.linspace(K * 0.6, K * 1.4, 100)
    profit = []
    profit_pct = []

    for S_T in S_range:
        price_T, _, _ = bs_price(S_T, K, final_T, r, sigma, option_type)
        intrinsic_value = max(S_T - K, 0) if option_type == 'call' else max(K - S_T, 0)
        received = intrinsic_value / ratio
        pnl = received - buy_price - fee
        pnl_pct = (pnl / (buy_price + fee)) * 100 if (buy_price + fee) > 0 else 0
        profit.append(pnl)
        profit_pct.append(pnl_pct)

    df_pnl = pd.DataFrame({
        'Stock Price at Maturity': S_range,
        'Profit/Loss (VND)': profit,
        'Profit/Loss (%)': profit_pct
    })

    plot_mode = st.radio("Select chart mode", ["VND", "Percentage (%)"])
    if plot_mode == "VND":
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=df_pnl['Stock Price at Maturity'], y=df_pnl['Profit/Loss (VND)'],
                                     fill='tozeroy', fillcolor='rgba(0,255,0,0.2)', line=dict(color='green'),
                                     name='Profit'))
        fig_pnl.update_layout(title="Profit/Loss (VND) at Maturity",
                              xaxis_title="Stock Price", yaxis_title="Profit/Loss (VND)")
    else:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=df_pnl['Stock Price at Maturity'], y=df_pnl['Profit/Loss (%)'],
                                     fill='tozeroy', fillcolor='rgba(0,0,255,0.2)', line=dict(color='blue'),
                                     name='Profit (%)'))
        fig_pnl.update_layout(title="Profit/Loss (%) at Maturity",
                              xaxis_title="Stock Price", yaxis_title="Profit/Loss (%)")

    st.plotly_chart(fig_pnl, use_container_width=True)

    try:
        breakeven_idx = np.argmin(np.abs(np.array(profit)))
        breakeven_price = S_range[breakeven_idx]
        st.success(f"Estimated breakeven stock price: approximately **{breakeven_price:.2f} VND/share**")
    except:
        st.warning("No breakeven point found in the simulated range.")
else:
    st.warning("Please upload or select a valid CW data file.")

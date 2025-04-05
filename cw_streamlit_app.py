import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io

# Hàm d?nh giá Black-Scholes
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

# Hàm tính Greek
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

# Giao di?n chính
#st.title("Cong cu dinh gia chung quyen (Covered Warrant) - Black-Scholes")
#st.markdown("Tai file du lieu CW (CSV hoac Excel).")

#uploaded_file = st.file_uploader("Tai file Excel hoac CSV", type=["xlsx", "csv"])
st.markdown("## 📁 Data Input")

file_option = st.radio("📌 Choose data source", ["📂 Use sample file (data.csv)", "⬆️ Upload new file"])

if file_option == "📂 Use sample file (data.csv)":
    try:
        df = pd.read_csv("data.csv")  # You can change the path if needed
        st.success("✅ Successfully loaded the sample file 'data.csv'")
    except FileNotFoundError:
        st.error("❌ File 'data.csv' not found. Make sure it's in the same directory as this app.")
        df = None
else:
    uploaded_file = st.file_uploader("📥 Upload Excel or CSV file", type=["xlsx", "csv"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.success("✅ Successfully uploaded your file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = None
    else:
        df = None

# Main Interface
st.title("Covered Warrant (CW) Valuation Tool - Black-Scholes Model")

# File upload already handled before this block (as in previous message)

if df is not None:
    st.write("📄 Initial Data Preview:")
    st.dataframe(df)

    # Calculate option price and Greeks for each row
    results = []
    for _, row in df.iterrows():
        try:
            result = greeks(row['S'], row['K'], row['T'], row['r'], row['sigma'], row['option_type'].lower())
            results.append(result)
        except:
            results.append({'Price': None, 'Delta': None, 'Gamma': None, 'Vega': None, 'Theta': None, 'Rho': None})
    
    greek_df = pd.DataFrame(results)
    final_df = pd.concat([df, greek_df], axis=1)

    st.success("✅ Calculation completed!")
    st.dataframe(final_df)

    # Interactive Plot Section
    st.markdown("---")
    st.subheader("📊 Interactive Chart - Analyze CW Price Sensitivity")

    selected_index = st.number_input("🔢 Select a CW row to analyze", min_value=0, max_value=len(final_df)-1, value=0, step=1)
    selected_row = final_df.iloc[selected_index]

    var_to_plot = st.selectbox("📈 Choose a variable to analyze", ['S', 'T', 'sigma'], index=0)

    # Generate the range of values for plotting
    if var_to_plot == 'S':
        x_range = np.linspace(selected_row['K'] * 0.6, selected_row['K'] * 1.4, 50)
    elif var_to_plot == 'T':
        x_range = np.linspace(0.01, 2, 50)
    elif var_to_plot == 'sigma':
        x_range = np.linspace(0.05, 1.0, 50)

    # Recalculate CW price based on selected variable
    prices = []
    for x in x_range:
        args = {
            'S': selected_row['S'],
            'K': selected_row['K'],
            'T': selected_row['T'],
            'r': selected_row['r'],
            'sigma': selected_row['sigma'],
            'option_type': selected_row['option_type'].lower()
        }
        args[var_to_plot] = x
        price, _, _ = bs_price(**args)
        prices.append(price)

    df_plot = pd.DataFrame({var_to_plot: x_range, 'CW Price': prices})
    fig = px.line(df_plot, x=var_to_plot, y='CW Price',
                  title=f"Sensitivity of CW Price to {var_to_plot}",
                  labels={var_to_plot: var_to_plot, 'CW Price': 'Covered Warrant Price'})
    st.plotly_chart(fig, use_container_width=True)

    # Export results to Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        final_df.to_excel(writer, index=False, sheet_name='CW Results')
    st.download_button("📥 Download Result (Excel)", data=output.getvalue(), file_name="cw_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# Profit & Loss Simulation Section
st.markdown("---")
st.subheader("📈 CW Profit/Loss Simulation at Maturity")

with st.expander("🛠️ Input Simulation Parameters"):
    buy_price = st.number_input("💰 CW Purchase Price (VND)", value=3000.0, step=100.0)
    ratio = st.number_input("🔄 Conversion Ratio (e.g. enter 5 for 5:1)", value=5.0, min_value=0.1)
    fee = st.number_input("💸 One-way Transaction Fee (VND)", value=0.0, step=100.0)
    final_T = st.number_input("⏳ Time to Maturity (Years)", value=selected_row['T'], min_value=0.01)
    K = selected_row['K']
    r = selected_row['r']
    sigma = selected_row['sigma']
    option_type = selected_row['option_type'].lower()

# Simulate stock price range at maturity
S_range = np.linspace(K * 0.6, K * 1.4, 100)
profit = []

for S_T in S_range:
    price_T, _, _ = bs_price(S_T, K, final_T, r, sigma, option_type)
    intrinsic_value = max(S_T - K, 0) if option_type == 'call' else max(K - S_T, 0)
    received = intrinsic_value / ratio
    pnl = received - buy_price - fee
    profit.append(pnl)

df_pnl = pd.DataFrame({'Stock Price at Maturity': S_range, 'Profit/Loss (VND)': profit})
fig_pnl = px.line(df_pnl, x='Stock Price at Maturity', y='Profit/Loss (VND)',
                  title='📊 Profit/Loss Simulation Holding CW until Maturity',
                  labels={'Stock Price at Maturity': 'Stock Price at Maturity', 'Profit/Loss (VND)': 'P&L in VND'})

st.plotly_chart(fig_pnl, use_container_width=True)

# Show breakeven point
try:
    breakeven_idx = np.argmin(np.abs(np.array(profit)))
    breakeven_price = S_range[breakeven_idx]
    st.success(f"💡 Estimated Breakeven Stock Price: **{breakeven_price:.2f} VND**")
except:
    st.warning("No breakeven point found within the simulated price range.")

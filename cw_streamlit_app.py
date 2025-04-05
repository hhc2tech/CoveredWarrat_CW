import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import io

# H�m d?nh gi� Black-Scholes
def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

# H�m t�nh Greek
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

# Giao di?n ch�nh
st.title("?? C�ng c? d?nh gi� Ch?ng quy?n (Covered Warrant) - Black-Scholes")
st.markdown("T?i file d? li?u CW (CSV ho?c Excel), h? th?ng s? t�nh gi� v� c�c h? s? Greek.")

uploaded_file = st.file_uploader("?? T?i file Excel ho?c CSV", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("? D? li?u ban d?u:")
        st.dataframe(df)

        results = []
        for _, row in df.iterrows():
            try:
                result = greeks(row['S'], row['K'], row['T'], row['r'], row['sigma'], row['option_type'].lower())
                results.append(result)
            except:
                results.append({'Price': None, 'Delta': None, 'Gamma': None, 'Vega': None, 'Theta': None, 'Rho': None})
        
        greek_df = pd.DataFrame(results)
        final_df = pd.concat([df, greek_df], axis=1)

        st.success("? T�nh to�n ho�n t?t!")
        st.dataframe(final_df)
        st.markdown("---")
        st.subheader("?? �? th? tuong t�c - Ph�n t�ch bi?n d?ng gi� CW")

        selected_index = st.number_input("?? Ch?n d�ng CW d? kh?o s�t", min_value=0, max_value=len(final_df)-1, value=0, step=1)
        selected_row = final_df.iloc[selected_index]

        var_to_plot = st.selectbox("?? Ch?n bi?n d? ph�n t�ch", ['S', 'T', 'sigma'], index=0)

        # T?o ph?m vi gi� tr? d? v? d? th?
        if var_to_plot == 'S':
            x_range = np.linspace(selected_row['K'] * 0.6, selected_row['K'] * 1.4, 50)
        elif var_to_plot == 'T':
            x_range = np.linspace(0.01, 2, 50)
        elif var_to_plot == 'sigma':
            x_range = np.linspace(0.05, 1.0, 50)

        # T�nh l?i gi� theo bi?n d� ch?n
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
                      title=f"Bi?n d?ng Gi� CW theo {var_to_plot}",
                      labels={var_to_plot: var_to_plot, 'CW Price': 'Gi� CW'})
        st.plotly_chart(fig, use_container_width=True)

        # T?i v? file k?t qu?
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            final_df.to_excel(writer, index=False, sheet_name='CW Results')
        st.download_button("?? T?i k?t qu? Excel", data=output.getvalue(), file_name="cw_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"L?i x? l�: {e}")

st.markdown("---")
st.subheader("?? M� ph?ng l�i/l? CW d?n d�o h?n")

with st.expander("?? Nh?p th�ng s? m� ph?ng"):
    buy_price = st.number_input("?? Gi� mua CW (d?ng)", value=3000.0, step=100.0)
    ratio = st.number_input("?? T? l? chuy?n d?i (v� d? 5:1 nh?p 5)", value=5.0, min_value=0.1)
    fee = st.number_input("?? Ph� giao d?ch 1 chi?u (d?ng)", value=0.0, step=100.0)
    final_T = st.number_input("? Th?i gian c�n l?i d?n d�o h?n (nam)", value=selected_row['T'], min_value=0.01)
    K = selected_row['K']
    r = selected_row['r']
    sigma = selected_row['sigma']
    option_type = selected_row['option_type'].lower()

# T?o d?i gi� c? phi?u t?i d�o h?n
S_range = np.linspace(K * 0.6, K * 1.4, 100)
profit = []

for S_T in S_range:
    price_T, _, _ = bs_price(S_T, K, final_T, r, sigma, option_type)
    intrinsic_value = max(S_T - K, 0) if option_type == 'call' else max(K - S_T, 0)
    received = intrinsic_value / ratio
    pnl = received - buy_price - fee
    profit.append(pnl)

df_pnl = pd.DataFrame({'S_T (Gi� CP d�o h?n)': S_range, 'L�i/L? (VND)': profit})
fig_pnl = px.line(df_pnl, x='S_T (Gi� CP d�o h?n)', y='L�i/L? (VND)',
                  title='?? L?i nhu?n/L? khi n?m gi? CW d?n d�o h?n',
                  labels={'S_T (Gi� CP d�o h?n)': 'Gi� CP d?n d�o h?n', 'L�i/L? (VND)': 'L?i nhu?n/VND'})

st.plotly_chart(fig_pnl, use_container_width=True)

# Hi?n th? di?m h�a v?n
try:
    breakeven_idx = np.argmin(np.abs(np.array(profit)))
    breakeven_price = S_range[breakeven_idx]
    st.success(f"?? �i?m h�a v?n u?c t�nh: kho?ng **{breakeven_price:.2f} d?ng/c? phi?u**")
except:
    st.warning("Kh�ng t�m th?y di?m h�a v?n trong kho?ng gi� kh?o s�t.")



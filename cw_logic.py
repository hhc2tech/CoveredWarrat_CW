import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2

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

def simulate_pnl(S_range, K, T, r, sigma, option_type, buy_price, ratio, fee):
    profit = []
    profit_pct = []
    for S_T in S_range:
        price_T, _, _ = bs_price(S_T, K, T, r, sigma, option_type)
        intrinsic_value = max(S_T - K, 0) if option_type == 'call' else max(K - S_T, 0)
        received = intrinsic_value / ratio
        pnl = received - buy_price - fee
        pnl_pct = (pnl / (buy_price + fee)) * 100 if (buy_price + fee) > 0 else 0
        profit.append(pnl)
        profit_pct.append(pnl_pct)
    return profit, profit_pct
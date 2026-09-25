import numpy as np
from statistics import NormalDist

# =========================
# Données de marché (yfinance)
# =========================

import yfinance as yf


def search_tickers(query, max_results=8):
    """Cherche des tickers correspondant à `query` sur Yahoo Finance.
    Renvoie une liste de dicts {symbol, name, exchange}.
    """
    try:
        resultats = yf.Search(query, max_results=max_results).quotes
    except Exception:
        return []

    return [
        {
            "symbol": r.get("symbol"),
            "name": r.get("shortname") or r.get("longname") or r.get("symbol"),
            "exchange": r.get("exchange"),
        }
        for r in resultats
        if r.get("symbol")
    ]


def get_spot(ticker):
    """Dernier prix connu du sous-jacent."""
    return float(yf.Ticker(ticker).fast_info["last_price"])


def get_devise(ticker):
    """Devise de cotation du ticker (ex: 'USD', 'EUR')."""
    return yf.Ticker(ticker).fast_info["currency"]


def get_vix():
    """Dernier niveau du VIX (volatilité implicite du S&P 500), en points de %."""
    return float(yf.Ticker("^VIX").fast_info["last_price"])

# =========================
# Fonction Mathématique
# =========================

def Ncdf(x):
    return NormalDist().cdf(x)

def Npdf(x):
    return NormalDist().pdf(x)

def _d1_d2(S0, K, T, r, q, sigma):
    """Calcule d1 et d2 communs à Black-Scholes et aux grecques."""
    d1 = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

# =========================
# BLACK-SCHOLES
# =========================

def black_scholes(S0, K, T, r, q, sigma, option_type):
    
    d1, d2 = _d1_d2(S0, K, T, r, q, sigma)
    
    if option_type == "call":
        prix = S0 * np.exp(-q*T) * Ncdf(d1) - K * np.exp(-r*T) * Ncdf(d2)
    else:
        prix = K * np.exp(-r*T) * Ncdf(-d2) - S0 * np.exp(-q*T) * Ncdf(-d1)

    return prix


# =========================
# MONTE CARLO
# =========================

def monte_carlo(S0, K, T, r, q, sigma, option_type, n_simulations):

    Z = np.random.normal(0, 1, n_simulations)

    ST = S0 * np.exp(
        (r - q - sigma**2 / 2) * T
        + sigma * np.sqrt(T) * Z
    )

    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    prix = np.exp(-r*T) * np.mean(payoff)

    return prix


# =========================
# LONGSTAFF-SCHWARTZ
# OPTION AMÉRICAINE
# =========================

def longstaff_schwartz(S0, K, T, r, q, sigma, option_type, n_simulations, n_steps=100):

    dt = T / n_steps

    # Simulation des prix
    prix = np.zeros((n_simulations, n_steps + 1))
    prix[:, 0] = S0

    for t in range(1, n_steps + 1):

        Z = np.random.normal(0, 1, n_simulations)

        prix[:, t] = prix[:, t-1] * np.exp(
            (r - q - sigma**2 / 2) * dt
            + sigma * np.sqrt(dt) * Z
        )

    # Payoff
    if option_type == "call":
        payoff = np.maximum(prix - K, 0)
    else:
        payoff = np.maximum(K - prix, 0)

    cashflow = payoff[:, -1]

    # On remonte dans le temps
    for t in range(n_steps - 1, 0, -1):

        cashflow *= np.exp(-r * dt)

        itm = payoff[:, t] > 0

        if np.sum(itm) > 2:

            X = prix[itm, t]
            Y = cashflow[itm]

            # --- standardisation avant la régression ---
            X_mean = X.mean()
            X_std = X.std()
            X_norm = (X - X_mean) / X_std

            regression = np.polyfit(X_norm, Y, 2)
            continuation = np.polyval(regression, X_norm)

            exercice = payoff[itm, t] > continuation

            indices = np.where(itm)[0]
            cashflow[indices[exercice]] = payoff[indices[exercice], t]

    prix_lsm = np.mean(cashflow) * np.exp(-r * dt)

    # --- plancher d'exercice immédiat ---
    if option_type == "call":
        payoff_immediat = max(S0 - K, 0)
    else:
        payoff_immediat = max(K - S0, 0)

    return max(prix_lsm, payoff_immediat)

# =========================
# Grec Primaire
# =========================

def grecs_primaires_bs(S0, K, T, r, q, sigma, option_type):

    d1, d2 = _d1_d2(S0, K, T, r, q, sigma)
    
    if option_type == "call" :
        Delta = (np.exp(-q*T))*Ncdf(d1)
    else:
        Delta = (np.exp(-q*T))*(Ncdf(d1)-1)
        
    Gamma = ((np.exp(-q*T))*(Npdf(d1)))/(S0*sigma*np.sqrt(T))
    
    Vega = S0*np.exp(-q*T)*Npdf(d1)*np.sqrt(T)
    
    if option_type == "call" : 
        Theta = - (S0 * sigma * np.exp(-q*T) * Npdf(d1)) / (2 * np.sqrt(T))-r*K*np.exp(-r*T)*Ncdf(d2)+q*S0*np.exp(-q*T)*Ncdf(d1)
    else:
        Theta = -(S0*sigma*np.exp(-q*T)*Npdf(d1))/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*Ncdf(-d2) - q*S0*np.exp(-q*T)*Ncdf(-d1)
    
    if option_type == "call" :
        Rho = K*T*np.exp(-r*T)*Ncdf(d2)
    else:
        Rho = -K*T*np.exp(-r*T)*Ncdf(-d2)


    return {
        "Delta": Delta,
        "Gamma": Gamma,
        "Vega": Vega,
        "Theta": Theta,
        "Rho": Rho
    }

# =========================
# Grec Secondaire
# =========================

def grecs_secondaires_bs(S0, K, T, r, q, sigma, option_type):

    d1, d2 = _d1_d2(S0, K, T, r, q, sigma)
    Vega = S0*np.exp(-q*T)*Npdf(d1)*np.sqrt(T)
    Gamma = ((np.exp(-q*T))*(Npdf(d1)))/(S0*sigma*np.sqrt(T))
    
    if option_type == "call" :
        Delta = (np.exp(-q*T))*Ncdf(d1)
    else:
        Delta = (np.exp(-q*T))*(Ncdf(d1)-1)
     

    Vanna = (-np.exp(-q*T)*Npdf(d1)*d2)/sigma
    
    Vomma = (Vega*d1*d2)/sigma
    
    if option_type == "call" :
        Charm = q*np.exp(-q*T)*Ncdf(d1)-np.exp(-q*T)*Npdf(d1)*(2*(r-q)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
    else:
        Charm = -q*np.exp(-q*T)*Ncdf(-d1)-np.exp(-q*T)*Npdf(d1)*(2*(r-q)*T-d2*sigma*np.sqrt(T))/(2*T*sigma*np.sqrt(T))
        
    Color = -np.exp(-q*T)*Npdf(d1)/(2*S0*T*sigma*np.sqrt(T))*(2*q*T+1+(2*(r-q)*T-d2*sigma*np.sqrt(T)))*d1/(sigma*np.sqrt(T))

    Speed = -Gamma/S0*(d1/(sigma*np.sqrt(T))+1)
    
    Zomma = Gamma*(d1*d2-1)/sigma
    Ultima = -Vega*(d1*d2*(1-d1*d2)+d1**2+d2**2)/sigma**2
    prix = black_scholes(S0, K, T, r, q, sigma, option_type)
    Lambda = Delta * S0 / prix
    
    return {
        "Vanna": Vanna,
        "Vomma": Vomma,
        "Charm": Charm,
        "Color": Color,
        "Speed": Speed,
        "Zomma": Zomma,
        "Ultima": Ultima,
        "Lambda": Lambda
    }

# =========================
# Futures / Forward
# =========================

def forward_price(S0, T, r, q):
    """Prix à terme théorique F."""
    F=S0*np.exp((r-q)*T)
    return F

def forward_value(S0, K, T, r, q, side):
    """Valeur actuelle d'un contrat forward déjà engagé à K. side = 'buy' ou 'sell'."""
    F = forward_price(S0, T, r, q)

    if side =="buy":
        valeur = (F-K)*np.exp(-r*T)
    else:
        valeur = (K-F)*np.exp(-r*T)
    
    return valeur

def forward_payoff(ST, K, side):
    """Payoff à maturité pour un prix final ST donné. side = 'buy' ou 'sell'."""
    if side == "buy": 
        payoff = ST-K
    else:
        payoff = K-ST
    return payoff

# =========================
# Legs
# =========================

def make_position(option_type, side, K, T, qty=1):
    """Construit une leg option unique (couvre long/short call, long/short put)."""
    return {
        "instrument": "option",
        "option_type": option_type,
        "side": side,
        "K": K,
        "T": T,
        "qty": qty,
    }

def make_stock_position(side, qty=1):
    """Construit une leg action (sans strike, sans maturité, sans option_type)."""
    return {
        "instrument": "stock",
        "side": side,
        "qty": qty,
    }

def price_leg(leg, S0, r, q, sigma):
    """Calcule la prime d'engagement d'une seule leg, signée selon buy/sell."""
    sign = 1 if leg["side"] == "buy" else -1

    if leg["instrument"] == "stock":
        prix_unitaire = S0
    else:
        prix_unitaire = black_scholes(S0, leg["K"], leg["T"], r, q, sigma, leg["option_type"])

    return sign * leg["qty"] * prix_unitaire

# =========================
# 
# =========================

def payoff_leg(leg, ST):
    """Payoff à maturité d'une seule leg, pour une grille de prix finaux ST (tableau numpy)."""
    sign = 1 if leg["side"] == "buy" else -1

    if leg["instrument"] == "stock":
        payoff = ST
    else:
        if leg["option_type"] == "call":
            payoff = np.maximum(ST - leg["K"], 0)
        else:
            payoff = np.maximum(leg["K"] - ST, 0)

    return sign * leg["qty"] * payoff

def payoff_strategie(legs, S0, r, q, sigma, ST_grid):
    """Payoff et P&L agrégés de la stratégie entière, sur une grille de prix finaux."""
    payoff_total = np.zeros_like(ST_grid)
    cout_total = 0

    for leg in legs:
        payoff_total = payoff_total + payoff_leg(leg, ST_grid)
        cout_total = cout_total + price_leg(leg, S0, r, q, sigma)

    pnl = payoff_total - cout_total

    return {
        "payoff": payoff_total,
        "pnl": pnl
    }

def greeks_strategie(legs, S0, r, q, sigma):
    """Grecques primaires agrégées de la stratégie (Delta, Gamma, Vega, Theta, Rho)."""

    total = {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0}

    for leg in legs:
        sign = 1 if leg["side"] == "buy" else -1

        if leg["instrument"] == "stock":
            total["Delta"] = total["Delta"] + sign * leg["qty"]
            # une action n'a pas de Gamma/Vega/Theta/Rho, donc rien d'autre à faire ici
        else:
            grecs = grecs_primaires_bs(S0, leg["K"], leg["T"], r, q, sigma, leg["option_type"])
            total["Delta"] = total["Delta"] + sign * leg["qty"] * grecs["Delta"]
            total["Gamma"] = total["Gamma"] + sign * leg["qty"] * grecs["Gamma"]
            total["Vega"]  = total["Vega"]  + sign * leg["qty"] * grecs["Vega"]
            total["Theta"] = total["Theta"] + sign * leg["qty"] * grecs["Theta"]
            total["Rho"]   = total["Rho"]   + sign * leg["qty"] * grecs["Rho"]

    return total

# =========================
# Stratégies
# =========================

def make_covered_call(K, T, qty=1):
    """Action longue + call vendu."""
    return [
        make_stock_position("buy", qty=qty),
        make_position("call", "sell", K, T, qty=qty),
    ]

def make_protective_put(K, T, qty=1):
    """Action longue + put acheté (assurance à la baisse)."""
    return [
        make_stock_position("buy", qty=qty),
        make_position("put", "buy", K, T, qty=qty),
    ]

def make_straddle(K, T, side="buy", qty=1):
    """Call + put, même strike, même maturité, même sens."""
    return [
        make_position("call", side, K, T, qty=qty),
        make_position("put", side, K, T, qty=qty),
    ]

def make_strangle(K_put, K_call, T, side="buy", qty=1):
    """Call + put, strikes différents (K_put < K_call), même sens."""
    return [
        make_position("call", side, K_call, T, qty=qty),
        make_position("put", side, K_put, T, qty=qty),
    ]

def make_call_spread(K1, K2, T, qty=1):
    """Bull call spread : achat call K1 (bas), vente call K2 (haut), K1 < K2."""
    return [
        make_position("call", "buy", K1, T, qty=qty),
        make_position("call", "sell", K2, T, qty=qty),
    ]

def make_put_spread(K1, K2, T, qty=1):
    """Bear put spread : achat put K1 (haut), vente put K2 (bas), K1 > K2."""
    return [
        make_position("put", "buy", K1, T, qty=qty),
        make_position("put", "sell", K2, T, qty=qty),
    ]

def make_collar(K_put, K_call, T, qty=1):
    """Action longue + put acheté (K_put) + call vendu (K_call), K_put < K_call."""
    return [
        make_stock_position("buy", qty=qty),
        make_position("put", "buy", K_put, T, qty=qty),
        make_position("call", "sell", K_call, T, qty=qty),
    ]
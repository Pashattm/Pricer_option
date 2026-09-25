from flask import Flask, render_template, request, jsonify
import numpy as np

from pricer_engine import (
    black_scholes, monte_carlo, longstaff_schwartz,
    grecs_primaires_bs, grecs_secondaires_bs,
    forward_price, forward_value, forward_payoff,
    payoff_strategie, greeks_strategie,
    make_covered_call, make_protective_put, make_straddle, make_strangle,
    make_call_spread, make_put_spread, make_collar,
    search_tickers, get_spot, get_devise, get_vix,
)

app = Flask(__name__)


# =========================
# Utilitaires communs
# =========================

def _grid(centres, n_points=121, low=0.4, high=1.6):
    """Construit une grille de ST autour d'une liste de niveaux de référence
    (S0, strikes...), avec une marge de 40% en dessous et 60% au-dessus.
    """
    lo = max(0.01, min(centres) * low)
    hi = max(centres) * high
    return np.linspace(lo, hi, n_points)


def _find_breakevens(ST_grid, pnl):
    """Détecte les points où le P&L change de signe (interpolation linéaire),
    pour gérer les stratégies à plusieurs breakevens (straddle, strangle...).

    Interpolation manuelle (pas np.interp) : np.interp exige un deuxième
    argument croissant, or le P&L peut aussi bien passer de + à - que de -
    à + selon le breakeven — np.interp donnerait un résultat faux, sans
    erreur, dans le premier cas.
    """
    breakevens = []
    signs = np.sign(pnl)
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0:
            be = ST_grid[i] + (0 - pnl[i]) * (ST_grid[i + 1] - ST_grid[i]) / (pnl[i + 1] - pnl[i])
            breakevens.append(round(float(be), 2))
    return breakevens


def _err(message, code=400):
    return jsonify({"error": message}), code


# =========================
# Marché : recherche de ticker & données live
# =========================

@app.get("/api/search_ticker")
def api_search_ticker():
    query = request.args.get("q", "").strip()
    if len(query) < 1:
        return jsonify({"results": []})
    try:
        return jsonify({"results": search_tickers(query)})
    except Exception as e:
        return _err(str(e))


@app.get("/api/market/<ticker>")
def api_market(ticker):
    try:
        S0 = get_spot(ticker)
        devise = get_devise(ticker)
        vix = get_vix()
        return jsonify({
            "S0": round(S0, 4),
            "devise": devise,
            "vix": round(vix, 4),
        })
    except Exception as e:
        return _err(f"Impossible de récupérer les données pour '{ticker}' : {e}")


# =========================
# Page principale
# =========================

@app.get("/")
def index():
    return render_template("index.html")


# =========================
# Option européenne / américaine
# =========================

@app.post("/api/option")
def api_option():
    try:
        data = request.get_json(force=True)

        S0 = float(data["S0"])
        K = float(data["K"])
        T = float(data["T"])
        r = float(data["r"]) / 100
        q = float(data["q"]) / 100
        sigma = float(data["sigma"]) / 100
        option_type = data["option_type"]
        style = data["style"]
        side = data.get("side", "buy")
        method = data.get("method", "auto")
        n_simulations = int(data.get("n_simulations", 30000))

        if option_type not in ("call", "put"):
            return _err("Type d'option invalide.")
        if style not in ("european", "american"):
            return _err("Style d'exercice invalide.")
        if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return _err("Les paramètres doivent être strictement positifs.")

        if style == "european":
            if method == "monte_carlo":
                price = monte_carlo(S0, K, T, r, q, sigma, option_type, n_simulations)
                used = "monte_carlo"
            else:
                price = black_scholes(S0, K, T, r, q, sigma, option_type)
                used = "black_scholes"
        else:
            price = longstaff_schwartz(S0, K, T, r, q, sigma, option_type, n_simulations)
            used = "longstaff_schwartz"

        primaires = grecs_primaires_bs(S0, K, T, r, q, sigma, option_type)
        secondaires = grecs_secondaires_bs(S0, K, T, r, q, sigma, option_type)

        sign = 1 if side == "buy" else -1

        ST_grid = _grid([S0, K])
        if option_type == "call":
            payoff = np.maximum(ST_grid - K, 0)
        else:
            payoff = np.maximum(K - ST_grid, 0)
        payoff = sign * payoff
        pnl = payoff - sign * price

        breakevens = _find_breakevens(ST_grid, pnl)

        return jsonify({
            "price": round(float(price), 4),
            "model_used": used,
            "primaires": {k: round(float(v), 6) for k, v in primaires.items()},
            "secondaires": {k: round(float(v), 6) for k, v in secondaires.items()},
            "breakevens": breakevens,
            "ST_grid": [round(float(x), 4) for x in ST_grid],
            "payoff": [round(float(x), 4) for x in payoff],
            "pnl": [round(float(x), 4) for x in pnl],
        })

    except KeyError as e:
        return _err(f"Paramètre manquant : {e}")
    except Exception as e:
        return _err(str(e))


# =========================
# Forward / Future
# =========================

@app.post("/api/forward")
def api_forward():
    try:
        data = request.get_json(force=True)

        S0 = float(data["S0"])
        K = float(data["K"])
        T = float(data["T"])
        r = float(data["r"]) / 100
        q = float(data["q"]) / 100
        side = data.get("side", "buy")

        if S0 <= 0 or T <= 0:
            return _err("S0 et T doivent être strictement positifs.")

        F = forward_price(S0, T, r, q)
        valeur = forward_value(S0, K, T, r, q, side)

        ST_grid = _grid([S0, K])
        payoff = np.array([forward_payoff(st, K, side) for st in ST_grid])
        # Le P&L d'un forward par rapport à aujourd'hui, c'est le payoff à
        # maturité moins la valeur actuelle du contrat (le forward n'a pas de
        # "prime" au sens option ; il se règle contre la valeur déjà engagée).
        pnl = payoff - valeur

        breakevens = _find_breakevens(ST_grid, pnl)

        return jsonify({
            "forward_price": round(float(F), 4),
            "valeur_actuelle": round(float(valeur), 4),
            "breakevens": breakevens,
            "ST_grid": [round(float(x), 4) for x in ST_grid],
            "payoff": [round(float(x), 4) for x in payoff],
            "pnl": [round(float(x), 4) for x in pnl],
        })

    except KeyError as e:
        return _err(f"Paramètre manquant : {e}")
    except Exception as e:
        return _err(str(e))


# =========================
# Stratégies
# =========================

STRATEGY_BUILDERS = {
    "covered_call":   lambda p: make_covered_call(p["K"], p["T"], qty=p["qty"]),
    "protective_put": lambda p: make_protective_put(p["K"], p["T"], qty=p["qty"]),
    "straddle":       lambda p: make_straddle(p["K"], p["T"], side=p["side"], qty=p["qty"]),
    "strangle":       lambda p: make_strangle(p["K_put"], p["K_call"], p["T"], side=p["side"], qty=p["qty"]),
    "call_spread":    lambda p: make_call_spread(p["K1"], p["K2"], p["T"], qty=p["qty"]),
    "put_spread":     lambda p: make_put_spread(p["K1"], p["K2"], p["T"], qty=p["qty"]),
    "collar":         lambda p: make_collar(p["K_put"], p["K_call"], p["T"], qty=p["qty"]),
}

# Les strikes pertinents par stratégie, pour calibrer la grille de ST.
STRATEGY_STRIKES = {
    "covered_call":   ["K"],
    "protective_put": ["K"],
    "straddle":       ["K"],
    "strangle":       ["K_put", "K_call"],
    "call_spread":    ["K1", "K2"],
    "put_spread":     ["K1", "K2"],
    "collar":         ["K_put", "K_call"],
}


@app.post("/api/strategy")
def api_strategy():
    try:
        data = request.get_json(force=True)

        strategy = data.get("strategy")
        if strategy not in STRATEGY_BUILDERS:
            return _err("Stratégie inconnue.")

        S0 = float(data["S0"])
        r = float(data["r"]) / 100
        q = float(data["q"]) / 100
        sigma = float(data["sigma"]) / 100
        T = float(data["T"])
        qty = float(data.get("qty", 1))
        side = data.get("side", "buy")

        if S0 <= 0 or T <= 0 or sigma <= 0:
            return _err("Les paramètres doivent être strictement positifs.")

        params = {"T": T, "qty": qty, "side": side}
        for cle in STRATEGY_STRIKES[strategy]:
            params[cle] = float(data[cle])

        legs = STRATEGY_BUILDERS[strategy](params)

        centres = [S0] + [params[cle] for cle in STRATEGY_STRIKES[strategy]]
        ST_grid = _grid(centres)

        resultat = payoff_strategie(legs, S0, r, q, sigma, ST_grid)
        grecques = greeks_strategie(legs, S0, r, q, sigma)

        breakevens = _find_breakevens(ST_grid, resultat["pnl"])

        return jsonify({
            "greeks": {k: round(float(v), 6) for k, v in grecques.items()},
            "breakevens": breakevens,
            "ST_grid": [round(float(x), 4) for x in ST_grid],
            "payoff": [round(float(x), 4) for x in resultat["payoff"]],
            "pnl": [round(float(x), 4) for x in resultat["pnl"]],
            "n_legs": len(legs),
        })

    except KeyError as e:
        return _err(f"Paramètre manquant : {e}")
    except Exception as e:
        return _err(str(e))


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
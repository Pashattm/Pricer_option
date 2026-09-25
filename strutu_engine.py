import numpy as np


# =========================
# Simulation aux dates d'observation
# =========================

def simulate_observation_dates(S0, r, q, sigma, obs_dates, n_simulations):
    """
    Simule le prix du sous-jacent à chaque date d'observation.

    obs_dates : liste des dates d'observation en années, ex: [0.25, 0.5, 0.75, 1.0]
    Renvoie un tableau numpy de forme (n_simulations, len(obs_dates) + 1) :
    colonne 0 = S0, colonne i = prix à obs_dates[i-1].
    """
    n_dates = len(obs_dates)
    prix = np.zeros((n_simulations, n_dates + 1))
    prix[:, 0] = S0

    t_precedent = 0
    for i, t in enumerate(obs_dates):
        dt = t - t_precedent
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i+1] = prix[:, i] * np.exp(
            (r - q - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Z
        )
        t_precedent = t

    return prix


# =========================
# Phoenix générique (standard / memory / bonus / step-up / sans autocall / Custom Strike)
# =========================

def price_phoenix(S0, r, q, sigma, obs_dates, K_auto, B_phoenix, B_pdi, coupon,
                   n_simulations, memory=True, has_autocall=True,
                   bonus_threshold=None, bonus_amount=0.0, strike_pct=1.0):
    """
    Prix d'un Phoenix générique par Monte Carlo, couvrant :
    - Phoenix coupon standard   : memory=False
    - Phoenix memory            : memory=True
    - Phoenix bonus à maturité  : bonus_threshold + bonus_amount
    - Phoenix coupon-only autocall : memory=False, has_autocall=True
    - Phoenix sans autocall     : has_autocall=False
    - Phoenix step-up coupon    : coupon = liste/array croissante au lieu d'un nombre
    - Phoenix Custom Strike     : strike_pct != 1.0 (toutes les barrières sont
      recalculées en % de custom_strike = strike_pct * S0, pas de S0 directement)

    K_auto, B_phoenix, B_pdi : niveaux en % du strike de référence.
    coupon : un nombre (coupon fixe chaque période) ou un tableau de longueur
             len(obs_dates) (step-up).
    """
    n_dates = len(obs_dates)
    prix = simulate_observation_dates(S0, r, q, sigma, obs_dates, n_simulations)

    custom_strike = strike_pct * S0
    K_auto_level = K_auto * custom_strike
    B_phoenix_level = B_phoenix * custom_strike
    B_pdi_level = B_pdi * custom_strike

    if np.isscalar(coupon):
        coupon_schedule = np.full(n_dates, coupon)
    else:
        coupon_schedule = np.asarray(coupon)

    cum_coupon = np.concatenate(([0.0], np.cumsum(coupon_schedule)))

    cashflow = np.zeros(n_simulations)
    called = np.zeros(n_simulations, dtype=bool)
    last_coupon_date = np.zeros(n_simulations, dtype=int)

    for k in range(1, n_dates + 1):
        t_k = obs_dates[k-1]
        S_k = prix[:, k]
        discount = np.exp(-r * t_k)

        actifs = ~called

        # L'autocall ne s'applique qu'aux observations STRICTEMENT avant
        # l'échéance. À la toute dernière date, franchir le callable level
        # ne constitue pas un "rappel anticipé" — c'est simplement le
        # règlement normal à maturité (qui doit rester éligible au bonus,
        # au coupon final, etc.). Sans cette distinction, un produit qui
        # finit sa vie normalement au-dessus du callable level était traité
        # à tort comme rappelé, et perdait injustement son bonus.
        if has_autocall and k < n_dates:
            autocall_now = actifs & (S_k >= K_auto_level)
        else:
            autocall_now = np.zeros(n_simulations, dtype=bool)

        coupon_du = actifs & (S_k >= B_phoenix_level)

        if memory:
            montant_si_du = cum_coupon[k] - cum_coupon[last_coupon_date]
        else:
            montant_si_du = coupon_schedule[k-1]

        montant_coupon = np.where(coupon_du, montant_si_du, 0.0)

        cashflow += np.where(autocall_now, (1.0 + montant_coupon) * discount, 0.0)
        cashflow += np.where(actifs & ~autocall_now & coupon_du, montant_coupon * discount, 0.0)

        last_coupon_date = np.where(actifs & coupon_du, k, last_coupon_date)
        called = called | autocall_now

    S_T = prix[:, n_dates]
    discount_T = np.exp(-r * obs_dates[-1])
    non_rappelees = ~called

    remboursement_T = np.where(S_T >= B_pdi_level, 1.0, S_T / custom_strike)

    if bonus_threshold is not None:
        bonus_du = non_rappelees & (S_T >= bonus_threshold * custom_strike)
        remboursement_T = remboursement_T + np.where(bonus_du, bonus_amount, 0.0)

    cashflow += np.where(non_rappelees, remboursement_T * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# Zenith (barrières et coupons étagés)
# =========================

def price_zenith(S0, r, q, sigma, obs_dates, K_levels, C_levels, B_pdi,
                  n_simulations, strike_pct=1.0):
    """
    Prix d'un Zenith par Monte Carlo : plusieurs barrières emboîtées K1 < K2 < ... < Kn,
    chacune associée à un coupon C1 < C2 < ... < Cn. Seul le palier le plus haut
    franchi déclenche l'autocall (+ le coupon du palier le plus haut) ; les paliers
    intermédiaires ne versent qu'un coupon, sans rappel.

    K_levels, C_levels : listes de même longueur, dans l'ordre croissant.
    """
    assert len(K_levels) == len(C_levels), "K_levels et C_levels doivent avoir la même longueur"

    n_dates = len(obs_dates)
    prix = simulate_observation_dates(S0, r, q, sigma, obs_dates, n_simulations)

    custom_strike = strike_pct * S0
    K_levels_abs = [k * custom_strike for k in K_levels]
    B_pdi_level = B_pdi * custom_strike
    n_paliers = len(K_levels_abs)

    cashflow = np.zeros(n_simulations)
    called = np.zeros(n_simulations, dtype=bool)

    for k in range(1, n_dates + 1):
        t_k = obs_dates[k-1]
        S_k = prix[:, k]
        discount = np.exp(-r * t_k)
        actifs = ~called

        montant_coupon = np.zeros(n_simulations)
        palier_du = np.zeros(n_simulations, dtype=bool)
        autocall_now = np.zeros(n_simulations, dtype=bool)

        # On part du palier le plus haut et on redescend : le premier palier
        # franchi (en partant du haut) est celui qui s'applique à cette trajectoire.
        for i in range(n_paliers - 1, -1, -1):
            franchi = actifs & ~palier_du & (S_k >= K_levels_abs[i])
            montant_coupon = np.where(franchi, C_levels[i], montant_coupon)
            palier_du = palier_du | franchi
            if i == n_paliers - 1:
                autocall_now = franchi  # seul le palier le plus haut déclenche le rappel

        cashflow += np.where(autocall_now, (1.0 + montant_coupon) * discount, 0.0)
        cashflow += np.where(actifs & ~autocall_now & palier_du, montant_coupon * discount, 0.0)

        called = called | autocall_now

    S_T = prix[:, n_dates]
    discount_T = np.exp(-r * obs_dates[-1])
    non_rappelees = ~called

    remboursement_T = np.where(S_T >= B_pdi_level, 1.0, S_T / custom_strike)
    cashflow += np.where(non_rappelees, remboursement_T * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# Phoenix DM (Double Memory)
# =========================

def price_phoenix_dm(S0, r, q, sigma, obs_dates, K_auto, B_phoenix, B_pdi, coupon,
                      n_simulations, dm_version="A", dm_window=3, dm_stat="mean",
                      K_auto_memory=None, memory=True, strike_pct=1.0):
    """
    Prix d'un Phoenix DM par Monte Carlo.

    dm_version "A" : mémoire sur l'autocall — le produit est aussi rappelé si la
        moyenne (ou le max, selon dm_stat) des `dm_window` dernières observations
        dépasse K_auto_memory, un seuil DISTINCT et plus bas que K_auto (sinon
        cette condition ne peut mathématiquement jamais se déclencher en plus de
        la condition ponctuelle : la moyenne de plusieurs points ne dépasse jamais
        leur maximum, donc si la moyenne franchit K_auto, un point l'aurait déjà
        franchi individuellement à une date antérieure). Si K_auto_memory n'est
        pas fourni, on prend K_auto par défaut, ce qui revient alors à désactiver
        la mémoire (comportement identique à un Phoenix standard) — à toi de
        fixer un niveau plus bas pour que la fonctionnalité soit active.
    dm_version "B" : mémoire sur la barrière coupon — le coupon est décidé sur la
        moyenne (ou le max) des `dm_window` dernières observations plutôt que sur
        le seul fixing du jour. Ce cas-ci reste valide avec le même seuil
        B_phoenix, car il ne s'agit pas d'une condition "OU" avec le point actuel,
        mais d'un remplacement : on ne regarde plus jamais le point seul.
    dm_window : nombre d'observations récentes prises en compte.
    dm_stat   : "mean" ou "max".
    """
    n_dates = len(obs_dates)
    prix = simulate_observation_dates(S0, r, q, sigma, obs_dates, n_simulations)

    custom_strike = strike_pct * S0
    K_auto_level = K_auto * custom_strike
    K_auto_memory_level = (K_auto_memory if K_auto_memory is not None else K_auto) * custom_strike
    B_phoenix_level = B_phoenix * custom_strike
    B_pdi_level = B_pdi * custom_strike

    if np.isscalar(coupon):
        coupon_schedule = np.full(n_dates, coupon)
    else:
        coupon_schedule = np.asarray(coupon)
    cum_coupon = np.concatenate(([0.0], np.cumsum(coupon_schedule)))

    cashflow = np.zeros(n_simulations)
    called = np.zeros(n_simulations, dtype=bool)
    last_coupon_date = np.zeros(n_simulations, dtype=int)

    def stat_fenetre(k):
        """Moyenne (ou max) du prix sur les `dm_window` dernières dates (1..k incluses)."""
        debut = max(1, k - dm_window + 1)
        fenetre = prix[:, debut:k+1]
        return fenetre.max(axis=1) if dm_stat == "max" else fenetre.mean(axis=1)

    for k in range(1, n_dates + 1):
        t_k = obs_dates[k-1]
        S_k = prix[:, k]
        discount = np.exp(-r * t_k)
        actifs = ~called

        stat_k = stat_fenetre(k)

        if dm_version == "A":
            autocall_now = actifs & ((S_k >= K_auto_level) | (stat_k >= K_auto_memory_level))
            coupon_du = actifs & (S_k >= B_phoenix_level)
        else:
            autocall_now = actifs & (S_k >= K_auto_level)
            coupon_du = actifs & (stat_k >= B_phoenix_level)

        if memory:
            montant_si_du = cum_coupon[k] - cum_coupon[last_coupon_date]
        else:
            montant_si_du = coupon_schedule[k-1]

        montant_coupon = np.where(coupon_du, montant_si_du, 0.0)

        cashflow += np.where(autocall_now, (1.0 + montant_coupon) * discount, 0.0)
        cashflow += np.where(actifs & ~autocall_now & coupon_du, montant_coupon * discount, 0.0)

        last_coupon_date = np.where(actifs & coupon_du, k, last_coupon_date)
        called = called | autocall_now

    S_T = prix[:, n_dates]
    discount_T = np.exp(-r * obs_dates[-1])
    non_rappelees = ~called
    remboursement_T = np.where(S_T >= B_pdi_level, 1.0, S_T / custom_strike)
    cashflow += np.where(non_rappelees, remboursement_T * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# Phoenix Japan 2 (coupon journalier + PDI américaine)
# =========================

def price_phoenix_japan2(S0, r, q, sigma, T, obs_dates,
                          K_auto, B_coupon_low, B_pdi, coupon,
                          n_simulations, strike_pct=1.0, trading_days_per_year=252):
    """
    Prix d'un Phoenix Japan 2 par Monte Carlo à fréquence journalière.

    obs_dates       : dates d'observation de l'autocall (en années), comme price_phoenix.
                      Les périodes de coupon sont délimitées directement par ces dates
                      (traduites en indices de jours sur la grille journalière) — aucun
                      paramètre séparé de "jours par période" n'est nécessaire, la grille
                      journalière donne déjà le nombre exact de jours dans chaque période.
    B_coupon_low    : barrière de coupon basse, observée quotidiennement, en % du strike.
    B_pdi           : barrière PDI américaine — franchie une seule fois = activée
                      définitivement pour le reste de la vie du produit.
    coupon          : coupon maximal (plein) par période ; versé en proportion n/N des
                      jours où le sous-jacent est au-dessus de B_coupon_low.

    Hypothèse : grille journalière régulière sur toute la durée T (trading_days_per_year
    jours par an) ; les dates d'obs_dates sont arrondies au jour le plus proche de cette
    grille pour déterminer les dates d'autocall.
    """
    n_days = int(round(T * trading_days_per_year))
    dt = T / n_days

    prix = np.zeros((n_simulations, n_days + 1))
    prix[:, 0] = S0
    for i in range(1, n_days + 1):
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i] = prix[:, i-1] * np.exp((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

    custom_strike = strike_pct * S0
    K_auto_level = K_auto * custom_strike
    B_coupon_level = B_coupon_low * custom_strike
    B_pdi_level = B_pdi * custom_strike

    # PDI américaine : franchie si le spot est passé sous la barrière n'importe quel jour.
    pdi_touchee = (prix <= B_pdi_level).any(axis=1)

    cashflow = np.zeros(n_simulations)
    called = np.zeros(n_simulations, dtype=bool)

    obs_days = [int(round(t / dt)) for t in obs_dates]

    jour_debut_periode = 0
    for k, jour_fin in enumerate(obs_days, start=1):
        t_k = obs_dates[k-1]
        discount = np.exp(-r * t_k)
        actifs = ~called

        periode = prix[:, jour_debut_periode+1:jour_fin+1]
        n_jours_periode = periode.shape[1]
        jours_au_dessus = (periode >= B_coupon_level).sum(axis=1)
        fraction = jours_au_dessus / n_jours_periode
        montant_coupon = fraction * coupon

        S_k = prix[:, jour_fin]
        autocall_now = actifs & (S_k >= K_auto_level)

        cashflow += np.where(autocall_now, (1.0 + montant_coupon) * discount, 0.0)
        cashflow += np.where(actifs & ~autocall_now, montant_coupon * discount, 0.0)

        called = called | autocall_now
        jour_debut_periode = jour_fin

    non_rappelees = ~called
    S_T = prix[:, n_days]
    discount_T = np.exp(-r * T)

    # Une fois la PDI touchée, le capital est exposé à la baisse du sous-jacent
    # (S_T/strike), mais ça reste une protection dégradée, pas une participation
    # à la hausse : le remboursement ne doit jamais dépasser 100% même si le
    # sous-jacent a fini par remonter au-dessus du strike après avoir franchi
    # la barrière en cours de vie.
    remboursement_T = np.where(pdi_touchee & non_rappelees, np.minimum(S_T / custom_strike, 1.0), 1.0)
    cashflow += np.where(non_rappelees, remboursement_T * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# Plain Autocall (+ variantes no KI / no KO / no KO-KI)
# =========================

def price_plain_autocall(S0, r, q, sigma, obs_dates, K_auto, B_pdi, coupon_annuel,
                          n_simulations, has_autocall=True, has_pdi=True, strike_pct=1.0):
    """
    Prix d'un Plain Autocall par Monte Carlo.

    Contrairement au Phoenix, il n'y a pas de coupon périodique : le coupon
    est cumulatif, versé en une fois à la date où le produit se termine
    (rappel anticipé OU maturité), proportionnel au nombre d'années écoulées :
    paiement = 100% + k·C, où k = temps écoulé en années à cette date.

    has_autocall=False -> "Autocall no KO" : jamais de rappel anticipé, le
        produit va systématiquement à maturité (Reverse Convertible incrémental).
    has_pdi=False -> "Autocall no KI" : pas de barrière de perte, capital
        toujours protégé à 100% si jamais rappelé.
    Les deux à False -> "Autocall no KO/KI" : note linéaire sans optionalité
        (coupon et capital garantis, aucune dépendance au sous-jacent).
    """
    n_dates = len(obs_dates)
    prix = simulate_observation_dates(S0, r, q, sigma, obs_dates, n_simulations)

    custom_strike = strike_pct * S0
    K_auto_level = K_auto * custom_strike
    B_pdi_level = B_pdi * custom_strike

    cashflow = np.zeros(n_simulations)
    called = np.zeros(n_simulations, dtype=bool)

    if has_autocall:
        for k in range(1, n_dates + 1):
            t_k = obs_dates[k-1]
            S_k = prix[:, k]
            discount = np.exp(-r * t_k)
            actifs = ~called

            autocall_now = actifs & (S_k >= K_auto_level)
            cashflow += np.where(autocall_now, (1.0 + t_k * coupon_annuel) * discount, 0.0)
            called = called | autocall_now

    non_rappeles = ~called
    T = obs_dates[-1]
    S_T = prix[:, n_dates]
    discount_T = np.exp(-r * T)

    if has_pdi:
        remboursement = np.where(S_T >= B_pdi_level, 1.0, S_T / custom_strike)
    else:
        remboursement = np.ones(n_simulations)

    cashflow += np.where(non_rappeles, (remboursement + T * coupon_annuel) * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# Autocall Ben (booster / step-down)
# =========================

def price_autocall_ben(S0, r, q, sigma, obs_dates, K0, delta_K, B_pdi, coupon_annuel,
                        n_simulations, has_pdi=True, strike_pct=1.0):
    """
    Prix d'un Autocall Ben (Booster Enhanced Note) par Monte Carlo — un Plain
    Autocall dont le callable level décroît à chaque date d'observation :

        K_auto(t_k) = K0 - (k-1)·delta_K

    (k = indice de la date d'observation, 1 à la première date). Le reste de
    la mécanique (coupon cumulatif, PDI à maturité) est identique au Plain
    Autocall.
    """
    n_dates = len(obs_dates)
    prix = simulate_observation_dates(S0, r, q, sigma, obs_dates, n_simulations)

    custom_strike = strike_pct * S0
    B_pdi_level = B_pdi * custom_strike

    cashflow = np.zeros(n_simulations)
    called = np.zeros(n_simulations, dtype=bool)

    for k in range(1, n_dates + 1):
        t_k = obs_dates[k-1]
        S_k = prix[:, k]
        discount = np.exp(-r * t_k)
        actifs = ~called

        K_auto_level_k = (K0 - (k - 1) * delta_K) * custom_strike
        autocall_now = actifs & (S_k >= K_auto_level_k)
        cashflow += np.where(autocall_now, (1.0 + t_k * coupon_annuel) * discount, 0.0)
        called = called | autocall_now

    non_rappeles = ~called
    T = obs_dates[-1]
    S_T = prix[:, n_dates]
    discount_T = np.exp(-r * T)

    if has_pdi:
        remboursement = np.where(S_T >= B_pdi_level, 1.0, S_T / custom_strike)
    else:
        remboursement = np.ones(n_simulations)

    cashflow += np.where(non_rappeles, (remboursement + T * coupon_annuel) * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# Autocall Daily (observation journalière du rappel)
# =========================

def price_autocall_daily(S0, r, q, sigma, T, K_auto, B_pdi, coupon_annuel, n_simulations,
                          guaranteed_period=0.0, strike_pct=1.0, trading_days_per_year=252):
    """
    Prix d'un Autocall Daily par Monte Carlo à fréquence journalière.

    Même mécanique de coupon cumulatif que le Plain Autocall (100% + k·C au
    rappel, k en années écoulées), mais le rappel est testé CHAQUE JOUR de
    bourse (au lieu de quelques dates espacées) dès la fin de la
    guaranteed_period. Dès qu'un jour dépasse le callable level, le produit
    est rappelé immédiatement.
    """
    n_days = int(round(T * trading_days_per_year))
    dt = T / n_days

    prix = np.zeros((n_simulations, n_days + 1))
    prix[:, 0] = S0
    for i in range(1, n_days + 1):
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i] = prix[:, i-1] * np.exp((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

    custom_strike = strike_pct * S0
    K_auto_level = K_auto * custom_strike
    B_pdi_level = B_pdi * custom_strike

    debut = int(round(guaranteed_period / dt)) if guaranteed_period > 0 else 0
    debut = min(debut, n_days)

    # Détection vectorisée du premier jour de franchissement (plus rapide
    # qu'une boucle Python jour par jour sur ~1250 jours) : argmax sur un
    # tableau booléen renvoie l'indice du premier True.
    fenetre = prix[:, debut:] >= K_auto_level
    a_franchi = fenetre.any(axis=1)
    premier_jour_relatif = np.argmax(fenetre, axis=1)
    jour_rappel = np.where(a_franchi, premier_jour_relatif + debut, -1)

    called = a_franchi
    t_rappel = jour_rappel * dt

    cashflow = np.zeros(n_simulations)
    discount_rappel = np.exp(-r * t_rappel)
    cashflow += np.where(called, (1.0 + t_rappel * coupon_annuel) * discount_rappel, 0.0)

    non_rappeles = ~called
    S_T = prix[:, n_days]
    discount_T = np.exp(-r * T)
    remboursement = np.where(S_T >= B_pdi_level, 1.0, S_T / custom_strike)
    cashflow += np.where(non_rappeles, (remboursement + T * coupon_annuel) * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# Autocall Asia (observation moyenne)
# =========================

def price_autocall_asia(S0, r, q, sigma, obs_dates, K_auto, B_pdi, coupon_annuel,
                         n_simulations, avg_days=20, average_pdi=True,
                         strike_pct=1.0, trading_days_per_year=252):
    """
    Prix d'un Autocall Asia par Monte Carlo à fréquence journalière.

    Le niveau comparé au callable level (et, si average_pdi=True, à la PDI
    à maturité) n'est pas le fixing ponctuel S_t, mais la moyenne des
    `avg_days` derniers jours de bourse — ce qui lisse le sous-jacent et
    réduit sa volatilité effective (protège la PDI, pénalise l'autocall).
    """
    T = obs_dates[-1]
    n_days = int(round(T * trading_days_per_year))
    dt = T / n_days

    prix = np.zeros((n_simulations, n_days + 1))
    prix[:, 0] = S0
    for i in range(1, n_days + 1):
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i] = prix[:, i-1] * np.exp((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

    custom_strike = strike_pct * S0
    K_auto_level = K_auto * custom_strike
    B_pdi_level = B_pdi * custom_strike

    called = np.zeros(n_simulations, dtype=bool)
    cashflow = np.zeros(n_simulations)
    obs_days = [int(round(t / dt)) for t in obs_dates]

    for k, jour in enumerate(obs_days, start=1):
        t_k = obs_dates[k-1]
        debut = max(0, jour - avg_days + 1)
        moyenne = prix[:, debut:jour+1].mean(axis=1)
        actifs = ~called

        autocall_now = actifs & (moyenne >= K_auto_level)
        discount = np.exp(-r * t_k)
        cashflow += np.where(autocall_now, (1.0 + t_k * coupon_annuel) * discount, 0.0)
        called = called | autocall_now

    non_rappeles = ~called
    if average_pdi:
        debut_T = max(0, n_days - avg_days + 1)
        S_T_ref = prix[:, debut_T:].mean(axis=1)
    else:
        S_T_ref = prix[:, n_days]

    discount_T = np.exp(-r * T)
    remboursement = np.where(S_T_ref >= B_pdi_level, 1.0, S_T_ref / custom_strike)
    cashflow += np.where(non_rappeles, (remboursement + T * coupon_annuel) * discount_T, 0.0)

    return np.mean(cashflow)


# =========================
# DRA — Daily Range Accrual (standard + mémoire)
# =========================

def price_dra(S0, r, q, sigma, T, B_down, B_up, coupon_max_periode, n_periodes,
              n_simulations, memory=False, strike_pct=1.0, trading_days_per_year=252):
    """
    Prix d'un DRA (Daily Range Accrual) par Monte Carlo à fréquence journalière.

    Coupon de la période k = (n_k/N_k) * coupon_max_periode, où n_k = nombre
    de jours dans [B_down, B_up] durant la période, N_k = nombre total de
    jours de la période. Capital toujours remboursé à 100% à maturité (pas
    de PDI dans la version standard).

    memory=True (DRA Daily à mémoire) : la fraction non payée d'une période
    est mise en réserve et peut être rattrapée lors d'une période ultérieure
    plus favorable — cf. cours III.II : Mémoire_k = M_{k-1} + (1-n_k/N_k)*C,
    Coupon_k = min(n_k/N_k*C + M_{k-1}, C). Le cours note que la règle exacte
    de plafonnement du rattrapage "varie selon les émetteurs" ; on retient
    ici la version la plus simple et auto-cohérente : la mémoire non
    consommée (parce que le versement est plafonné à un coupon plein par
    période) est reportée à la période suivante.
    """
    n_days = int(round(T * trading_days_per_year))
    dt = T / n_days

    prix = np.zeros((n_simulations, n_days + 1))
    prix[:, 0] = S0
    for i in range(1, n_days + 1):
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i] = prix[:, i-1] * np.exp((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

    custom_strike = strike_pct * S0
    B_down_level = B_down * custom_strike
    B_up_level = B_up * custom_strike

    jours_par_periode = n_days // n_periodes
    memoire = np.zeros(n_simulations)
    cashflow = np.zeros(n_simulations)

    for k in range(1, n_periodes + 1):
        debut = (k - 1) * jours_par_periode + 1
        fin = n_days if k == n_periodes else k * jours_par_periode
        fenetre = prix[:, debut:fin+1]
        N_k = fenetre.shape[1]
        n_k = ((fenetre >= B_down_level) & (fenetre <= B_up_level)).sum(axis=1)
        fraction_k = n_k / N_k

        t_k = fin * dt
        discount = np.exp(-r * t_k)

        if memory:
            # Paiement : le manque des périodes précédentes peut compléter
            # le coupon de cette période, plafonné à un coupon plein (C).
            payable = fraction_k * coupon_max_periode + memoire
            coupon_k = np.minimum(payable, coupon_max_periode)
            memoire_consommee = coupon_k - fraction_k * coupon_max_periode  # >= 0
            # Mémoire pour la période suivante : ce qui restait après
            # paiement, PLUS le nouveau manque propre à cette période
            # (formule du cours : Mémoire_k = M_{k-1} + (1-fraction_k)·C).
            memoire = (memoire - memoire_consommee) + (1 - fraction_k) * coupon_max_periode
        else:
            coupon_k = fraction_k * coupon_max_periode

        cashflow += coupon_k * discount

    cashflow += 1.0 * np.exp(-r * T)  # capital à 100%, pas de PDI dans le DRA standard
    return np.mean(cashflow)


# =========================
# Double Decker (deux ranges emboîtés)
# =========================

def price_double_decker(S0, r, q, sigma, T, B_down1, B_up1, C1, B_down2, B_up2, C2,
                         n_periodes, n_simulations, strike_pct=1.0, trading_days_per_year=252):
    """
    Prix d'un Double Decker (Layered Range) par Monte Carlo à fréquence
    journalière. Coupon de la période = (n1/N)*C1 + (n2/N)*C2, où n1 = jours
    dans le range intérieur [B_down1,B_up1] (coupon C1, le plus élevé), n2 =
    jours dans le range extérieur [B_down2,B_up2] mais hors de l'intérieur
    (coupon C2 < C1). Hors des deux ranges : rien. Capital 100% à maturité.
    """
    n_days = int(round(T * trading_days_per_year))
    dt = T / n_days

    prix = np.zeros((n_simulations, n_days + 1))
    prix[:, 0] = S0
    for i in range(1, n_days + 1):
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i] = prix[:, i-1] * np.exp((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

    custom_strike = strike_pct * S0
    B_down1_level, B_up1_level = B_down1 * custom_strike, B_up1 * custom_strike
    B_down2_level, B_up2_level = B_down2 * custom_strike, B_up2 * custom_strike

    jours_par_periode = n_days // n_periodes
    cashflow = np.zeros(n_simulations)

    for k in range(1, n_periodes + 1):
        debut = (k - 1) * jours_par_periode + 1
        fin = n_days if k == n_periodes else k * jours_par_periode
        fenetre = prix[:, debut:fin+1]
        N_k = fenetre.shape[1]

        dans_interieur = (fenetre >= B_down1_level) & (fenetre <= B_up1_level)
        dans_exterieur = (fenetre >= B_down2_level) & (fenetre <= B_up2_level) & ~dans_interieur
        n1 = dans_interieur.sum(axis=1)
        n2 = dans_exterieur.sum(axis=1)

        t_k = fin * dt
        discount = np.exp(-r * t_k)
        cashflow += (n1/N_k * C1 + n2/N_k * C2) * discount

    cashflow += 1.0 * np.exp(-r * T)
    return np.mean(cashflow)


# =========================
# Accumulator (Koda) — GP, leverage GP/NGP, Bonus Coupon, Limited Loss
# =========================

def price_accumulator(S0, r, q, sigma, T, strike_pct, ko_pct, n_shares_per_day,
                       n_simulations, guaranteed_period=0.0, leverage_gp=2.0,
                       leverage_normal=2.0, bonus_coupon=0.0, stop_loss_pct=None,
                       trading_days_per_year=252):
    """
    Valeur nette (en % du notionnel de référence n·N·S0) d'un Accumulator
    (Koda) par Monte Carlo, décomposé jour par jour comme le fait le cours
    (V.I.c) : chaque jour est équivalent à Long call(K,KO) - Short 2·put(K,KO)
    à expiration journalière — chaque achat est donc valorisé à SA propre
    date, actualisée, indépendamment de ce qui se passe ensuite.

    Couvre plusieurs variantes via les paramètres :
    - guaranteed_period > 0 : "Guaranteed Period" — le Knock-Out est
      désactivé pendant cette période initiale.
    - leverage_gp / leverage_normal : facteur d'achat doublé (2 par défaut)
      quand S_j < strike, pouvant différer pendant/hors GP -> couvre
      "GP leverage / NGP leverage".
    - bonus_coupon > 0 : coupon cash additionnel (en % du notionnel de
      référence), versé et actualisé à la date de déclenchement du KO —
      "Bonus Coupon".
    - stop_loss_pct : barrière basse symétrique qui arrête aussi
      l'accumulation si franchie, plafonnant la perte maximale — "Limited
      Loss Accumulator". None = pas de plafond (comportement standard,
      cf. cours : "sans aucune limite de perte explicite").

    Non modélisé : "Immediate Redemption" (livraison immédiate à la date de
    KO plutôt qu'à maturité) est une différence de TIMING DE RÈGLEMENT
    PHYSIQUE, pas de valorisation risque-neutre — la décomposition en
    options à expiration journalière du cours valorise déjà chaque achat à
    sa propre date d'achat, donc cette variante ne change pas le prix ici.
    """
    n_days = int(round(T * trading_days_per_year))
    dt = T / n_days

    prix = np.zeros((n_simulations, n_days + 1))
    prix[:, 0] = S0
    for i in range(1, n_days + 1):
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i] = prix[:, i-1] * np.exp((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

    K = strike_pct * S0
    B_KO = ko_pct * S0
    B_stop = stop_loss_pct * S0 if stop_loss_pct is not None else None
    guaranteed_days = int(round(guaranteed_period / dt)) if guaranteed_period > 0 else 0

    active = np.ones(n_simulations, dtype=bool)
    ko_triggered = np.zeros(n_simulations, dtype=bool)
    ko_day = np.zeros(n_simulations, dtype=int)
    valeur = np.zeros(n_simulations)

    for j in range(1, n_days + 1):
        S_j = prix[:, j]
        discount = np.exp(-r * j * dt)

        leverage = leverage_gp if j <= guaranteed_days else leverage_normal
        parts = np.where(S_j < K, leverage, 1.0) * n_shares_per_day

        valeur += np.where(active, parts * (S_j - K) * discount, 0.0)

        ko_now = active & (j > guaranteed_days) & (S_j >= B_KO)
        stop_now = active & (S_j <= B_stop) if B_stop is not None else np.zeros(n_simulations, dtype=bool)

        ko_triggered = ko_triggered | ko_now
        ko_day = np.where(ko_now, j, ko_day)
        active = active & ~ko_now & ~stop_now

    if bonus_coupon > 0:
        t_ko = np.where(ko_triggered, ko_day * dt, T)
        valeur += np.where(ko_triggered, bonus_coupon * np.exp(-r * t_ko), 0.0)

    notionnel_ref = n_shares_per_day * n_days * S0
    return np.mean(valeur) / notionnel_ref


def price_decumulator(S0, r, q, sigma, T, strike_pct, ko_pct, n_shares_per_day,
                       n_simulations, guaranteed_period=0.0, leverage_gp=2.0,
                       leverage_normal=2.0, bonus_coupon=0.0, stop_loss_pct=None,
                       trading_days_per_year=252):
    """
    Miroir exact de price_accumulator (cf. cours V.III, tableau de synthèse) :
    l'investisseur VEND n actions/jour au strike K (typiquement au-DESSUS du
    spot), double la vente à 2n si S_j > K (au lieu de < K), et le
    Knock-Out se déclenche à la BAISSE (S_j <= B_KO, avec B_KO < spot) au
    lieu de la hausse. Mêmes paramètres et mêmes variantes que l'Accumulator.
    """
    n_days = int(round(T * trading_days_per_year))
    dt = T / n_days

    prix = np.zeros((n_simulations, n_days + 1))
    prix[:, 0] = S0
    for i in range(1, n_days + 1):
        Z = np.random.normal(0, 1, n_simulations)
        prix[:, i] = prix[:, i-1] * np.exp((r - q - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

    K = strike_pct * S0
    B_KO = ko_pct * S0
    B_stop = stop_loss_pct * S0 if stop_loss_pct is not None else None
    guaranteed_days = int(round(guaranteed_period / dt)) if guaranteed_period > 0 else 0

    active = np.ones(n_simulations, dtype=bool)
    ko_triggered = np.zeros(n_simulations, dtype=bool)
    ko_day = np.zeros(n_simulations, dtype=int)
    valeur = np.zeros(n_simulations)

    for j in range(1, n_days + 1):
        S_j = prix[:, j]
        discount = np.exp(-r * j * dt)

        leverage = leverage_gp if j <= guaranteed_days else leverage_normal
        parts = np.where(S_j > K, leverage, 1.0) * n_shares_per_day

        valeur += np.where(active, parts * (K - S_j) * discount, 0.0)

        ko_now = active & (j > guaranteed_days) & (S_j <= B_KO)
        stop_now = active & (S_j >= B_stop) if B_stop is not None else np.zeros(n_simulations, dtype=bool)

        ko_triggered = ko_triggered | ko_now
        ko_day = np.where(ko_now, j, ko_day)
        active = active & ~ko_now & ~stop_now

    if bonus_coupon > 0:
        t_ko = np.where(ko_triggered, ko_day * dt, T)
        valeur += np.where(ko_triggered, bonus_coupon * np.exp(-r * t_ko), 0.0)

    notionnel_ref = n_shares_per_day * n_days * S0
    return np.mean(valeur) / notionnel_ref


# =========================
# Issuer Callable Airbag
# =========================

def price_issuer_callable_airbag(S0, r, q, sigma, obs_dates, K_call, B_airbag, coupon_annuel,
                                  n_simulations, strike_pct=1.0):
    """
    Prix d'un Issuer Callable Airbag par Monte Carlo.

    APPROXIMATION IMPORTANTE : le vrai rappel de la banque est discrétionnaire
    et suit une politique d'exercice optimal côté émetteur (le cours suggère
    un Monte Carlo américain façon Longstaff-Schwartz, mais du point de vue
    de la banque) — un moteur distinct de tout ce qu'on a construit jusqu'ici.
    Ce n'est PAS implémenté ici : on approxime le rappel par un déclenchement
    mécanique (même mécanique que l'autocall standard, K_call), pas par une
    vraie décision optimisée de la banque. À ne pas utiliser comme prix de
    référence pour un vrai produit issuer callable.

    Coupon cumulatif à l'identique du Plain Autocall (100% + k·C au rappel).
    Si jamais rappelé, le payoff à maturité utilise la formule airbag :
    100% si S_T >= B_airbag, sinon 100%·S_T/B_airbag (l'airbag amortit la
    perte en redéfinissant un "strike effectif" plus bas que le nominal).
    """
    n_dates = len(obs_dates)
    prix = simulate_observation_dates(S0, r, q, sigma, obs_dates, n_simulations)

    custom_strike = strike_pct * S0
    K_call_level = K_call * custom_strike
    B_airbag_level = B_airbag * custom_strike

    cashflow = np.zeros(n_simulations)
    called = np.zeros(n_simulations, dtype=bool)

    for k in range(1, n_dates + 1):
        t_k = obs_dates[k-1]
        S_k = prix[:, k]
        discount = np.exp(-r * t_k)
        actifs = ~called

        rappel_now = actifs & (S_k >= K_call_level)
        cashflow += np.where(rappel_now, (1.0 + t_k * coupon_annuel) * discount, 0.0)
        called = called | rappel_now

    non_rappeles = ~called
    T = obs_dates[-1]
    S_T = prix[:, n_dates]
    discount_T = np.exp(-r * T)

    remboursement = np.where(S_T >= B_airbag_level, 1.0, S_T / B_airbag_level)
    cashflow += np.where(non_rappeles, (remboursement + T * coupon_annuel) * discount_T, 0.0)

    return np.mean(cashflow)
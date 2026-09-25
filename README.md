# Pricer d’options — comprendre le code et savoir le défendre

Pricer pédagogique en Python : options européennes (Black-Scholes et Monte-Carlo), américaines (Longstaff-Schwartz), forwards, sept stratégies et API Flask.

Ce guide privilégie des explications courtes et des exemples concrets : **que signifie l’écriture, pourquoi l’utiliser ici, et que se passe-t-il si l’on change un argument ?** Les extraits correspondent aux fichiers actuels ; les propositions de correction ne modifient pas le code.

## Sommaire

- [Projet et lancement](#projet)
- [Comprendre les indices et les arguments](#syntaxe)
- [Les formules à connaître](#finance)
- [pricer_engine.py, bloc par bloc](#engine)
- [main.py, bloc par bloc](#main)
- [Limites et questions d’oral](#oral)

<a id="projet"></a>
## 1. Projet et lancement

| Fichier | Rôle |
|---|---|
| `pricer_engine.py` | Calcule les prix, les sensibilités, les stratégies et récupère des données Yahoo |
| `main.py` | Reçoit les requêtes, convertit les entrées et renvoie du JSON |
| `templates/index.html` | Page appelée par Flask ; présente, mais non vérifiée dans un navigateur |
| `strutu_engine.py` | Présent dans le dossier, mais non importé par ce `main.py` ; hors périmètre |

**Circuit :** client → Flask → moteur → Flask → réponse JSON. Le moteur n’a pas besoin de Flask, mais son import nécessite NumPy et yfinance.

Depuis le dossier du projet, première installation :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy flask yfinance
python main.py
```

Les fois suivantes : `source .venv/bin/activate`, puis `python main.py`. Adresse locale : [127.0.0.1:5000](http://127.0.0.1:5000). Le mode debug actuel est destiné au développement.

| Entrée | Dans le JSON envoyé à Flask | Dans le moteur Python |
|---|---:|---:|
| Spot `S0`, strike `K` | 100 | 100 |
| Maturité `T`, en années | 1 | 1 |
| Taux `r` de 5 % | 5 | 0.05 |
| Rendement de dividende `q` de 2 % | 2 | 0.02 |
| Volatilité `sigma` de 20 % | 20 | 0.20 |

`qty` représente des unités, sans multiplicateur de contrat automatique. `q` est un rendement continu, pas un dividende en euros. Le code ne convertit ni devises ni dates calendaires.

Exemple d’appel :

```bash
curl -X POST http://127.0.0.1:5000/api/option \
  -H 'Content-Type: application/json' \
  -d '{"S0":100,"K":100,"T":1,"r":5,"q":2,"sigma":20,"option_type":"call","style":"european"}'
```

Le prix BS correspondant est environ 9,2270. Les routes sont : `GET /api/search_ticker?q=...`, `GET /api/market/<ticker>`, `POST /api/option`, `POST /api/forward`, `POST /api/strategy` et `GET /`.

<a id="syntaxe"></a>
## 2. Comprendre les indices et les arguments

### 2.1 Ton exemple : `iloc[0]`, `iloc[1]`, `iloc[-1]`

`iloc` appartient à pandas et sélectionne **par position**. Pour sélectionner, on écrit `df.iloc[1]`, avec des crochets, et non `df.iloc(1)`.

| Expression | Résultat |
|---|---|
| `df.iloc[0]` | Première ligne |
| `df.iloc[1]` | Deuxième ligne |
| `df.iloc[2]` | Troisième ligne |
| `df.iloc[-1]` | Dernière ligne |
| `df.iloc[-2]` | Avant-dernière ligne |
| `df.iloc[k]` | Ligne de position `k`, comptée depuis 0 |
| `df.iloc[0, 1]` | Première ligne, deuxième colonne |
| `df.iloc[:, 1]` | Toutes les lignes, deuxième colonne |
| `df.iloc[1:3]` | Positions 1 et 2 ; la borne 3 est exclue |

Pour cinq lignes, les indices scalaires valides vont de `0` à `4`, ou de `-5` à `-1`. `df.iloc[5]` dépasse le tableau et provoque une erreur. Un indice fractionnaire comme `1.5` n’est pas une position valide. `iloc` ignore les étiquettes de lignes : la position 1 reste la deuxième ligne, même si son étiquette est une date.

**Aucun `iloc` n’apparaît dans ces deux fichiers.** Le code utilise surtout des tableaux NumPy, avec une logique d’indices comparable. [Référence pandas](https://pandas.pydata.org/docs/user_guide/indexing.html).

### 2.2 `prix[:, 0]`, `prix[:, -1]` et `prix[itm, t]`

Dans le pricer, **ligne = trajectoire**, **colonne = date**. Prenons ce petit exemple indépendant :

```python
prix = np.array([
    [100, 110, 120],
    [100,  90,  80],
])
```

| Expression | Résultat | Ce que l’on sélectionne |
|---|---|---|
| `prix[0]` | `[100, 110, 120]` | Première trajectoire entière |
| `prix[1]` | `[100, 90, 80]` | Deuxième trajectoire entière |
| `prix[:, 0]` | `[100, 100]` | Tous les spots initiaux |
| `prix[:, 1]` | `[110, 90]` | Tous les spots à la date intermédiaire |
| `prix[:, -1]` | `[120, 80]` | Tous les spots finaux |
| `prix[0, -1]` | `120` | Spot final de la première trajectoire |
| `prix[1, 0]` | `100` | Spot initial de la deuxième trajectoire |
| `prix[:, :2]` | Les deux premières colonnes | Tranche dont la fin est exclue |

La virgule sépare les axes. `:` seul signifie « tout cet axe ». `t-1` signifie « la colonne précédente », alors que `-1` seul signifie « la dernière colonne ».

**Changer le code :** remplacer `payoff[:, -1]` par `payoff[:, 0]` commencerait LSM avec les payoffs initiaux, au lieu des payoffs à maturité : l’algorithme serait faux. Remplacer par `[:, 1]` prendrait la première date après aujourd’hui.

`prix[:, 0] = S0` remplit toute la première colonne avec le même nombre. C’est une affectation, pas une extraction. Sur cet exemple, `prix[:, 3]` et `prix[2]` dépassent les dimensions. [Référence NumPy sur les indices](https://numpy.org/doc/stable/user/basics.indexing).

### 2.3 Masque booléen, `np.where(itm)[0]` et double sélection

```python
itm = np.array([False, True, False, True])
np.where(itm)       # (array([1, 3]),)
np.where(itm)[0]    # array([1, 3])
```

`itm` indique les trajectoires dans la monnaie. Un masque conserve les positions `True`. `np.where(itm)` renvoie un **tuple de tableaux**, un par axe. Ici le masque a un seul axe, donc le tuple contient un seul tableau.

- Le `[0]` prend ce tableau complet : `[1, 3]`. Il ne prend pas seulement la première trajectoire.
- `[1]` provoquerait une erreur : pas de deuxième tableau dans ce tuple.
- `[-1]` prendrait aussi l’unique tableau ; `[0]` exprime plus clairement le premier axe.
- `np.where(itm)[0][0]` prendrait seulement l’indice `1`.

Puis, si `exercice = [True, False]`, `indices[exercice]` donne `[1]`. La ligne `cashflow[indices[exercice]] = ...` modifie donc la trajectoire 1 dans le tableau complet.

**Pourquoi deux masques ?** `itm` filtre toutes les trajectoires ; `exercice` filtre seulement celles déjà retenues. Leur longueur n’est pas la même.

### 2.4 `range(début, fin, pas)` : pourquoi ces nombres ?

La borne de fin est toujours exclue. `list(...)` sert ici uniquement à montrer les valeurs produites.

| Expression | Valeurs |
|---|---|
| `range(4)` | 0, 1, 2, 3 |
| `range(1, 4)` | 1, 2, 3 |
| `range(1, 5)` | 1, 2, 3, 4 |
| `range(3, 0, -1)` | 3, 2, 1 |
| `range(3, -1, -1)` | 3, 2, 1, 0 |
| `range(0, 5, 2)` | 0, 2, 4 |
| `range(3, 0, 1)` | Rien : le pas va dans le mauvais sens |
| `range(3, 0, 0)` | Erreur : un pas nul est interdit |

Avec `n_steps=4` :

- `range(1, n_steps+1)` remplit les colonnes 1 à 4 ; la colonne 0 existe déjà.
- `range(n_steps-1, 0, -1)` traite 3, 2, 1 : la date finale est connue, la date 0 sera traitée séparément.
- Remplacer le pas `-1` par `-2` sauterait des dates ; l’actualisation prévue pour un seul pas ne correspondrait plus aux sauts.

Dans les breakevens, `range(len(signs)-1)` évite de lire `signs[i+1]` après la dernière case.

### 2.5 `np.zeros((N, M+1))`, `zeros_like` et dimensions

`np.zeros((2, 3))` crée deux lignes et trois colonnes de zéros. La paire `(2, 3)` est un tuple décrivant la forme. `np.zeros(3)` crée seulement un vecteur de trois cases.

Dans LSM, `M` intervalles nécessitent `M+1` dates, car on ajoute aujourd’hui. Pour 100 intervalles : 101 colonnes, numérotées de 0 à 100. Supprimer `+1` ferait dépasser le tableau au dernier tour de simulation.

`np.zeros_like(ST_grid)` copie la forme et, par défaut, le type de la grille, puis remplit de zéros. Cela prépare un accumulateur de payoff. Augmenter `N` ou `M` consomme plus de mémoire ; un nombre négatif de lignes est invalide.

### 2.6 `np.random.normal(0, 1, N)` : les trois arguments

| Argument | Sens | Si on le change |
|---|---|---|
| `0` | Moyenne des chocs | `1` décale les chocs et fausse le drift prévu par la formule |
| `1` | Écart-type des chocs | `2` double leur écart-type, donc quadruple leur variance |
| `N` | Nombre de tirages | Plus grand : moyenne généralement plus stable, calcul plus coûteux |

Le deuxième argument n’est **pas la variance**. `normal(0, 0, N)` donne des zéros ; un écart-type négatif est invalide. `N=0` donne un tableau vide dont la moyenne n’est pas un prix exploitable.

Ici, la volatilité est déjà appliquée avec `sigma*sqrt(T)*Z` : il faut donc un `Z` d’écart-type 1. Doubler l’écart-type de `Z` sans adapter les autres termes ne revient pas à modifier correctement le modèle.

### 2.7 `np.maximum`, `np.mean`, `np.sum` et `axis`

```python
np.maximum(np.array([-5, 0, 8]), 0)  # [0, 0, 8]
np.mean([2, 4, 9])                  # 5.0
np.sum([False, True, True])          # 2
```

`maximum` compare chaque case à 0. Le remplacer par `minimum` garderait les pertes au lieu du payoff positif. Remplacer le 0 par 1 imposerait un versement minimal de 1 : ce serait un autre produit.

`mean` additionne puis divise par le nombre d’éléments ; `sum` ne divise pas. Pour des booléens, `True` vaut 1 et `False` vaut 0 : `sum(itm)` compte les trajectoires retenues.

Pour `A = [[1, 3], [5, 7]]` :

| Calcul | Résultat |
|---|---|
| `np.mean(A)` | `4` : moyenne de toutes les cases |
| `np.mean(A, axis=0)` | `[3, 5]` : on réduit les lignes, une moyenne par colonne |
| `np.mean(A, axis=1)` | `[2, 6]` : on réduit les colonnes, une moyenne par ligne |
| `np.mean(A, axis=-1)` | Même chose que `axis=1` pour ce tableau à deux axes |

Dans le code, `cashflow` et les payoffs MC terminaux sont des vecteurs : la moyenne sans `axis` donne un seul prix.

### 2.8 `X.mean()`, `X.std()` et normalisation

`X.mean()` est la moyenne. `X.std()` est l’écart-type, pas la variance ni la volatilité financière `sigma`. Par défaut, il utilise le diviseur `n` ; `X.std(ddof=1)` utiliserait `n-1`.

```python
X = np.array([90, 100, 110])
X_norm = (X - X.mean()) / X.std()
# environ [-1.2247, 0, 1.2247]
```

Soustraire la moyenne recentre les valeurs ; diviser par l’écart-type met leur dispersion à une échelle commune. Si toutes les valeurs sont identiques, l’écart-type vaut 0 : la division échoue numériquement. Le code actuel ne protège pas ce cas.

### 2.9 `np.polyfit(X_norm, Y, 2)` et `np.polyval`

`polyfit` ajuste un polynôme à des observations. `X_norm` contient les entrées ; `Y`, les valeurs à approcher ; **`2` est le degré**, pas le nombre d’observations ni une colonne.

| Degré | Forme ajustée | Coefficients |
|---|---|---:|
| `0` | Constante `c` | 1 |
| `1` | Droite `a*x+b` | 2 |
| `2` | Parabole `a*x**2+b*x+c` | 3 |
| `3` | Cubique `a*x**3+b*x**2+c*x+d` | 4 |
| `-1` | Invalide | — |

Un degré plus élevé peut mieux épouser les données mais aussi surajuster et rendre le calcul instable. Changer `2` en `3` demanderait notamment de revoir `np.sum(itm)>2` : trois points ne suffisent pas à identifier quatre coefficients. Même un nombre suffisant de points ne garantit pas le rang si les abscisses se répètent.

Avec un degré 2, le résultat est `[a, b, c]`, dans l’ordre des puissances décroissantes : `regression[0]` est le coefficient de `x**2`, `[1]` celui de `x`, `[2]` ou `[-1]` la constante.

```python
np.polyval([2, 3, 4], 5)  # 2*5**2 + 3*5 + 4 = 69
```

`polyfit` **apprend** les coefficients ; `polyval` **utilise** les coefficients. Dans LSM, le résultat approche la continuation, que l’on compare au payoff immédiat. [Référence NumPy](https://numpy.org/doc/2.0/reference/generated/numpy.polyfit.html).

### 2.10 `np.linspace(lo, hi, 121)` et les arrondis

`linspace` crée des valeurs régulièrement espacées, bornes incluses par défaut.

```python
np.linspace(0, 10, 3)  # [0, 5, 10]
np.linspace(0, 10, 5)  # [0, 2.5, 5, 7.5, 10]
```

Le `121` est un nombre de **points**, donc 120 intervalles. Plus de points affine le tracé mais ne crée pas plus de simulations MC. Avec 1 point, on obtient seulement la borne basse ; avec 0, une grille vide ; un nombre négatif est invalide.

`round(1.23456, 4)` donne `1.2346`. Le `4` désigne les décimales, pas les chiffres significatifs. `round(1234.56, 0)` donne `1235.0` et `round(1234.56, -2)` donne `1200.0`. L’arrondi ne rend pas le modèle plus précis.

### 2.11 `data['K']`, `.get(...)`, `or` et conversions

| Écriture | Fonctionnement |
|---|---|
| `data['K']` | Lit K ; erreur `KeyError` si la clé manque |
| `data.get('side', 'buy')` | Lit side ; utilise buy seulement si la clé manque |
| `data.get('side')` | Donne `None` si la clé manque |
| `a or b` | Prend a si sa valeur est considérée vraie ; sinon b |
| `.strip()` | Retire les espaces au début et à la fin d’une chaîne |
| `float('20')` | Convertit le texte en nombre `20.0` |
| `int(2.9)` | Tronque en `2` ; ce n’est pas un contrôle d’intégralité |

Une clé présente avec `None` ou `''` n’active pas le défaut de `.get`. En revanche, `a or b` passe à b pour `None`, `''` ou `0`. Dans la recherche Yahoo, cette logique choisit le premier nom non vide.

`def f(qty=1)` définit une valeur par défaut ; appeler `f(qty=3)` la remplace. Dans `make_position(..., qty=qty)`, le `qty` de gauche est le nom du paramètre reçu, celui de droite la variable transmise.

### 2.12 `+1`, `-1`, `0`, `**`, `*=` et listes

`sign = 1 if side == 'buy' else -1` transforme le sens en multiplicateur. Une prime de 4 à quantité 3 donne un coût signé de +12 à l’achat et −12 à la vente. `qty=0` annule la position ; une quantité négative inverse son sens et devrait être encadrée.

`=` affecte ; `==` compare ; `**2` élève au carré ; `*` multiplie. `cashflow *= facteur` modifie le tableau existant. Comme `cashflow = payoff[:, -1]` crée une vue, cette modification touche aussi la dernière colonne de `payoff`. `.copy()` rendrait le vecteur indépendant.

Avec des listes, `[1, 2] + [3]` donne `[1, 2, 3]`. Avec des tableaux NumPy, `np.array([1, 2]) + 3` donne `[4, 5]` : le scalaire s’applique à chaque case. Le type de l’objet détermine donc le sens de l’opération.

### 2.13 Compréhensions, lambda, exceptions et décorateurs

```python
[round(float(x), 4) for x in ST_grid]
```

Se lit : « pour chaque x de la grille, convertir, arrondir, ajouter à la liste ». Avec `{k: v for ...}`, on construit un dictionnaire ; `.items()` fournit les couples clé/valeur.

`lambda p: make_straddle(p['K'], p['T'])` définit une petite fonction ; son corps attend l’appel. `STRATEGY_BUILDERS[strategy](params)` sélectionne la fonction **puis** l’appelle.

`try` exécute un bloc ; `except KeyError as e` traite une clé manquante ; `except Exception` traite plus largement les exceptions. Un avertissement numérique n’est pas forcément une exception.

`@app.post('/api/option')` associe une route POST à la fonction placée dessous. `get_json(force=True)` tente le décodage JSON même sans le bon en-tête HTTP ; il ne valide pas les champs. `force=False` rétablit le contrôle du type de contenu, sans remplacer les validations métier. [Référence Flask](https://flask.palletsprojects.com/en/stable/api/#flask.Request.get_json).

<a id="finance"></a>
## 3. Les formules à connaître

### Black-Scholes et mouvement brownien géométrique

$$
d_1=\frac{\ln(S_0/K)+(r-q+\sigma^2/2)T}{\sigma\sqrt T},\qquad d_2=d_1-\sigma\sqrt T.
$$
$$
C=S_0e^{-qT}N(d_1)-Ke^{-rT}N(d_2),\qquad
P=Ke^{-rT}N(-d_2)-S_0e^{-qT}N(-d_1).
$$

`N` est la fonction de répartition normale ; `phi` sa densité. Le modèle suppose notamment des taux et une volatilité constants, des prix continus et un marché sans frictions.

Sous la mesure risque-neutre :

$$
\frac{dS_t}{S_t}=(r-q)dt+\sigma dW_t^{\mathbb Q},\qquad
S_T=S_0e^{(r-q-\sigma^2/2)T+\sigma\sqrt T Z},\quad Z\sim\mathcal N(0,1).
$$

Le drift est `r-q` pour valoriser, pas pour prévoir le rendement historique. `-sigma**2/2` est la correction d’Itô ; `sqrt(T)` vient de la variance du brownien, égale à T.

### Monte-Carlo et Longstaff-Schwartz

$$
V_0=e^{-rT}\mathbb E^{\mathbb Q}[h(S_T)]
\approx e^{-rT}\frac1N\sum_i h(S_T^{(i)}).
$$

L’erreur-type MC décroît comme $1/\sqrt N$ : quatre fois plus de simulations pour environ deux fois moins d’erreur. Aucun intervalle de confiance n’est renvoyé actuellement.

Pour une américaine, LSM remonte les dates et exerce si $h(S_t)>\widehat C_t(S_t)$. La continuation est approchée par :

$$
\widehat\beta=\arg\min_{\beta}\sum_i[Y_i-(\beta_0+\beta_1x_i+\beta_2x_i^2)]^2.
$$

X = spots actuels des trajectoires ITM ; Y = leurs flux futurs actualisés à la date courante. La grille d’exercice et la régression rendent le résultat approximatif.

> **À dire à l’oral** — « La régression estime ce que vaut continuer, en moyenne conditionnellement au spot présent. Elle ne donne pas au détenteur la connaissance parfaite du futur. »

### Grecques : définition et unités

| Nom | Dérivée | Sens |
|---|---|---|
| Delta | $\partial_S V$ | Effet du spot |
| Gamma | $\partial_S^2 V$ | Variation de Delta avec le spot |
| Vega | $\partial_\sigma V$ | Effet de la volatilité |
| Theta | $-\partial_T V$ | Effet du temps écoulé |
| Rho | $\partial_r V$ | Effet du taux |
| Vanna | $\partial_\sigma\Delta$ | Effet de la volatilité sur Delta |
| Vomma | $\partial_\sigma\mathrm{Vega}$ | Courbure en volatilité |
| Charm | $-\partial_T\Delta$ | Effet du temps écoulé sur Delta |
| Color calendaire | $-\partial_T\Gamma$ | Effet du temps écoulé sur Gamma ; ligne actuelle erronée |
| Speed | $\partial_S\Gamma$ | Effet du spot sur Gamma |
| Zomma | $\partial_\sigma\Gamma$ | Effet de la volatilité sur Gamma |
| Ultima | $\partial_\sigma\mathrm{Vomma}$ | Dérivée troisième en volatilité |
| Lambda | $S\Delta/V$ | Élasticité du prix au spot |

Vega et Rho sont bruts : multiplier par `0.01` pour une hausse de **1 point** de volatilité ou de taux. Theta est annuel ; diviser par 365 donne une approximation quotidienne si cette convention est retenue. Le groupe « secondaires » n’est pas exclusivement composé de dérivées secondes.

### Forward, payoff et P&L

$$
F=S_0e^{(r-q)T},\qquad V_{long}=e^{-rT}(F-K),\qquad h_{long}(S_T)=S_T-K.
$$

F = prix de livraison équitable ; V = valeur actuelle ; h = flux final. Le code ne modélise pas le règlement quotidien des futures.

Pour des legs de sens $s_j$ et quantités $n_j$ :

$$
H(S_T)=\sum_j s_jn_jh_j(S_T),\qquad C_0=\sum_j s_jn_jV_j(0).
$$

Le code affiche $H-C_0$. Un résultat terminal financé, sans flux intermédiaires, serait $H-C_0e^{rT}$. Les dividendes des actions nécessitent un traitement supplémentaire. Exemple : call de strike 100 et prime 8 → breakeven simple 108 ; la vente inverse le P&L mais conserve ce zéro.

<a id="engine"></a>
## 4. `pricer_engine.py`, bloc par bloc

Chaque bloc explique directement les écritures utiles à sa compréhension.

### 4.1 Importer les outils numériques

**Lignes 1–2 de `pricer_engine.py`**

```python
import numpy as np
from statistics import NormalDist
```

`as np` donne un nom court à NumPy. `from ... import NormalDist` importe seulement l’outil de loi normale. Ces lignes rendent les outils disponibles ; elles ne calculent aucun prix.

### 4.2 yfinance et recherche de symboles

**Lignes 8–18 de `pricer_engine.py`**

```python
import yfinance as yf


def search_tickers(query, max_results=8):
    """Cherche des tickers correspondant à `query` sur Yahoo Finance.
    Renvoie une liste de dicts {symbol, name, exchange}.
    """
    try:
        resultats = yf.Search(query, max_results=max_results).quotes
    except Exception:
        return []
```

`yf` est l’alias de yfinance. `max_results=8` est un défaut : passer 3 demande au plus trois résultats ; passer 20 en demande davantage, selon le fournisseur. `.quotes` récupère les résultats de recherche. Toute exception renvoie `[]` : panne et absence de résultat deviennent indistinguables.

### 4.3 Nettoyer les résultats Yahoo

**Lignes 20–28 de `pricer_engine.py`**

```python
    return [
        {
            "symbol": r.get("symbol"),
            "name": r.get("shortname") or r.get("longname") or r.get("symbol"),
            "exchange": r.get("exchange"),
        }
        for r in resultats
        if r.get("symbol")
    ]
```

La compréhension parcourt les résultats, conserve ceux ayant un symbole et produit trois champs. Les `or` choisissent le premier nom non vide. Ici, `r` est un résultat Yahoo, pas le taux financier. `.get('symbol')` lit la clé et renvoie `None` si elle manque, contrairement à `r['symbol']` qui déclencherait une erreur. `a or b` garde a si sa valeur est considérée vraie, sinon prend b : ici, un nom vide ou absent fait passer au suivant.

### 4.4 Dernier prix et devise

**Lignes 31–38 de `pricer_engine.py`**

```python
def get_spot(ticker):
    """Dernier prix connu du sous-jacent."""
    return float(yf.Ticker(ticker).fast_info["last_price"])


def get_devise(ticker):
    """Devise de cotation du ticker (ex: 'USD', 'EUR')."""
    return yf.Ticker(ticker).fast_info["currency"]
```

`Ticker(ticker)` désigne le titre demandé. `fast_info['last_price']` lit son dernier cours accessible ; `['currency']`, sa devise. Ces crochets utilisent des clés textuelles, pas des positions : les remplacer par `0` ne signifie pas « premier cours ». `float` convertit le cours en nombre.

### 4.5 VIX : donnée de contexte, pas volatilité calibrée

**Lignes 41–43 de `pricer_engine.py`**

```python
def get_vix():
    """Dernier niveau du VIX (volatilité implicite du S&P 500), en points de %."""
    return float(yf.Ticker("^VIX").fast_info["last_price"])
```

`^VIX` fixe le symbole interrogé. Le niveau 20 signifie 20 points de pourcentage. Le VIX concerne la volatilité attendue à environ 30 jours du S&P 500 : ce n’est pas la volatilité de toute action. Il est renvoyé comme contexte, sans alimenter automatiquement `sigma`.

### 4.6 Fonction de répartition et densité normales

**Lignes 49–53 de `pricer_engine.py`**

```python
def Ncdf(x):
    return NormalDist().cdf(x)

def Npdf(x):
    return NormalDist().pdf(x)
```

`NormalDist()` sans argument représente une normale de moyenne 0 et d’écart-type 1. `cdf(x)` donne P(Z ≤ x), donc `cdf(0)=0.5`. `pdf(x)` donne une densité, pas une probabilité ponctuelle. Ces fonctions évitent de répéter les appels complets.

### 4.7 Factoriser d1 et d2

**Lignes 55–59 de `pricer_engine.py`**

```python
def _d1_d2(S0, K, T, r, q, sigma):
    """Calcule d1 et d2 communs à Black-Scholes et aux grecques."""
    d1 = (np.log(S0 / K) + (r - q + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2
```

Le `_` signale une fonction interne par convention. `log(S0/K)` mesure la moneyness ; `sigma**2` est le carré de la volatilité ; `sqrt(T)` la racine de la maturité. `return d1, d2` renvoie une paire. Remplacer `/2` par `/3` changerait la formule financière, pas un réglage numérique.

### 4.8 Prix Black-Scholes-Merton

**Lignes 65–74 de `pricer_engine.py`**

```python
def black_scholes(S0, K, T, r, q, sigma, option_type):
    
    d1, d2 = _d1_d2(S0, K, T, r, q, sigma)
    
    if option_type == "call":
        prix = S0 * np.exp(-q*T) * Ncdf(d1) - K * np.exp(-r*T) * Ncdf(d2)
    else:
        prix = K * np.exp(-r*T) * Ncdf(-d2) - S0 * np.exp(-q*T) * Ncdf(-d1)

    return prix
```

`d1, d2 = ...` décompose la paire renvoyée. `if` distingue le call ; `else` traite tout le reste comme un put, même une chaîne invalide en appel direct. Les exponentielles intègrent dividende continu et actualisation. Le résultat est la prime unitaire longue ; le sens vendeur sera traité ailleurs.

### 4.9 Monte-Carlo : simuler uniquement la date finale

**Lignes 81–88 de `pricer_engine.py`**

```python
def monte_carlo(S0, K, T, r, q, sigma, option_type, n_simulations):

    Z = np.random.normal(0, 1, n_simulations)

    ST = S0 * np.exp(
        (r - q - sigma**2 / 2) * T
        + sigma * np.sqrt(T) * Z
    )
```

Dans `normal(0,1,N)`, 0 est la moyenne, 1 l’écart-type et N le nombre de tirages. Mettre 2 à la place de 1 doublerait l’écart-type des chocs ; la volatilité étant déjà appliquée ensuite, cela fausserait le modèle. Augmenter N augmente le coût et réduit généralement le bruit statistique. L’exponentielle simule directement les prix finaux : aucune trajectoire intermédiaire n’est nécessaire pour une vanilla européenne. Pas de graine dans la fonction : deux appels peuvent donner des résultats différents.

### 4.10 Monte-Carlo : payoff, moyenne et actualisation

**Lignes 90–97 de `pricer_engine.py`**

```python
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)

    prix = np.exp(-r*T) * np.mean(payoff)

    return prix
```

`maximum(...,0)` applique le droit de ne pas exercer. `mean` estime l’espérance, puis `exp(-r*T)` l’actualise. Remplacer `mean` par `sum` multiplierait l’estimation par le nombre de simulations. `np.maximum` compare chaque case à zéro : `[-5,0,8]` devient `[0,0,8]`. `np.mean` additionne les cases puis divise par leur nombre ; `np.sum` les additionne seulement. Ces opérations portent ici sur tous les scénarios.

### 4.11 LSM : calendrier et matrice de trajectoires

**Lignes 105–111 de `pricer_engine.py`**

```python
def longstaff_schwartz(S0, K, T, r, q, sigma, option_type, n_simulations, n_steps=100):

    dt = T / n_steps

    # Simulation des prix
    prix = np.zeros((n_simulations, n_steps + 1))
    prix[:, 0] = S0
```

`dt=T/n_steps` est la durée d’un pas. `n_steps=100` signifie 100 intervalles et 101 dates, grâce au `+1`. `prix[:,0]=S0` remplit la colonne initiale. Avec 200 pas, l’exercice est examiné plus souvent, mais le coût augmente ; 0 provoquerait une division par zéro. Dans `zeros((N,M+1))`, le tuple indique N lignes et M+1 colonnes. Dans `prix[:,0]`, `:` sélectionne toutes les lignes et 0 la première colonne ; 1 désignerait la deuxième, −1 la dernière. Ici, chaque ligne est une trajectoire et chaque colonne une date.

### 4.12 LSM : avancer dans le temps

**Lignes 113–120 de `pricer_engine.py`**

```python
    for t in range(1, n_steps + 1):

        Z = np.random.normal(0, 1, n_simulations)

        prix[:, t] = prix[:, t-1] * np.exp(
            (r - q - sigma**2 / 2) * dt
            + sigma * np.sqrt(dt) * Z
        )
```

La boucle remplit les dates 1 à `n_steps`, incluses. `t-1` lit la date précédente ; `t` écrit la nouvelle. Chaque pas utilise de nouveaux chocs. La transition MBG est exacte entre deux dates ; c’est l’ensemble des dates d’exercice qui est discret. `range(1,n_steps+1)` part de 1 et exclut sa borne finale : avec 4 pas, il produit 1,2,3,4. Sans le `+1`, la dernière date ne serait pas calculée. Le pas omis vaut +1 ; un pas de 2 sauterait une date sur deux.

### 4.13 LSM : valeurs intrinsèques et condition terminale

**Lignes 123–128 de `pricer_engine.py`**

```python
    if option_type == "call":
        payoff = np.maximum(prix - K, 0)
    else:
        payoff = np.maximum(K - prix, 0)

    cashflow = payoff[:, -1]
```

Le payoff est calculé pour toutes les dates. `[:, -1]` prend la dernière colonne, donc les flux à maturité. C’est une vue : les modifications de `cashflow` affectent cette colonne. `.copy()` éviterait cet alias ; les colonnes antérieures restent indépendantes de cette modification.

### 4.14 LSM : actualiser puis sélectionner les trajectoires ITM

**Lignes 131–137 de `pricer_engine.py`**

```python
    for t in range(n_steps - 1, 0, -1):

        cashflow *= np.exp(-r * dt)

        itm = payoff[:, t] > 0

        if np.sum(itm) > 2:
```

On revient de l’avant-dernière date jusqu’à la date 1. `*=` actualise tous les flux d’un pas. `>0` repère les options dans la monnaie ; `sum(itm)>2` signifie au moins trois observations, pour trois coefficients quadratiques. Changer `>` en `>=` autoriserait seulement deux observations, insuffisantes pour identifier trois coefficients.

### 4.15 LSM : variables explicative et cible

**Lignes 139–145 de `pricer_engine.py`**

```python
            X = prix[itm, t]
            Y = cashflow[itm]

            # --- standardisation avant la régression ---
            X_mean = X.mean()
            X_std = X.std()
            X_norm = (X - X_mean) / X_std
```

Le masque `itm` sélectionne X et Y pour les mêmes trajectoires. X contient leurs spots actuels ; Y, les flux futurs déjà actualisés. Le centrage-réduction améliore la stabilité de la régression. `X.std()` peut valoir zéro ; aucun garde-fou n’est prévu. `X.mean()` calcule la moyenne ; `X.std()` calcule l’écart-type, avec un diviseur égal au nombre d’observations par défaut. `(X-X_mean)/X_std` recentre les valeurs sur zéro et ramène leur dispersion à une échelle commune. Si tous les X sont identiques, la division par zéro rend le résultat inutilisable.

### 4.16 LSM : régression et décision d’exercice

**Lignes 147–153 de `pricer_engine.py`**

```python
            regression = np.polyfit(X_norm, Y, 2)
            continuation = np.polyval(regression, X_norm)

            exercice = payoff[itm, t] > continuation

            indices = np.where(itm)[0]
            cashflow[indices[exercice]] = payoff[indices[exercice], t]
```

Le `2` de `polyfit` signifie polynôme quadratique. `polyval` calcule la continuation estimée. `exercice` teste si l’intrinsèque la dépasse ; `where(itm)[0]` retrouve les indices globaux. On remplace le flux futur par le flux d’exercice, on ne les additionne pas. `polyfit(X,Y,2)` ajuste trois coefficients : a*x²+b*x+c ; 1 donnerait une droite, 0 une constante et 3 un polynôme cubique. `polyval` évalue ces coefficients dans l’ordre décroissant des puissances. `where(itm)` renvoie un tuple contenant un tableau d’indices ; `[0]` extrait ce tableau entier, pas son premier indice. Pour `[False,True,False,True]`, il vaut `[1,3]`. `[1]` échouerait car ce tuple ne contient qu’un élément. `indices[exercice]` conserve ensuite les indices où le masque exercice est vrai.

### 4.17 LSM : revenir à aujourd’hui et imposer l’intrinsèque

**Lignes 155–163 de `pricer_engine.py`**

```python
    prix_lsm = np.mean(cashflow) * np.exp(-r * dt)

    # --- plancher d'exercice immédiat ---
    if option_type == "call":
        payoff_immediat = max(S0 - K, 0)
    else:
        payoff_immediat = max(K - S0, 0)

    return max(prix_lsm, payoff_immediat)
```

Après la boucle, les flux sont à la date 1 : le dernier facteur les ramène à aujourd’hui. Le `max` final impose la valeur d’exercice immédiat. Remplacer par `min` pourrait donner un prix inférieur à ce qu’un détenteur reçoit en exerçant maintenant. Le plancher ne garantit pas la précision du modèle.

### 4.18 Delta, Gamma et Vega

**Lignes 169–180 de `pricer_engine.py`**

```python
def grecs_primaires_bs(S0, K, T, r, q, sigma, option_type):

    d1, d2 = _d1_d2(S0, K, T, r, q, sigma)
    
    if option_type == "call" :
        Delta = (np.exp(-q*T))*Ncdf(d1)
    else:
        Delta = (np.exp(-q*T))*(Ncdf(d1)-1)
        
    Gamma = ((np.exp(-q*T))*(Npdf(d1)))/(S0*sigma*np.sqrt(T))
    
    Vega = S0*np.exp(-q*T)*Npdf(d1)*np.sqrt(T)
```

Delta change avec call/put ; Gamma et Vega ont ici la même formule pour les deux types. `Npdf(d1)` est la densité normale. Delta est la variation locale du prix par unité de spot ; Gamma est la variation de Delta par unité de spot ; Vega est la variation du prix par unité de volatilité décimale. Pour +1 point de volatilité, on multiplie Vega par 0,01. Le `-1` dans le Delta put est une soustraction issue de la formule, pas un indice de tableau.

### 4.19 Theta et Rho

**Lignes 182–190 de `pricer_engine.py`**

```python
    if option_type == "call" : 
        Theta = - (S0 * sigma * np.exp(-q*T) * Npdf(d1)) / (2 * np.sqrt(T))-r*K*np.exp(-r*T)*Ncdf(d2)+q*S0*np.exp(-q*T)*Ncdf(d1)
    else:
        Theta = -(S0*sigma*np.exp(-q*T)*Npdf(d1))/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*Ncdf(-d2) - q*S0*np.exp(-q*T)*Ncdf(-d1)
    
    if option_type == "call" :
        Rho = K*T*np.exp(-r*T)*Ncdf(d2)
    else:
        Rho = -K*T*np.exp(-r*T)*Ncdf(-d2)
```

Theta mesure le temps qui passe, donc l’opposé de la dérivée en maturité restante. Rho mesure l’effet du taux décimal. Les branches call/put ajustent les signes des termes de taux et dividende. `2*sqrt(T)` est un dénominateur mathématique, pas une valeur de précision à choisir.

### 4.20 Retourner les cinq sensibilités

**Lignes 193–199 de `pricer_engine.py`**

```python
    return {
        "Delta": Delta,
        "Gamma": Gamma,
        "Vega": Vega,
        "Theta": Theta,
        "Rho": Rho
    }
```

Les accolades créent un dictionnaire : nom de grecque → valeur. On lit ensuite `grecs['Delta']`. Les clés sont sensibles à la casse. Aucun arrondi ni multiplication par le sens de la position n’a lieu ici.

### 4.21 Grecques complémentaires : intermédiaires communs

**Lignes 205–214 de `pricer_engine.py`**

```python
def grecs_secondaires_bs(S0, K, T, r, q, sigma, option_type):

    d1, d2 = _d1_d2(S0, K, T, r, q, sigma)
    Vega = S0*np.exp(-q*T)*Npdf(d1)*np.sqrt(T)
    Gamma = ((np.exp(-q*T))*(Npdf(d1)))/(S0*sigma*np.sqrt(T))
    
    if option_type == "call" :
        Delta = (np.exp(-q*T))*Ncdf(d1)
    else:
        Delta = (np.exp(-q*T))*(Ncdf(d1)-1)
```

On recalcule d1, d2, Vega, Gamma et Delta pour les formules suivantes. Ces variables locales évitent de recopier leurs expressions partout. Une factorisation avec les grecques primaires réduirait encore les répétitions.

### 4.22 Vanna et Vomma

**Lignes 217–219 de `pricer_engine.py`**

```python
    Vanna = (-np.exp(-q*T)*Npdf(d1)*d2)/sigma
    
    Vomma = (Vega*d1*d2)/sigma
```

Vanna mesure comment Delta varie avec sigma ; Vomma, comment Vega varie avec sigma. Les divisions par sigma supposent une volatilité non nulle. Pour un mouvement de 1 point de volatilité, la variation de Vega est approximativement `Vomma*0.01`.

### 4.23 Charm : le Delta évolue même à spot constant

**Lignes 221–224 de `pricer_engine.py`**

```python
    if option_type == "call" :
        Charm = q*np.exp(-q*T)*Ncdf(d1)-np.exp(-q*T)*Npdf(d1)*(2*(r-q)*T - d2*sigma*np.sqrt(T)) / (2*T*sigma*np.sqrt(T))
    else:
        Charm = -q*np.exp(-q*T)*Ncdf(-d1)-np.exp(-q*T)*Npdf(d1)*(2*(r-q)*T-d2*sigma*np.sqrt(T))/(2*T*sigma*np.sqrt(T))
```

Charm est la dérivée de Delta en temps écoulé. Il peut faire varier la couverture même si le spot reste fixe. `Ncdf(-d1)` évalue la fonction au nombre opposé de d1 : ce `-` n’a aucun lien avec l’accès au dernier élément d’un tableau.

### 4.24 Color : anomalie de formule à connaître

**Lignes 226–226 de `pricer_engine.py`**

```python
    Color = -np.exp(-q*T)*Npdf(d1)/(2*S0*T*sigma*np.sqrt(T))*(2*q*T+1+(2*(r-q)*T-d2*sigma*np.sqrt(T)))*d1/(sigma*np.sqrt(T))
```

**Erreur à corriger :** le facteur `d1/(sigma*sqrt(T))` multiplie trop de termes. Avec $B=2(r-q)T-d_2\sigma\sqrt T$, la dérivée correcte en maturité est :

$$
\partial_T\Gamma=-\frac{\Gamma}{2T}\left[2qT+1+\frac{d_1B}{\sigma\sqrt T}\right].
$$

La convention calendaire prend l’opposé. La ligne actuelle ne correspond généralement à aucune des deux. Les parenthèses déterminent donc le sens mathématique, pas seulement la lisibilité.

### 4.25 Speed, Zomma, Ultima et Lambda

**Lignes 228–233 de `pricer_engine.py`**

```python
    Speed = -Gamma/S0*(d1/(sigma*np.sqrt(T))+1)
    
    Zomma = Gamma*(d1*d2-1)/sigma
    Ultima = -Vega*(d1*d2*(1-d1*d2)+d1**2+d2**2)/sigma**2
    prix = black_scholes(S0, K, T, r, q, sigma, option_type)
    Lambda = Delta * S0 / prix
```

Speed = effet du spot sur Gamma ; Zomma = effet de sigma sur Gamma ; Ultima = effet de sigma sur Vomma. `**2` calcule des carrés. Lambda divise Delta fois spot par la prime : une prime presque nulle rend cette élasticité instable ou indéfinie.

### 4.26 Exporter les sensibilités complémentaires

**Lignes 235–244 de `pricer_engine.py`**

```python
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
```

Le dictionnaire renvoie les huit résultats, y compris Color malgré l’erreur signalée. Il ne filtre pas les valeurs infinies ou `NaN`. Arrondir plus tard ne corrige pas une valeur non finie.

### 4.27 Prix forward théorique

**Lignes 250–253 de `pricer_engine.py`**

```python
def forward_price(S0, T, r, q):
    """Prix à terme théorique F."""
    F=S0*np.exp((r-q)*T)
    return F
```

`exp((r-q)*T)` capitalise le spot au coût de portage net. F est le prix de livraison qui donne une valeur initiale nulle au nouveau forward ; ce n’est pas la valeur d’un contrat déjà engagé.

### 4.28 Valeur d’un forward déjà engagé

**Lignes 255–264 de `pricer_engine.py`**

```python
def forward_value(S0, K, T, r, q, side):
    """Valeur actuelle d'un contrat forward déjà engagé à K. side = 'buy' ou 'sell'."""
    F = forward_price(S0, T, r, q)

    if side =="buy":
        valeur = (F-K)*np.exp(-r*T)
    else:
        valeur = (K-F)*np.exp(-r*T)
    
    return valeur
```

On compare F au strike contractuel K, puis on actualise l’écart. `buy` donne la valeur longue ; toute autre chaîne entre dans la branche vendeuse. `K=F` donne zéro. Une validation de `side` manque.

### 4.29 Payoff forward et différence avec les futures

**Lignes 266–272 de `pricer_engine.py`**

```python
def forward_payoff(ST, K, side):
    """Payoff à maturité pour un prix final ST donné. side = 'buy' ou 'sell'."""
    if side == "buy": 
        payoff = ST-K
    else:
        payoff = K-ST
    return payoff
```

Le long reçoit `ST-K` et le short `K-ST`. Pas de plancher à zéro : le forward engage les deux parties, même si son flux devient négatif. La même soustraction peut fonctionner sur un nombre ou un tableau NumPy.

### 4.30 Décrire une leg optionnelle

**Lignes 278–287 de `pricer_engine.py`**

```python
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
```

Ce constructeur décrit une leg sans la valoriser. `qty=1` est le défaut ; passer 2 double sa taille. Le dictionnaire conserve instrument, type, sens, strike, maturité et quantité. Aucun champ ne prévoit un style américain : les stratégies utilisent BS.

### 4.31 Décrire une leg action

**Lignes 289–295 de `pricer_engine.py`**

```python
def make_stock_position(side, qty=1):
    """Construit une leg action (sans strike, sans maturité, sans option_type)."""
    return {
        "instrument": "stock",
        "side": side,
        "qty": qty,
    }
```

Une action n’a ni strike ni maturité d’option : son dictionnaire conserve seulement instrument, sens et quantité. Le type `stock` permet ensuite de choisir la bonne formule sans paramètres fictifs.

### 4.32 Prime ou coût signé d’une leg

**Lignes 297–306 de `pricer_engine.py`**

```python
def price_leg(leg, S0, r, q, sigma):
    """Calcule la prime d'engagement d'une seule leg, signée selon buy/sell."""
    sign = 1 if leg["side"] == "buy" else -1

    if leg["instrument"] == "stock":
        prix_unitaire = S0
    else:
        prix_unitaire = black_scholes(S0, leg["K"], leg["T"], r, q, sigma, leg["option_type"])

    return sign * leg["qty"] * prix_unitaire
```

Le signe transforme achat/vente en +1/−1. L’action vaut S0 ; l’option reçoit un prix BS. Le produit `sign*qty*prix_unitaire` est un coût signé : positif pour payer, négatif pour recevoir. Le flux initial de trésorerie est son opposé. `1 if ... else -1` est une expression conditionnelle : elle renvoie 1 si le sens vaut buy, −1 sinon. Une prime de 4 à quantité 3 donne donc +12 à l’achat et −12 à la vente. Une quantité nulle annule le coût ; une quantité négative inverse le sens.

### 4.33 Payoff signé d’une leg

**Lignes 312–324 de `pricer_engine.py`**

```python
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
```

L’action renvoie ST, sa valeur terminale brute ; l’option renvoie son intrinsèque. Le signe et la quantité s’appliquent ensuite à chaque point du tableau. Écrire `ST-S0` ici soustrairait deux fois le coût, car il est déjà retiré au niveau de la stratégie.

### 4.34 Additionner payoffs et coûts

**Lignes 326–340 de `pricer_engine.py`**

```python
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
```

`zeros_like` prépare un total pour chaque spot final ; `cout_total=0` prépare un scalaire. La boucle additionne les legs, puis soustrait le coût à chaque scénario par broadcasting. Le P&L ignore financement et dividendes reçus sur l’action. Il suppose aussi une échéance de comparaison commune.

### 4.35 Grecques d’une stratégie : initialisation et action

**Lignes 342–354 de `pricer_engine.py`**

```python
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
```

On initialise cinq totaux à zéro. Une action de valeur spot S a Delta 1 et les quatre autres dérivées nulles dans ce cadre. Pour une option, on appelle les grecques BS. `total['Delta']` sélectionne une clé, pas un indice.

### 4.36 Grecques d’une stratégie : agrégation signée

**Lignes 355–361 de `pricer_engine.py`**

```python
            total["Delta"] = total["Delta"] + sign * leg["qty"] * grecs["Delta"]
            total["Gamma"] = total["Gamma"] + sign * leg["qty"] * grecs["Gamma"]
            total["Vega"]  = total["Vega"]  + sign * leg["qty"] * grecs["Vega"]
            total["Theta"] = total["Theta"] + sign * leg["qty"] * grecs["Theta"]
            total["Rho"]   = total["Rho"]   + sign * leg["qty"] * grecs["Rho"]

    return total
```

Chaque ligne ajoute `sens × quantité × grecque unitaire`. Ainsi, vendre inverse bien les dérivées de position dans les stratégies. Les sensibilités s’additionnent parce que la dérivée d’une somme est la somme des dérivées. Les grecques secondaires ne sont pas agrégées ici.

### 4.37 Covered call

**Lignes 367–372 de `pricer_engine.py`**

```python
def make_covered_call(K, T, qty=1):
    """Action longue + call vendu."""
    return [
        make_stock_position("buy", qty=qty),
        make_position("call", "sell", K, T, qty=qty),
    ]
```

La liste contient action achetée et call vendu. Payoff unitaire : `min(ST,K)`. La prime réduit le coût d’entrée, mais plafonne la hausse sans supprimer la perte en cas de baisse de l’action. Le constructeur fixe les sens.

### 4.38 Protective put

**Lignes 374–379 de `pricer_engine.py`**

```python
def make_protective_put(K, T, qty=1):
    """Action longue + put acheté (assurance à la baisse)."""
    return [
        make_stock_position("buy", qty=qty),
        make_position("put", "buy", K, T, qty=qty),
    ]
```

Action achetée + put acheté : payoff `max(ST,K)`. Le put crée un plancher de valeur terminale, au prix d’une prime. Ce plancher n’est pas un gain net : le coût initial reste à soustraire.

### 4.39 Straddle

**Lignes 381–386 de `pricer_engine.py`**

```python
def make_straddle(K, T, side="buy", qty=1):
    """Call + put, même strike, même maturité, même sens."""
    return [
        make_position("call", side, K, T, qty=qty),
        make_position("put", side, K, T, qty=qty),
    ]
```

Call et put de mêmes strike, échéance et sens. À l’achat, payoff `abs(ST-K)` ; il faut un déplacement suffisant pour couvrir les deux primes. Passer `side='sell'` inverse les deux legs. Passer `qty=2` double leurs tailles.

### 4.40 Strangle

**Lignes 388–393 de `pricer_engine.py`**

```python
def make_strangle(K_put, K_call, T, side="buy", qty=1):
    """Call + put, strikes différents (K_put < K_call), même sens."""
    return [
        make_position("call", side, K_call, T, qty=qty),
        make_position("put", side, K_put, T, qty=qty),
    ]
```

Call au strike haut et put au strike bas, normalement `K_put<K_call`. Entre les strikes, le payoff long est nul. Le code ne vérifie pas cet ordre. Le sens est transmis aux deux options, contrairement aux spreads à sens fixés.

### 4.41 Bull call spread

**Lignes 395–400 de `pricer_engine.py`**

```python
def make_call_spread(K1, K2, T, qty=1):
    """Bull call spread : achat call K1 (bas), vente call K2 (haut), K1 < K2."""
    return [
        make_position("call", "buy", K1, T, qty=qty),
        make_position("call", "sell", K2, T, qty=qty),
    ]
```

Call acheté bas K1, call vendu haut K2 : on attend `K1<K2`. Payoff borné entre 0 et `K2-K1`. Échanger les strikes change l’exposition et ne correspond plus au bull call spread annoncé.

### 4.42 Bear put spread

**Lignes 402–407 de `pricer_engine.py`**

```python
def make_put_spread(K1, K2, T, qty=1):
    """Bear put spread : achat put K1 (haut), vente put K2 (bas), K1 > K2."""
    return [
        make_position("put", "buy", K1, T, qty=qty),
        make_position("put", "sell", K2, T, qty=qty),
    ]
```

Put acheté haut K1, put vendu bas K2 : on attend cette fois `K1>K2`. Payoff borné entre 0 et `K1-K2`. L’ordre est donc l’inverse de celui attendu pour le call spread.

### 4.43 Collar

**Lignes 409–415 de `pricer_engine.py`**

```python
def make_collar(K_put, K_call, T, qty=1):
    """Action longue + put acheté (K_put) + call vendu (K_call), K_put < K_call."""
    return [
        make_stock_position("buy", qty=qty),
        make_position("put", "buy", K_put, T, qty=qty),
        make_position("call", "sell", K_call, T, qty=qty),
    ]
```

Action longue, put acheté bas, call vendu haut : la valeur terminale est encadrée par les strikes. Le call aide à financer le put, mais les primes ne se compensent pas forcément. La liste a trois legs, quelle que soit `qty`.

<a id="main"></a>
## 5. `main.py`, bloc par bloc

Chaque bloc explique directement les écritures utiles à sa compréhension.

### 5.1 Imports et création de l’application

**Lignes 1–14 de `main.py`**

```python
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
```

Flask reçoit les requêtes ; `render_template` prépare le HTML ; `request` lit les entrées ; `jsonify` produit les réponses. L’import parenthésé rassemble les fonctions du moteur. `Flask(__name__)` crée l’application et fournit son identité de module pour localiser les ressources.

### 5.2 Construire la grille de prix finaux

**Lignes 21–27 de `main.py`**

```python
def _grid(centres, n_points=121, low=0.4, high=1.6):
    """Construit une grille de ST autour d'une liste de niveaux de référence
    (S0, strikes...), avec une marge de 40% en dessous et 60% au-dessus.
    """
    lo = max(0.01, min(centres) * low)
    hi = max(centres) * high
    return np.linspace(lo, hi, n_points)
```

`min` et `max` prennent les niveaux extrêmes. `low=0.4` signifie 40 % du minimum, donc 60 % en dessous : la docstring annonce à tort −40 %. Passer `low=0.6` donnerait réellement −40 %. Le plancher 0.01 évite zéro ; `np.linspace(lo,hi,121)` crée 121 points régulièrement espacés, bornes incluses, donc 120 intervalles. Avec 3 points entre 0 et 10, on obtient `[0,5,10]`. Augmenter ce nombre affine le graphique, pas les simulations MC.

### 5.3 Breakevens : préparer les signes

**Lignes 30–40 de `main.py`**

```python
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
```

`np.sign(pnl)` renvoie −1 pour une perte, 0 pour zéro et +1 pour un gain. `breakevens=[]` prépare la liste des racines. La docstring explique pourquoi l’interpolation est écrite directement : un P&L peut croître ou décroître entre deux points.

### 5.4 Breakevens : interpolation linéaire et défaut des zéros exacts

**Lignes 41–45 de `main.py`**

```python
    for i in range(len(signs) - 1):
        if signs[i] != signs[i + 1] and signs[i] != 0 and signs[i + 1] != 0:
            be = ST_grid[i] + (0 - pnl[i]) * (ST_grid[i + 1] - ST_grid[i]) / (pnl[i + 1] - pnl[i])
            breakevens.append(round(float(be), 2))
    return breakevens
```

On inspecte des paires voisines sans dépasser le tableau. La formule trouve le zéro de la droite joignant les deux points. `round(...,2)` garde deux décimales. **Défaut :** les tests `!=0` excluent un zéro exact : `[-10,0,10]` n’est pas détecté. Hors grille, deux racines dans un intervalle et plateaux posent aussi problème.

### 5.5 Une réponse d’erreur commune

**Lignes 48–49 de `main.py`**

```python
def _err(message, code=400):
    return jsonify({"error": message}), code
```

Le retour est une paire `(réponse, statut HTTP)`. `code=400` est le défaut pour une requête invalide ; `code=500` annoncerait une erreur serveur. Changer le nombre change le statut, pas le message JSON. Tous les appels actuels utilisent le défaut, même pour certaines pannes externes.

### 5.6 Route de recherche de ticker

**Lignes 56–64 de `main.py`**

```python
@app.get("/api/search_ticker")
def api_search_ticker():
    query = request.args.get("q", "").strip()
    if len(query) < 1:
        return jsonify({"results": []})
    try:
        return jsonify({"results": search_tickers(query)})
    except Exception as e:
        return _err(str(e))
```

Le décorateur enregistre une route GET. `q` est ici une recherche textuelle, pas le dividende. `.strip()` enlève les espaces de bord ; une recherche vide donne une liste vide. Les exceptions sont converties en message. `.get('q','')` lit q et utilise une chaîne vide seulement si la clé manque. `.strip()` transforme par exemple `' Apple '` en `'Apple'`. `@app.get` associe la fonction à une requête HTTP GET ; `try` tente le traitement et `except Exception as e` récupère une éventuelle exception dans e.

### 5.7 Route de données de marché

**Lignes 67–79 de `main.py`**

```python
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
```

`<ticker>` est une partie variable de l’URL transmise à la fonction. Trois lectures se suivent : spot, devise, VIX. Une seule panne fait échouer l’ensemble. `f'...{ticker}...{e}'` insère les valeurs dans le message ; `round(...,4)` limite les décimales affichées.

### 5.8 Page principale

**Lignes 86–88 de `main.py`**

```python
@app.get("/")
def index():
    return render_template("index.html")
```

GET `/` appelle le template `templates/index.html`, présent dans le dossier. Changer le nom demanderait un fichier correspondant. La présence du HTML ne prouve pas que les interactions avec l’API fonctionnent.

### 5.9 Route option : JSON et conversions financières

**Lignes 95–105 de `main.py`**

```python
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
```

Le JSON est lu puis les nombres sont convertis. Diviser r, q et sigma par 100 transforme des pourcentages en décimaux : envoyer sigma=20 signifie 20 %. `get_json(force=True)` tente de décoder le corps en JSON même si l’en-tête HTTP ne l’annonce pas. Avec `force=False`, Flask contrôle le type de contenu. Ni l’un ni l’autre ne vérifie que le JSON contient les bons paramètres financiers. Un champ obligatoire absent provoque `KeyError`.

### 5.10 Route option : valeurs par défaut et validations

**Lignes 106–117 de `main.py`**

```python
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
```

Les défauts sont buy, auto et 30 000. `not in (...)` refuse les types/styles inconnus ; `or` rejette dès qu’une condition de positivité échoue. Il manque des contrôles sur side, method, finitude et nombre de simulations. `int(...)` peut tronquer au lieu de refuser une valeur non entière.

### 5.11 Sélection du modèle réellement appliqué

**Lignes 119–128 de `main.py`**

```python
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
```

Européenne + `monte_carlo` → MC ; européenne + toute autre valeur → BS ; américaine → LSM quel que soit method. `used` conserve le modèle effectivement appliqué. Ajouter `n_steps` au JSON ne le change pas : la route ne lit pas ce champ.

### 5.12 Grecques et profil terminal d’une option

**Lignes 130–143 de `main.py`**

```python
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
```

Les grecques restent BS même pour une américaine. Le signe s’applique seulement aux courbes : en vente, les grecques renvoyées restent longues. La grille représente l’intrinsèque terminal moins la prime, pas les flux d’exercice anticipé d’une américaine.

### 5.13 Sérialiser la réponse option

**Lignes 145–154 de `main.py`**

```python
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
```

Les dictionnaires gardent les noms des grecques ; les tableaux deviennent des listes. `float` normalise les scalaires, `round` ajuste l’affichage. Prix/courbes : 4 décimales ; grecques : 6. `price` reste la prime unitaire non signée. `round(x,4)` garde quatre décimales : 1,23456 devient 1,2346 ; 2 garderait deux décimales et −2 arrondirait à la centaine. `[... for x in ST_grid]` construit une liste en transformant chaque x. `{k: ... for k,v in primaires.items()}` fait la même chose pour un dictionnaire : `.items()` fournit les couples nom/valeur.

### 5.14 Exceptions de la route option

**Lignes 156–159 de `main.py`**

```python
    except KeyError as e:
        return _err(f"Paramètre manquant : {e}")
    except Exception as e:
        return _err(str(e))
```

La première branche traite les clés manquantes ; la seconde traite les autres exceptions. L’ordre va du spécifique au général. Les deux renvoient HTTP 400. Un avertissement NumPy ou un `NaN` peut ne pas déclencher ces branches.

### 5.15 Route forward : lecture et valorisation

**Lignes 166–182 de `main.py`**

```python
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
```

Même lecture JSON, sans volatilité : le forward n’en utilise pas dans ce modèle. Seuls S0 et T sont contrôlés positifs. `F` est un prix de livraison ; `valeur` dépend aussi de K et du sens. Les deux résultats ne doivent pas être confondus.

### 5.16 Route forward : tableau des flux et convention de P&L

**Lignes 184–191 de `main.py`**

```python
        ST_grid = _grid([S0, K])
        payoff = np.array([forward_payoff(st, K, side) for st in ST_grid])
        # Le P&L d'un forward par rapport à aujourd'hui, c'est le payoff à
        # maturité moins la valeur actuelle du contrat (le forward n'a pas de
        # "prime" au sens option ; il se règle contre la valeur déjà engagée).
        pnl = payoff - valeur

        breakevens = _find_breakevens(ST_grid, pnl)
```

La compréhension calcule un payoff par spot puis `np.array` crée le tableau. `pnl=payoff-valeur` soustrait une valeur actuelle à un flux terminal : convention simplifiée. Pour un contrat acquis aujourd’hui et financé, on retrancherait `valeur*exp(r*T)` ; l’historique d’un contrat déjà détenu est une autre question.

### 5.17 Route forward : réponse et erreurs

**Lignes 193–205 de `main.py`**

```python
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
```

Le JSON distingue F, valeur actuelle signée et profils de résultat. Il n’expose pas de grecques forward. Les conversions et exceptions suivent le même mécanisme que la route option.

### 5.18 Table de dispatch des stratégies

**Lignes 212–220 de `main.py`**

```python
STRATEGY_BUILDERS = {
    "covered_call":   lambda p: make_covered_call(p["K"], p["T"], qty=p["qty"]),
    "protective_put": lambda p: make_protective_put(p["K"], p["T"], qty=p["qty"]),
    "straddle":       lambda p: make_straddle(p["K"], p["T"], side=p["side"], qty=p["qty"]),
    "strangle":       lambda p: make_strangle(p["K_put"], p["K_call"], p["T"], side=p["side"], qty=p["qty"]),
    "call_spread":    lambda p: make_call_spread(p["K1"], p["K2"], p["T"], qty=p["qty"]),
    "put_spread":     lambda p: make_put_spread(p["K1"], p["K2"], p["T"], qty=p["qty"]),
    "collar":         lambda p: make_collar(p["K_put"], p["K_call"], p["T"], qty=p["qty"]),
}
```

Chaque clé choisit une lambda, appelée ensuite avec params. Seuls straddle et strangle transmettent le side global ; les autres recettes fixent leurs achats/ventes. Les parenthèses après une fonction l’appellent ; sans cet appel, on conserve simplement l’objet fonction.

### 5.19 Table des strikes requis

**Lignes 223–231 de `main.py`**

```python
STRATEGY_STRIKES = {
    "covered_call":   ["K"],
    "protective_put": ["K"],
    "straddle":       ["K"],
    "strangle":       ["K_put", "K_call"],
    "call_spread":    ["K1", "K2"],
    "put_spread":     ["K1", "K2"],
    "collar":         ["K_put", "K_call"],
}
```

Chaque liste indique les clés de strike à lire. `['K']` contient un nom, pas un strike chiffré. Les spreads attendent K1/K2, strangle/collar K_put/K_call. Cette table sert aussi à la grille mais ne valide pas l’ordre des strikes.

### 5.20 Route stratégie : vérifier le nom

**Lignes 234–241 de `main.py`**

```python
@app.post("/api/strategy")
def api_strategy():
    try:
        data = request.get_json(force=True)

        strategy = data.get("strategy")
        if strategy not in STRATEGY_BUILDERS:
            return _err("Stratégie inconnue.")
```

`.get('strategy')` vaut None si la clé est absente. `strategy not in STRATEGY_BUILDERS` teste les noms disponibles. La route accepte sept recettes, pas une liste libre de legs transmise par le client.

### 5.21 Route stratégie : entrées communes

**Lignes 243–252 de `main.py`**

```python
        S0 = float(data["S0"])
        r = float(data["r"]) / 100
        q = float(data["q"]) / 100
        sigma = float(data["sigma"]) / 100
        T = float(data["T"])
        qty = float(data.get("qty", 1))
        side = data.get("side", "buy")

        if S0 <= 0 or T <= 0 or sigma <= 0:
            return _err("Les paramètres doivent être strictement positifs.")
```

Les taux sont convertis comme pour l’option. `qty` devient un flottant : quantités fractionnaires, nulles ou négatives ne sont pas rejetées. Le test vérifie seulement spot, maturité et volatilité. Lire `side` ne signifie pas que chaque recette l’utilise.

### 5.22 Route stratégie : construire legs et grille

**Lignes 254–261 de `main.py`**

```python
        params = {"T": T, "qty": qty, "side": side}
        for cle in STRATEGY_STRIKES[strategy]:
            params[cle] = float(data[cle])

        legs = STRATEGY_BUILDERS[strategy](params)

        centres = [S0] + [params[cle] for cle in STRATEGY_STRIKES[strategy]]
        ST_grid = _grid(centres)
```

On crée params puis ajoute les strikes exigés. `STRATEGY_BUILDERS[strategy](params)` choisit et exécute le constructeur. `[S0]+[...]` concatène des listes pour former les centres de grille ; par exemple `[100]+[95,110]` donne `[100,95,110]`. Le signe + concatène ici des listes. Sur des tableaux NumPy, + additionnerait au contraire les valeurs selon leurs dimensions.

### 5.23 Route stratégie : déléguer les calculs

**Lignes 263–266 de `main.py`**

```python
        resultat = payoff_strategie(legs, S0, r, q, sigma, ST_grid)
        grecques = greeks_strategie(legs, S0, r, q, sigma)

        breakevens = _find_breakevens(ST_grid, resultat["pnl"])
```

Le moteur calcule les courbes et les grecques, puis l’utilitaire cherche les zéros. Les options des stratégies restent européennes BS : un champ JSON `style='american'` supplémentaire serait ignoré.

### 5.24 Route stratégie : résultats et gestion des erreurs

**Lignes 268–280 de `main.py`**

```python
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
```

`greeks` contient les sensibilités agrégées ; `n_legs=len(legs)` compte les composantes, donc 2 ou 3, pas leur quantité. Le coût initial et les legs ne sont pas renvoyés séparément. Les exceptions utilisent le format commun.

### 5.25 Point d’entrée du programme

**Lignes 283–284 de `main.py`**

```python
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
```

Le test démarre le serveur seulement lorsque le fichier est exécuté directement. En cas d’import, les routes sont déclarées mais `app.run` n’est pas appelé. `debug=True` active le débogage ; `use_reloader=False` désactive seulement le redémarrage automatique, pas le debug.

<a id="oral"></a>
## 6. Limites et questions d’oral

### Ce que l’on peut améliorer

| Point actuel | Correction ou amélioration |
|---|---|
| Color mal parenthésé | Corriger la formule et annoncer le sens du temps |
| Grecques américaines calculées en BS | Utiliser un modèle américain cohérent ou indiquer l’approximation |
| Grecques de l’option seule non signées | Distinguer risque unitaire et risque de position ; traiter Lambda comme une élasticité |
| Zéros exacts ignorés pour les breakevens | Traiter zéros, plateaux et segments délimités par les strikes |
| P&L sans financement ni dividendes de l’action | Fixer la date de mesure et intégrer les flux nécessaires |
| Régression LSM sur le même échantillon que l’évaluation | Apprendre et évaluer sur des trajectoires séparées pour limiter le biais |
| Pas de protection contre `X.std()==0` ou prix BS presque nul | Définir des branches de secours pour régression et Lambda |
| Validations incomplètes | Vérifier nombres finis, catégories, quantités, strikes et budget de simulations |
| Pas de graine ni erreur-type MC | Rendre les simulations reproductibles et mesurer leur incertitude |
| T=0 et sigma=0 non pris en charge | Implémenter les cas limites plutôt que diviser par zéro |
| Données Yahoo sans horodatage exposé | Signaler leur fraîcheur et les pannes ; ne pas promettre du temps réel |

Les modèles supposent notamment une volatilité constante et des dividendes continus. Le code ne traite pas les smiles, sauts, coûts de transaction ou stratégies multi-échéances complètes. Augmenter les simulations ne supprime pas ces limites de modèle.

### Quelques breakevens à savoir retrouver

Pour une unité, en P&L simple et avec des primes hypothétiques cohérentes :

| Position | Zéro du P&L |
|---|---|
| Call long, prime c | K+c |
| Put long, prime p | K−p, si non négatif |
| Straddle long, coût D | K−D et K+D, si dans le domaine S ≥ 0 |
| Strangle long, coût D | K_put−D et K_call+D, si non négatifs |
| Bull call spread, débit D | K1+D, si entre les strikes |
| Bear put spread, débit D | K1−D, si entre les strikes |

Covered call : payoff `min(ST,K)` ; protective put : `max(ST,K)` ; collar : valeur terminale encadrée par les strikes. Leur breakeven dépend du coût total, action comprise. Un segment entier peut aussi avoir un P&L nul : tous les zéros ne sont pas des points isolés.

### Réponses courtes à retenir

**Pourquoi séparer les fichiers ?**  
« Je peux vérifier un calcul financier indépendamment d’une requête web. »

**Pourquoi les indices commencent-ils à 0 ?**  
« C’est la convention Python/NumPy : la première position est 0. L’indice −1 part de la fin. »

**Pourquoi `np.where(itm)[0]` et pas `[1]` ?**  
« Le masque est à un seul axe. `where` renvoie un tuple contenant un seul tableau ; je récupère ce tableau. »

**Pourquoi le degré 2 ?**  
« Il apporte une courbure avec trois coefficients. C’est un compromis numérique à vérifier, pas une vérité financière. »

**Pourquoi r−q dans les simulations ?**  
« Je travaille sous la mesure de valorisation risque-neutre ; je ne prédis pas le rendement historique. »

**Pourquoi garder BS et MC ?**  
« BS sert de référence rapide pour vérifier la simulation, que l’on pourra étendre à d’autres payoffs. »

**Pourquoi remonter le temps en LSM ?**  
« La décision d’exercer aujourd’hui dépend de ce que vaut continuer. Je pars du payoff final connu. »

**Un call américain sans dividende doit-il être exercé tôt ?**  
« Dans le modèle standard sans frictions, sans dividende et avec taux non négatifs, ce n’est pas avantageux. Ces conditions comptent. »

**Pourquoi Vega n’est-il pas déjà divisé par 100 ?**  
« Le moteur dérive par rapport à une volatilité décimale. Une variation d’un point vaut 0,01. »

**Changer un nombre change-t-il toujours la finance ?**  
« Non : le 4 de round change l’affichage, le 121 de linspace change la grille, le 2 de polyfit change l’approximation, et le 2 de sigma**2 appartient à la formule. »

### Vérifications et références

Les blocs reproduisent les fichiers actuels dans leur ordre, avec couverture de toutes les lignes non vides hors commentaires de séparation. Les petits exemples d’indices, masques, boucles et calculs numériques ont été contrôlés séparément. Cela ne constitue pas un test complet de Flask, de Yahoo ou de l’interface.

Les vérifications précédentes du moteur donnent, pour S=K=100, T=1, r=0,05, q=0,02, sigma=0,20 : prix BS ≈ 9,22700551, Delta ≈ 0,58685115 et Vega ≈ 37,90115751. Elles ont aussi confirmé le défaut de Color et celui des breakevens exacts. Un test futur pertinent comparerait les grecques à des différences finies et LSM à un arbre américain.

Pour approfondir les fondements : [MIT — valorisation risque-neutre et Black-Scholes](https://ocw.mit.edu/courses/18-642-topics-in-mathematics-with-applications-in-finance-fall-2024/resources/mit18_642_f24_lec21_pdf/), [Longstaff et Schwartz — article original](https://escholarship.org/uc/item/43n1k4jb), [Cboe — définition du VIX](https://cdn.cboe.com/resources/vix/VIX_Methodology.pdf).

**Usage :** placer ce README à côté des deux fichiers Python. Les exemples pédagogiques sont signalés comme tels ; ils ne remplacent pas les sources et n’ajoutent pas pandas au projet.

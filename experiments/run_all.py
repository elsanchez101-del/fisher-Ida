"""
run_all.py
----------
Reproduce todos los experimentos con datos sintéticos (semillas fijas).
USO: python experiments/run_all.py

  1. Ejemplo a mano (6 puntos).
  2. Experimento A: dos clases gaussianas correlacionadas; Fisher frente a la diferencia de medias
     y a PCA, con 10 repeticiones.
  3. Experimento B: pocos datos y muchas variables (S_W singular); efecto de regularizar.

Guarda las tablas en results/ y las figuras en figures/.
"""
import sys
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SK

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fisher import FisherLDA, class_stats, fisher_J, fisher_direction, theoretical_accuracy, within_scatter

# ----------------------------- PARÁMETROS ---------------------------------
SEED = 2026
FIG, RES = ROOT / "figures", ROOT / "results"
SIG = np.array([[4.0, 1.8], [1.8, 1.0]])        # covarianza común de las dos clases
MU1, MU2 = np.array([0., 0.]), np.array([1., 0.])  # medias de las dos clases
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
C1c, C2c = "#1f77b4", "#ff7f0e"                  # colores de las clases
COLORES = {"Diferencia de medias": "#2ca02c", "Fisher": "k", "PCA (PC1)": "#d62728"}


# ----------------------------- FUNCIONES ----------------------------------
def unit(v):
    """Vector de norma 1 (solo importa la dirección)."""
    return v / np.linalg.norm(v)


def gen(n, rng, mu2=MU2, S=SIG):
    """n puntos de cada clase: clase 0 ~ N(MU1, S), clase 1 ~ N(mu2, S)."""
    X = np.vstack([rng.multivariate_normal(MU1, S, n), rng.multivariate_normal(mu2, S, n)])
    y = np.r_[np.zeros(n), np.ones(n)].astype(int)
    return X, y


def pc1(X):
    """Primera componente principal: autovector de mayor autovalor (no usa las etiquetas)."""
    Xc = X - X.mean(0)
    autovectores = np.linalg.eigh(Xc.T @ Xc)[1]
    return unit(autovectores[:, -1])


def clasificador(w, Xtr, ytr):
    """La misma regla para cualquier dirección w: umbral en el punto medio de las medias
    proyectadas del entrenamiento. Devuelve una función que predice 0 o 1."""
    m0 = (Xtr[ytr == 0] @ w).mean()
    m1 = (Xtr[ytr == 1] @ w).mean()
    t = (m0 + m1) / 2
    signo = 1 if m0 < m1 else -1                 # de qué lado del umbral queda la clase 1
    return lambda X: ((X @ w - t) * signo > 0).astype(int)


def confusion(y, y_pred):
    """Matriz de confusión: filas = clase real, columnas = clase predicha."""
    return np.array([[np.sum((y == i) & (y_pred == j)) for j in (0, 1)] for i in (0, 1)])


def direcciones(Xtr, ytr):
    """Las tres direcciones que comparamos, calculadas con el entrenamiento."""
    return {"Diferencia de medias": unit(Xtr[ytr == 0].mean(0) - Xtr[ytr == 1].mean(0)),
            "Fisher": fisher_direction(Xtr, ytr),
            "PCA (PC1)": pc1(Xtr)}


def angulo(u, v):
    """Ángulo en grados entre dos direcciones (sin importar el signo)."""
    return float(np.degrees(np.arccos(min(1, abs(u @ v)))))


# =========================== 1) EJEMPLO A MANO =============================
C1 = np.array([[1, 1], [2, 1], [2, 2]], float)
C2 = np.array([[4, 3], [5, 3], [5, 4]], float)
X = np.vstack([C1, C2])
y = np.array([0, 0, 0, 1, 1, 1])
w = fisher_direction(X, y)
w4 = w / w[1]                                    # escalada para que se vea (4, 1)

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(*C1.T, c=C1c, label="Clase 1")
ax.scatter(*C2.T, c=C2c, label="Clase 2")
xs = np.linspace(0.5, 5.5, 50)
ax.plot(xs, 15 - 4 * xs, "k--", label=r"$4x_1+x_2=15$")     # frontera: 4 x1 + x2 = 15
ax.set_xlim(.5, 5.5); ax.set_ylim(.5, 4.8)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend()
ax.set_title("Ejemplo a mano: frontera de Fisher")
fig.tight_layout(); fig.savefig(FIG / "ej_mano.png"); plt.close(fig)

exp1 = {"w_proporcional": list(map(float, w4)),
        "proyecciones": (X @ np.array([4, 1])).tolist(),
        "J": fisher_J(w, X, y)}
_, mus, Ss, _ = class_stats(X, y)
d = mus[0] - mus[1]
exp1["J_mahalanobis"] = float(d @ np.linalg.solve(sum(Ss), d))   # máximo teórico d^T S_W^{-1} d


# ============ 2) EXPERIMENTO A: gaussianas correlacionadas ==================
filas = []
for rep in range(10):
    rng = np.random.default_rng(SEED + rep)
    Xa, ya = gen(150, rng)
    idx = rng.permutation(len(ya))
    tr, te = idx[:150], idx[150:]                # mitad entrenamiento, mitad prueba
    Xtr, ytr, Xte, yte = Xa[tr], ya[tr], Xa[te], ya[te]
    dirs = direcciones(Xtr, ytr)
    for nombre, wv in dirs.items():
        clf = clasificador(wv, Xtr, ytr)
        filas.append(dict(rep=rep, metodo=nombre,
                          J_train=fisher_J(wv, Xtr, ytr) * (len(ytr) - 2),   # J con S_W/(N-2)
                          acc_test=float(np.mean(clf(Xte) == yte)),
                          angulo_vs_fisher=angulo(wv, dirs["Fisher"])))
    # verificación: la dirección de Fisher coincide con la de scikit-learn
    wsk = unit(SK(solver="lsqr").fit(Xtr, ytr).coef_[0])
    filas[-1]["dif_sklearn"] = float(1 - abs(dirs["Fisher"] @ wsk))

df = pd.DataFrame(filas)
df.to_csv(RES / "exp_gauss_reps.csv", index=False)
summ = df.groupby("metodo")[["J_train", "acc_test", "angulo_vs_fisher"]].agg(["mean", "std"]).round(3)
summ.to_csv(RES / "exp_gauss_resumen.csv")
sk_diff = float(df["dif_sklearn"].max())

# Exactitud teórica (con infinitos datos), usando las medias y la covarianza verdaderas
Delta = MU1 - MU2
w_teo = {"Diferencia de medias": Delta,
         "Fisher": np.linalg.solve(SIG, Delta),                                   # óptimo de Bayes
         "PCA (PC1)": np.linalg.eigh(SIG + np.outer(Delta, Delta) / 4)[1][:, -1]}  # PC1 de la mezcla
acc_teo = {k: theoretical_accuracy(v, MU1, MU2, SIG) for k, v in w_teo.items()}
acc_obt = df.groupby("metodo")["acc_test"].mean().to_dict()
J_teo = float(Delta @ np.linalg.solve(SIG, Delta))

# --- Figuras de la repetición 0 ---
rng = np.random.default_rng(SEED)
Xa, ya = gen(150, rng)
idx = rng.permutation(300)
tr, te = idx[:150], idx[150:]
Xtr, ytr, Xte, yte = Xa[tr], ya[tr], Xa[te], ya[te]
dirs = direcciones(Xtr, ytr)

# Figura: las nubes de puntos y las tres direcciones
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(*Xtr[ytr == 0].T, s=12, c=C1c, alpha=.5, label="Clase 1 (train)")
ax.scatter(*Xtr[ytr == 1].T, s=12, c=C2c, alpha=.5, label="Clase 2 (train)")
centro = Xtr.mean(0)
for nombre, wv in dirs.items():
    ax.annotate("", centro + 4.5 * wv, centro, arrowprops=dict(arrowstyle="->", color=COLORES[nombre], lw=2))
    ax.plot([], [], c=COLORES[nombre], label=nombre)
wf = dirs["Fisher"]
modelo = FisherLDA().fit(Xtr, ytr)
perp = np.array([-wf[1], wf[0]])                 # la frontera es perpendicular a w
p0 = wf * modelo.t_
ax.plot(*(np.c_[p0 - 8 * perp, p0 + 8 * perp]), "k--", lw=1, label="Frontera Fisher")
ax.set_xlim(-6, 7); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Gaussianas correlacionadas: direcciones")
fig.tight_layout(); fig.savefig(FIG / "gauss_direcciones.png"); plt.close(fig)

# Figura: histogramas de las proyecciones en prueba
fig, axs = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for ax, (nombre, wv) in zip(axs, dirs.items()):
    z = Xte @ wv
    ax.hist(z[yte == 0], 25, alpha=.6, color=C1c)
    ax.hist(z[yte == 1], 25, alpha=.6, color=C2c)
    ax.set_title(f"{nombre}\nJ={fisher_J(wv, Xtr, ytr)*(len(ytr)-2):.2f}", fontsize=9)
fig.suptitle("Histogramas de proyecciones (prueba)", y=1.02, fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "gauss_histogramas.png", bbox_inches="tight"); plt.close(fig)

cm = {nombre: confusion(yte, clasificador(wv, Xtr, ytr)(Xte)).tolist() for nombre, wv in dirs.items()}

# Figura: exactitud de cada método en las 10 repeticiones
fig, ax = plt.subplots(figsize=(5, 3.2))
for nombre in dirs:
    s = df[df.metodo == nombre]
    ax.plot(s.rep + 1, s.acc_test, "o-", label=nombre, c=COLORES[nombre], ms=4)
ax.set_xlabel("Repetición"); ax.set_ylabel("Exactitud (prueba)"); ax.legend(fontsize=7)
ax.set_title("10 repeticiones (semilla 2026+k)")
fig.tight_layout(); fig.savefig(FIG / "gauss_reps.png"); plt.close(fig)


# ========= 3) EXPERIMENTO B: pocos datos, muchas variables (S_W singular) =========
def gen_hd(n, p, rng, dprime=1.5):
    """Las 2 primeras variables como antes; luego 1 copia con ruido de x1 y p - 3 de puro ruido."""
    X, y = gen(n, rng, mu2=np.array([dprime, 0.]))
    redundante = X[:, [0]] + 0.01 * rng.normal(size=(len(y), 1))
    ruido = rng.normal(size=(len(y), p - 3))
    return np.hstack([X, redundante, ruido]), y


gammas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1, 3, 10, 30, 100]


def cv_gamma(Xtr, ytr, rng, k=5):
    """Elige gamma con validación cruzada de k pliegues usando SOLO el entrenamiento."""
    pliegues = np.array_split(rng.permutation(len(ytr)), k)
    puntaje = np.zeros(len(gammas))
    for f in pliegues:
        m = np.ones(len(ytr), bool)
        m[f] = False                             # m = datos para ajustar; f = datos para validar
        for i, g in enumerate(gammas):
            puntaje[i] += FisherLDA(g).fit(Xtr[m], ytr[m]).score(Xtr[f], ytr[f])
    return gammas[int(np.argmax(puntaje))]       # si hay empate, gana el gamma menor


n, p, reps = 15, 40, 10                           # 15 datos por clase y 40 variables
# Con gamma = 0 y N < p, S_W es singular y fisher_direction avisa: aquí ese caso es intencional.
warnings.filterwarnings("ignore", message="S_W es numéricamente singular")
acc = {g: [] for g in gammas}
chosen, acc_cv, acc_pinv, acc_dm = [], [], [], []
for rep in range(reps):
    rng = np.random.default_rng(SEED + 100 + rep)
    Xtr, ytr = gen_hd(n, p, rng)
    Xte, yte = gen_hd(500, p, rng)
    for g in gammas:                             # Fisher con S_W + lambda I para cada gamma
        acc[g].append(FisherLDA(g).fit(Xtr, ytr).score(Xte, yte))
    acc_pinv.append(FisherLDA(method="pinv").fit(Xtr, ytr).score(Xte, yte))   # pseudoinversa (solo para comparar)
    w_dm = unit(Xtr[ytr == 0].mean(0) - Xtr[ytr == 1].mean(0))               # referencia: diferencia de medias
    acc_dm.append(float(np.mean(clasificador(w_dm, Xtr, ytr)(Xte) == yte)))
    g = cv_gamma(Xtr, ytr, rng)
    chosen.append(g)
    acc_cv.append(FisherLDA(g).fit(Xtr, ytr).score(Xte, yte))
    if rep == 0:                                 # diagnóstico de la singularidad de S_W
        SW = within_scatter(Xtr, ytr)
        ev = np.linalg.eigvalsh(SW)
        sing = dict(N=int(len(ytr)), p=p, rango_max=int(len(ytr) - 2),
                    n_autovalores_nulos=int(np.sum(ev < 1e-9 * ev.max())), cond=float(np.linalg.cond(SW)))

rg = pd.DataFrame({"gamma": gammas,
                   "acc_media": [np.mean(acc[g]) for g in gammas],
                   "acc_std": [np.std(acc[g], ddof=1) for g in gammas]})
rg.to_csv(RES / "exp_reg.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 3.2))
xs = np.arange(len(gammas))
ax.errorbar(xs, rg.acc_media, rg.acc_std, fmt="o-", capsize=3, label=r"Fisher con $S_W+\lambda I$")
ax.axhline(np.mean(acc_pinv), color="#d62728", ls="--", lw=1, label="Pseudoinversa (sin regularizar)")
ax.axhline(np.mean(acc_dm), color="#2ca02c", ls=":", lw=1.5, label="Diferencia de medias")
ax.legend(fontsize=7, loc="lower right")
ax.set_xticks(xs); ax.set_xticklabels(["0"] + [f"{g:g}" for g in gammas[1:]], fontsize=7)
ax.set_xlabel(r"$\gamma$  ($\lambda=\gamma\,\mathrm{tr}(S_W)/p$)"); ax.set_ylabel("Exactitud (prueba)")
ax.set_title(f"Regularización: N={2*n}, p={p}")
fig.tight_layout(); fig.savefig(FIG / "regularizacion.png"); plt.close(fig)


# ============================ RESUMEN ======================================
resumen = dict(manual=exp1, sklearn_max_dif=sk_diff, confusion_rep0=cm,
               exactitud_teorica=acc_teo, exactitud_obtenida=acc_obt, J_teorico=J_teo, singularidad=sing,
               gamma_cv=[float(g) for g in chosen],
               acc_cv_media=float(np.mean(acc_cv)), acc_cv_std=float(np.std(acc_cv, ddof=1)),
               acc_pinv_media=float(np.mean(acc_pinv)), acc_pinv_std=float(np.std(acc_pinv, ddof=1)),
               acc_dif_medias_media=float(np.mean(acc_dm)), acc_dif_medias_std=float(np.std(acc_dm, ddof=1)))
json.dump(resumen, open(RES / "resumen.json", "w"), indent=1, default=float)

print(summ.to_string())
print(rg.round(3).to_string())
print(json.dumps(dict(sing=sing, cm=cm, sk=sk_diff, cv=float(np.mean(acc_cv)), g=[float(g) for g in chosen], J=exp1),
                 default=float))

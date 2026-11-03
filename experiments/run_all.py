"""Reproduce todos los experimentos. Uso: python experiments/run_all.py  (semilla fija)."""
import sys, pathlib, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SK
from fisher import *

SEED = 2026
FIG, RES = ROOT / "figures", ROOT / "results"
SIG = np.array([[4.0, 1.8], [1.8, 1.0]])
MU1, MU2 = np.array([0., 0.]), np.array([1., 0.])
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
C1c, C2c = "#1f77b4", "#ff7f0e"

def unit(v): return v / np.linalg.norm(v)

def gen(n, rng, mu2=MU2, S=SIG):
    X = np.vstack([rng.multivariate_normal(MU1, S, n), rng.multivariate_normal(mu2, S, n)])
    return X, np.r_[np.zeros(n), np.ones(n)].astype(int)

def pc1(X):
    Xc = X - X.mean(0)
    return unit(np.linalg.eigh(Xc.T @ Xc)[1][:, -1])

def thresh_clf(w, Xtr, ytr):
    """Misma regla para todo w: umbral en el punto medio de medias proyectadas (entrenamiento)."""
    m0, m1 = (Xtr[ytr == 0] @ w).mean(), (Xtr[ytr == 1] @ w).mean()
    t, s = (m0 + m1) / 2, (1 if m0 < m1 else -1)
    return lambda X: ((X @ w - t) * s > 0).astype(int)

def confusion(y, yh): return np.array([[np.sum((y == i) & (yh == j)) for j in (0, 1)] for i in (0, 1)])

# ---------- Exp 1: ejemplo a mano ----------
C1 = np.array([[1,1],[2,1],[2,2]], float); C2 = np.array([[4,3],[5,3],[5,4]], float)
X = np.vstack([C1, C2]); y = np.array([0,0,0,1,1,1])
w = fisher_direction(X, y); w4 = w / w[1]
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(*C1.T, c=C1c, label="Clase 1"); ax.scatter(*C2.T, c=C2c, label="Clase 2")
xs = np.linspace(0.5, 5.5, 50); ax.plot(xs, 15 - 4 * xs, "k--", label=r"$4x_1+x_2=15$")
ax.set_xlim(.5, 5.5); ax.set_ylim(.5, 4.8); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.legend()
ax.set_title("Ejemplo a mano: frontera de Fisher"); fig.tight_layout(); fig.savefig(FIG / "ej_mano.png"); plt.close(fig)
exp1 = {"w_proporcional": list(map(float, w4)), "proyecciones": (X @ np.array([4, 1])).tolist(),
        "J": fisher_J(w, X, y)}
_, mus, Ss, _ = class_stats(X, y); d = mus[0] - mus[1]
exp1["J_mahalanobis"] = float(d @ np.linalg.solve(sum(Ss), d))

# ---------- Exp 2-4: gaussianas correlacionadas, 10 repeticiones ----------
rows = []
for rep in range(10):
    rng = np.random.default_rng(SEED + rep)
    Xa, ya = gen(150, rng); idx = rng.permutation(len(ya)); tr, te = idx[:150], idx[150:]
    Xtr, ytr, Xte, yte = Xa[tr], ya[tr], Xa[te], ya[te]
    dirs = {"Diferencia de medias": unit(Xtr[ytr == 0].mean(0) - Xtr[ytr == 1].mean(0)),
            "Fisher": fisher_direction(Xtr, ytr), "PCA (PC1)": pc1(Xtr)}
    for name, wv in dirs.items():
        clf = thresh_clf(wv, Xtr, ytr)
        rows.append(dict(rep=rep, metodo=name, J_train=fisher_J(wv, Xtr, ytr) * (len(ytr) - 2),
                         acc_test=float(np.mean(clf(Xte) == yte)),
                         angulo_vs_fisher=float(np.degrees(np.arccos(min(1, abs(wv @ dirs["Fisher"])))))))
    # verificación contra sklearn
    wsk = unit(SK(solver="lsqr").fit(Xtr, ytr).coef_[0])
    rows[-1]["dif_sklearn"] = float(1 - abs(dirs["Fisher"] @ wsk))
import pandas as pd
df = pd.DataFrame(rows); df.to_csv(RES / "exp_gauss_reps.csv", index=False)
summ = df.groupby("metodo")[["J_train", "acc_test", "angulo_vs_fisher"]].agg(["mean", "std"]).round(3)
summ.to_csv(RES / "exp_gauss_resumen.csv")
sk_diff = float(df["dif_sklearn"].max())

# Repetición 0: figuras detalladas
rng = np.random.default_rng(SEED); Xa, ya = gen(150, rng); idx = rng.permutation(300); tr, te = idx[:150], idx[150:]
Xtr, ytr, Xte, yte = Xa[tr], ya[tr], Xa[te], ya[te]
dirs = {"Diferencia de medias": unit(Xtr[ytr == 0].mean(0) - Xtr[ytr == 1].mean(0)),
        "Fisher": fisher_direction(Xtr, ytr), "PCA (PC1)": pc1(Xtr)}
cols = {"Diferencia de medias": "#2ca02c", "Fisher": "k", "PCA (PC1)": "#d62728"}
fig, ax = plt.subplots(figsize=(6.2, 4.6))
ax.scatter(*Xtr[ytr == 0].T, s=12, c=C1c, alpha=.5, label="Clase 1 (train)")
ax.scatter(*Xtr[ytr == 1].T, s=12, c=C2c, alpha=.5, label="Clase 2 (train)")
cen = Xtr.mean(0)
for n, wv in dirs.items(): ax.annotate("", cen + 4.5 * wv, cen, arrowprops=dict(arrowstyle="->", color=cols[n], lw=2)); ax.plot([], [], c=cols[n], label=n)
wf = dirs["Fisher"]; clf = FisherLDA().fit(Xtr, ytr)
perp = np.array([-wf[1], wf[0]]); p0 = wf * clf.t_
ax.plot(*(np.c_[p0 - 8 * perp, p0 + 8 * perp]), "k--", lw=1, label="Frontera Fisher")
ax.set_xlim(-6, 7); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal"); ax.legend(fontsize=7, loc="lower right")
ax.set_title("Gaussianas correlacionadas: direcciones"); fig.tight_layout(); fig.savefig(FIG / "gauss_direcciones.png"); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for ax, (n, wv) in zip(axs, dirs.items()):
    z = Xte @ wv
    ax.hist(z[yte == 0], 25, alpha=.6, color=C1c); ax.hist(z[yte == 1], 25, alpha=.6, color=C2c)
    ax.set_title(f"{n}\nJ={fisher_J(wv, Xtr, ytr)*(len(ytr)-2):.2f}", fontsize=9)
fig.suptitle("Histogramas de proyecciones (prueba)", y=1.02, fontsize=10); fig.tight_layout(); fig.savefig(FIG / "gauss_histogramas.png", bbox_inches="tight"); plt.close(fig)

cm = {n: confusion(yte, thresh_clf(wv, Xtr, ytr)(Xte)).tolist() for n, wv in dirs.items()}

fig, ax = plt.subplots(figsize=(5, 3.2))
for n in dirs:
    s = df[df.metodo == n]; ax.plot(s.rep + 1, s.acc_test, "o-", label=n, c=cols[n], ms=4)
ax.set_xlabel("Repetición"); ax.set_ylabel("Exactitud (prueba)"); ax.legend(fontsize=7); ax.set_title("10 repeticiones (semilla 2026+k)")
fig.tight_layout(); fig.savefig(FIG / "gauss_reps.png"); plt.close(fig)

# ---------- Exp 5: regularización (pocas muestras + variable redundante) ----------
def gen_hd(n, p, rng, dprime=1.5):
    """Primeras 2 variables como antes; el resto: 1 redundante (copia ruidosa de x1) + ruido independiente."""
    X, y = gen(n, rng, mu2=np.array([dprime, 0.]))
    red = X[:, [0]] + 0.01 * rng.normal(size=(len(y), 1))
    noise = rng.normal(size=(len(y), p - 3))
    return np.hstack([X, red, noise]), y

gammas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1, 3]
def cv_gamma(Xtr, ytr, rng, k=5):
    folds = np.array_split(rng.permutation(len(ytr)), k); sc = np.zeros(len(gammas))
    for f in folds:
        m = np.ones(len(ytr), bool); m[f] = False
        for i, g in enumerate(gammas):
            sc[i] += FisherLDA(g).fit(Xtr[m], ytr[m]).score(Xtr[f], ytr[f])
    return gammas[int(np.argmax(sc))]  # empates -> menor gamma

n, p, reps = 15, 40, 10
acc = {g: [] for g in gammas}; chosen = []; acc_cv = []
for rep in range(reps):
    rng = np.random.default_rng(SEED + 100 + rep)
    Xtr, ytr = gen_hd(n, p, rng); Xte, yte = gen_hd(500, p, rng)
    for g in gammas: acc[g].append(FisherLDA(g).fit(Xtr, ytr).score(Xte, yte))
    g = cv_gamma(Xtr, ytr, rng); chosen.append(g); acc_cv.append(FisherLDA(g).fit(Xtr, ytr).score(Xte, yte))
    if rep == 0:
        SW = within_scatter(Xtr, ytr); ev = np.linalg.eigvalsh(SW)
        sing = dict(N=int(len(ytr)), p=p, rango_max=int(len(ytr) - 2), n_autovalores_nulos=int(np.sum(ev < 1e-9 * ev.max())), cond=float(np.linalg.cond(SW)))
rg = pd.DataFrame({"gamma": gammas, "acc_media": [np.mean(acc[g]) for g in gammas], "acc_std": [np.std(acc[g], ddof=1) for g in gammas]})
rg.to_csv(RES / "exp_reg.csv", index=False)
fig, ax = plt.subplots(figsize=(5, 3.2))
xs = np.arange(len(gammas)); ax.errorbar(xs, rg.acc_media, rg.acc_std, fmt="o-", capsize=3)
ax.set_xticks(xs); ax.set_xticklabels(["0"] + [f"{g:g}" for g in gammas[1:]], fontsize=7)
ax.set_xlabel(r"$\gamma$  ($\lambda=\gamma\,\mathrm{tr}(S_W)/p$)"); ax.set_ylabel("Exactitud (prueba)")
ax.set_title(f"Regularización: N={2*n}, p={p}"); fig.tight_layout(); fig.savefig(FIG / "regularizacion.png"); plt.close(fig)

json.dump(dict(manual=exp1, sklearn_max_dif=sk_diff, confusion_rep0=cm, singularidad=sing, gamma_cv=[float(g) for g in chosen],
               acc_cv_media=float(np.mean(acc_cv)), acc_cv_std=float(np.std(acc_cv, ddof=1))), open(RES / "resumen.json", "w"), indent=1, default=float)
print(summ.to_string()); print(rg.round(3).to_string()); print(json.dumps(dict(sing=sing, cm=cm, sk=sk_diff, cv=float(np.mean(acc_cv)), g=[float(g) for g in chosen], J=exp1), default=float))

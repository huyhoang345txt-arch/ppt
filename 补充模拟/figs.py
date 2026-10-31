import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Polygon

OUT = "assets"
os.makedirs(OUT, exist_ok=True)

INK = "#1B2333"
SLATE = "#5B6575"
JADE = "#16866E"
SEAL = "#B83A2E"
AMBER = "#C98A1E"
PURPLE = "#6B5CA5"
GRID = "#E3E6EA"

for f in font_manager.findSystemFonts():
    if "wqy-zenhei" in f.lower() or "wqy_zenhei" in f.lower():
        font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["WenQuanYi Zen Hei", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.edgecolor": "#9AA3AF",
    "axes.linewidth": 0.8,
    "font.size": 13,
})


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------- phase diagram 1: fixed m ----------------
def phase1():
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=220)
    ax.add_patch(Polygon([(0, 0), (0.5, 0), (0, 0.5)], color="#CFE9E0", lw=0))
    ax.add_patch(Polygon([(0, 0.5), (0.5, 0), (0, 1)], color="#F4E0B6", lw=0))
    ax.add_patch(Polygon([(0, 1), (0.5, 0), (0.6, 0), (0.6, 1)], facecolor="#EEF0F2", edgecolor="#C3C8CF", hatch="///", lw=0))
    ax.plot([0, 0.5], [0.5, 0], color=JADE, lw=2.6)
    ax.plot([0, 0.5], [1, 0], color=SEAL, lw=2.4, ls=(0, (5, 3)))
    ax.text(0.07, 0.15, "textGrain 可检测\n求和检验已最优", fontsize=13, color="#0D5747", weight="bold")
    ax.text(0.012, 0.515, "Gumbel-max /\nmaximal coupling 可检测\ntextGrain 不可检测", fontsize=10, color="#7A5210", weight="bold", linespacing=1.35)
    ax.text(0.45, 0.50, "三者都\n不可检测", fontsize=12.5, color=SLATE, weight="bold", ha="center")
    ax.plot([0], [2 / 3], "o", ms=9, mfc="white", mec=INK, mew=2, clip_on=False, zorder=5)
    ax.annotate("Example 1：$(p,q)=(0,\\,2/3)$", xy=(0.008, 2 / 3), xytext=(0.13, 0.80), fontsize=10.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#C3C8CF", lw=0.8),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=JADE, lw=2.6, label="$p+q=1/2$：textGrain（$\\beta_n\\asymp 1$）"),
                       Line2D([], [], color=SEAL, lw=2.2, ls=(0, (5, 3)), label="$2p+q=1$：Gumbel-max / max. coupling")],
              loc="upper right", fontsize=10.5, frameon=True, facecolor="white", edgecolor="#C3C8CF", framealpha=1)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1)
    ax.set_xlabel("$p$：编辑强度   $\\varepsilon_n\\asymp n^{-p}$")
    ax.set_ylabel("$q$：模型确定性   $\\Delta_n\\asymp n^{-q}$")
    clean(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/phase1.png", transparent=False, facecolor="white")
    plt.close(fig)


# ---------------- phase diagram 2: growing m, s = 0.4 ----------------
def phase2(s=0.4):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=220)
    kink = ((1 - s) / 2, s)
    top = (1 + s) / 2
    ax.add_patch(Polygon([(0, 0), (0.5, 0), (0, 0.5)], color="#CFE9E0", lw=0))
    ax.add_patch(Polygon([(0, 0.5), (0.5, 0), kink, (0, top)], color="#DCD6EE", lw=0))
    ax.add_patch(Polygon([(0, top), kink, (0, 1)], color="#F2F3F5", lw=0))
    ax.add_patch(Polygon([(0, 1), (0.5, 0), (0.6, 0), (0.6, 1)], facecolor="#EEF0F2", edgecolor="#C3C8CF", hatch="///", lw=0))
    ax.plot([0, 0.5], [0.5, 0], color=JADE, lw=2.6)
    ax.plot([0.5, kink[0], 0], [0, kink[1], top], color=PURPLE, lw=2.6)
    ax.plot([0, 0.5], [1, 0], color=SEAL, lw=2.0, ls=(0, (5, 3)))
    ax.plot(*kink, "o", ms=8, mfc="white", mec=PURPLE, mew=2, zorder=5)
    ax.text(0.05, 0.13, "求和检验与\nAlgorithm 1 都可以", fontsize=12.5, color="#0D5747", weight="bold")
    ax.text(0.135, 0.43, "可检测\n但求和检验失效\n只有 Alg. 1 能做到", fontsize=10.5, color="#3F3473", weight="bold", va="center", linespacing=1.25)
    ax.text(0.012, 0.725, "$s=0.4$ 时\n不可检测", fontsize=10, color=SLATE)
    ax.text(0.47, 0.52, "任何 $s$ 都\n不可检测", fontsize=12.5, color=SLATE, weight="bold", ha="center")
    ax.annotate("拐点 $q=s$", xy=kink, xytext=(0.36, 0.33), fontsize=11, color=PURPLE,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"),
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=1))
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=JADE, lw=2.6, label="$p+q=1/2$：求和检验的边界"),
                       Line2D([], [], color=PURPLE, lw=2.6, label="$2p+2q-\\min(s,q)=1$：最优边界"),
                       Line2D([], [], color=SEAL, lw=2.0, ls=(0, (5, 3)), label="$2p+q=1$")],
              loc="upper right", fontsize=10.5, frameon=True, facecolor="white", edgecolor="#C3C8CF", framealpha=1)
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 1)
    ax.set_xlabel("$p$：编辑强度   $\\varepsilon_n\\asymp n^{-p}$")
    ax.set_ylabel("$q$：模型确定性   $\\Delta_n\\asymp n^{-q}$")
    ax.set_title("列数 $m_n\\asymp n^{s}$，$s=0.4$，$\\beta_t\\geq\\beta_0$", fontsize=12.5, color=SLATE, loc="left")
    clean(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/phase2.png", facecolor="white")
    plt.close(fig)


# ---------------- simulation: signal strength ----------------
def signal(res):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=220)
    ax = axes[0]
    bins = np.array(res["dens_bins"]); mid = (bins[:-1] + bins[1:]) / 2
    y = np.linspace(0.001, 1, 400)
    ax.axhline(1, color="#9AA3AF", lw=2, ls=(0, (4, 3)))
    ax.plot(y, y ** (1 / 0.7 - 1) + y ** (1 / 0.3 - 1), color=SEAL, lw=2.2)
    ax.step(mid, res["dens_tg"], where="mid", color=JADE, lw=2.2)
    ax.text(0.03, 1.06, "$H_0$：$U(0,1)$", color=SLATE, fontsize=12)
    ax.text(0.50, 0.35, "Gumbel-max", color=SEAL, fontsize=12.5, weight="bold")
    ax.text(0.05, 0.80, "textGrain", color=JADE, fontsize=12.5, weight="bold")
    ax.set_xlabel("枢轴量 $Y$"); ax.set_ylabel("密度 $f_1(y)$")
    ax.set_title("(a) 带水印时 $Y$ 的密度（$\\Delta=0.3$，$B=m=2$）", fontsize=12.5, loc="left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 2.1)
    ax.grid(axis="y", color=GRID, lw=0.8); clean(ax)

    ax = axes[1]
    d = np.array(res["deltas"]); tg = np.array([t[0] for t in res["tg"]]); gm = np.array(res["gm"])
    ax.loglog(d, gm, "o-", color=SEAL, lw=2, ms=7)
    ax.loglog(d, tg, "s-", color=JADE, lw=2, ms=7)
    ax.text(0.006, 0.075, "Gumbel-max\n$\\approx\\Delta\\log(1/\\Delta)$", color=SEAL, fontsize=12, weight="bold")
    ax.text(0.16, 0.022, "textGrain\n$\\approx 0.74\\,\\Delta$", color=JADE, fontsize=12, weight="bold")
    r0 = gm[0] / tg[0]; r1 = gm[-1] / tg[-1]
    ax.text(0.03, 0.05, f"两者之比：{r0:.1f}（$\\Delta$=0.2） → {r1:.1f}（$\\Delta$=0.002）",
            transform=ax.transAxes, fontsize=11.5, color=INK, va="bottom")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.2", "0.1", "0.05", "0.02", "0.01", "0.005", "0.002"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.001, 0.003, 0.01, 0.03, 0.1, 0.3]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.001", "0.003", "0.01", "0.03", "0.1", "0.3"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("$\\Delta(P)=1-\\max_w P_w$"); ax.set_ylabel("$\\mathbb{E}_1 h(Y)-\\mathbb{E}_0 h(Y)$")
    ax.set_title("(b) 求和得分 $h(y)=-\\log(1-y)$ 的均值漂移", fontsize=12.5, loc="left")
    ax.grid(which="major", color=GRID, lw=0.8); clean(ax)
    ax.invert_xaxis()
    fig.tight_layout(w_pad=3)
    fig.savefig(f"{OUT}/signal.png", facecolor="white")
    plt.close(fig)


def power(res):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=220)
    cfg = [
        ("A", res["epsA"], res["pwA"], f"(A) 固定列数：$m=2$，$\\Delta=0.2$，$n={res['nA']}$"),
        ("B", res["epsB"], res["pwB"], f"(B) 列数很大：$m=4096$，$\\Delta=1/4096$，$n=3\\times10^5$"),
    ]
    for ax, (k, eps, pw, title) in zip(axes, cfg):
        eps = np.array(eps); s = np.array([p[0] for p in pw]); c = np.array([p[1] for p in pw])
        ax.axhline(0.05, color="#9AA3AF", lw=1.4, ls=(0, (4, 3)))
        ax.plot(eps, s, "o-", color=JADE, lw=2.2, ms=7)
        ax.plot(eps, c, "D-", color=PURPLE, lw=2.2, ms=6.5)
        ax.set_ylim(0, 1.02); ax.set_xlim(eps.min(), eps.max())
        ax.set_xlabel("保留水印的比例 $\\varepsilon$"); ax.set_ylabel("功效（拒绝 $H_0$ 的频率）")
        ax.set_title(title, fontsize=12.5, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8); clean(ax)
        ax.text(eps.max(), 0.07, "$\\alpha=0.05$", color=SLATE, fontsize=11, ha="right")
        i = len(eps) - 2
        if k == "A":
            ax.text(eps[i] - 0.02, s[i] + 0.06, "求和检验 $T_n$", color=JADE, fontsize=12.5, weight="bold", ha="right")
            ax.text(eps[i] + 0.0, c[i] - 0.13, "Algorithm 1", color=PURPLE, fontsize=12.5, weight="bold")
        else:
            ax.text(0.60, 0.27, "求和检验 $T_n$", color=JADE, fontsize=12.5, weight="bold")
            ax.text(eps[i] - 0.03, c[i] + 0.07, "Algorithm 1", color=PURPLE, fontsize=12.5, weight="bold", ha="right")
    fig.tight_layout(w_pad=3)
    fig.savefig(f"{OUT}/power.png", facecolor="white")
    plt.close(fig)


# ---------------- formulas ----------------
F = {
    "entropy": r"$H(P)=-\sum_{w}P_w\log P_w,\qquad \Delta(P)=1-\max_w P_w$",
    "rg": r"$z=\frac{|s|_G-\gamma T}{\sqrt{T\gamma(1-\gamma)}}$",
    "gm": r"$W=\mathrm{arg\,max}_{w}\ \{\log P_w+Z_w\}\ \Rightarrow\ \mathbb{P}(W=w)=P_w$",
    "tg1": r"$\pi_{P,\beta}\in\mathrm{arg\,max}_{\pi}\ \sum_{b=1}^{B}\sum_{j=1}^{m}\pi(b,j)\,G_{bj}$",
    "tg2": r"$\mathrm{s.t.}\quad \pi\in\Pi(P^{\mathrm{blk}},m^{-1}\mathbf{1}),\quad D_{\mathrm{KL}}(\pi\,\|\,P^{\mathrm{blk}}\otimes m^{-1}\mathbf{1})\leq\beta H(P)$",
    "tg3": r"$Q(w\mid J=j,g,G)=m\,\pi_{P,\beta}(g(w),j)\cdot\frac{P_w}{P^{\mathrm{blk}}_{g(w)}}$",
    "unb": r"$\mathbb{P}(W=w\mid g,G)=\sum_{j=1}^{m}\frac{1}{m}\cdot m\,\pi(g(w),j)\,\frac{P_w}{P^{\mathrm{blk}}_{g(w)}}=P^{\mathrm{blk}}_{g(w)}\cdot\frac{P_w}{P^{\mathrm{blk}}_{g(w)}}=P_w$",
    "beta0": r"$\beta=0\ \Rightarrow\ \pi_{P,0}=P^{\mathrm{blk}}\otimes m^{-1}\mathbf{1}$",
    "pivot": r"$Y_t=F_G\left(Z_{t,\,g(W_t),\,J_t}\right),\qquad F_G(z)=e^{-e^{-z}}$",
    "pit": r"$H_0:\ Y_t\sim U(0,1)\ \Rightarrow\ -\log(1-Y_t)\sim\mathrm{Exp}(1)$",
    "sum": r"$T_n=-\sum_{t=1}^{n}\log(1-Y_t)\ \sim\ \mathrm{Gamma}(n,1)$",
    "fisher": r"$2T_n=-2\sum_{t=1}^{n}\log p_t\ \sim\ \chi^2_{2n},\qquad p_t=1-Y_t$",
    "h0": r"$H_0:\ Y_t\mid\mathcal{F}_{t-1}\ \sim\ U(0,1),\quad 1\leq t\leq n$",
    "h1": r"$H_1:\ Y_t\mid\mathcal{F}_{t-1}\ \sim\ (1-\varepsilon_n)+\varepsilon_n\,f_{1,t}$",
    "rates": r"$\varepsilon_n\asymp n^{-p},\qquad \Delta_n\asymp n^{-q},\qquad \beta_n\asymp n^{-r}$",
    "risk": r"$R_n=\inf_{\phi_n}\left\{\mathbb{E}_0\phi_n+\sup_{P_1\in\mathcal{H}_{1,n}}\mathbb{E}_1(1-\phi_n)\right\}$",
    "chimix": r"$\chi^2\left(1-\varepsilon+\varepsilon f\ \|\ U\right)=\varepsilon^2\,\chi^2(f\,\|\,U)$",
    "In": r"$I_n=n\,\varepsilon_n^2\,\Delta_n\,\min\{\Delta_n,\ \beta_n H(\Delta_n)\}$",
    "thm24": r"$R_n\to 0\ \Leftrightarrow\ I_n\to\infty,\qquad I_n\to 0\ \Rightarrow\ R_n\to 1$",
    "chitg": r"$\chi^2(f_{1,P,\beta}\,\|\,U)\ \asymp\ \Delta(P)\cdot\min\{\Delta(P),\ \beta H(P)\}$",
    "shift": r"$\mathbb{E}_1h(Y)-\mathbb{E}_0h(Y)\ \asymp\ \sqrt{\Delta(P)\cdot\min\{\Delta(P),\ \beta H(P)\}}$",
    "ratio": r"$\frac{\mathrm{shift}_{\mathrm{tG}}}{\mathrm{shift}_{\mathrm{GM}}}\ \asymp\ \frac{\Delta(P)}{H(P)}\ \asymp\ \frac{1}{\log(1/\Delta)}\ \to\ 0$",
    "hbound": r"$H(P)\geq\Delta\log(1/\Delta)\gg\Delta\ \Rightarrow\ \min\{\Delta,\beta H\}=\Delta\quad(\beta\asymp 1)$",
    "poly": r"$\beta_n\asymp 1:\quad I_n\asymp n^{\,1-2p-2q}$",
    "bd": r"$2p+2q+r=1$",
    "ex1": r"$\Delta_n=n^{-2/3},\ \ \varepsilon_n=1$",
    "ex1tg": r"$I_n^{\mathrm{tG}}\asymp n\Delta_n^2=n^{-1/3}\to 0\quad\Rightarrow\quad R_n\to 1$",
    "ex1rg": r"$\inf_{\phi}\{\mathbb{E}_0\phi+\mathbb{E}_{\mathrm{RG}}(1-\phi)\}\to 0$",
    "thm31": r"$\phi_{n,h}=\mathbf{1}\left\{\sum_{t=1}^{n}h(Y_t)>n\int_0^1h+\sigma_h\sqrt{n}\,a_n\right\},\quad a_n\to\infty,\ a_n=o(\sqrt{I_n})$",
    "thm31b": r"$\mathbf{1}\{T_n>c_{n,\alpha_n}\},\ \ \alpha_n=\frac{\alpha_0}{\log(n+e)}:\quad \frac{I_n}{\log\log n}\to\infty\ \Rightarrow\ \mathrm{error}\to 0$",
    "chim": r"$\chi^2\ \asymp\ \Delta\cdot\min\{m\Delta,\ 1\}$",
    "thm28": r"$\chi^2(f_{1,P_t,\beta_t}\,\|\,U)=n^{-2q+\min(s,q)+o(1)},\qquad m_n\asymp n^{s}$",
    "bd28": r"$2p+2q-\min(s,q)=1$",
    "prop34": r"$\sqrt{n}\,\varepsilon_n\Delta_n\log(e/\Delta_n)\to 0\ \Rightarrow\ \sup_{P_1}\ \left|P_0(T_n>c_n)+P_1(T_n\leq c_n)-1\right|\to 0$",
    "prop34s": r"$\sqrt{n}\,\varepsilon_n\Delta_n\log(e/\Delta_n)\to 0\ \Rightarrow\ T_n\ \mathrm{powerless}$",
    "stein": r"$\limsup_{n}\ -\frac{1}{n}\log\sup_{P_1}\mathbb{E}_1(1-\psi_n)\ \leq\ \inf_{q\in\mathcal{Q}_\Theta}\left\{-\int_0^1\log q(y)\,dy\right\}$",
    "stein2": r"$-\int_0^1\log q(y)\,dy=D_{\mathrm{KL}}(U\,\|\,q)$",
    "cramer": r"$\lim_{n}\ -\frac{1}{n}\log P_q(T_n\leq c_{n,\alpha})=\sup_{\lambda\geq 0}\left\{-\lambda-\log\int_0^1(1-y)^{\lambda}q(y)\,dy\right\}$",
    "strict": r"$0\ <\ \sup_{\lambda\geq 0}\{\cdots\}\ <\ -\int_0^1\log q(y)\,dy$",
    "grid": r"$\mathcal{U}_n=\{2^{-j}:1\leq j\leq\lceil\log_2 n\rceil\}\ \cup\ \{k/L_n:1\leq k<L_n\}$",
    "w": r"$w_n(u)=\frac{1}{2|\mathcal{U}_n|}+\frac{1}{2}\mathbf{1}\{u=1/2\}$",
    "kl": r"$\mathrm{kl}(a,u)=a\log\frac{a}{u}+(1-a)\log\frac{1-a}{1-u}$",
    "Tn": r"$T_n^{\mathrm{scan}}=\max_{u\in\mathcal{U}_n,\ \hat{F}_n(u)>u}\ \left\{n\,\mathrm{kl}\left(\hat{F}_n(u),u\right)+\log w_n(u)\right\}$",
    "rej": r"$\phi_{n,\alpha}=\mathbf{1}\left\{T_n^{\mathrm{scan}}>\log(1/\alpha)\right\},\qquad \hat{F}_n(u)=\frac{1}{n}\sum_{t}\mathbf{1}\{p_t\leq u\}$",
    "size1": r"$P_0\left(\hat{F}_n(u)\geq a\right)\leq e^{-n\,\mathrm{kl}(a,u)},\qquad a>u$",
    "size2": r"$P_0\left(T_n^{\mathrm{scan}}>\log\frac{1}{\alpha}\right)\leq\sum_{u\in\mathcal{U}_n}P_0\left(n\,\mathrm{kl}>\log\frac{1}{\alpha\,w_n(u)}\right)\leq\sum_{u}\alpha\,w_n(u)=\alpha$",
    "power1": r"$\int_{1-u}^{1}f_{1,t}(y)\,dy\geq u+d\quad\mathrm{and}\quad\frac{n\varepsilon_n^2d^2}{8Mu}>\log\frac{1}{\alpha\,w_n(u)}$",
    "power2": r"$\Rightarrow\quad P_1(\phi_{n,\alpha}=0)\leq\frac{4Mu}{n\varepsilon_n^2d^2}$",
}


def render_formulas(color=INK, suffix=""):
    sizes = {}
    for k, tex in F.items():
        fig = plt.figure(figsize=(0.01, 0.01), dpi=300)
        fig.text(0, 0, tex, fontsize=24, color=color)
        path = f"{OUT}/f_{k}{suffix}.png"
        fig.savefig(path, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        from PIL import Image
        w, h = Image.open(path).size
        sizes[k] = [w / 300, h / 300]
    return sizes


if __name__ == "__main__":
    res = json.load(open("sim_results.json"))
    phase1(); phase2(); signal(res); power(res)
    sizes = render_formulas()
    json.dump(sizes, open(f"{OUT}/formula_sizes.json", "w"), indent=1)
    for k, v in sizes.items():
        print(k, [round(x, 2) for x in v])

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
from numpy import linalg as LA
import scipy.linalg as scipy_la
from scipy.sparse.linalg import eigs as sparse_eigs
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
import sympy as sp
from sympy import Matrix as SMatrix, latex, symbols, Rational
import warnings, json, time, random, math, os
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN TOKENS  — "Forge" theme: dark steel + molten amber
# ══════════════════════════════════════════════════════════════════════════════
BG       = "#0b0c10"
BG2      = "#141519"
BG3      = "#1c1e25"
BG4      = "#23262f"
AMBER    = "#ffc107"
AMBER2   = "#ff8f00"
CYAN     = "#00e5ff"
VIOLET   = "#7c4dff"
GREEN    = "#69ff47"
RED      = "#ff1744"
PINK     = "#f06292"
TEAL     = "#1de9b6"
TEXT     = "#eceff1"
TEXT2    = "#90a4ae"
TEXT3    = "#546e7a"
BORDER   = "#2a2d37"
GRID_C   = "#1e2028"
SEL_BG   = "#1a1c24"

MONO  = ("Consolas", 10)
MONO_S = ("Consolas", 9)
MONO_L = ("Consolas", 12, "bold")
TITLE  = ("Georgia", 14, "bold")
LABEL  = ("Consolas", 10, "bold")


# ══════════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════════════════════════
def apply_theme():
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor":   BG2,
        "axes.edgecolor":   BORDER,
        "axes.labelcolor":  TEXT2,
        "axes.titlecolor":  AMBER,
        "axes.titlesize":   11,
        "axes.grid":        True,
        "grid.color":       GRID_C,
        "grid.linewidth":   0.5,
        "xtick.color":      TEXT2,
        "ytick.color":      TEXT2,
        "text.color":       TEXT,
        "lines.linewidth":  2.0,
        "font.family":      "monospace",
        "legend.facecolor": BG3,
        "legend.edgecolor": BORDER,
        "legend.fontsize":  8,
    })

apply_theme()

COLORS = [AMBER, CYAN, VIOLET, GREEN, RED, PINK, TEAL,
          "#ff6e40", "#40c4ff", "#b2ff59", "#ea80fc", "#ffab40"]


# ══════════════════════════════════════════════════════════════════════════════
#  PARSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def parse_matrix(text: str) -> np.ndarray:
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").replace(";", " ").split()
        rows.append([float(eval(p)) for p in parts])
    if not rows:
        raise ValueError("Empty input")
    lens = set(len(r) for r in rows)
    if len(lens) != 1:
        raise ValueError(f"Inconsistent column counts: {sorted(lens)}")
    return np.array(rows, dtype=float)

def parse_vector(text: str) -> np.ndarray:
    parts = text.replace(",", " ").replace(";", " ").split()
    return np.array([float(eval(p)) for p in parts if p], dtype=float)

def fmt(v, dp=4):
    if isinstance(v, complex):
        r = f"{v.real:.{dp}f}"
        i = f"{abs(v.imag):.{dp}f}"
        sign = "+" if v.imag >= 0 else "−"
        return f"{r} {sign} {i}i"
    return f"{v:.{dp}f}"

def mat_str(M, dp=4):
    lines = []
    for row in M:
        lines.append("  ".join(fmt(v, dp) for v in row))
    return "\n".join(lines)

def vec_str(v, dp=4):
    return "  ".join(fmt(x, dp) for x in v)

def row_sep(width=58):
    return "─" * width


# ══════════════════════════════════════════════════════════════════════════════
#  CORE LINEAR ALGEBRA ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class LA_Engine:

    # ── Systems & Elimination ────────────────────────────────────────────────
    @staticmethod
    def gaussian_elim(A, b=None, *, full_pivot=False):
        M = np.hstack([A, b.reshape(-1,1)]) if b is not None else A.copy()
        M = M.astype(float)
        m, n = M.shape
        steps, pivots = [], []
        pr = 0
        for col in range(n - (1 if b is not None else 0)):
            if full_pivot:
                sub = np.abs(M[pr:, col:n-(1 if b is not None else 0)])
                ri, ci = np.unravel_index(sub.argmax(), sub.shape)
                ri += pr; ci += col
                if M[ri, ci] < 1e-12: continue
                if ri != pr:
                    M[[pr, ri]] = M[[ri, pr]]
                    steps.append(f"R{pr+1} ↔ R{ri+1}\n{mat_str(M)}")
                if ci != col:
                    M[:, [col, ci]] = M[:, [ci, col]]
                    steps.append(f"C{col+1} ↔ C{ci+1}\n{mat_str(M)}")
            else:
                max_row = pr + np.argmax(np.abs(M[pr:, col]))
                if abs(M[max_row, col]) < 1e-12: continue
                if max_row != pr:
                    M[[pr, max_row]] = M[[max_row, pr]]
                    steps.append(f"R{pr+1} ↔ R{max_row+1}\n{mat_str(M)}")
            pivot = M[pr, col]
            M[pr] /= pivot
            steps.append(f"R{pr+1} ÷ {pivot:.4f}\n{mat_str(M)}")
            for r in range(m):
                if r != pr and abs(M[r, col]) > 1e-12:
                    f_ = M[r, col]
                    M[r] -= f_ * M[pr]
                    steps.append(f"R{r+1} −= {f_:.4f}·R{pr+1}\n{mat_str(M)}")
            pivots.append((pr, col))
            pr += 1
            if pr == m: break
        return M, steps, pivots

    @staticmethod
    def solve(A, b):
        m, n = A.shape
        rA  = np.linalg.matrix_rank(A)
        rAb = np.linalg.matrix_rank(np.column_stack([A, b]))
        res = {"rA": rA, "rAb": rAb, "m": m, "n": n}
        if rA != rAb:
            res["type"] = "inconsistent"
        elif rA == n:
            res["type"] = "unique"
            res["x"] = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            res["type"] = "infinite"
            res["x"] = np.linalg.lstsq(A, b, rcond=None)[0]
            res["null"] = scipy_la.null_space(A)
        return res

    @staticmethod
    def least_squares(A, b):
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        proj = A @ x
        res_vec = b - proj
        return {"x": x, "residuals": res_vec, "rank": rank,
                "singular_vals": s, "fitted": proj,
                "r2": 1 - np.sum(res_vec**2)/max(np.sum((b-b.mean())**2), 1e-15)}

    # ── Decompositions ───────────────────────────────────────────────────────
    @staticmethod
    def lu(A):     return scipy_la.lu(A)
    @staticmethod
    def qr(A):     return np.linalg.qr(A)
    @staticmethod
    def svd(A):    return np.linalg.svd(A, full_matrices=True)
    @staticmethod
    def cholesky(A):
        try:    return np.linalg.cholesky(A), None
        except: return None, "Not positive definite"
    @staticmethod
    def schur(A):
        try:    return scipy_la.schur(A)
        except: return None, None, "Schur failed"
    @staticmethod
    def hessenberg(A):  return scipy_la.hessenberg(A, calc_q=True)

    # ── Eigenanalysis ────────────────────────────────────────────────────────
    @staticmethod
    def eigen(A):       return np.linalg.eig(A)
    @staticmethod
    def eigen_sym(A):   return np.linalg.eigh(A)   # guaranteed real for Hermitian

    @staticmethod
    def jordan_form(A):
        try:
            M = SMatrix(A.tolist())
            P, J = M.jordan_form()
            return np.array(J.tolist(), dtype=complex), np.array(P.tolist(), dtype=complex), None
        except Exception as e:
            return None, None, str(e)

    # ── Matrix Functions ─────────────────────────────────────────────────────
    @staticmethod
    def mat_exp(A):     return scipy_la.expm(A)
    @staticmethod
    def mat_sqrt(A):
        try:    return scipy_la.sqrtm(A), None
        except: return None, "Matrix square root failed"
    @staticmethod
    def mat_log(A):
        try:    return scipy_la.logm(A), None
        except: return None, "Matrix log failed"
    @staticmethod
    def mat_power(A, k): return np.linalg.matrix_power(A, k)

    # ── Properties ───────────────────────────────────────────────────────────
    @staticmethod
    def properties(A):
        sq = A.shape[0] == A.shape[1]
        r  = np.linalg.matrix_rank(A)
        ns = scipy_la.null_space(A)
        sym   = sq and np.allclose(A, A.T, atol=1e-8)
        skew  = sq and np.allclose(A, -A.T, atol=1e-8)
        orth  = sq and np.allclose(A@A.T, np.eye(A.shape[0]), atol=1e-6)
        inv_e = sq and np.linalg.matrix_rank(A) == A.shape[0]
        pd    = sq and sym and bool(np.all(np.linalg.eigvalsh(A) > 0))
        idem  = sq and np.allclose(A@A, A, atol=1e-8)
        p = {
            "shape":     A.shape, "rank": r,
            "nullity":   A.shape[1] - r,
            "trace":     np.trace(A) if sq else None,
            "det":       np.linalg.det(A) if sq else None,
            "cond":      np.linalg.cond(A),
            "frob":      np.linalg.norm(A, "fro"),
            "spectral":  np.linalg.norm(A, 2),
            "nuclear":   np.sum(np.linalg.svd(A, compute_uv=False)),
            "symmetric": sym, "skew_sym": skew, "orthogonal": orth,
            "invertible":inv_e, "pos_def": pd, "idempotent": idem,
            "null_space": ns,
        }
        if sq and inv_e:
            p["inv"] = np.linalg.inv(A)
        p["pinv"] = np.linalg.pinv(A)
        return p

    # ── Gram-Schmidt ─────────────────────────────────────────────────────────
    @staticmethod
    def gram_schmidt(A):
        Q = np.zeros_like(A, dtype=float)
        for i in range(A.shape[1]):
            v = A[:, i].copy()
            for j in range(i):
                v -= np.dot(Q[:,j], A[:,i]) * Q[:,j]
            n = np.linalg.norm(v)
            Q[:,i] = v/n if n > 1e-12 else 0
        return Q

    # ── PCA ──────────────────────────────────────────────────────────────────
    @staticmethod
    def pca(A):
        Ac = A - A.mean(axis=0)
        cov = np.cov(Ac.T)
        if cov.ndim == 0: cov = np.array([[float(cov)]])
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        total = vals.sum()
        return vecs, vals, vals/total if total else vals, Ac @ vecs

    # ── Vector Operations ────────────────────────────────────────────────────
    @staticmethod
    def vector_ops(u, v):
        ops = {
            "dot":       np.dot(u, v),
            "norm_u":    np.linalg.norm(u),
            "norm_v":    np.linalg.norm(v),
            "angle_deg": math.degrees(math.acos(
                np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-15), -1, 1))),
            "u+v":       u + v,
            "u-v":       u - v,
            "proj_v_on_u": (np.dot(u,v)/max(np.dot(u,u),1e-15))*u,
        }
        if len(u) == 3 and len(v) == 3:
            ops["cross"] = np.cross(u, v)
        return ops

    # ── Markov / Stochastic ──────────────────────────────────────────────────
    @staticmethod
    def markov_steady_state(P):
        """Find steady-state distribution for stochastic matrix P."""
        n = P.shape[0]
        # Eigenvalue method
        vals, vecs = np.linalg.eig(P.T)
        idx = np.argmin(np.abs(vals - 1.0))
        ss = np.abs(vecs[:, idx])
        return ss / ss.sum()

    @staticmethod
    def markov_power(P, k):
        """P^k via matrix power."""
        return np.linalg.matrix_power(P, k)

    # ── Iterative Solvers ────────────────────────────────────────────────────
    @staticmethod
    def jacobi(A, b, max_iter=200, tol=1e-8):
        n = len(b)
        x = np.zeros(n)
        history = []
        for it in range(max_iter):
            x_new = np.zeros(n)
            for i in range(n):
                s = sum(A[i,j]*x[j] for j in range(n) if j != i)
                x_new[i] = (b[i]-s) / A[i,i]
            history.append(np.linalg.norm(x_new - x))
            x = x_new
            if history[-1] < tol:
                break
        return x, history

    @staticmethod
    def gauss_seidel(A, b, max_iter=200, tol=1e-8):
        n = len(b)
        x = np.zeros(n)
        history = []
        for it in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                s1 = sum(A[i,j]*x[j]   for j in range(i))
                s2 = sum(A[i,j]*x_old[j] for j in range(i+1,n))
                x[i] = (b[i]-s1-s2) / A[i,i]
            err = np.linalg.norm(x - x_old)
            history.append(err)
            if err < tol: break
        return x, history

    # ── SVD Image Compression ────────────────────────────────────────────────
    @staticmethod
    def svd_compress(A, k):
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        Ak = U[:,:k] @ np.diag(s[:k]) @ Vt[:k,:]
        ratio = k*(A.shape[0]+A.shape[1]+1) / (A.shape[0]*A.shape[1])
        return Ak, ratio

    # ── Quadratic Form ───────────────────────────────────────────────────────
    @staticmethod
    def quadratic_form(A, x):
        return float(x.T @ A @ x)

    # ── Linear Regression ────────────────────────────────────────────────────
    @staticmethod
    def linear_regression(X, y):
        A = np.column_stack([np.ones(len(X)), X])
        res = LA_Engine.least_squares(A, y)
        return res["x"][0], res["x"][1:], res["r2"]


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def heatmap(ax, A, title="Heatmap", cmap="plasma"):
    im = ax.imshow(A.real, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, pad=0.02)
    ax.set_title(title)
    m, n = A.shape
    if m <= 12 and n <= 12:
        for i in range(m):
            for j in range(n):
                v = A.real[i,j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=max(6, 9-max(m,n)//2),
                        color="white" if abs(v) < np.abs(A).max()*0.6 else "black")
    ax.set_xlabel("col"); ax.set_ylabel("row")

def vectors_2d(ax, vecs, labels=None, title="Vectors", origin=None):
    ax.set_title(title)
    O = origin or np.zeros(2)
    for j, v in enumerate(vecs):
        v2 = np.array(v[:2], dtype=float)
        c  = COLORS[j % len(COLORS)]
        ax.annotate("", xy=O+v2, xytext=O,
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
        lbl = labels[j] if labels else f"v{j+1}"
        ax.text(O[0]+v2[0]*1.08, O[1]+v2[1]*1.08, lbl, color=c, fontsize=10, fontweight="bold")
    all_pts = np.array(vecs)[:, :2]
    lim = max(1.2, np.abs(all_pts).max()*1.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color=BORDER, lw=1); ax.axvline(0, color=BORDER, lw=1)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")

def eigen_spectrum(ax, vals, title="Eigenvalue Spectrum"):
    re, im = vals.real, vals.imag
    ax.scatter(re, im, c=RED, s=120, zorder=5, edgecolors="white", lw=0.8)
    θ = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(θ), np.sin(θ), color=BORDER, ls="--", lw=1)
    ax.axhline(0, color=BORDER, lw=0.8); ax.axvline(0, color=BORDER, lw=0.8)
    ax.set_title(title); ax.set_aspect("equal")
    ax.set_xlabel("Re"); ax.set_ylabel("Im")
    for i,(r,c) in enumerate(zip(re,im)):
        ax.annotate(f"λ{i+1}", (r,c), xytext=(5,4),
                    textcoords="offset points", color=TEXT2, fontsize=8)

def transform_viz(ax, A, title="Linear Transform"):
    θ = np.linspace(0, 2*np.pi, 300)
    circ = np.vstack([np.cos(θ), np.sin(θ)])
    sq   = np.array([[0,1,1,0,0],[0,0,1,1,0]], float)
    if A.shape == (2,2):
        tc = A @ circ; ts = A @ sq
        ax.plot(*circ, color=BORDER, lw=1.2, ls="--", label="Original circle")
        ax.plot(*tc,   color=CYAN,   lw=2,            label="Transformed circle")
        ax.fill(*ts,   color=VIOLET, alpha=0.15)
        ax.plot(*ts,   color=VIOLET, lw=2,            label="Transformed square")
        # basis vectors
        for j, col in enumerate(A.T):
            ax.annotate("", xy=col, xytext=[0,0],
                        arrowprops=dict(arrowstyle="->",color=COLORS[j],lw=2.5))
    else:
        ax.text(0.5,0.5,"Only for 2×2",ha="center",va="center",
                transform=ax.transAxes,color=TEXT2)
    ax.axhline(0,color=BORDER,lw=0.6); ax.axvline(0,color=BORDER,lw=0.6)
    ax.set_title(title); ax.legend(fontsize=7); ax.set_aspect("equal")

def singular_bar(ax, s, title="Singular Values"):
    bars = ax.bar(range(1,len(s)+1), s, color=AMBER, edgecolor=BG, lw=0.8, alpha=0.9)
    ax.set_title(title); ax.set_xlabel("Index"); ax.set_ylabel("σ")
    for bar, v in zip(bars, s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+s.max()*0.01,
                f"{v:.3f}", ha="center", fontsize=7, color=TEXT2)

def convergence_plot(ax, histories, labels, title="Iterative Convergence"):
    for h, lbl, c in zip(histories, labels, COLORS):
        ax.semilogy(h, color=c, lw=2, label=lbl)
    ax.set_title(title); ax.set_xlabel("Iteration"); ax.set_ylabel("‖Δx‖")
    ax.legend()

def pca_biplot(ax, scores, vecs, labels=None, title="PCA Biplot"):
    ax.scatter(scores[:,0], scores[:,1] if scores.shape[1]>1 else np.zeros(len(scores)),
               c=CYAN, s=50, alpha=0.7, edgecolors=AMBER, lw=0.5)
    scale = np.abs(scores).max() * 0.5
    for i, v in enumerate(vecs.T[:min(4, vecs.shape[1])]):
        ax.annotate("", xy=v[:2]*scale, xytext=[0,0],
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2))
        ax.text(v[0]*scale*1.1, v[1]*scale*1.1,
                labels[i] if labels else f"f{i+1}", color=RED, fontsize=8)
    ax.axhline(0,color=BORDER,lw=0.6); ax.axvline(0,color=BORDER,lw=0.6)
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")


# ══════════════════════════════════════════════════════════════════════════════
#  TOOLTIP
# ══════════════════════════════════════════════════════════════════════════════
class Tip:
    def __init__(self, w, txt):
        self.w   = w
        self.txt = txt
        self.tip = None
        w.bind("<Enter>", self.show); w.bind("<Leave>", self.hide)
    def show(self, _=None):
        x = self.w.winfo_rootx() + 22
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tk.Toplevel(self.w)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.txt, bg=BG4, fg=TEXT2,
                 relief="solid", bd=1, font=MONO_S, padx=7, pady=4).pack()
    def hide(self, _=None):
        if self.tip: self.tip.destroy(); self.tip = None


# ══════════════════════════════════════════════════════════════════════════════
#  FORGE BUTTON  (amber-accented)
# ══════════════════════════════════════════════════════════════════════════════
class ForgeBtn(tk.Label):
    def __init__(self, parent, text, cmd, fg=AMBER, dim=False, **kw):
        bg_ = BG3 if not dim else BG2
        super().__init__(parent, text=text, bg=bg_, fg=fg,
                         font=("Consolas", 9, "bold"),
                         cursor="hand2", pady=5, padx=10,
                         relief="flat", bd=0, **kw)
        self.cmd = cmd; self.fg_ = fg; self.bg_ = bg_
        self.bind("<Button-1>", lambda e: cmd())
        self.bind("<Enter>",    lambda e: self.config(bg=BG4, fg="white"))
        self.bind("<Leave>",    lambda e: self.config(bg=self.bg_, fg=self.fg_))


# ══════════════════════════════════════════════════════════════════════════════
#  HISTORY MANAGER
# ══════════════════════════════════════════════════════════════════════════════
class HistoryManager:
    def __init__(self, max_items=50):
        self.stack = []
        self.max   = max_items
    def push(self, op, matrix_txt, result_txt):
        self.stack.append({
            "ts": time.strftime("%H:%M:%S"),
            "op": op,
            "matrix": matrix_txt,
            "result": result_txt,
        })
        if len(self.stack) > self.max:
            self.stack.pop(0)
    def get_all(self):
        return self.stack[::-1]
    def clear(self):
        self.stack.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class MatrixForge(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⟨⟨ MATRIX FORGE — Linear Algebra Workbench ⟩⟩")
        self.configure(bg=BG)
        self.minsize(1280, 800)
        try: self.state("zoomed")
        except: self.geometry("1440x900")

        self.history = HistoryManager()
        self._setup_style()
        self._build_ui()

    # ── ttk style ─────────────────────────────────────────────────────────────
    def _setup_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TNotebook",      background=BG,  borderwidth=0)
        s.configure("TNotebook.Tab",  background=BG3, foreground=TEXT2,
                    font=("Consolas", 9, "bold"), padding=[14, 7])
        s.map("TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", AMBER)])
        s.configure("TFrame",  background=BG)
        s.configure("TSeparator", background=BORDER)

    # ── Overall layout ────────────────────────────────────────────────────────
    def _build_ui(self):
        self._header()
        tk.Frame(self, bg=AMBER, height=2).pack(fill="x")
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self._left_panel(body)
        self._right_notebook(body)

    # ── Header ────────────────────────────────────────────────────────────────
    def _header(self):
        hdr = tk.Frame(self, bg=BG2, height=52)
        hdr.pack(fill="x"); hdr.pack_propagate(False)
        tk.Label(hdr, text="⟨⟨ MATRIX FORGE ⟩⟩",
                 bg=BG2, fg=AMBER, font=TITLE).pack(side="left", padx=22, pady=8)
        subtitle = "NumPy · SciPy · SymPy · Matplotlib  |  Linear Algebra Workbench"
        tk.Label(hdr, text=subtitle, bg=BG2, fg=TEXT3, font=MONO_S).pack(side="right", padx=22)
        # status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(hdr, textvariable=self.status_var, bg=BG2, fg=TEAL,
                 font=MONO_S).pack(side="right", padx=30)

    # ── Left Panel ────────────────────────────────────────────────────────────
    def _left_panel(self, parent):
        lf = tk.Frame(parent, bg=BG2, width=310)
        lf.pack(side="left", fill="y")
        lf.pack_propagate(False)

        # ── Matrix A ────────────────────────────────────────────────────────
        self._sec_label(lf, "▸ MATRIX  A")
        self.mat_a = self._text_box(lf, h=8, fg=AMBER)
        self.mat_a.insert("1.0", "2  1  -1\n-3 -1  2\n-2  1  2")

        # ── Vector b ────────────────────────────────────────────────────────
        self._sec_label(lf, "▸ VECTOR  b  (for Ax=b / least squares)")
        self.vec_b = self._entry(lf, fg=CYAN)
        self.vec_b.insert(0, "8  11  3")

        # ── Matrix B ────────────────────────────────────────────────────────
        self._sec_label(lf, "▸ MATRIX  B  (for A±B, A·B, etc.)")
        self.mat_b = self._text_box(lf, h=5, fg=PINK)
        self.mat_b.insert("1.0", "1  0  0\n0  1  0\n0  0  1")

        # ── Vector u / v ────────────────────────────────────────────────────
        self._sec_label(lf, "▸ VECTOR  u  (for vector ops)")
        self.vec_u = self._entry(lf, fg=GREEN)
        self.vec_u.insert(0, "1  2  3")
        self._sec_label(lf, "▸ VECTOR  v")
        self.vec_v = self._entry(lf, fg=TEAL)
        self.vec_v.insert(0, "4  5  6")

        # ── Options ─────────────────────────────────────────────────────────
        of = tk.Frame(lf, bg=BG2); of.pack(fill="x", padx=10, pady=(6,2))
        tk.Label(of, text="Decimals:", bg=BG2, fg=TEXT2, font=MONO_S).pack(side="left")
        self.dp = tk.IntVar(value=4)
        tk.Spinbox(of, from_=1, to=12, textvariable=self.dp,
                   width=4, bg=BG3, fg=AMBER, font=MONO_S,
                   relief="flat", bd=0, buttonbackground=BG3).pack(side="left", padx=5)
        tk.Label(of, text="Full pivot:", bg=BG2, fg=TEXT2, font=MONO_S).pack(side="left", padx=(10,2))
        self.full_piv = tk.BooleanVar()
        tk.Checkbutton(of, variable=self.full_piv,
                       bg=BG2, fg=AMBER, selectcolor=BG3,
                       activebackground=BG2, relief="flat").pack(side="left")

        tk.Frame(lf, bg=BORDER, height=1).pack(fill="x", padx=8, pady=6)

        # ── Buttons ─────────────────────────────────────────────────────────
        self._buttons(lf)

    def _sec_label(self, parent, text):
        tk.Label(parent, text=text, bg=BG2, fg=TEXT2,
                 font=("Consolas", 9, "bold"), anchor="w").pack(
                     fill="x", padx=10, pady=(8,1))

    def _text_box(self, parent, h=7, fg=AMBER):
        t = scrolledtext.ScrolledText(parent, height=h,
                                      bg=BG3, fg=fg, insertbackground=fg,
                                      font=MONO, relief="flat", bd=0,
                                      padx=7, pady=5)
        t.pack(fill="x", padx=10, pady=2)
        return t

    def _entry(self, parent, fg=AMBER):
        e = tk.Entry(parent, bg=BG3, fg=fg, insertbackground=fg,
                     font=MONO, relief="flat", bd=0)
        e.pack(fill="x", padx=10, ipady=5, pady=2)
        return e

    def _buttons(self, parent):
        cats = [
            ("SYSTEMS", AMBER, [
                ("▶ Solve  Ax = b",          self.op_solve,       Tip, "Unique / infinite / inconsistent"),
                ("▶ Gaussian Elimination",    self.op_gauss,       Tip, "Partial or full pivot RREF"),
                ("▶ RREF",                    self.op_rref,        Tip, "Reduced row echelon form"),
                ("▶ Least Squares  Ax ≈ b",  self.op_lsq,         Tip, "Overdetermined system"),
                ("▶ Iterative Solvers",       self.op_iterative,   Tip, "Jacobi + Gauss-Seidel"),
            ]),
            ("DECOMPOSITIONS", CYAN, [
                ("◈ LU  (PA = LU)",           self.op_lu,          Tip, "Partial-pivot LU via SciPy"),
                ("◈ QR  (Householder)",        self.op_qr,          Tip, "Orthogonal × upper-triangular"),
                ("◈ SVD",                      self.op_svd,         Tip, "U Σ Vᵀ — singular values"),
                ("◈ Cholesky",                 self.op_cholesky,    Tip, "Requires positive definite A"),
                ("◈ Schur",                    self.op_schur,       Tip, "A = QTQ^H"),
                ("◈ Hessenberg",               self.op_hessenberg,  Tip, "Upper Hessenberg form"),
            ]),
            ("EIGENANALYSIS", VIOLET, [
                ("◉ Eigenvalues / Vectors",    self.op_eigen,       Tip, "General complex eigenvalues"),
                ("◉ Hermitian Eigen",           self.op_eigen_sym,   Tip, "Real eigenvalues guaranteed"),
                ("◉ Jordan Normal Form",        self.op_jordan,      Tip, "Symbolic via SymPy"),
            ]),
            ("MATRIX FUNCTIONS", PINK, [
                ("⊛ Matrix Exponential  eᴬ",  self.op_exp,         Tip, "scipy.linalg.expm"),
                ("⊛ Matrix Square Root √A",   self.op_sqrt,        Tip, "scipy.linalg.sqrtm"),
                ("⊛ Matrix Logarithm  ln(A)", self.op_log,         Tip, "scipy.linalg.logm"),
                ("⊛ Matrix Power  Aᵏ",        self.op_power,       Tip, "k: integer or type in dialog"),
            ]),
            ("ANALYSIS", TEAL, [
                ("⊠ Matrix Properties",        self.op_props,       Tip, "Rank, norms, symmetry, …"),
                ("⊠ Gram-Schmidt",             self.op_gs,          Tip, "Orthonormalise columns"),
                ("⊠ PCA",                      self.op_pca,         Tip, "Principal component analysis"),
                ("⊠ Vector Operations",        self.op_vecops,      Tip, "Dot, cross, angle, projection"),
                ("⊠ Markov Chain",             self.op_markov,      Tip, "Steady-state from stochastic A"),
            ]),
            ("APPLICATIONS", GREEN, [
                ("✦ Arithmetic  A±B, A·B",    self.op_arith,       Tip, "Matrix algebra with B"),
                ("✦ Symbolic (SymPy exact)",   self.op_sym,         Tip, "Exact rational / symbolic"),
                ("✦ SVD Image Compress",       self.op_img_svd,     Tip, "Low-rank approximation demo"),
                ("✦ Quadratic Form  xᵀAx",    self.op_quad,        Tip, "Evaluate quadratic form"),
                ("✦ Linear Regression",        self.op_regression,  Tip, "Fit y = a + bx from data"),
            ]),
        ]

        canvas = tk.Canvas(parent, bg=BG2, highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame  = tk.Frame(canvas, bg=BG2)
        frame.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(0,0))
        scroll.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(1 if e.delta>0 else -1),"units"))

        for cat_name, cat_color, btns in cats:
            tk.Label(frame, text=f"  {cat_name}", bg=BG2, fg=cat_color,
                     font=("Consolas", 9, "bold"), anchor="w").pack(
                         fill="x", padx=6, pady=(8,1))
            for txt, cmd, tip_cls, tip_txt in btns:
                b = ForgeBtn(frame, txt, cmd, fg=cat_color)
                b.pack(fill="x", padx=8, pady=1)
                tip_cls(b, tip_txt)

        # ── Utility strip ────────────────────────────────────────────────────
        util = tk.Frame(parent, bg=BG2)
        util.pack(fill="x", padx=8, pady=6)
        ForgeBtn(util, "📊 All Graphs",    self.plot_all,     fg=AMBER).pack(side="left", expand=True, fill="x", padx=1)
        ForgeBtn(util, "🗑 Clear",          self.clear_all,    fg=RED  ).pack(side="left", expand=True, fill="x", padx=1)
        ForgeBtn(util, "🎲 Sample 5×5",    self.load_5x5,     fg=TEXT2).pack(side="left", expand=True, fill="x", padx=1)
        ForgeBtn(util, "💾 Export",         self.export_res,   fg=TEXT2).pack(side="left", expand=True, fill="x", padx=1)

    # ── Right Notebook ────────────────────────────────────────────────────────
    def _right_notebook(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)

        self.tab_res   = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_res,   text="  📄 Results  ")
        self.tab_steps = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_steps, text="  🔢 Steps  ")
        self.tab_graph = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_graph, text="  📊 Graphs  ")
        self.tab_vec   = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_vec,   text="  ↗ Vectors  ")
        self.tab_hist  = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_hist,  text="  🕓 History  ")
        self.tab_ref   = tk.Frame(self.nb, bg=BG); self.nb.add(self.tab_ref,   text="  📖 Reference  ")

        self._build_res_tab()
        self._build_steps_tab()
        self._build_graph_placeholder()
        self._build_vec_tab()
        self._build_hist_tab()
        self._build_ref_tab()

    # ── Tab builders ─────────────────────────────────────────────────────────
    def _build_res_tab(self):
        hdr = tk.Frame(self.tab_res, bg=BG2, height=32); hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  COMPUTATION RESULTS", bg=BG2, fg=AMBER,
                 font=LABEL).pack(side="left", pady=6)
        ForgeBtn(hdr, "⎘ Copy", self.copy_res, fg=TEXT2).pack(side="right", padx=8, pady=4)
        self.res = scrolledtext.ScrolledText(self.tab_res, bg=BG2, fg=TEXT,
                                             font=MONO, relief="flat", bd=0,
                                             padx=14, pady=10)
        self.res.pack(fill="both", expand=True)
        for tag, clr, fnt in [
            ("H", AMBER, ("Consolas",12,"bold")),
            ("h", CYAN,  ("Consolas",10,"bold")),
            ("v", GREEN, MONO),
            ("e", RED,   MONO),
            ("d", TEXT2, MONO_S),
            ("n", TEXT,  MONO),
        ]:
            self.res.tag_configure(tag, foreground=clr, font=fnt)
        self._wr([("  ▷ Welcome to Matrix Forge\n\n"
                   "  Enter matrix A (and optional b, B, u, v) on the left\n"
                   "  then press any operation button.\n\n"
                   "  • Least Squares   • Iterative Solvers (Jacobi, Gauss-Seidel)\n"
                   "  • Schur / Hessenberg / Jordan decompositions\n"
                   "  • Matrix Exp / Log / Sqrt  •  Markov Chains\n"
                   "  • Vector Operations (dot, cross, angle, projection)\n"
                   "  • SVD Image Compression demo\n"
                   "  • Quadratic Form evaluator\n"
                   "  • Linear Regression via Least Squares\n"
                   "  • History tab  •  Reference sheet  •  Export\n", "d")])

    def _build_steps_tab(self):
        hdr = tk.Frame(self.tab_steps, bg=BG2, height=32); hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  STEP-BY-STEP ELIMINATION", bg=BG2, fg=CYAN,
                 font=LABEL).pack(side="left", pady=6)
        self.steps = scrolledtext.ScrolledText(self.tab_steps, bg=BG2, fg=TEXT2,
                                               font=MONO, relief="flat", bd=0,
                                               padx=14, pady=10)
        self.steps.pack(fill="both", expand=True)

    def _build_graph_placeholder(self):
        self._graph_canvas = None
        lbl = tk.Label(self.tab_graph,
                       text="Press  「 📊 All Graphs 」  or run any operation.",
                       bg=BG, fg=TEXT3, font=("Consolas",12))
        lbl.pack(expand=True)
        self._graph_ph = lbl

    def _build_vec_tab(self):
        self._vec_canvas = None
        tk.Label(self.tab_vec,
                 text="Run  「 ⊠ Vector Operations 」  to populate this tab.",
                 bg=BG, fg=TEXT3, font=("Consolas",12)).pack(expand=True)

    def _build_hist_tab(self):
        hdr = tk.Frame(self.tab_hist, bg=BG2, height=32); hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  OPERATION HISTORY", bg=BG2, fg=PINK,
                 font=LABEL).pack(side="left", pady=6)
        ForgeBtn(hdr, "🗑 Clear", lambda: (self.history.clear(), self._refresh_hist()),
                 fg=RED).pack(side="right", padx=8, pady=4)
        self.hist_text = scrolledtext.ScrolledText(
            self.tab_hist, bg=BG2, fg=TEXT2, font=MONO,
            relief="flat", bd=0, padx=14, pady=10)
        self.hist_text.pack(fill="both", expand=True)
        self.hist_text.tag_configure("ts", foreground=TEXT3)
        self.hist_text.tag_configure("op", foreground=AMBER, font=("Consolas",10,"bold"))
        self.hist_text.tag_configure("m",  foreground=TEXT2)

    def _build_ref_tab(self):
        ref = scrolledtext.ScrolledText(self.tab_ref, bg=BG2, fg=TEXT2,
                                        font=MONO, relief="flat", bd=0,
                                        padx=20, pady=12)
        ref.pack(fill="both", expand=True)
        ref.insert("1.0", REFERENCE_TEXT)
        ref.config(state="disabled")

    # ── Result writer ─────────────────────────────────────────────────────────
    def _wr(self, items):
        self.res.config(state="normal")
        self.res.delete("1.0","end")
        for txt, tag in items:
            self.res.insert("end", txt, tag)

    def _app(self, txt, tag="n"):
        self.res.config(state="normal")
        self.res.insert("end", txt, tag)

    def _set_steps(self, steps):
        self.steps.config(state="normal"); self.steps.delete("1.0","end")
        for i, s in enumerate(steps, 1):
            self.steps.insert("end", f"── Step {i} ──────────────────\n")
            self.steps.insert("end", s + "\n\n")

    def _status(self, msg):
        self.status_var.set(msg)
        self.after(4000, lambda: self.status_var.set("Ready"))

    # ── History refresh ───────────────────────────────────────────────────────
    def _refresh_hist(self):
        self.hist_text.config(state="normal"); self.hist_text.delete("1.0","end")
        for item in self.history.get_all():
            self.hist_text.insert("end", f"[{item['ts']}] ", "ts")
            self.hist_text.insert("end", f"{item['op']}\n", "op")
            self.hist_text.insert("end", item['matrix'][:120] + "\n\n", "m")

    # ── Matrix getters ────────────────────────────────────────────────────────
    def _A(self):  return parse_matrix(self.mat_a.get("1.0","end"))
    def _b(self):
        t = self.vec_b.get().strip()
        return parse_vector(t) if t else None
    def _B(self):
        t = self.mat_b.get("1.0","end").strip()
        return parse_matrix(t) if t else None
    def _u(self):  return parse_vector(self.vec_u.get())
    def _v(self):  return parse_vector(self.vec_v.get())
    def _dp(self): return self.dp.get()

    # ── Error wrapper ─────────────────────────────────────────────────────────
    def _run(self, fn):
        try: fn()
        except Exception as ex:
            self._wr([("  ✕ ERROR\n\n", "e"), (f"  {type(ex).__name__}: {ex}\n", "e")])

    # ══════════════════════════════════════════════════════════════════════════
    #  OPERATIONS
    # ══════════════════════════════════════════════════════════════════════════

    # ── Solve ─────────────────────────────────────────────────────────────────
    def op_solve(self): self._run(self._solve)
    def _solve(self):
        A, b = self._A(), self._b()
        if b is None: raise ValueError("Please enter vector b")
        if len(b) != A.shape[0]: raise ValueError(f"A has {A.shape[0]} rows, b has {len(b)}")
        info = LA_Engine.solve(A, b); dp = self._dp()
        out = [
            ("  SOLVE  Ax = b\n\n", "H"),
            (f"  A  ({A.shape[0]}×{A.shape[1]}):\n", "h"), (mat_str(A,dp)+"\n\n", "n"),
            (f"  b: {vec_str(b,dp)}\n\n", "d"),
            (f"  rank(A) = {info['rA']}   rank([A|b]) = {info['rAb']}"
             f"   unknowns = {info['n']}\n\n", "d"),
        ]
        if info["type"] == "inconsistent":
            out += [("  ✕  INCONSISTENT — no solution.\n", "e")]
        elif info["type"] == "unique":
            x = info["x"]
            out += [("  ✓  UNIQUE SOLUTION\n", "v"),
                    ("  x = "+vec_str(x,dp)+"\n\n","v"),
                    (f"  Residual ‖Ax−b‖ = {LA.norm(A@x-b):.2e}\n", "d")]
        else:
            out += [("  ∞  INFINITE SOLUTIONS\n","h"),
                    ("  Particular: x₀ = "+vec_str(info["x"],dp)+"\n\n","v"),
                    ("  Null-space basis:\n","h"),
                    (mat_str(info["null"],dp)+"\n","d")]
        self._wr(out)
        _, steps, _ = LA_Engine.gaussian_elim(A, b, full_pivot=self.full_piv.get())
        self._set_steps(steps)
        self.history.push("Solve Ax=b", mat_str(A)[:60], str(out[4][0]))
        self._refresh_hist(); self._status("Solved ✓")
        self.nb.select(0); self._render_graphs(A)

    # ── Gaussian ──────────────────────────────────────────────────────────────
    def op_gauss(self): self._run(self._gauss)
    def _gauss(self):
        A = self._A(); b = self._b(); dp = self._dp()
        rref, steps, pivots = LA_Engine.gaussian_elim(A, b, full_pivot=self.full_piv.get())
        out = [("  GAUSSIAN ELIMINATION\n\n","H"),
               (f"  Steps taken: {len(steps)}\n","d"),
               ("  RREF result:\n","h"), (mat_str(rref,dp)+"\n\n","v"),
               (f"  Pivots: {pivots}\n  Rank = {len(pivots)}\n","d")]
        self._wr(out); self._set_steps(steps)
        self.history.push("Gaussian", mat_str(A)[:60], f"rank={len(pivots)}")
        self._refresh_hist(); self._status("Elimination done ✓")
        self.nb.select(0); self._render_graphs(A)

    # ── RREF ──────────────────────────────────────────────────────────────────
    def op_rref(self): self._run(self._rref)
    def _rref(self):
        A = self._A(); dp = self._dp()
        rref, steps, pivots = LA_Engine.gaussian_elim(A, full_pivot=self.full_piv.get())
        self._wr([("  RREF\n\n","H"),(mat_str(rref,dp)+"\n\n","v"),
                  (f"  Rank = {len(pivots)}   Nullity = {A.shape[1]-len(pivots)}\n","d")])
        self._set_steps(steps); self.nb.select(0)

    # ── Least Squares ─────────────────────────────────────────────────────────
    def op_lsq(self): self._run(self._lsq)
    def _lsq(self):
        A, b = self._A(), self._b(); dp = self._dp()
        if b is None: raise ValueError("Enter vector b")
        info = LA_Engine.least_squares(A, b)
        self._wr([
            ("  LEAST SQUARES  Ax ≈ b\n\n","H"),
            (f"  Optimal  x̂ = {vec_str(info['x'],dp)}\n\n","v"),
            (f"  Fitted  Ax̂ = {vec_str(info['fitted'],dp)}\n","d"),
            (f"  Residual    = {vec_str(info['residuals'],dp)}\n","d"),
            (f"  ‖residual‖  = {LA.norm(info['residuals']):.{dp}f}\n","d"),
            (f"  R²          = {info['r2']:.{dp}f}\n","v"),
        ])
        self.nb.select(0)

    # ── Iterative ─────────────────────────────────────────────────────────────
    def op_iterative(self): self._run(self._iterative)
    def _iterative(self):
        A, b = self._A(), self._b(); dp = self._dp()
        if b is None: raise ValueError("Enter b")
        if A.shape[0] != A.shape[1]: raise ValueError("Square matrix required")
        xj, hj = LA_Engine.jacobi(A, b)
        xg, hg = LA_Engine.gauss_seidel(A, b)
        self._wr([
            ("  ITERATIVE SOLVERS\n\n","H"),
            ("  Jacobi:\n","h"),
            (f"  x = {vec_str(xj,dp)}\n","v"),
            (f"  Converged in {len(hj)} iterations, final err = {hj[-1]:.2e}\n\n","d"),
            ("  Gauss-Seidel:\n","h"),
            (f"  x = {vec_str(xg,dp)}\n","v"),
            (f"  Converged in {len(hg)} iterations, final err = {hg[-1]:.2e}\n\n","d"),
        ])
        self.nb.select(0)
        self._render_iter_graphs(A, hj, hg)

    # ── LU ────────────────────────────────────────────────────────────────────
    def op_lu(self): self._run(self._lu)
    def _lu(self):
        A = self._A(); dp = self._dp()
        P, L, U = LA_Engine.lu(A)
        self._wr([("  LU  PA=LU\n\n","H"),
                  ("  P:\n","h"),(mat_str(P,dp)+"\n\n","d"),
                  ("  L:\n","h"),(mat_str(L,dp)+"\n\n","v"),
                  ("  U:\n","h"),(mat_str(U,dp)+"\n\n","v"),
                  (f"  ‖PA−LU‖ = {LA.norm(P@A-L@U):.2e}\n","d")])
        self.nb.select(0); self._render_graphs(A)

    # ── QR ────────────────────────────────────────────────────────────────────
    def op_qr(self): self._run(self._qr)
    def _qr(self):
        A = self._A(); dp = self._dp()
        Q, R = LA_Engine.qr(A)
        self._wr([("  QR\n\n","H"),
                  ("  Q (orthogonal):\n","h"),(mat_str(Q,dp)+"\n\n","v"),
                  ("  R (upper triangular):\n","h"),(mat_str(R,dp)+"\n\n","v"),
                  (f"  ‖QᵀQ−I‖ = {LA.norm(Q.T@Q-np.eye(Q.shape[1])):.2e}\n","d"),
                  (f"  ‖A−QR‖  = {LA.norm(A-Q@R):.2e}\n","d")])
        self.nb.select(0); self._render_graphs(A)

    # ── SVD ───────────────────────────────────────────────────────────────────
    def op_svd(self): self._run(self._svd)
    def _svd(self):
        A = self._A(); dp = self._dp()
        U, s, Vt = LA_Engine.svd(A)
        self._wr([("  SVD  A = UΣVᵀ\n\n","H"),
                  ("  U:\n","h"),(mat_str(U,dp)+"\n\n","v"),
                  ("  σ: ","h"),(vec_str(s,dp)+"\n\n","v"),
                  ("  Vᵀ:\n","h"),(mat_str(Vt,dp)+"\n\n","v"),
                  (f"  Effective rank = {np.sum(s>1e-10)}\n","d"),
                  (f"  Cond# = {s[0]/s[-1]:.4f}\n" if s[-1]>1e-12 else "  Singular matrix\n","d")])
        self.nb.select(0); self._render_svd_graphs(A, s, U, Vt)

    # ── Cholesky ──────────────────────────────────────────────────────────────
    def op_cholesky(self): self._run(self._cholesky)
    def _cholesky(self):
        A = self._A(); dp = self._dp(); L, err = LA_Engine.cholesky(A)
        if err:
            self._wr([("  CHOLESKY\n\n","H"),("  ✕ "+err+"\n","e")]); return
        self._wr([("  CHOLESKY  A=LLᵀ\n\n","H"),
                  ("  L:\n","h"),(mat_str(L,dp)+"\n\n","v"),
                  (f"  ‖A−LLᵀ‖ = {LA.norm(A-L@L.T):.2e}\n","d")])
        self.nb.select(0)

    # ── Schur ────────────────────────────────────────────────────────────────
    def op_schur(self): self._run(self._schur)
    def _schur(self):
        A = self._A(); dp = self._dp()
        T, Z, err = LA_Engine.schur(A)
        if err: self._wr([("  ✕ "+str(err)+"\n","e")]); return
        self._wr([("  SCHUR  A=QTQ^H\n\n","H"),
                  ("  T (quasi-upper triangular):\n","h"),(mat_str(T.real,dp)+"\n\n","v"),
                  ("  Q (unitary):\n","h"),(mat_str(Z.real,dp)+"\n\n","v"),
                  (f"  ‖A−QTQ^H‖ = {LA.norm(A-Z@T@Z.conj().T):.2e}\n","d")])
        self.nb.select(0)

    # ── Hessenberg ───────────────────────────────────────────────────────────
    def op_hessenberg(self): self._run(self._hessenberg)
    def _hessenberg(self):
        A = self._A(); dp = self._dp()
        H, Q = LA_Engine.hessenberg(A)
        self._wr([("  HESSENBERG  A=QHQᵀ\n\n","H"),
                  ("  H:\n","h"),(mat_str(H,dp)+"\n\n","v"),
                  ("  Q:\n","h"),(mat_str(Q,dp)+"\n\n","v"),
                  (f"  ‖A−QHQᵀ‖ = {LA.norm(A-Q@H@Q.T):.2e}\n","d")])
        self.nb.select(0)

    # ── Eigenanalysis ─────────────────────────────────────────────────────────
    def op_eigen(self): self._run(self._eigen)
    def _eigen(self):
        A = self._A(); dp = self._dp()
        if A.shape[0]!=A.shape[1]: raise ValueError("Square matrix required")
        vals, vecs = LA_Engine.eigen(A)
        out = [("  EIGENANALYSIS\n\n","H")]
        for i,(v,ev) in enumerate(zip(vals, vecs.T)):
            out += [(f"  λ{i+1} = {fmt(v,dp)}\n","v"),
                    ("  vec = "+vec_str(ev.real,dp)+"\n","d")]
        out += [("\n","n"),
                (f"  Trace = {np.trace(A):.{dp}f}   "
                 f"Det = {LA.det(A):.{dp}f}   "
                 f"ρ(A) = {np.max(np.abs(vals)):.{dp}f}\n","d")]
        self._wr(out); self.nb.select(0)
        self._render_eigen_graphs(A, vals, vecs)

    def op_eigen_sym(self): self._run(self._eigen_sym)
    def _eigen_sym(self):
        A = self._A(); dp = self._dp()
        vals, vecs = LA_Engine.eigen_sym(A)
        out = [("  HERMITIAN EIGEN  (real-valued)\n\n","H")]
        for i,(v,ev) in enumerate(zip(vals, vecs.T)):
            out += [(f"  λ{i+1} = {v:.{dp}f}\n","v"),
                    ("  vec = "+vec_str(ev,dp)+"\n","d")]
        self._wr(out); self.nb.select(0)

    # ── Jordan ────────────────────────────────────────────────────────────────
    def op_jordan(self): self._run(self._jordan)
    def _jordan(self):
        A = self._A(); dp = self._dp()
        J, P, err = LA_Engine.jordan_form(A)
        if err: self._wr([("  JORDAN\n\n","H"),("  ✕ "+err+"\n","e")]); return
        self._wr([("  JORDAN NORMAL FORM  A=PJP⁻¹  (SymPy)\n\n","H"),
                  ("  J:\n","h"),(mat_str(J.real,dp)+"\n\n","v"),
                  ("  P:\n","h"),(mat_str(P.real,dp)+"\n\n","v")])
        self.nb.select(0)

    # ── Matrix Functions ──────────────────────────────────────────────────────
    def op_exp(self): self._run(self._exp)
    def _exp(self):
        A = self._A(); dp = self._dp()
        E = LA_Engine.mat_exp(A)
        self._wr([("  MATRIX EXPONENTIAL  eᴬ\n\n","H"),(mat_str(E,dp)+"\n","v")])
        self.nb.select(0)

    def op_sqrt(self): self._run(self._sqrt)
    def _sqrt(self):
        A = self._A(); dp = self._dp(); S, err = LA_Engine.mat_sqrt(A)
        if err: self._wr([("  ✕ "+err+"\n","e")]); return
        self._wr([("  MATRIX SQUARE ROOT  √A\n\n","H"),(mat_str(S.real,dp)+"\n\n","v"),
                  (f"  Verify ‖√A·√A−A‖ = {LA.norm(S@S-A):.2e}\n","d")])
        self.nb.select(0)

    def op_log(self): self._run(self._log)
    def _log(self):
        A = self._A(); dp = self._dp(); L, err = LA_Engine.mat_log(A)
        if err: self._wr([("  ✕ "+err+"\n","e")]); return
        self._wr([("  MATRIX LOGARITHM  ln(A)\n\n","H"),(mat_str(L.real,dp)+"\n","v")])
        self.nb.select(0)

    def op_power(self): self._run(self._power)
    def _power(self):
        A = self._A(); dp = self._dp()
        k = tk.simpledialog.askinteger("Power", "Enter integer k:", parent=self,
                                        initialvalue=2, minvalue=-20, maxvalue=100)
        if k is None: return
        R = LA_Engine.mat_power(A, k)
        self._wr([("  MATRIX POWER  A^{"+str(k)+"}\n\n","H"),(mat_str(R,dp)+"\n","v")])
        self.nb.select(0)

    # ── Properties ────────────────────────────────────────────────────────────
    def op_props(self): self._run(self._props)
    def _props(self):
        A = self._A(); dp = self._dp(); p = LA_Engine.properties(A)
        yes = lambda b: ("✓ Yes","v") if b else ("✗ No","d")
        out = [("  MATRIX PROPERTIES\n\n","H"),
               (f"  Shape     : {p['shape'][0]} × {p['shape'][1]}\n","d"),
               (f"  Rank      : {p['rank']}   Nullity : {p['nullity']}\n","d")]
        if p["trace"]   is not None: out.append((f"  Trace     : {p['trace']:.{dp}f}\n","d"))
        if p["det"]     is not None: out.append((f"  Det       : {p['det']:.{dp}f}\n","d"))
        out += [(f"  Cond #    : {p['cond']:.{dp}f}\n","d"),
                (f"  Frobenius : {p['frob']:.{dp}f}\n","d"),
                (f"  Spectral  : {p['spectral']:.{dp}f}\n","d"),
                (f"  Nuclear   : {p['nuclear']:.{dp}f}\n\n","d"),
                ("  Structural:\n","h")]
        for name, key in [("Symmetric","symmetric"),("Skew-sym","skew_sym"),
                           ("Orthogonal","orthogonal"),("Pos. Def.","pos_def"),
                           ("Invertible","invertible"),("Idempotent","idempotent")]:
            txt, tag = yes(p[key])
            out.append((f"    {name:<12}: {txt}\n", tag))
        if "inv" in p:
            out += [("\n  Inverse:\n","h"),(mat_str(p["inv"],dp)+"\n\n","v")]
        out += [("\n  Pseudoinverse:\n","h"),(mat_str(p["pinv"],dp)+"\n","d")]
        if p["null_space"].size:
            out += [("\n  Null-space basis:\n","h"),
                    (mat_str(p["null_space"],dp)+"\n","d")]
        self._wr(out); self.nb.select(0); self._render_graphs(A)

    # ── Gram-Schmidt ──────────────────────────────────────────────────────────
    def op_gs(self): self._run(self._gs)
    def _gs(self):
        A = self._A(); dp = self._dp()
        Q = LA_Engine.gram_schmidt(A)
        self._wr([("  GRAM-SCHMIDT\n\n","H"),
                  ("  Q (orthonormal):\n","h"),(mat_str(Q,dp)+"\n\n","v"),
                  ("  QᵀQ:\n","d"),(mat_str(Q.T@Q,dp)+"\n","d")])
        self.nb.select(0); self._render_graphs(A)

    # ── PCA ───────────────────────────────────────────────────────────────────
    def op_pca(self): self._run(self._pca)
    def _pca(self):
        A = self._A(); dp = self._dp()
        vecs, vals, var_exp, scores = LA_Engine.pca(A)
        out = [("  PCA\n\n","H"),
               (f"  {A.shape[0]} samples × {A.shape[1]} features\n\n","d"),
               ("  PC directions:\n","h"),(mat_str(vecs,dp)+"\n\n","v"),
               ("  Variance explained:\n","h")]
        for i, ve in enumerate(var_exp):
            bar = "█"*int(ve*30) + "░"*(30-int(ve*30))
            out.append((f"  PC{i+1}  {bar}  {ve*100:.2f}%\n","v"))
        self._wr(out); self.nb.select(0)
        self._render_pca_graphs(A, vecs, vals, var_exp, scores)

    # ── Vector Ops ────────────────────────────────────────────────────────────
    def op_vecops(self): self._run(self._vecops)
    def _vecops(self):
        u, v = self._u(), self._v(); dp = self._dp()
        ops = LA_Engine.vector_ops(u, v)
        out = [("  VECTOR OPERATIONS\n\n","H"),
               (f"  u = {vec_str(u,dp)}\n","d"),
               (f"  v = {vec_str(v,dp)}\n\n","d"),
               (f"  u·v (dot)       = {ops['dot']:.{dp}f}\n","v"),
               (f"  ‖u‖             = {ops['norm_u']:.{dp}f}\n","v"),
               (f"  ‖v‖             = {ops['norm_v']:.{dp}f}\n","v"),
               (f"  angle(u,v)      = {ops['angle_deg']:.{dp}f}°\n","v"),
               (f"  u + v           = {vec_str(ops['u+v'],dp)}\n","d"),
               (f"  u - v           = {vec_str(ops['u-v'],dp)}\n","d"),
               (f"  proj_v onto u   = {vec_str(ops['proj_v_on_u'],dp)}\n","d")]
        if "cross" in ops:
            out.append((f"  u × v (cross)   = {vec_str(ops['cross'],dp)}\n","v"))
        self._wr(out); self.nb.select(0)
        self._render_vec_tab(u, v, ops)

    # ── Markov ────────────────────────────────────────────────────────────────
    def op_markov(self): self._run(self._markov)
    def _markov(self):
        A = self._A(); dp = self._dp()
        # validate stochastic
        row_sums = A.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=0.05):
            self._wr([("  MARKOV CHAIN\n\n","H"),
                      ("  ⚠ Rows do not sum to 1. Normalising...\n\n","e")])
            A = A / row_sums[:, None]
        ss = LA_Engine.markov_steady_state(A)
        P10 = LA_Engine.markov_power(A, 10)
        P50 = LA_Engine.markov_power(A, 50)
        self._wr([("  MARKOV CHAIN\n\n","H"),
                  ("  Transition matrix A:\n","h"),(mat_str(A,dp)+"\n\n","d"),
                  ("  Steady-state distribution π:\n","h"),
                  (vec_str(ss,dp)+"\n","v"),
                  ("\n  A^10:\n","h"),(mat_str(P10,dp)+"\n\n","d"),
                  ("  A^50:\n","h"),(mat_str(P50,dp)+"\n","d")])
        self.nb.select(0)
        self._render_markov_graph(A, ss)

    # ── Arithmetic ────────────────────────────────────────────────────────────
    def op_arith(self): self._run(self._arith)
    def _arith(self):
        A = self._A(); B = self._B(); dp = self._dp()
        out = [("  MATRIX ARITHMETIC\n\n","H"),
               ("  Aᵀ:\n","h"),(mat_str(A.T,dp)+"\n\n","d")]
        if A.shape[0]==A.shape[1]:
            out += [("  A²:\n","h"),(mat_str(A@A,dp)+"\n\n","v")]
        if B is not None:
            if A.shape==B.shape:
                out += [("  A+B:\n","h"),(mat_str(A+B,dp)+"\n\n","v"),
                        ("  A-B:\n","h"),(mat_str(A-B,dp)+"\n\n","v")]
            if A.shape[1]==B.shape[0]:
                out += [("  A·B:\n","h"),(mat_str(A@B,dp)+"\n\n","v")]
            out += [("  Hadamard A⊙B (if same shape):\n","h")]
            if A.shape==B.shape:
                out.append((mat_str(A*B,dp)+"\n","d"))
            else:
                out.append(("  Shape mismatch\n","d"))
        self._wr(out); self.nb.select(0)

    # ── Symbolic ──────────────────────────────────────────────────────────────
    def op_sym(self): self._run(self._sym)
    def _sym(self):
        A_np = self._A(); b_np = self._b()
        A_sym = SMatrix([[Rational(v).limit_denominator(1000) for v in row]
                         for row in A_np.tolist()])
        rref_sym, pivots = A_sym.rref()
        out = [("  SYMBOLIC (SymPy)\n\n","H"),
               ("  A:\n","h"),(str(A_sym)+"\n\n","d"),
               ("  RREF:\n","h"),(str(rref_sym)+"\n\n","v"),
               (f"  Pivots: {pivots}\n\n","d")]
        if A_sym.shape[0]==A_sym.shape[1]:
            det = A_sym.det()
            out += [(f"  Det (exact) = {det}\n\n","v")]
            try:
                evs = A_sym.eigenvals()
                out += [("  Eigenvalues:\n","h")]
                for val, mult in evs.items():
                    out.append((f"  λ={val}  (mult {mult})\n","v"))
            except: pass
        if b_np is not None:
            b_sym = SMatrix([[Rational(x).limit_denominator(1000)] for x in b_np])
            try:
                sol = A_sym.solve(b_sym)
                out += [("\n  Exact solution:\n","h"),(str(sol)+"\n","v")]
            except Exception as e:
                out.append((f"  Solve failed: {e}\n","e"))
        self._wr(out); self.nb.select(0)

    # ── SVD Image ─────────────────────────────────────────────────────────────
    def op_img_svd(self): self._run(self._img_svd)
    def _img_svd(self):
        A = self._A()
        self._wr([("  SVD IMAGE COMPRESSION DEMO\n\n","H"),
                  ("  Using matrix A as a grayscale image...\n\n","d")])
        k_values = [1, 2, 5, min(10, min(A.shape)//2)]
        apply_theme()
        self._clear_graph_tab()
        n_plots = 1 + len(k_values)
        fig = Figure(figsize=(14, 4), facecolor=BG)
        axes = fig.subplots(1, n_plots)
        heatmap(axes[0], A, "Original", cmap="gray")
        for ax, k in zip(axes[1:], k_values):
            Ak, ratio = LA_Engine.svd_compress(A, k)
            heatmap(ax, Ak, f"k={k}  ({ratio*100:.0f}% size)", cmap="gray")
        fig.suptitle("SVD Low-Rank Approximation", color=AMBER, fontsize=12, fontweight="bold")
        self._embed_fig(fig); self.nb.select(2)
        self._app(f"\n  See Graphs tab for k={k_values} approximations.\n","d")

    # ── Quadratic Form ────────────────────────────────────────────────────────
    def op_quad(self): self._run(self._quad)
    def _quad(self):
        A, u = self._A(), self._u(); dp = self._dp()
        if A.shape[0]!=A.shape[1]: raise ValueError("Square matrix required")
        if len(u)!=A.shape[0]: raise ValueError(f"u must have {A.shape[0]} components")
        val = LA_Engine.quadratic_form(A, u)
        self._wr([("  QUADRATIC FORM  xᵀAx\n\n","H"),
                  (f"  A ({A.shape[0]}×{A.shape[1]}):\n","h"),(mat_str(A,dp)+"\n\n","d"),
                  (f"  x = {vec_str(u,dp)}\n\n","d"),
                  (f"  xᵀAx = {val:.{dp}f}\n","v")])
        self.nb.select(0)

    # ── Linear Regression ─────────────────────────────────────────────────────
    def op_regression(self): self._run(self._regression)
    def _regression(self):
        # treat columns of A as features, vector b as target
        A, b = self._A(), self._b(); dp = self._dp()
        if b is None: raise ValueError("Enter y-values in vector b")
        intercept, coefs, r2 = LA_Engine.linear_regression(A, b)
        self._wr([("  LINEAR REGRESSION  y = β₀ + Xβ\n\n","H"),
                  (f"  Intercept β₀ = {intercept:.{dp}f}\n","v"),
                  (f"  Coefficients = {vec_str(coefs,dp)}\n","v"),
                  (f"  R²           = {r2:.{dp}f}\n","v")])
        self.nb.select(0)
        self._render_regression_graph(A, b, intercept, coefs)

    # ══════════════════════════════════════════════════════════════════════════
    #  GRAPH RENDERING
    # ══════════════════════════════════════════════════════════════════════════
    def _clear_graph_tab(self):
        for w in self.tab_graph.winfo_children(): w.destroy()

    def _embed_fig(self, fig):
        self._clear_graph_tab()
        c = FigureCanvasTkAgg(fig, master=self.tab_graph)
        c.draw(); c.get_tk_widget().pack(fill="both", expand=True)
        tb = NavigationToolbar2Tk(c, self.tab_graph)
        tb.update(); tb.config(background=BG2)

    def plot_all(self): self._run(lambda: self._render_graphs(self._A(), go_tab=True))

    def _render_graphs(self, A, go_tab=False):
        apply_theme()
        fig = Figure(figsize=(15,9), facecolor=BG)
        gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.36,
                                left=0.05, right=0.97, top=0.93, bottom=0.07)
        axs = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(4)]

        heatmap(axs[0], A, "Matrix A")

        if A.shape[0]>=2 and A.shape[1]>=1:
            vectors_2d(axs[1], [A[:,j] for j in range(min(A.shape[1],6))],
                       labels=[f"c{j+1}" for j in range(min(A.shape[1],6))],
                       title="Column Vectors")
        else: axs[1].set_visible(False)

        _, s, _ = np.linalg.svd(A)
        singular_bar(axs[2], s, "Singular Values")

        transform_viz(axs[3], A, "Transform (2×2)")

        if A.shape[0]==A.shape[1]:
            vals, _ = np.linalg.eig(A)
            eigen_spectrum(axs[4], vals, "Eigenvalue Spectrum")
        else:
            axs[4].text(0.5,0.5,"Non-square",ha="center",va="center",
                        transform=axs[4].transAxes,color=TEXT2)

        # Heatmap of A^2 if square
        if A.shape[0]==A.shape[1]:
            heatmap(axs[5], A@A, "A²")
        else:
            heatmap(axs[5], A.T@A, "AᵀA")

        # Norm comparison
        norms = {"Frob": LA.norm(A,"fro"), "Spectral": LA.norm(A,2),
                 "Nuclear": np.sum(s), "Max": np.max(np.abs(A))}
        axs[6].bar(list(norms.keys()), list(norms.values()),
                   color=[AMBER,CYAN,VIOLET,GREEN], edgecolor=BG, lw=0.7)
        axs[6].set_title("Matrix Norms"); axs[6].set_ylabel("Value")

        # Condition number bar
        cond = LA.cond(A)
        axs[7].barh(["κ(A)"], [min(cond, 1e6)], color=AMBER2)
        axs[7].set_title(f"Condition # = {cond:.4g}")
        axs[7].set_xlabel("value (capped 1e6)")

        fig.suptitle("Linear Algebra Dashboard — Matrix Forge",
                     color=AMBER, fontsize=13, fontweight="bold")
        self._embed_fig(fig)
        if go_tab: self.nb.select(2)

    def _render_svd_graphs(self, A, s, U, Vt):
        apply_theme()
        fig = Figure(figsize=(14,8), facecolor=BG)
        gs  = gridspec.GridSpec(2,4, figure=fig, hspace=0.45, wspace=0.36,
                                left=0.05, right=0.97, top=0.93, bottom=0.07)
        heatmap(fig.add_subplot(gs[0,0]), A,  "A")
        heatmap(fig.add_subplot(gs[0,1]), U,  "U")
        singular_bar(fig.add_subplot(gs[0,2]), s, "Singular Values σ")
        heatmap(fig.add_subplot(gs[0,3]), Vt, "Vᵀ")

        ax_cum = fig.add_subplot(gs[1,:2])
        cum = np.cumsum(s**2)/np.sum(s**2)*100
        ax_cum.plot(range(1,len(s)+1), cum, "o-", color=AMBER, lw=2)
        ax_cum.fill_between(range(1,len(s)+1), cum, alpha=0.2, color=AMBER)
        ax_cum.set_title("Cumulative Energy (%)"); ax_cum.set_xlabel("k"); ax_cum.set_ylim(0,105)

        # Low-rank approx error
        ax_err = fig.add_subplot(gs[1,2:])
        errs = [LA.norm(A - U[:,:k]@np.diag(s[:k])@Vt[:k,:])
                for k in range(1, len(s)+1)]
        ax_err.semilogy(range(1,len(s)+1), errs, "s-", color=CYAN)
        ax_err.set_title("‖A−Aₖ‖ vs. k"); ax_err.set_xlabel("k")

        fig.suptitle("SVD Analysis", color=AMBER, fontsize=13, fontweight="bold")
        self._embed_fig(fig); self.nb.select(2)

    def _render_eigen_graphs(self, A, vals, vecs):
        apply_theme()
        fig = Figure(figsize=(14,8), facecolor=BG)
        gs  = gridspec.GridSpec(2,3, figure=fig, hspace=0.45, wspace=0.4,
                                left=0.06, right=0.97, top=0.93, bottom=0.07)

        heatmap(fig.add_subplot(gs[0,0]), A, "Matrix A")
        eigen_spectrum(fig.add_subplot(gs[0,1]), vals, "Eigenvalue Spectrum")
        heatmap(fig.add_subplot(gs[0,2]), vecs.real, "Eigenvectors (Re)")

        ax_mag = fig.add_subplot(gs[1,0])
        ax_mag.bar(range(1,len(vals)+1), np.abs(vals), color=RED, edgecolor=BG)
        ax_mag.set_title("|Eigenvalues|"); ax_mag.set_xlabel("Index")

        ax_tr = fig.add_subplot(gs[1,1])
        transform_viz(ax_tr, A, "2×2 Transform")

        # Spectral radius convergence
        ax_sp = fig.add_subplot(gs[1,2])
        rho = np.max(np.abs(vals))
        ax_sp.axhline(rho, color=AMBER, ls="--", lw=1.5, label=f"ρ(A)={rho:.4f}")
        ax_sp.bar(range(1,len(vals)+1), np.abs(vals), color=VIOLET, edgecolor=BG, alpha=0.8)
        ax_sp.set_title("Spectral Radius"); ax_sp.legend()

        fig.suptitle("Eigenvalue Analysis", color=AMBER, fontsize=13, fontweight="bold")
        self._embed_fig(fig); self.nb.select(2)

    def _render_pca_graphs(self, A, vecs, vals, var_exp, scores):
        apply_theme()
        fig = Figure(figsize=(14,8), facecolor=BG)
        gs  = gridspec.GridSpec(2,3, figure=fig, hspace=0.45, wspace=0.4,
                                left=0.06, right=0.97, top=0.93, bottom=0.07)

        heatmap(fig.add_subplot(gs[0,0]), A, "Data Matrix")

        ax_bar = fig.add_subplot(gs[0,1])
        pcs = [f"PC{i+1}" for i in range(len(var_exp))]
        ax_bar.bar(pcs, var_exp*100, color=VIOLET, alpha=0.85)
        ax_bar.plot(pcs, np.cumsum(var_exp)*100, "o-", color=AMBER, lw=2)
        ax_bar.set_title("Variance Explained (%)"); ax_bar.set_ylim(0,110)

        ax_hm = fig.add_subplot(gs[0,2])
        heatmap(ax_hm, vecs, "PC Directions")

        ax_sc = fig.add_subplot(gs[1,:2])
        if scores.shape[1]>=2:
            pca_biplot(ax_sc, scores, vecs, title="PCA Biplot (PC1-PC2)")
        else:
            ax_sc.scatter(scores[:,0], np.zeros(len(scores)), c=CYAN)
            ax_sc.set_title("PC1 scores")

        ax_cor = fig.add_subplot(gs[1,2])
        heatmap(ax_cor, np.corrcoef(A.T), "Correlation Matrix", cmap="coolwarm")

        fig.suptitle("PCA Analysis", color=AMBER, fontsize=13, fontweight="bold")
        self._embed_fig(fig); self.nb.select(2)

    def _render_iter_graphs(self, A, hj, hg):
        apply_theme()
        fig = Figure(figsize=(12,5), facecolor=BG)
        gs  = gridspec.GridSpec(1,2, figure=fig, hspace=0.3, wspace=0.38)
        convergence_plot(fig.add_subplot(gs[0,0]),
                         [hj, hg], ["Jacobi","Gauss-Seidel"],
                         "Convergence: ‖Δx‖ per iteration")

        ax_h = fig.add_subplot(gs[0,1])
        heatmap(ax_h, A, "Matrix A (Diag Dominance?)")
        fig.suptitle("Iterative Solver Analysis", color=AMBER)
        self._embed_fig(fig); self.nb.select(2)

    def _render_markov_graph(self, A, ss):
        apply_theme()
        fig = Figure(figsize=(12,5), facecolor=BG)
        gs  = gridspec.GridSpec(1,3, figure=fig, wspace=0.38)

        heatmap(fig.add_subplot(gs[0,0]), A, "Transition Matrix")

        ax_ss = fig.add_subplot(gs[0,1])
        states = [f"s{i+1}" for i in range(len(ss))]
        ax_ss.bar(states, ss, color=AMBER, edgecolor=BG)
        ax_ss.set_title("Steady-State Distribution π"); ax_ss.set_ylabel("Probability")

        ax_pw = fig.add_subplot(gs[0,2])
        ks = range(1,31)
        # track state 0 prob starting from uniform
        x0 = np.ones(A.shape[0]) / A.shape[0]
        trace = []
        xk = x0.copy()
        for _ in ks:
            trace.append(xk[0])
            xk = A.T @ xk
        ax_pw.plot(ks, trace, "o-", color=CYAN, lw=2)
        ax_pw.axhline(ss[0], color=AMBER, ls="--", lw=1.5, label=f"π₁={ss[0]:.4f}")
        ax_pw.set_title("State 1 prob converging to π₁")
        ax_pw.set_xlabel("Steps"); ax_pw.legend()

        fig.suptitle("Markov Chain Analysis", color=AMBER)
        self._embed_fig(fig); self.nb.select(2)

    def _render_vec_tab(self, u, v, ops):
        for w in self.tab_vec.winfo_children(): w.destroy()
        apply_theme()
        fig = Figure(figsize=(12,5), facecolor=BG)
        gs  = gridspec.GridSpec(1,3, figure=fig, wspace=0.38)

        ax1 = fig.add_subplot(gs[0,0])
        vecs_2d = [u[:2], v[:2]]
        vectors_2d(ax1, vecs_2d, ["u","v"], "u and v (2D proj)")

        ax2 = fig.add_subplot(gs[0,1])
        vectors_2d(ax2, [u[:2], v[:2], ops["proj_v_on_u"][:2]],
                   ["u","v","proj"], "Projection")

        ax3 = fig.add_subplot(gs[0,2])
        labels = ["‖u‖","‖v‖","u·v"]
        vals_  = [ops["norm_u"], ops["norm_v"], ops["dot"]]
        ax3.bar(labels, vals_, color=[CYAN,TEAL,AMBER], edgecolor=BG)
        ax3.set_title("Magnitudes & Dot Product")

        fig.suptitle(f"Vector Ops — angle = {ops['angle_deg']:.2f}°",
                     color=AMBER, fontsize=12)
        c = FigureCanvasTkAgg(fig, master=self.tab_vec)
        c.draw(); c.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(c, self.tab_vec).update()
        self.nb.select(3)

    def _render_regression_graph(self, A, b, intercept, coefs):
        apply_theme()
        fig = Figure(figsize=(12,5), facecolor=BG)
        gs  = gridspec.GridSpec(1,2, figure=fig, wspace=0.38)

        ax1 = fig.add_subplot(gs[0,0])
        x_ = A[:,0]
        ax1.scatter(x_, b, c=CYAN, s=60, label="data", edgecolors=AMBER, lw=0.5)
        xs = np.linspace(x_.min(), x_.max(), 200)
        ys = intercept + coefs[0]*xs + (coefs[1]*xs**2 if len(coefs)>1 else 0)
        ax1.plot(xs, ys, color=AMBER, lw=2, label=f"fit")
        ax1.set_title("Regression Fit"); ax1.legend(); ax1.set_xlabel("x"); ax1.set_ylabel("y")

        ax2 = fig.add_subplot(gs[0,1])
        y_pred = intercept + A @ coefs
        residuals = b - y_pred
        ax2.scatter(y_pred, residuals, c=RED, s=50, edgecolors=BORDER, lw=0.5)
        ax2.axhline(0, color=AMBER, ls="--", lw=1.5)
        ax2.set_title("Residuals"); ax2.set_xlabel("ŷ"); ax2.set_ylabel("residual")

        fig.suptitle("Linear Regression via Least Squares", color=AMBER)
        self._embed_fig(fig); self.nb.select(2)

    # ══════════════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ══════════════════════════════════════════════════════════════════════════
    def clear_all(self):
        self.mat_a.delete("1.0","end")
        self.mat_b.delete("1.0","end")
        self.vec_b.delete(0,"end")
        self.vec_u.delete(0,"end"); self.vec_v.delete(0,"end")
        self.res.config(state="normal"); self.res.delete("1.0","end")
        self.steps.config(state="normal"); self.steps.delete("1.0","end")

    def load_5x5(self):
        self.mat_a.delete("1.0","end")
        self.mat_a.insert("1.0",
            "4  2  0  1  3\n"
            "2  6  1  0  2\n"
            "0  1  5  2  1\n"
            "1  0  2  7  3\n"
            "3  2  1  3  8")
        self.vec_b.delete(0,"end"); self.vec_b.insert(0,"10 14 9 13 17")

    def copy_res(self):
        txt = self.res.get("1.0","end")
        self.clipboard_clear(); self.clipboard_append(txt)
        self._status("Copied ✓")

    def export_res(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text","*.txt"),("All","*.*")],
            title="Export results")
        if path:
            with open(path, "w") as f:
                f.write(self.res.get("1.0","end"))
            self._status(f"Saved → {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCE SHEET
# ══════════════════════════════════════════════════════════════════════════════
REFERENCE_TEXT = """
╔══════════════════════════════════════════════════════════════════════════╗
║                 MATRIX FORGE — Quick Reference Sheet                    ║
╚══════════════════════════════════════════════════════════════════════════╝

INPUT FORMAT
  • Matrix A: rows separated by newlines, values by spaces or commas
  • Supports Python expressions e.g. "1/3  sqrt(2)  pi" (eval'd)
  • # at start of line = comment

SYSTEMS
  Ax = b  solve:    Unique (rank A = rank [A|b] = n)
                    Infinite (rank A = rank [A|b] < n)
                    Inconsistent (rank A ≠ rank [A|b])
  Least Squares:    minimise ‖Ax−b‖₂  →  x̂ = A⁺b = (AᵀA)⁻¹Aᵀb

DECOMPOSITIONS
  LU:    PA = LU   (P permutation, L lower, U upper triangular)
  QR:    A = QR    (Q orthogonal, R upper triangular)
  SVD:   A = UΣVᵀ  (U,V orthogonal, Σ diagonal non-neg)
  Chol:  A = LLᵀ   (A must be symmetric positive definite)
  Schur: A = QTQ*  (T quasi-upper triangular, Q unitary)

EIGENVALUES
  Ax = λx   →   det(A − λI) = 0  (characteristic equation)
  Trace(A) = Σλᵢ,  Det(A) = Πλᵢ
  Spectral radius ρ(A) = max|λᵢ|
  A is stable iff ρ(A) < 1

MATRIX PROPERTIES
  Symmetric:    A = Aᵀ                (all real eigenvalues)
  Orthogonal:   AᵀA = I              (eigenvalues on unit circle)
  Pos. Def.:    xᵀAx > 0 ∀x≠0       (all eigenvalues > 0)
  Idempotent:   A² = A               (eigenvalues 0 or 1)
  Nilpotent:    Aᵏ = 0 for some k    (all eigenvalues = 0)

NORMS
  Frobenius:  ‖A‖F  = √(Σaᵢⱼ²) = √(Σσᵢ²)
  Spectral:   ‖A‖₂  = σ₁  (largest singular value)
  Nuclear:    ‖A‖*  = Σσᵢ
  Condition:  κ(A)  = σ₁/σₙ  (sensitivity to perturbations)

ITERATIVE SOLVERS  (require diagonally dominant A)
  Jacobi:       xᵢ⁽ᵏ⁺¹⁾ = (bᵢ − Σⱼ≠ᵢ aᵢⱼxⱼ⁽ᵏ⁾) / aᵢᵢ
  Gauss-Seidel: uses updated values immediately → faster

VECTOR OPERATIONS
  Dot:       u·v = Σuᵢvᵢ = ‖u‖‖v‖cos θ
  Cross:     u×v = det([e₁ e₂ e₃; u; v])   (3D only)
  Projection of v onto u: (u·v / u·u)·u

PCA
  Centre data: X̃ = X − mean(X)
  Covariance: C = X̃ᵀX̃/(n-1)
  Eigen decompose C → principal directions
  Variance explained: λᵢ / Σλᵢ

MARKOV CHAINS
  P: row-stochastic matrix (each row sums to 1)
  Steady state π: Pᵀπ = π,  Σπᵢ = 1
  Converges as Pᵏ → rows all equal π

MATRIX FUNCTIONS  (via Padé approximation / Schur decomposition)
  eᴬ  = I + A + A²/2! + …   (scipy.linalg.expm)
  √A   : scipy.linalg.sqrtm
  ln A : scipy.linalg.logm

SVD COMPRESSION
  Rank-k approx: Aₖ = U[:,1:k] Σ[1:k,1:k] Vᵀ[1:k,:]
  Storage ratio: k(m+n+1) / mn

USEFUL IDENTITIES
  (AB)ᵀ = BᵀAᵀ         (AB)⁻¹ = B⁻¹A⁻¹
  det(AB) = det(A)det(B) rank(AB) ≤ min(rank A, rank B)
  Cauchy-Schwarz: |u·v| ≤ ‖u‖‖v‖
  Sylvester: rank(A)+rank(B)−n ≤ rank(AB) ≤ min(rank A, rank B)

KEYBOARD SHORTCUTS  (future)
  Ctrl+Enter : run last operation
  Ctrl+S     : save results
  Ctrl+L     : clear all
"""


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import tkinter.simpledialog   # needed for matrix power dialog
    MatrixForge().mainloop()

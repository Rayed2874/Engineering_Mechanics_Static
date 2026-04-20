"""
Van Hatch FBD Solver — Problem 3/57  (symbolic + numeric)
─────────────────────────────────────────────────────────
Features:
  • Scrollable left panel with mouse-wheel support
  • Two-column input layout
  • Popup FBD window (resizable, closeable, remembers position)
  • Symbolic inputs generate a drawable approximated FBD
  • Corner label panel (key geometry summary)
  • Result box on right side
  • Clean reset / close behavior
"""

import math, tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
from sympy import abc

# ── palette ────────────────────────────────────────────────────────────────────
BG      = "#f0efe9"
CARD    = "#ffffff"
BLUE    = "#2578c8"
GREEN   = "#1a9e5f"
RED     = "#c83025"
ORANGE  = "#e07b00"
PURPLE  = "#7c3fb5"
SUBTLE  = "#f7f6f2"
BORDER  = "#d8d8d8"
FS      = 11          # base font size
G_NUM   = 9.81
G_SYM   = sp.Float(9.81)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_sym(text: str):
    text = text.strip()
    if not text:
        raise ValueError("Empty field.")
    try:
        return sp.Float(float(text))
    except ValueError:
        pass
    local = {n: sp.Symbol(n) for n in abc.__dict__ if not n.startswith("_")}
    local.update({k: v for k, v in sp.__dict__.items()
                  if isinstance(v, (sp.core.function.UndefinedFunction, type))
                  and not k.startswith("_")})
    try:
        return sp.sympify(text, locals=local)
    except Exception:
        raise ValueError(f"Cannot parse '{text}' as a number or expression.")

def is_numeric(expr):  return len(expr.free_symbols) == 0

def try_float(expr):
    try:
        v = complex(expr)
        if abs(v.imag) < 1e-12:
            return float(v.real)
    except Exception:
        pass
    return None

def sym_cos(d): return sp.cos(sp.pi * d / 180)
def sym_sin(d): return sp.sin(sp.pi * d / 180)


# ══════════════════════════════════════════════════════════════════════════════
#  PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

def compute(P, mass, AO, OB, AB, theta_deg, dist_P):
    simp = lambda e: sp.simplify(sp.trigsimp(sp.expand(e)))
    cos_a = simp((AO**2 + AB**2 - OB**2) / (2 * AO * AB))
    cos_a_f = try_float(cos_a)
    if cos_a_f is not None and not (-1.0 <= cos_a_f <= 1.0):
        raise ValueError("Invalid triangle: AO, OB, AB cannot form a triangle.")
    alpha     = sp.acos(cos_a)
    sin_alpha = sp.sqrt(1 - cos_a**2)
    denom     = simp(2 * AO * sin_alpha)
    W         = mass * G_SYM
    numer     = simp(P * dist_P + W * AO * sym_cos(theta_deg - 180 * alpha / sp.pi))
    F         = simp(numer / denom)
    return {"F": F, "alpha": alpha, "W": W, "cos_a": cos_a}


# ══════════════════════════════════════════════════════════════════════════════
#  FBD DRAWING
# ══════════════════════════════════════════════════════════════════════════════

# def draw_fbd(ax, P_f, mass_f, AO_f, OB_f, AB_f, theta_f, dist_P_f, F_f, alpha_f,
#              sym_labels=None):
#     import numpy as np
#     import math
#     import matplotlib.patches as mpatches
#     from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

#     ax.clear()
#     ax.set_aspect("equal")
#     ax.axis("off")
#     ax.set_facecolor("white")

#     # =====================================================
#     # BASIC VALUES
#     # =====================================================
#     W = mass_f * 9.81
#     theta = math.radians(theta_f)

#     # Geometry points
#     C = np.array([0.0, 0.0])              # hinge
#     B = np.array([0.0, -OB_f])            # lower mount

#     A = np.array([
#         AO_f * math.cos(theta),
#         AO_f * math.sin(theta)
#     ])

#     Pp = np.array([
#         dist_P_f * math.cos(theta),
#         dist_P_f * math.sin(theta)
#     ])

#     G = A * 0.62

#     # =====================================================
#     # HELPERS
#     # =====================================================
#     def vec_arrow(p1, p2, txt=None, lw=2.4, fs=12):
#         arr = FancyArrowPatch(
#             p1, p2,
#             arrowstyle='Simple,head_length=12,head_width=12,tail_width=2',
#             color="black",
#             linewidth=lw
#         )
#         ax.add_patch(arr)
#         if txt:
#             mid = (np.array(p1) + np.array(p2)) / 2
#             ax.text(mid[0], mid[1], txt, fontsize=fs, fontweight="bold")

#     def dim_arrow(p1, p2, txt, offset=(0, 0), fs=11):
#         arr = FancyArrowPatch(
#             p1, p2,
#             arrowstyle='<->',
#             mutation_scale=12,
#             linewidth=1.2,
#             linestyle='--',
#             color="gray"
#         )
#         ax.add_patch(arr)

#         mid = (np.array(p1) + np.array(p2)) / 2
#         ax.text(mid[0] + offset[0], mid[1] + offset[1],
#                 txt, fontsize=fs, color="black")

#     # =====================================================
#     # HATCH CURVED BODY
#     # =====================================================
#     xs = np.linspace(C[0], Pp[0], 80)
#     ys = 0.18*np.sin(np.linspace(0, math.pi, 80)) + np.linspace(C[1], Pp[1], 80)

#     ax.plot(xs, ys, lw=16, color="#7f97a8", solid_capstyle="round")
#     ax.plot(xs, ys, lw=10, color="#b7cad8", solid_capstyle="round")
#     ax.plot(xs, ys + 0.015, lw=1.2, color="black")
#     ax.plot(xs, ys - 0.015, lw=1.2, color="black")

#     # actual A point on hatch
#     idxA = int(len(xs)*AO_f/dist_P_f)
#     # A = np.array([xs[idxA], ys[idxA]])
#     idxA = max(0, min(len(xs)-1, idxA))
#     A = np.array([xs[idxA], ys[idxA]])

#     # =====================================================
#     # STRUT BODY
#     # =====================================================
#     ax.plot([B[0], A[0]], [B[1], A[1]], lw=8, color="#7f97a8")
#     ax.plot([B[0], A[0]], [B[1], A[1]], lw=4, color="#dce8f0")

#     # piston rod
#     mid = B + (A-B)*0.55
#     ax.plot([mid[0], A[0]], [mid[1], A[1]], lw=5, color="#d0d0d0")

#     # =====================================================
#     # JOINTS
#     # =====================================================
#     for pt, r in [(C, 0.018), (B, 0.016), (A, 0.012)]:
#         ax.add_patch(Circle(pt, r, fc="gray", ec="black", lw=1.5))
#         ax.add_patch(Circle(pt, r*0.45, fc="black"))

#     # wall bracket
#     ax.add_patch(Rectangle((-0.03, B[1]-0.05), 0.025, 0.10,
#                            fc="#b0b0b0", ec="black"))

#     # =====================================================
#     # LABEL POINTS
#     # =====================================================
#     ax.text(C[0]-0.03, C[1]+0.03, "C", fontsize=18)
#     ax.text(B[0]+0.02, B[1]-0.01, "B", fontsize=18)
#     ax.text(A[0]+0.02, A[1]+0.02, "A", fontsize=18)
#     ax.text(G[0]+0.02, G[1], "G", fontsize=18)

#     # =====================================================
#     # FORCES
#     # =====================================================
#     # Applied force P
#     vec_arrow(Pp + np.array([0, 0.16]), Pp + np.array([0, 0.01]))
#     ax.text(Pp[0]+0.03, Pp[1]+0.10, r"$P = %.0f\ \mathrm{N}$" % P_f,
#             fontsize=16, fontweight="bold")

#     # Weight
#     vec_arrow(G + np.array([0, 0.10]), G + np.array([0, -0.08]))
#     ax.text(G[0]-0.04, G[1]-0.14,
#             r"$W = mg \approx %.0f\ \mathrm{N}$" % W,
#             fontsize=16, fontweight="bold")

#     # Strut force
#     AB = A-B
#     ABu = AB / np.linalg.norm(AB)

#     start = B + ABu*0.10
#     end   = start + ABu*0.18

#     vec_arrow(start, end)
#     vec_arrow(start + np.array([0.03, 0.02]),
#               end + np.array([0.03, 0.02]))

#     # ax.text(end[0]+0.02, end[1]+0.02, r"$F_A$", fontsize=18, fontweight="bold")
#     ax.text(end[0]+0.02, end[1]+0.02, f"F = {abs(F_f):.1f} N")
#     ax.text(start[0]+0.06, start[1]+0.02, r"$F_A$", fontsize=18, fontweight="bold")

#     # Reactions at hinge
#     vec_arrow(C + np.array([-0.12, 0]), C + np.array([-0.02, 0]))
#     ax.text(C[0]-0.16, C[1]-0.02, r"$C_x$", fontsize=18)

#     vec_arrow(C + np.array([0, -0.13]), C + np.array([0, -0.02]))
#     ax.text(C[0]-0.02, C[1]-0.17, r"$C_y$", fontsize=18)

#     vec_arrow(C + np.array([0, 0.02]), C + np.array([0, 0.13]))
#     ax.text(C[0]-0.02, C[1]+0.15, r"$C_y$", fontsize=18)

#     # =====================================================
#     # DIMENSIONS
#     # =====================================================
#     # top distance
#     dim_arrow((C[0], A[1]+0.18), (Pp[0], A[1]+0.18),
#               f"{dist_P_f*1000:.0f} mm", (0, 0.02))

#     # vertical CG
#     ax.plot([A[0], A[0]], [A[1], G[1]], ls='--', lw=1.2, color='gray')
#     # ax.text(A[0]+0.03, (A[1]+G[1])/2, "37.5 mm", fontsize=15)
#     cg_mm = np.linalg.norm(A - G) * 1000
#     ax.text(A[0]+0.03, (A[1]+G[1])/2, f"{cg_mm:.1f} mm", fontsize=15)

#     # OB
#     ax.text(B[0]+0.05, (B[1]+C[1])/2, f"{OB_f*1000:.0f} mm", fontsize=14)

#     # =====================================================
#     # ANGLES
#     # =====================================================
#     arc = mpatches.Arc(C, 0.30, 0.30,
#                        theta1=0, theta2=theta_f,
#                        lw=1.5)
#     ax.add_patch(arc)
#     ax.text(0.12, 0.03, f"{theta_f:.0f}°", fontsize=14)

#     # =====================================================
#     # AXES
#     # =====================================================
#     vec_arrow((-0.22, -0.30), (-0.22, -0.16))
#     vec_arrow((-0.22, -0.30), (-0.08, -0.30))
#     ax.text(-0.23, -0.15, "y", fontsize=18)
#     ax.text(-0.07, -0.31, "x", fontsize=18)

#     # =====================================================
#     # TITLE
#     # =====================================================
#     ax.set_title("Free Body Diagram: Van Hatch",
#                  fontsize=32, pad=20)

#     # =====================================================
#     # VIEW LIMITS
#     # =====================================================
#     ax.set_xlim(-0.28, max(Pp[0]+0.20, 0.95))
#     ax.set_ylim(B[1]-0.18, A[1]+0.35)

def draw_fbd(ax, P_f, mass_f, AO_f, OB_f, AB_f, theta_f, dist_P_f, F_f, alpha_f,
             sym_labels=None):
    import numpy as np
    import math
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, Circle, Rectangle

    ax.clear()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("white")

    sl = sym_labels or {}

    # ─────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────
    def txt(key, fallback):
        return str(sl.get(key, fallback))

    def mm(v):
        return f"{v * 1000:.0f} mm"

    def force(v):
        return f"{abs(v):.1f} N"

    W = mass_f * 9.81
    theta = math.radians(theta_f)
    alpha_deg = math.degrees(alpha_f)

    # ─────────────────────────────────────────────
    # geometry
    # ─────────────────────────────────────────────
    C = np.array([0.0, 0.0])
    B = np.array([0.0, -OB_f])

    Pp = np.array([
        dist_P_f * math.cos(theta),
        dist_P_f * math.sin(theta)
    ])

    xs = np.linspace(C[0], Pp[0], 90)
    ys = np.linspace(C[1], Pp[1], 90) + 0.08 * np.sin(np.linspace(0, math.pi, 90))

    idxA = int((AO_f / max(dist_P_f, 1e-9)) * (len(xs) - 1))
    idxA = max(0, min(len(xs) - 1, idxA))

    A = np.array([xs[idxA], ys[idxA]])
    G = A * 0.62

    # ─────────────────────────────────────────────
    # arrow helper
    # ─────────────────────────────────────────────
    def arr(p1, p2, lw=2.2):
        patch = FancyArrowPatch(
            p1, p2,
            arrowstyle='Simple,head_length=10,head_width=10,tail_width=2',
            color="black",
            linewidth=lw
        )
        ax.add_patch(patch)

    # ─────────────────────────────────────────────
    # hatch body
    # ─────────────────────────────────────────────
    ax.plot(xs, ys, lw=16, color="#7f97a8", solid_capstyle="round")
    ax.plot(xs, ys, lw=10, color="#b7cad8", solid_capstyle="round")
    ax.plot(xs, ys + 0.01, lw=1, color="black")
    ax.plot(xs, ys - 0.01, lw=1, color="black")

    # ─────────────────────────────────────────────
    # strut
    # ─────────────────────────────────────────────
    ax.plot([B[0], A[0]], [B[1], A[1]], lw=8, color="#7f97a8")
    ax.plot([B[0], A[0]], [B[1], A[1]], lw=4, color="#e8eef2")

    rod = B + (A - B) * 0.58
    ax.plot([rod[0], A[0]], [rod[1], A[1]], lw=5, color="#d0d0d0")

    # wall bracket
    ax.add_patch(Rectangle(
        (-0.03, B[1] - 0.05), 0.025, 0.10,
        fc="#b0b0b0", ec="black"
    ))

    # ─────────────────────────────────────────────
    # joints
    # ─────────────────────────────────────────────
    for pt, r in [(C, 0.018), (B, 0.016), (A, 0.014)]:
        ax.add_patch(Circle(pt, r, fc="gray", ec="black", lw=1.2))
        ax.add_patch(Circle(pt, r * 0.45, fc="black"))

    # ─────────────────────────────────────────────
    # point labels only
    # ─────────────────────────────────────────────
    ax.text(C[0] - 0.025, C[1] + 0.03, "C",
            fontsize=13, fontweight="bold")

    ax.text(B[0] + 0.02, B[1] - 0.01, "B",
            fontsize=13, fontweight="bold")

    ax.text(A[0] + 0.02, A[1] + 0.02, "A",
            fontsize=13, fontweight="bold")

    ax.text(G[0] + 0.02, G[1], "G",
            fontsize=12)

    # ─────────────────────────────────────────────
    # arrows only (no labels)
    # ─────────────────────────────────────────────
    # Applied force P
    arr(Pp + np.array([0, 0.14]), Pp + np.array([0, 0.01]))

    # Weight
    arr(G + np.array([0, 0.10]), G + np.array([0, -0.08]))

    # Strut force
    AB = A - B
    ABu = AB / (np.linalg.norm(AB) or 1e-9)

    sign = 1 if F_f >= 0 else -1

    s1 = B + ABu * 0.08
    s2 = s1 + ABu * (0.18 * sign)

    arr(s1, s2)

    # hinge reactions
    arr(C + np.array([-0.12, 0]), C + np.array([-0.02, 0]))
    arr(C + np.array([0, -0.13]), C + np.array([0, -0.02]))

    # ─────────────────────────────────────────────
    # angle arcs only
    # ─────────────────────────────────────────────
    arc1 = mpatches.Arc(
        C, 0.28, 0.28,
        theta1=0,
        theta2=theta_f,
        lw=1.2
    )
    ax.add_patch(arc1)

    arc2 = mpatches.Arc(
        A, 0.18, 0.18,
        theta1=-90,
        theta2=-90 + alpha_deg,
        lw=1.0
    )
    ax.add_patch(arc2)

    # ─────────────────────────────────────────────
    # corner legend
    # ─────────────────────────────────────────────
    if "mass" in sl:
        w_text = f"{txt('mass', mass_f)}·g"
    else:
        w_text = force(W)

    if "F_result" in sl:
        f_text = "symbolic"
    else:
        f_text = force(F_f)

    info = (
        f"P   = {txt('P', f'{P_f:.0f} N')}\n"
        f"W   = {w_text}\n"
        f"F   = {f_text}\n"
        f"AO  = {txt('AO', mm(AO_f))}\n"
        f"OB  = {txt('OB', mm(OB_f))}\n"
        f"AB  = {txt('AB', mm(AB_f))}\n"
        f"dP  = {txt('dist_P', mm(dist_P_f))}\n"
        f"θ   = {txt('theta', f'{theta_f:.0f}°')}\n"
        f"α   = {txt('alpha_result', f'{alpha_deg:.1f}°')}\n"
        f"Mode= {'Compression' if F_f >= 0 else 'Tension'}"
    )

    ax.text(
        0.01, 0.99,
        info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(
            boxstyle="round,pad=0.45",
            fc="white",
            ec="#999999",
            alpha=0.95
        )
    )

    # ─────────────────────────────────────────────
    # title
    # ─────────────────────────────────────────────
    note = " [symbolic]" if sl else ""

    ax.set_title(
        f"Free Body Diagram: Van Hatch{note}",
        fontsize=18,
        pad=14,
        fontweight="bold"
    )

    # ─────────────────────────────────────────────
    # autoscale
    # ─────────────────────────────────────────────
    xs_all = [C[0], B[0], A[0], Pp[0]]
    ys_all = [C[1], B[1], A[1], Pp[1]]

    ax.set_xlim(min(xs_all) - 0.22, max(xs_all) + 0.22)
    ax.set_ylim(min(ys_all) - 0.22, max(ys_all) + 0.25)
# ══════════════════════════════════════════════════════════════════════════════
#  SCROLL CANVAS HELPER
# ══════════════════════════════════════════════════════════════════════════════

class ScrollFrame(tk.Frame):
    """A Frame wrapped in a Canvas+Scrollbar with mouse-wheel support."""
    def __init__(self, parent, bg=BG, width=420, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._canvas = tk.Canvas(self, bg=bg, highlightthickness=0,
                                 width=width)
        self._vsb = tk.Scrollbar(self, orient="vertical",
                                 command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)
        self._vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.inner)

    def _on_inner_configure(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfig(self._win_id, width=e.width)

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>",   self._on_mousewheel, add="+")
        widget.bind("<Button-4>",     self._on_mousewheel, add="+")
        widget.bind("<Button-5>",     self._on_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_mousewheel(child)

    def _on_mousewheel(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(1, "units")
        else:
            self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def rebind(self, widget):
        """Call after adding new children so they also scroll."""
        self._bind_mousewheel(widget)


# ══════════════════════════════════════════════════════════════════════════════
#  FIELDS
# ══════════════════════════════════════════════════════════════════════════════

FIELDS = [
    ("P",      "Applied force P",           "N",   "40"),
    ("mass",   "Door mass",                 "kg",  "40"),
    ("AO",     "Length AO",                 "mm",  "550"),
    ("OB",     "Length OB",                 "mm",  "175"),
    ("AB",     "Strut length AB",           "mm",  "600"),
    ("theta",  "Door angle θ",              "deg", "30"),
    ("dist_P", "Distance P from hinge O",   "mm",  "1125"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Van Hatch — Hydraulic Strut Solver  ·  Problem 3/57")
        self.configure(bg=BG)
        self.geometry("1100x700")
        self.minsize(800, 560)
        self.resizable(True, True)
        self._entries   = {}
        self._fbd_win   = None   # reference to popup window
        self._fbd_fig   = None
        self._fbd_ax    = None
        self._fbd_canvas= None
        self._last_result = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────
    #  LAYOUT
    # ─────────────────────────────────────────────────────────────────────
    def _build(self):
        # ── outer panes ───────────────────────────────────────────────────
        self._pane = tk.PanedWindow(self, orient=tk.HORIZONTAL,
                                    sashwidth=6, sashrelief="flat",
                                    bg=BORDER)
        self._pane.pack(fill=tk.BOTH, expand=True)

        # LEFT  – scrollable input panel
        self._scroll_frame = ScrollFrame(self._pane, bg=BG, width=430)
        self._pane.add(self._scroll_frame, minsize=360)
        inner = self._scroll_frame.inner

        # RIGHT – results / info panel
        right = tk.Frame(self._pane, bg=SUBTLE)
        self._pane.add(right, minsize=300)

        self._build_left(inner)
        self._build_right(right)
        self._scroll_frame.rebind(inner)

    # ─── LEFT panel ───────────────────────────────────────────────────────
    def _build_left(self, p):
        pad = dict(padx=22, pady=0)

        # ── TITLE ──────────────────────────────────────────────
        tk.Label(
            p,
            text="Problem 3/57",
            bg=BG,
            font=("Helvetica", 22, "bold"),
            fg="#222"
        ).pack(anchor="w", padx=22, pady=(20, 0))

        tk.Label(
            p,
            text="Van Hatch  —  Hydraulic Strut Analysis",
            bg=BG,
            font=("Helvetica", FS),
            fg="#666"
        ).pack(anchor="w", **pad)

        tk.Label(
            p,
            text="Enter numbers or symbolic expressions (e.g. F, m, 2*a)",
            bg=BG,
            font=("Helvetica", 9),
            fg="#aaa"
        ).pack(anchor="w", padx=22, pady=(2, 14))

        # ── INPUT CARD ─────────────────────────────────────────
        card = tk.Frame(
            p,
            bg=CARD,
            highlightbackground=BORDER,
            highlightthickness=1,
            padx=18,
            pady=14
        )
        card.pack(fill=tk.X, padx=22, pady=(0, 6))

        tk.Label(
            card,
            text="Parameters",
            bg=CARD,
            font=("Helvetica", FS, "bold"),
            fg="#333"
        ).pack(anchor="w", pady=(0, 10))

        # ── LIST STYLE INPUTS ──────────────────────────────────
        for key, label, unit, default in FIELDS:
            row = tk.Frame(card, bg=CARD)
            row.pack(fill=tk.X, pady=4)

            tk.Label(
                row,
                text=label,
                bg=CARD,
                font=("Helvetica", 10),
                fg="#555",
                width=24,
                anchor="w"
            ).pack(side=tk.LEFT)

            e = tk.Entry(
                row,
                font=("Helvetica", FS),
                width=12,
                relief="solid",
                bd=1
            )
            e.insert(0, default)
            e.pack(side=tk.LEFT, padx=(6, 6))
            e.bind("<Return>", lambda _: self._solve())

            self._entries[key] = e

            tk.Label(
                row,
                text=unit,
                bg=CARD,
                font=("Helvetica", 9),
                fg="#aaa",
                width=5,
                anchor="w"
            ).pack(side=tk.LEFT)

        # ── GEOMETRY SUMMARY ───────────────────────────────────
        geo = tk.Frame(
            p,
            bg="#fffbe8",
            highlightbackground="#e8d88a",
            highlightthickness=1,
            padx=14,
            pady=10
        )
        geo.pack(fill=tk.X, padx=22, pady=(4, 0))

        tk.Label(
            geo,
            text="⚙ Geometry Summary",
            bg="#fffbe8",
            font=("Helvetica", 9, "bold"),
            fg="#8a6800"
        ).pack(anchor="w")

        self._geo_var = tk.StringVar(value="— enter values to populate")

        tk.Label(
            geo,
            textvariable=self._geo_var,
            bg="#fffbe8",
            font=("Courier", 9),
            fg="#6b5000",
            justify="left"
        ).pack(anchor="w", pady=(4, 0))

        # ── BUTTONS ────────────────────────────────────────────
        btn = tk.Frame(p, bg=BG)
        btn.pack(fill=tk.X, padx=22, pady=14)

        tk.Button(
            btn,
            text=" Solve ",
            command=self._solve,
            bg=BLUE,
            fg="white",
            relief="flat",
            font=("Helvetica", FS, "bold"),
            padx=14,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT)

        tk.Button(
            btn,
            text=" Open FBD ",
            command=self._open_fbd,
            bg=GREEN,
            fg="white",
            relief="flat",
            font=("Helvetica", FS, "bold"),
            padx=14,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            btn,
            text="Reset",
            command=self._reset,
            bg="#dedcd4",
            fg="#444",
            relief="flat",
            font=("Helvetica", FS),
            padx=10,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT)

        # ── NOTE ───────────────────────────────────────────────
        tk.Label(
            p,
            text="Symbolic inputs: FBD uses approximated geometry,\nexpressions shown as labels.",
            bg=BG,
            font=("Helvetica", 8),
            fg="#aaa",
            justify="left"
        ).pack(anchor="w", padx=22, pady=(0, 20))

    # ─── RIGHT panel ──────────────────────────────────────────────────────
    def _build_right(self, p):
        # ── HEADER ─────────────────────────────────────────────
        hdr = tk.Frame(p, bg="#e8e7e0", padx=18, pady=10)
        hdr.pack(fill=tk.X)

        tk.Label(
            hdr,
            text="Results",
            bg="#e8e7e0",
            font=("Helvetica", 14, "bold"),
            fg="#333"
        ).pack(anchor="w")

        tk.Label(
            hdr,
            text="Problem 3/57  —  Strut force analysis",
            bg="#e8e7e0",
            font=("Helvetica", 9),
            fg="#888"
        ).pack(anchor="w")

        # ── RESULT BOX ─────────────────────────────────────────
        res = tk.Frame(
            p,
            bg="#eef6f1",
            highlightbackground="#99ccaa",
            highlightthickness=1,
            padx=18,
            pady=14
        )
        res.pack(fill=tk.BOTH, padx=16, pady=(14, 0))

        tk.Label(
            res,
            text="RESULT",
            bg="#eef6f1",
            font=("Helvetica", 9, "bold"),
            fg="#666"
        ).pack(anchor="w")

        res_inner = tk.Frame(res, bg="#eef6f1")
        res_inner.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        r_scroll = tk.Scrollbar(res_inner)
        r_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._result_box = tk.Text(
            res_inner,
            height=8,
            wrap="word",
            font=("Courier", FS),
            bg="#eef6f1",
            fg="#1a7a4a",
            relief="flat",
            yscrollcommand=r_scroll.set
        )

        self._result_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        r_scroll.config(command=self._result_box.yview)

        self._result_box.insert("1.0", "Click Solve to compute.")
        self._result_box.config(state="disabled")

        # ── SYMBOLIC EXPRESSIONS BOX ──────────────────────────
        sym_box = tk.Frame(
            p,
            bg="#f0eeff",
            highlightbackground="#c0aaff",
            highlightthickness=1,
            padx=18,
            pady=14
        )
        sym_box.pack(fill=tk.BOTH, padx=16, pady=(10, 0))

        tk.Label(
            sym_box,
            text="SYMBOLIC EXPRESSIONS",
            bg="#f0eeff",
            font=("Helvetica", 9, "bold"),
            fg="#555"
        ).pack(anchor="w")

        sym_inner = tk.Frame(sym_box, bg="#f0eeff")
        sym_inner.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        s_scroll = tk.Scrollbar(sym_inner)
        s_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._sym_box = tk.Text(
            sym_inner,
            height=8,
            wrap="word",
            font=("Courier", 9),
            bg="#f0eeff",
            fg="#4a2a9a",
            relief="flat",
            yscrollcommand=s_scroll.set
        )

        self._sym_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        s_scroll.config(command=self._sym_box.yview)

        self._sym_box.insert("1.0", "—")
        self._sym_box.config(state="disabled")

        # ── STATUS BOX ────────────────────────────────────────
        status_box = tk.Frame(p, bg=SUBTLE, padx=18, pady=8)
        status_box.pack(fill=tk.X, padx=16, pady=(10, 0))

        self._status_var = tk.StringVar(value="No FBD generated yet.")

        tk.Label(
            status_box,
            textvariable=self._status_var,
            bg=SUBTLE,
            font=("Helvetica", 9),
            fg="#888",
            wraplength=360,
            justify="left"
        ).pack(anchor="w")

        # ── FILLER ────────────────────────────────────────────
        tk.Frame(p, bg=SUBTLE).pack(fill=tk.BOTH, expand=True)

        # ── FOOTER ────────────────────────────────────────────
        tk.Label(
            p,
            text="Van Hatch Solver  ·  symbolic + numeric",
            bg=SUBTLE,
            font=("Helvetica", 8),
            fg="#ccc"
        ).pack(side=tk.BOTTOM, pady=8)

    # ─────────────────────────────────────────────────────────────────────
    #  SOLVE
    # ─────────────────────────────────────────────────────────────────────
    def _solve(self):
        mm = sp.Rational(1, 1000)

        try:
            # ── read inputs ─────────────────────────────────────
            sv = {k: parse_sym(self._entries[k].get()) for k in self._entries}

            AO_s     = sv["AO"] * mm
            OB_s     = sv["OB"] * mm
            AB_s     = sv["AB"] * mm
            dist_P_s = sv["dist_P"] * mm

            # ── validations ─────────────────────────────────────
            def vp(expr, name):
                f = try_float(expr)
                if f is not None:
                    if f < 0:
                        raise ValueError(f"{name} cannot be negative.")
                    if f == 0:
                        raise ValueError(f"{name} cannot be zero.")

            vp(sv["P"], "Applied force P")
            vp(sv["mass"], "Mass")
            vp(AO_s, "AO")
            vp(OB_s, "OB")
            vp(AB_s, "AB")
            vp(dist_P_s, "Distance P")

            # ── solve physics ───────────────────────────────────
            sym_res = compute(
                sv["P"], sv["mass"],
                AO_s, OB_s, AB_s,
                sv["theta"], dist_P_s
            )

            def fmt(expr):
                f = try_float(expr)
                return f"{f:.6g}" if f is not None else str(sp.simplify(expr))

            F_f     = try_float(sym_res["F"])
            alpha_f = try_float(sym_res["alpha"])
            W_f     = try_float(sym_res["W"])

            mode_str = ""
            if F_f is not None:
                mode_str = (
                    f"\nMode    : "
                    f"{'Compression' if F_f >= 0 else 'Tension'}"
                )

            # ── RESULT BOX ──────────────────────────────────────
            self._result_box.config(state="normal")
            self._result_box.delete("1.0", tk.END)

            self._result_box.insert(
                "1.0",
                f"F       = {fmt(sym_res['F'])}"
                f"{' N' if is_numeric(sym_res['F']) else ''}\n"
                f"α       = {fmt(sym_res['alpha'])} rad\n"
                f"W       = {fmt(sym_res['W'])}"
                f"{' N' if is_numeric(sym_res['W']) else ''}"
                + mode_str
            )

            self._result_box.config(state="disabled")

            # ── SYMBOLIC BOX ────────────────────────────────────
            free = set()
            for expr in sym_res.values():
                free |= expr.free_symbols

            if free:
                symbolic_text = (
                    f"Free symbols: "
                    f"{', '.join(sorted(str(s) for s in free))}\n\n"
                    f"F     = {sp.simplify(sym_res['F'])}\n\n"
                    f"alpha = {sp.simplify(sym_res['alpha'])}"
                )
            else:
                symbolic_text = "All inputs numeric — no free symbols."

            self._sym_box.config(state="normal")
            self._sym_box.delete("1.0", tk.END)
            self._sym_box.insert("1.0", symbolic_text)
            self._sym_box.config(state="disabled")

            # ── geometry summary ────────────────────────────────
            self._geo_var.set(
                f"AO = {fmt(AO_s)} m   OB = {fmt(OB_s)} m\n"
                f"AB = {fmt(AB_s)} m   θ  = {fmt(sv['theta'])}°\n"
                f"dP = {fmt(dist_P_s)} m"
            )

            # ── fallback numeric values for drawing ─────────────
            DEF = {
                "P": 40,
                "mass": 40,
                "AO": 0.55,
                "OB": 0.175,
                "AB": 0.60,
                "theta": 30,
                "dist_P": 1.125
            }

            def fv(expr, key):
                f = try_float(expr)
                return f if f is not None else DEF[key]

            P_draw     = fv(sv["P"], "P")
            mass_draw  = fv(sv["mass"], "mass")
            AO_draw    = fv(AO_s, "AO")
            OB_draw    = fv(OB_s, "OB")
            AB_draw    = fv(AB_s, "AB")
            theta_draw = fv(sv["theta"], "theta")
            dist_draw  = fv(dist_P_s, "dist_P")

            F_draw = F_f if F_f is not None else 1.0

            if alpha_f is not None:
                alpha_draw = alpha_f
            else:
                alpha_draw = math.acos(
                    (AO_draw**2 + AB_draw**2 - OB_draw**2) /
                    max(2 * AO_draw * AB_draw, 1e-9)
                )

            # ── symbolic labels ─────────────────────────────────
            sym_labels = {}

            for k in ("P", "mass", "theta"):
                if not is_numeric(sv[k]):
                    sym_labels[k] = str(sp.simplify(sv[k]))

            if F_f is None:
                sym_labels["F_result"] = f"F = {fmt(sym_res['F'])}"

            if alpha_f is None:
                sym_labels["alpha_result"] = f"α = {fmt(sym_res['alpha'])}"

            if not sym_labels:
                sym_labels = None

            # ── store for popup FBD ─────────────────────────────
            self._last_result = dict(
                P=P_draw,
                mass=mass_draw,
                AO=AO_draw,
                OB=OB_draw,
                AB=AB_draw,
                theta=theta_draw,
                dist_P=dist_draw,
                F=F_draw,
                alpha=alpha_draw,
                sym_labels=sym_labels
            )

            all_numeric = all(
                is_numeric(e)
                for e in [
                    sv["P"], sv["mass"],
                    AO_s, OB_s, AB_s,
                    sv["theta"], dist_P_s
                ]
            )

            note = " (geometry approximated)" if not all_numeric else ""

            self._status_var.set(
                f"FBD ready{note}. Click Open FBD to view."
            )

            # auto refresh popup
            if self._fbd_win and self._fbd_win.winfo_exists():
                self._render_fbd()

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

        except Exception as e:
            messagebox.showerror("Unexpected Error", str(e))

    # ─────────────────────────────────────────────────────────────────────
    #  FBD POPUP
    # ─────────────────────────────────────────────────────────────────────
    def _open_fbd(self):
        if self._last_result is None:
            self._solve()
            if self._last_result is None:
                return

        if self._fbd_win and self._fbd_win.winfo_exists():
            self._fbd_win.lift()
            self._render_fbd()
            return

        win = tk.Toplevel(self)
        win.title("FBD — Van Hatch  ·  Problem 3/57")
        win.geometry("820x680")
        win.minsize(560, 440)
        win.configure(bg="#f4f6f8")
        win.protocol("WM_DELETE_WINDOW", self._close_fbd)
        self._fbd_win = win

        # toolbar
        toolbar = tk.Frame(win, bg="#e0e0d8", padx=10, pady=6)
        toolbar.pack(fill=tk.X)
        tk.Label(toolbar, text="Free-Body Diagram", bg="#e0e0d8",
                 font=("Helvetica", 11, "bold"), fg="#333"
                 ).pack(side=tk.LEFT)
        tk.Button(toolbar, text="✕  Close",
                  command=self._close_fbd,
                  bg="#c0392b", fg="white", relief="flat",
                  font=("Helvetica", 9), padx=8, pady=3,
                  cursor="hand2", activebackground="#922b21"
                  ).pack(side=tk.RIGHT)
        tk.Button(toolbar, text="↺  Refresh",
                  command=self._render_fbd,
                  bg="#555", fg="white", relief="flat",
                  font=("Helvetica", 9), padx=8, pady=3,
                  cursor="hand2", activebackground="#333"
                  ).pack(side=tk.RIGHT, padx=(0, 6))

        fig, ax = plt.subplots(figsize=(8, 6.5))
        fig.patch.set_facecolor("#f4f6f8")
        self._fbd_fig    = fig
        self._fbd_ax     = ax
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._fbd_canvas = canvas

        self._render_fbd()

    def _render_fbd(self):
        if self._last_result is None or self._fbd_ax is None:
            return
        r = self._last_result
        draw_fbd(self._fbd_ax,
                 r["P"], r["mass"], r["AO"], r["OB"],
                 r["AB"], r["theta"], r["dist_P"],
                 r["F"], r["alpha"],
                 sym_labels=r["sym_labels"])
        self._fbd_canvas.draw()

    def _close_fbd(self):
        if self._fbd_win and self._fbd_win.winfo_exists():
            self._fbd_win.destroy()
        self._fbd_win    = None
        self._fbd_fig    = None
        self._fbd_ax     = None
        self._fbd_canvas = None

    # ─────────────────────────────────────────────────────────────────────
    #  RESET
    # ─────────────────────────────────────────────────────────────────────
    def _reset(self):
        # ── restore default inputs ─────────────────────────────
        for key, _, _, default in FIELDS:
            self._entries[key].delete(0, tk.END)
            self._entries[key].insert(0, default)

        # ── clear result box ───────────────────────────────────
        self._result_box.config(state="normal")
        self._result_box.delete("1.0", tk.END)
        self._result_box.insert("1.0", "Click Solve to compute.")
        self._result_box.config(state="disabled")

        # ── clear symbolic box ────────────────────────────────
        self._sym_box.config(state="normal")
        self._sym_box.delete("1.0", tk.END)
        self._sym_box.insert("1.0", "—")
        self._sym_box.config(state="disabled")

        # ── reset geometry / status ───────────────────────────
        self._geo_var.set("—  enter values to populate")
        self._status_var.set("No FBD generated yet.")

        # ── clear stored result and close popup ───────────────
        self._last_result = None
        self._close_fbd()

        # ── optional auto-solve defaults after reset ──────────
        self.after(150, self._solve)

    # ─────────────────────────────────────────────────────────────────────
    #  CLOSE MAIN WINDOW
    # ─────────────────────────────────────────────────────────────────────
    def _on_close(self):
        self._close_fbd()
        plt.close("all")
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
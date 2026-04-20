"""
Resultant Force & Moment Solver  (symbolic + numeric)
Supports plain numbers OR sympy-compatible expressions / symbol names.
Requires: matplotlib, sympy  →  pip install matplotlib sympy
"""

import math
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp
from sympy import abc

FS = 16
BG   = "#f0efe9"
CARD = "#ffffff"
BLUE = "#2578c8"


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS  – parse a string into a sympy expression
# ──────────────────────────────────────────────────────────────────────────────

def parse_sym(text: str):
    """
    Try to return a sympy expression from *text*.
    Plain floats become sp.Float; symbolic strings use sympy's parser.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty field.")
    try:
        return sp.Float(float(text))
    except ValueError:
        pass
    # Allow implicit multiplication and standard math names
    local = {name: sp.Symbol(name) for name in abc.__dict__
         if not name.startswith("_")}
    local.update({k: v for k, v in sp.__dict__.items()
                  if isinstance(v, (sp.core.function.UndefinedFunction,
                                    type)) and not k.startswith("_")})
    try:
        expr = sp.sympify(text, locals=local)
        return expr
    except Exception:
        raise ValueError(f"Cannot parse '{text}' as a number or expression.")


def is_numeric(expr):
    """True when the sympy expression has no free symbols."""
    return len(expr.free_symbols) == 0


def sym_cos(angle_deg):
    return sp.cos(sp.pi * angle_deg / 180)

def sym_sin(angle_deg):
    return sp.sin(sp.pi * angle_deg / 180)

def sym_sqrt(x):
    return sp.sqrt(x)


# ──────────────────────────────────────────────────────────────────────────────
#  PHYSICS  (fully symbolic)
# ──────────────────────────────────────────────────────────────────────────────

def compute(F1, a1, F2, a2, F3, a3, xB, yB, xC, yC, xE, yE):
    """
    All arguments are sympy expressions (possibly numeric).
    Returns a dict of sympy expressions, simplified.
    """
    simp = lambda e: sp.nsimplify(sp.trigsimp(sp.expand(e)), rational=False)

    Fx1 = simp(F1 * sym_cos(a1));  Fy1 = simp(F1 * sym_sin(a1))
    Fx2 = simp(F2 * sym_cos(a2));  Fy2 = simp(F2 * sym_sin(a2))
    Fx3 = simp(F3 * sym_cos(a3));  Fy3 = simp(F3 * sym_sin(a3))

    Rx = simp(Fx1 + Fx2 + Fx3)
    Ry = simp(Fy1 + Fy2 + Fy3)
    R  = simp(sym_sqrt(Rx**2 + Ry**2))

    MB = simp(xB * Fy1 - yB * Fx1)
    MC = simp(xC * Fy2 - yC * Fx2)
    ME = simp(xE * Fy3 - yE * Fx3)
    M_total = simp(MB + MC + ME)

    x_int = sp.oo if Ry.equals(0) else simp(M_total / Ry)

    return {
        "Fx1": Fx1, "Fy1": Fy1,
        "Fx2": Fx2, "Fy2": Fy2,
        "Fx3": Fx3, "Fy3": Fy3,
        "Rx": Rx, "Ry": Ry, "R": R,
        "M_total": M_total,
        "x_int": x_int,
    }


def try_float(expr):
    """Convert a sympy expr to Python float if fully numeric, else return None."""
    try:
        v = complex(expr)
        if abs(v.imag) < 1e-12:
            return float(v.real)
    except Exception:
        pass
    return None

def draw_diagram(ax, vals_sym, vals_float, res_float, sym_res):
    ax.clear()
    ax.set_facecolor("#f9f9f7")
    ax.set_aspect("equal")

    xB, yB = vals_float["xB"], vals_float["yB"]
    xC, yC = vals_float["xC"], vals_float["yC"]
    xE, yE = vals_float["xE"], vals_float["yE"]

    max_f = max(abs(vals_float["F1"]), abs(vals_float["F2"]),
                abs(vals_float["F3"]), 1e-9)
    arrow_scale = 0.8 / max_f

    Ax, Ay = 0.0, 0.0
    x_end = xE + 0.4

    ax.plot([Ax, x_end], [Ay, Ay], color="#888888", lw=2, zorder=2)
    ax.plot([Ax, xC], [Ay, yC], color="#aaaaaa", lw=1.5, linestyle="--", zorder=2)
    ax.plot([xC, xE], [yC, yE], color="#aaaaaa", lw=1.5, linestyle="--", zorder=2)

    def expr_text(expr):
        f = try_float(expr)
        if f is not None:
            return f"{f:.3g}"
        return str(sp.simplify(expr))

    def force_arrow(x, y, fx, fy, color):
        dx = fx * arrow_scale
        dy = fy * arrow_scale

        ax.annotate(
            "",
            xy=(x + dx, y + dy),
            xytext=(x, y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=2,
                mutation_scale=14
            ),
            zorder=6
        )

    # Draw arrows only (no labels near arrows)
    force_arrow(xB, yB, res_float["Fx1"], res_float["Fy1"], "#2578c8")
    force_arrow(xC, yC, res_float["Fx2"], res_float["Fy2"], "#c83025")
    force_arrow(xE, yE, res_float["Fx3"], res_float["Fy3"], "#1a9e5f")

    xi_f = res_float.get("x_int")

    if xi_f is not None and math.isfinite(xi_f):
        ax.annotate(
            "",
            xy=(xi_f + res_float["Rx"] * arrow_scale,
                res_float["Ry"] * arrow_scale),
            xytext=(xi_f, 0),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#7c3fb5",
                lw=2,
                mutation_scale=14
            ),
            zorder=6
        )

        ax.plot(xi_f, 0, "D", color="#7c3fb5", markersize=8, zorder=8)

    def dot(x, y, c, lbl, dy=0.08):
        ax.plot(x, y, "o", color=c, markersize=7, zorder=8)
        ax.text(
            x, y + dy, lbl,
            color=c,
            fontsize=FS,
            fontweight="bold",
            ha="center",
            zorder=9
        )

    dot(Ax, Ay, "#333333", "A", dy=0.10)
    dot(xB, yB, "#2578c8", "B", dy=0.10)
    dot(xC, yC, "#c83025", "C", dy=-0.18)
    dot(xE, yE, "#1a9e5f", "E", dy=-0.18)

    ax.set_xlabel("x (m)", fontsize=FS)
    ax.set_ylabel("y (m)", fontsize=FS)
    ax.tick_params(labelsize=FS)
    ax.grid(True, linestyle=":", alpha=0.4)

    ax.set_xlim(min(Ax, xB, xC, xE) - 1, max(Ax, xB, xC, xE) + 1)
    ax.set_ylim(min(Ay, yB, yC, yE) - 1, max(Ay, yB, yC, yE) + 1)

    # ── CORNER LABEL BOX ─────────────────────────────
    info = (
        f"F1    = {expr_text(vals_sym['F1'])} kN\n"
        f"F2    = {expr_text(vals_sym['F2'])} kN\n"
        f"F3    = {expr_text(vals_sym['F3'])} kN\n"
        f"Rx    = {expr_text(sym_res['Rx'])} kN\n"
        f"Ry    = {expr_text(sym_res['Ry'])} kN\n"
        f"R     = {expr_text(sym_res['R'])} kN\n"
        f"M_A   = {expr_text(sym_res['M_total'])} kN·m\n"
        f"x-int = {expr_text(sym_res['x_int'])} m"
    )

    ax.text(
        0.98, 0.98,
        info,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  fc="white",
                  ec="#999999",
                  alpha=0.95)
    )
def draw_symbolic_placeholder(ax, sym_res):
    """Show symbolic result text on the canvas when diagram can't be drawn."""
    ax.clear()
    ax.axis("off")
    lines = [
        "Symbolic result — diagram requires numeric inputs",
        "",
        f"Rx  = {sp.pretty(sym_res['Rx'])}",
        f"Ry  = {sp.pretty(sym_res['Ry'])}",
        f"R   = {sp.pretty(sym_res['R'])}",
        f"M_A = {sp.pretty(sym_res['M_total'])}",
        f"x   = {sp.pretty(sym_res['x_int'])}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines),
            transform=ax.transAxes, fontsize=10,
            va="top", ha="left", family="monospace",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.6", fc="#f4f4ee",
                      ec="#cccccc", lw=1))


# ──────────────────────────────────────────────────────────────────────────────
#  FIELDS  (key, label, unit, default)
# ──────────────────────────────────────────────────────────────────────────────

FIELDS = [
    ("F1", "Force at B",  "kN",  "10"),
    ("a1", "Angle at B",  "deg", "-90"),
    ("F2", "Force at C",  "kN",  "3.2"),
    ("a2", "Angle at C",  "deg", "60"),
    ("F3", "Force at E",  "kN",  "4.8"),
    ("a3", "Angle at E",  "deg", "-90"),
    ("xB", "xB",          "m",   "1.2"),
    ("yB", "yB",          "m",   "0"),
    ("xC", "xC",          "m",   "1.719"),
    ("yC", "yC",          "m",   "-0.3"),
    ("xE", "xE",          "m",   "3.139"),
    ("yE", "yE",          "m",   "-0.6"),
]


# ──────────────────────────────────────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Resultant Force & Moment Solver  [symbolic + numeric]")
        self.configure(bg=BG)
        self.resizable(True, True)
        self._entries = {}
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def on_close(self):
        self.destroy()
        plt.close("all")
        self.quit()

    def _build(self):
        # ── SCROLLABLE MAIN CONTAINER ─────────────────────────────
        container = tk.Frame(self, bg=BG)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.configure(yscrollcommand=scrollbar.set)

        left = tk.Frame(canvas, bg=BG, padx=20, pady=18)
        canvas.create_window((0, 0), window=left, anchor="nw")

        left.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        # ── Mouse Scroll ─────────────────────────────────────────
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # ── HEADER ───────────────────────────────────────────────
        tk.Label(
            left,
            text="Resultant Force Solver",
            bg=BG,
            font=("Helvetica", FS, "bold"),
            fg="#222"
        ).pack(anchor="w")

        tk.Label(
            left,
            text="Numeric values OR symbolic expressions (e.g. F, alpha, 2*a)",
            bg=BG,
            font=("Helvetica", FS),
            fg="#777"
        ).pack(anchor="w", pady=(0, 16))

        # ── MAIN TWO COLUMN LAYOUT ──────────────────────────────
        main_row = tk.Frame(left, bg=BG)
        main_row.pack(fill=tk.BOTH, expand=True)

        # LEFT PANEL (inputs + buttons)
        left_panel = tk.Frame(main_row, bg=BG)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # RIGHT PANEL (results)
        res_frame = tk.Frame(
            main_row,
            bg="#eef6f1",
            highlightbackground="#99ccaa",
            highlightthickness=1,
            padx=14,
            pady=12
        )
        res_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(20, 0), anchor="n")

        # ── INPUT CARD ───────────────────────────────────────────
        card = tk.Frame(
            left_panel,
            bg=CARD,
            bd=0,
            highlightbackground="#d8d8d8",
            highlightthickness=1,
            padx=18,
            pady=16
        )
        card.pack(fill=tk.X)

        force_fields = [f for f in FIELDS if f[0] in ("F1", "a1", "F2", "a2", "F3", "a3")]
        coord_fields = [f for f in FIELDS if f[0] in ("xB", "yB", "xC", "yC", "xE", "yE")]

        tk.Label(
            card,
            text="Forces",
            bg=CARD,
            font=("Helvetica", FS, "bold"),
            fg="#333"
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        row = 1

        for key, label, unit, default in force_fields:
            tk.Label(
                card,
                text=label,
                bg=CARD,
                font=("Helvetica", FS),
                fg="#444",
                width=20,
                anchor="w"
            ).grid(row=row, column=0, sticky="w", pady=5)

            entry = tk.Entry(card, font=("Helvetica", FS), width=12)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=5)
            entry.bind("<Return>", lambda e: self._solve())

            self._entries[key] = entry

            tk.Label(
                card,
                text=unit,
                bg=CARD,
                font=("Helvetica", FS),
                fg="#999"
            ).grid(row=row, column=2, sticky="w", padx=(8, 0))

            row += 1

        tk.Label(
            card,
            text="Point Coordinates",
            bg=CARD,
            font=("Helvetica", FS, "bold"),
            fg="#333"
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(14, 8))

        row += 1

        for key, label, unit, default in coord_fields:
            tk.Label(
                card,
                text=label,
                bg=CARD,
                font=("Helvetica", FS),
                fg="#444",
                width=20,
                anchor="w"
            ).grid(row=row, column=0, sticky="w", pady=5)

            entry = tk.Entry(card, font=("Helvetica", FS), width=12)
            entry.insert(0, default)
            entry.grid(row=row, column=1, pady=5)
            entry.bind("<Return>", lambda e: self._solve())

            self._entries[key] = entry

            tk.Label(
                card,
                text=unit,
                bg=CARD,
                font=("Helvetica", FS),
                fg="#999"
            ).grid(row=row, column=2, sticky="w", padx=(8, 0))

            row += 1

        # ── BUTTONS ──────────────────────────────────────────────
        btn_row = tk.Frame(left_panel, bg=BG)
        btn_row.pack(fill=tk.X, pady=14)

        tk.Button(
            btn_row,
            text="Solve & Draw",
            command=self._solve,
            bg=BLUE,
            fg="white",
            relief="flat",
            font=("Helvetica", FS, "bold"),
            padx=12,
            pady=8
        ).pack(side=tk.LEFT)

        tk.Button(
            btn_row,
            text="Reset",
            command=self._reset,
            bg="#dedcd4",
            fg="#333",
            relief="flat",
            font=("Helvetica", FS),
            padx=12,
            pady=8
        ).pack(side=tk.LEFT, padx=10)

        # ── RESULT PANEL ────────────────────────────────────────
        tk.Label(
            res_frame,
            text="RESULT",
            bg="#eef6f1",
            font=("Helvetica", FS, "bold"),
            fg="#555"
        ).pack(anchor="w")

        text_frame = tk.Frame(res_frame, bg="#eef6f1")
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        scroll = tk.Scrollbar(text_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._result_box = tk.Text(
            text_frame,
            width=34,
            height=18,
            wrap="word",
            yscrollcommand=scroll.set,
            font=("Courier", FS),
            bg="#eef6f1",
            fg="#1a7a4a",
            relief="flat"
        )

        self._result_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self._result_box.yview)

        self._result_box.insert(
            "1.0",
            "Fill in the values above and click Solve & Draw."
        )
        self._result_box.config(state="disabled")
    # ── Solve ──────────────────────────────────────────────────────────────
    def _solve(self):
        try:
            sym_vals = {}
            for k in self._entries:
                raw = self._entries[k].get().strip()
                sym_vals[k] = parse_sym(raw)

            sym_res = compute(
                sym_vals["F1"], sym_vals["a1"],
                sym_vals["F2"], sym_vals["a2"],
                sym_vals["F3"], sym_vals["a3"],
                sym_vals["xB"], sym_vals["yB"],
                sym_vals["xC"], sym_vals["yC"],
                sym_vals["xE"], sym_vals["yE"],
            )

            def fmt(expr):
                f = try_float(expr)
                if f is not None:
                    return f"{f:.6g}"
                return str(sp.simplify(expr))

            x_str = fmt(sym_res["x_int"])

            self._result_box.config(state="normal")
            self._result_box.delete("1.0", tk.END)

            self._result_box.insert(
                tk.END,
                f"Rx      = {fmt(sym_res['Rx'])}\n"
                f"Ry      = {fmt(sym_res['Ry'])}\n"
                f"R       = {fmt(sym_res['R'])}\n"
                f"M_A     = {fmt(sym_res['M_total'])}\n"
                f"x-int   = {x_str}"
            )

            self._result_box.config(state="disabled")

            # plotting values
            fv = {}
            for k in sym_vals:
                val = try_float(sym_vals[k])
                fv[k] = val if val is not None else 1.0

            fr = {}
            for k in sym_res:
                val = try_float(sym_res[k])
                fr[k] = val if val is not None else 1.0

            if fr["x_int"] is None:
                fr["x_int"] = 1.0

            # separate window for diagram
            fig, ax = plt.subplots(figsize=(10, 8))
            draw_diagram(ax, sym_vals, fv, fr, sym_res)
            fig.tight_layout()
            plt.show()

        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
        except Exception as e:
            messagebox.showerror("Unexpected Error", str(e))


    # ── Reset ──────────────────────────────────────────────────────────────
    def _reset(self):
        for key, _, _, default in FIELDS:
            self._entries[key].delete(0, tk.END)
            self._entries[key].insert(0, default)

        self._result_box.config(state="normal")
        self._result_box.delete("1.0", tk.END)
        self._result_box.insert(
            "1.0",
            "Fill in the values above and click Solve & Draw."
        )
        self._result_box.config(state="disabled")

if __name__ == "__main__":
    app = App()
    app.mainloop()
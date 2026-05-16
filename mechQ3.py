# ============================================================
# BEAM REACTION SOLVER — Problem 5/111
# POINT LOAD + DISTRIBUTED LOAD + PULLEY TENSION
# WITH DYNAMIC GUI FBD
# ============================================================
#
# GEOMETRY (from left end):
#   0 ←1m→ A ←3m→ (end of UDL, L1=4m total) ←2m→ (end of tri, L2=2m) ←2m→ B ←3m→ C ←1m→ end
#   Total beam = 12 m
#
# LOADS:
#   R1 = w * L1          (UDL resultant), acts at L1/2 from left end
#   R2 = 0.5 * w * L2   (triangular resultant), acts at L1 + L2/3 from left end
#   T  = 0.5 * M * g    (pulley MA=2, upward at B)
#
# EQUILIBRIUM about A:
#   ΣMₐ = 0: -R1*(L1/2 - A) - R2*(L1 + L2/3 - A) + T*(B - A) + Nc*(C - A) = 0
#   ΣFy  = 0: Ay - R1 - R2 + T + Nc = 0
# ============================================================

import tkinter as tk
from tkinter import messagebox
import sympy as sp
import math


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def parse_expr(val, default=None):
    val = val.strip()
    if val == "":
        if default is None:
            raise ValueError("Empty input")
        return sp.sympify(default)
    return sp.sympify(val)


# ============================================================
# MAIN APPLICATION
# ============================================================

class BeamReactionSolver:

    def __init__(self, root):
        self.root = root
        self.root.title("Beam Reaction Solver — 5/111")
        self.root.geometry("1450x850")
        self.root.configure(bg="#F4F4F4")

        title = tk.Label(
            root,
            text="Beam Reaction Solver — Problem 5/111",
            font=("Segoe UI", 22, "bold"),
            bg="#F4F4F4"
        )
        title.pack(pady=10)

        main = tk.Frame(root, bg="#F4F4F4")
        main.pack(fill="both", expand=True)

        # ====================================================
        # LEFT PANEL
        # ====================================================
        left = tk.Frame(main, bg="#F4F4F4")
        left.pack(side="left", fill="y", padx=20)

        tk.Label(
            left,
            text="INPUTS",
            font=("Segoe UI", 16, "bold"),
            bg="#F4F4F4"
        ).pack(pady=10)

        self.entries = {}

        # NOTE: L1 = 4m (UDL spans from x=0 to x=4, i.e. 4 m total)
        #       L2 = 2m (triangular from x=4 to x=6)
        fields = [
            ("UDL Intensity w (kN/m)",          "15"),
            ("UDL Length L1 (m)",               "4"),   # ← corrected default: 4, not 3
            ("Triangular Length L2 (m)",         "2"),
            ("Mass M (kg)",                      "2000"),
            ("Distance A from left end (m)",     "1"),
            ("Distance B from left end (m)",     "8"),
            ("Distance C from left end (m)",     "11"),
            ("Beam total length (m)",            "12"),
        ]

        for label, default in fields:
            frame = tk.Frame(left, bg="#F4F4F4")
            frame.pack(anchor="w", pady=6)
            tk.Label(
                frame,
                text=label,
                font=("Segoe UI", 11),
                bg="#F4F4F4"
            ).pack(anchor="w")
            entry = tk.Entry(frame, width=28, font=("Consolas", 12))
            entry.insert(0, default)
            entry.pack()
            self.entries[label] = entry

        # Buttons
        btn_frame = tk.Frame(left, bg="#F4F4F4")
        btn_frame.pack(pady=15)

        tk.Button(
            btn_frame,
            text="Solve",
            command=self.solve,
            bg="#0078D7", fg="white",
            font=("Segoe UI", 12, "bold"),
            width=14
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_output,
            bg="#666666", fg="white",
            font=("Segoe UI", 12, "bold"),
            width=14
        ).grid(row=0, column=1, padx=5)

        self.output = tk.Text(
            left, width=58, height=32,
            font=("Consolas", 11), bg="white"
        )
        self.output.pack(pady=10)

        # ====================================================
        # RIGHT PANEL
        # ====================================================
        right = tk.Frame(main, bg="#F4F4F4")
        right.pack(side="right", fill="both", expand=True)

        tk.Label(
            right,
            text="Dynamic Free Body Diagram",
            font=("Segoe UI", 16, "bold"),
            bg="#F4F4F4"
        ).pack(pady=10)

        self.canvas = tk.Canvas(right, width=850, height=720, bg="white")
        self.canvas.pack(padx=10, pady=10)

        self.draw_fbd(15, 4, 2, 2000, 1, 8, 11, 12)

    # ========================================================
    # CLEAR
    # ========================================================

    def clear_output(self):
        self.output.delete("1.0", tk.END)

    # ========================================================
    # SOLVE
    # ========================================================

    def solve(self):
        try:
            w        = parse_expr(self.entries["UDL Intensity w (kN/m)"].get(),      "15")
            L1       = parse_expr(self.entries["UDL Length L1 (m)"].get(),           "4")
            L2       = parse_expr(self.entries["Triangular Length L2 (m)"].get(),    "2")
            M        = parse_expr(self.entries["Mass M (kg)"].get(),                 "2000")
            A        = parse_expr(self.entries["Distance A from left end (m)"].get(),"1")
            B        = parse_expr(self.entries["Distance B from left end (m)"].get(),"8")
            C        = parse_expr(self.entries["Distance C from left end (m)"].get(),"11")
            beam_len = parse_expr(self.entries["Beam total length (m)"].get(),       "12")

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            # w can be negative
            # all other quantities must remain positive

            for val in [L1, L2, M, A, B, C, beam_len]:

                if val.is_number and float(sp.N(val)) <= 0:

                    raise ValueError(
                        "All geometric quantities and mass must be > 0"
                    )

            if all(v.is_number for v in [A, B, C, beam_len]):
                if not (float(A) < float(B) < float(C) < float(beam_len)):
                    raise ValueError("Must satisfy A < B < C < Beam Length")

            # ------------------------------------------------
            # RESULTANTS
            # ------------------------------------------------

            g = sp.Float(9.81)

            # UDL: uniform intensity w over length L1 (starting from x=0)
            R1 = sp.simplify(w * L1)
            # acts at midpoint of UDL from left end
            x1_from_left = L1 / 2
            # moment arm from A
            d1 = sp.simplify(x1_from_left - A)   # will be negative if UDL centroid is left of A

            # Triangular: starts at x=L1, peak at left, zero at right
            R2 = sp.simplify(sp.Rational(1, 2) * w * L2)
            # centroid of triangle is at L1 + L2/3 from left end (peak at left → centroid at 1/3 from peak)
            x2_from_left = L1 + L2 / sp.Integer(3)
            d2 = sp.simplify(x2_from_left - A)

            # Pulley tension (movable pulley → MA = 2)
            T = sp.simplify((M * g) / 2 / 1000)   # convert N → kN

            # moment arms from A
            dT  = sp.simplify(B - A)
            dNc = sp.simplify(C - A)

            # ------------------------------------------------
            # EQUILIBRIUM
            # ------------------------------------------------
            # ΣMₐ = 0  (↑ positive, CCW positive)
            # Downward loads R1, R2 create CW moments about A → negative
            # T acts upward at B → CCW → positive
            # Nc acts upward at C → CCW → positive
            #
            #   -R1*d1 - R2*d2 + T*dT + Nc*dNc = 0

            Nc = sp.simplify(
                (R1 * d1 + R2 * d2 - T * dT) / dNc
            )

            # ΣFy = 0: Ay - R1 - R2 + T + Nc = 0
            Ay = sp.simplify(R1 + R2 - T - Nc)
            
            # ------------------------------------------------
            # PHYSICAL VALIDITY CHECK
            # ------------------------------------------------

            Nc_negative = False

            if Nc.is_number:

                if float(sp.N(Nc)) < 0:

                    Nc_negative = True
                    
                    
                    
            # ------------------------------------------------
            # NEGATIVE UDL CHECK
            # ------------------------------------------------

            negative_udl = False

            if w.is_number:

                if float(sp.N(w)) < 0:

                    negative_udl = True

            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------
            self.output.delete("1.0", tk.END)

            lines = [
                "====================================\n",
                "RESULTS\n",
                "====================================\n\n",
                f"UDL Resultant:\n",
                f"  R1 = w × L1 = {sp.simplify(R1)} kN\n",
                f"  Acts at {sp.simplify(x1_from_left)} m from left end\n",
                f"  Moment arm from A = {sp.simplify(d1)} m\n\n",
                f"Triangular Resultant:\n",
                f"  R2 = ½ × w × L2 = {sp.simplify(R2)} kN\n",
                f"  Acts at {sp.simplify(x2_from_left)} m from left end\n",
                f"  Moment arm from A = {sp.simplify(d2)} m\n\n",
                f"Pulley Tension (MA = 2):\n",
                f"  T = Mg/2 = {sp.simplify(T)} kN\n",
                f"  Moment arm from A = {sp.simplify(dT)} m\n\n",
                "------------------------------------\n",
                "EQUILIBRIUM (moments about A):\n",
                f"  -R1×{sp.simplify(d1)} - R2×{sp.simplify(d2)} + T×{sp.simplify(dT)} + Nc×{sp.simplify(dNc)} = 0\n\n",
                "SUPPORT REACTIONS\n",
                "------------------------------------\n\n",
                f"  Nc = {sp.simplify(Nc)} kN\n",
                f"  Ay = {sp.simplify(Ay)} kN\n\n",
            ]
            
            # ------------------------------------------------
            # COMBINED WARNING DIALOG
            # ------------------------------------------------

            warning_msg = ""

            if negative_udl:

                warning_msg += (

                    "NEGATIVE UDL DETECTED\n"
                    "======================\n\n"

                    "The distributed load is acting upward.\n\n"

                    "This may represent:\n"
                    "• suction loading\n"
                    "• aerodynamic lift\n"
                    "• upward distributed support force\n\n"

                    "For sufficiently large negative loads,\n"
                    "the beam may lose equilibrium or\n"
                    "lift off supports.\n\n"
                )

            if Nc_negative:

                warning_msg += (

                    "INVALID ROLLER REACTION\n"
                    "========================\n\n"

                    "Nc is negative.\n\n"

                    "Since support C is a roller support,\n"
                    "it cannot provide a downward pulling force.\n\n"

                    "Therefore, the beam loses contact at C.\n\n"

                    "Actual reaction at C = 0 kN.\n\n"

                    "The assumed support condition\n"
                    "is no longer valid.\n\n"
                )

            if warning_msg != "":

                messagebox.showwarning(
                    "Physical Validity Warning",
                    warning_msg
                )

            for line in lines:
                self.output.insert(tk.END, line)

            if Ay.is_number and Nc.is_number:
                self.output.insert(tk.END, "Approximate Values:\n\n")
                self.output.insert(tk.END, f"  Ay ≈ {float(sp.N(Ay)):.4f} kN\n")
                self.output.insert(tk.END, f"  Nc ≈ {float(sp.N(Nc)):.4f} kN\n")

                # Cross-check
                self.output.insert(tk.END, "\nCross-check ΣFy:\n")
                check = float(sp.N(Ay)) - float(sp.N(R1)) - float(sp.N(R2)) + float(sp.N(T)) + float(sp.N(Nc))
                self.output.insert(tk.END, f"  Ay - R1 - R2 + T + Nc = {check:.6f} kN  (≈0 ✓)\n")

            self.draw_fbd(w, L1, L2, M, A, B, C, beam_len)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ========================================================
    # DRAW FBD
    # ========================================================

    def draw_fbd(self, w, L1, L2, M, A, B, C, beam_len):
        c = self.canvas
        c.delete("all")

        try:
            beam_val = float(sp.N(beam_len))
            A_val    = float(sp.N(A))
            B_val    = float(sp.N(B))
            C_val    = float(sp.N(C))
            L1_val   = float(sp.N(L1))
            L2_val   = float(sp.N(L2))
        except:
            beam_val = 12; A_val = 1; B_val = 8; C_val = 11; L1_val = 4; L2_val = 2

        scale   = 55
        start_x = 70
        beam_y  = 420
        beam_end = start_x + beam_val * scale

        # ---- BEAM ----
        c.create_line(start_x, beam_y, beam_end, beam_y, width=8)

        # ---- SUPPORT A (pin) ----
        Ax = start_x + A_val * scale
        c.create_polygon(Ax, beam_y, Ax-18, beam_y+35, Ax+18, beam_y+35, fill="#87CEFA")
        c.create_text(Ax, beam_y+52, text="A", font=("Segoe UI", 12, "bold"))

        # ---- SUPPORT C (roller) ----
        Cx = start_x + C_val * scale
        c.create_polygon(Cx, beam_y, Cx-18, beam_y+35, Cx+18, beam_y+35, fill="#90EE90")
        c.create_text(Cx, beam_y+52, text="C", font=("Segoe UI", 12, "bold"))

        # ---- UDL (from x=0, length L1) ----

        udl_start = start_x
        udl_end   = start_x + L1_val * scale

        # positive UDL → downward
        # negative UDL → upward

        if float(sp.N(w)) >= 0:

            load_y = beam_y - 100
            arrow_dir = tk.LAST
            label_y = beam_y - 120

        else:

            load_y = beam_y + 100
            arrow_dir = tk.LAST
            label_y = beam_y + 120

        c.create_line(
            udl_start,
            load_y,
            udl_end,
            load_y,
            width=3,
            fill="red"
        )

        for x in range(
            int(udl_start),
            int(udl_end)+1,
            25
        ):

            c.create_line(
                x,
                load_y,
                x,
                beam_y,
                arrow=arrow_dir,
                fill="red",
                width=2
            )

        c.create_text(
            (udl_start+udl_end)/2,
            label_y,
            text=f"{w} kN/m  (L1={L1_val}m)",
            fill="red",
            font=("Segoe UI", 11, "bold")
        )

        # ---- TRIANGULAR LOAD (from x=L1, length L2) ----

        tri_start = udl_end
        tri_end   = tri_start + L2_val * scale

        # positive triangular load → downward
        # negative triangular load → upward

        if float(sp.N(w)) >= 0:

            tri_top_y = beam_y - 100
            tri_arrow = tk.LAST

            c.create_line(
                tri_start,
                tri_top_y,
                tri_start,
                beam_y,
                fill="red",
                width=2
            )

            c.create_line(
                tri_start,
                tri_top_y,
                tri_end,
                beam_y,
                fill="red",
                width=2
            )

            for i in range(6):

                x = tri_start + i*(tri_end-tri_start)/5

                top_y = beam_y - 100 + i*20

                c.create_line(
                    x,
                    top_y,
                    x,
                    beam_y,
                    arrow=tri_arrow,
                    fill="red",
                    width=2
                )

        else:

            tri_bottom_y = beam_y + 100
            tri_arrow = tk.FIRST

            c.create_line(
                tri_start,
                tri_bottom_y,
                tri_start,
                beam_y,
                fill="red",
                width=2
            )

            c.create_line(
                tri_start,
                tri_bottom_y,
                tri_end,
                beam_y,
                fill="red",
                width=2
            )

            for i in range(6):

                x = tri_start + i*(tri_end-tri_start)/5

                bottom_y = beam_y + 100 - i*20

                c.create_line(
                    x,
                    bottom_y,
                    x,
                    beam_y,
                    arrow=tri_arrow,
                    fill="red",
                    width=2
                )

        # ====================================================================
        # PULLEY SYSTEM — matches Problem 5/111 image exactly:
        #
        #  Ceiling: spans from above B to far right
        #  Fixed pulley 1 (P1): attached to ceiling, directly above B
        #  Fixed pulley 2 (P2): attached to ceiling, far right
        #  Movable pulley (Pm): attached to block, hangs below P2
        #    - left rope segment from Pm goes up to P2 bottom
        #    - right rope segment from Pm is fixed to ceiling (anchor)
        #  Rope path: beam@B → up → over P1 → horizontal right → over P2
        #             → down → under Pm (movable) → up → fixed to ceiling
        #  Result: 2 rope segments support block → T = Mg/2
        # ====================================================================

        Bx = start_x + B_val * scale
        r  = 18

        # Ceiling bracket — spans from above B to right edge of canvas
        ceil_y      = 60
        ceil_left   = Bx
        ceil_right  = beam_end + 30          # slightly past beam right end
        ceil_thick  = 10
        c.create_rectangle(
            ceil_left, ceil_y - ceil_thick,
            ceil_right, ceil_y,
            fill="#555555", outline=""
        )
        # hatch marks on top to indicate wall/ceiling
        for hx in range(int(ceil_left), int(ceil_right), 20):
            c.create_line(hx, ceil_y - ceil_thick,
                          hx - 12, ceil_y - ceil_thick - 12,
                          width=2, fill="#555555")

        # ---- Fixed Pulley P1 — directly above B ----
        P1x = Bx
        P1y = ceil_y + r
        c.create_line(P1x, ceil_y, P1x, P1y - r, width=3)          # bracket to ceiling
        c.create_oval(P1x-r, P1y-r, P1x+r, P1y+r,
                      width=3, fill="#D6ECFF", outline="#336699")
        c.create_text(P1x, P1y, text="P1", font=("Consolas", 8, "bold"), fill="#336699")

        # ---- Fixed Pulley P2 — far right, above block ----
        P2x = ceil_right - 30
        P2y = ceil_y + r
        c.create_line(P2x, ceil_y, P2x, P2y - r, width=3)
        c.create_oval(P2x-r, P2y-r, P2x+r, P2y+r,
                      width=3, fill="#D6ECFF", outline="#336699")
        c.create_text(P2x, P2y, text="P2", font=("Consolas", 8, "bold"), fill="#336699")

        # ---- Movable Pulley Pm — hangs below P2 ----
        Pmx = P2x
        Pmy = P2y + 130
        c.create_oval(Pmx-r, Pmy-r, Pmx+r, Pmy+r,
                      width=3, fill="#FFE5B4", outline="#CC6600")
        c.create_text(Pmx, Pmy, text="Pm", font=("Consolas", 8, "bold"), fill="#CC6600")

        # ---- Block hanging from movable pulley ----
        block_top = Pmy + r + 8
        c.create_line(Pmx, Pmy + r, Pmx, block_top, width=4)
        c.create_rectangle(
            Pmx - 30, block_top,
            Pmx + 30, block_top + 65,
            fill="#E8A07C", width=2
        )
        c.create_text(
            Pmx,
            block_top + 33,
            text="M",
            font=("Segoe UI", 11, "bold")
        )

        # ---- Rope anchor to ceiling (right of P2) ----
        anchor_x = P2x + 25
        c.create_line(anchor_x, ceil_y, anchor_x, Pmy + r,
                      width=3, fill="#333333")             # fixed end: ceiling → Pm right
        # small anchor marker
        c.create_oval(anchor_x-4, ceil_y-4, anchor_x+4, ceil_y+4,
                      fill="black")

        # ---- Rope: beam@B → up → P1 → horizontal → P2 → down → Pm left ----
        # segment 1: beam attachment point up to P1 bottom
        c.create_line(Bx, beam_y, Bx, P1y + r, width=3, fill="#333333")
        # arc under P1 (rope wraps under)
        c.create_arc(P1x-r, P1y-r, P1x+r, P1y+r,
                     start=0, extent=180, style=tk.ARC, width=3)
        # segment 2: horizontal P1 top → P2 top
        c.create_line(P1x, P1y - r, P2x, P2y - r, width=3, fill="#333333")
        # arc under P2 (rope wraps under)
        c.create_arc(P2x-r, P2y-r, P2x+r, P2y+r,
                     start=0, extent=180, style=tk.ARC, width=3)
        # segment 3: P2 bottom → Pm top-left
        c.create_line(P2x, P2y + r, Pmx - r, Pmy, width=3, fill="#333333")
        # arc under Pm (rope wraps under movable pulley)
        c.create_arc(Pmx-r, Pmy-r, Pmx+r, Pmy+r,
                     start=180, extent=180, style=tk.ARC, width=3)
        # segment 4: Pm top-right → ceiling anchor
        c.create_line(Pmx + r, Pmy, anchor_x, ceil_y, width=3, fill="#333333")

        # ---- TENSION ARROW at B (upward, dashed offset to distinguish from rope) ----
        c.create_line(Bx, beam_y, Bx, beam_y-110,
                      arrow=tk.LAST, width=5, fill="purple", dash=(6, 3))
        c.create_text(Bx-50, beam_y-60,
                      text="T", fill="purple", font=("Segoe UI", 10, "bold"))

        # ---- REACTION ARROWS ----
        c.create_line(Ax, beam_y+80, Ax, beam_y, arrow=tk.LAST, width=4, fill="green")
        c.create_text(Ax-25, beam_y+95, text="Ay", fill="green", font=("Segoe UI", 11, "bold"))

        c.create_line(Cx, beam_y+80, Cx, beam_y, arrow=tk.LAST, width=4, fill="green")
        c.create_text(Cx-25, beam_y+95, text="Nc", fill="green", font=("Segoe UI", 11, "bold"))

        # ---- DISTANCE LABELS ----
        for i in range(int(beam_val)+1):
            x = start_x + i*scale
            c.create_line(x, beam_y+5, x, beam_y+20)
            c.create_text(x, beam_y+35, text=str(i), font=("Consolas", 9))

        # ---- INFO PANEL ----
        c.create_rectangle(540, 520, 820, 690, fill="#F5F5F5", width=2)
        c.create_text(680, 545,  text="INPUT PARAMETERS",        font=("Segoe UI", 13, "bold"))
        c.create_text(680, 575,  text=f"w  = {w} kN/m",          font=("Consolas", 11))
        c.create_text(680, 600,  text=f"L1 = {L1_val} m (UDL)",  font=("Consolas", 11))
        c.create_text(680, 625,  text=f"L2 = {L2_val} m (tri)",  font=("Consolas", 11))
        c.create_text(680, 650,  text=f"M  = {M} kg",            font=("Consolas", 11))
        c.create_text(680, 675,  text="T = Mg/2  (MA=2 pulley)", font=("Consolas", 10), fill="#555555")


# ============================================================
# RUN
# ============================================================

root = tk.Tk()
app  = BeamReactionSolver(root)
root.mainloop()
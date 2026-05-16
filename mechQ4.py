# ============================================================
# PULLEY + FRICTION EQUILIBRIUM SOLVER
# WITH DYNAMIC FREE BODY DIAGRAM
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


def is_numeric(expr):

    try:
        return sp.sympify(expr).is_number
    except:
        return False


# ============================================================
# MAIN APPLICATION
# ============================================================

class PulleySolver:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Pulley Friction Equilibrium Solver"
        )

        self.root.geometry("1250x760")

        self.root.configure(bg="#f4f4f4")

        # ====================================================
        # TITLE
        # ====================================================

        title = tk.Label(
            root,
            text="Pulley + Friction Equilibrium Solver",
            font=("Segoe UI", 20, "bold"),
            bg="#f4f4f4"
        )

        title.pack(pady=10)

        # ====================================================
        # MAIN FRAME
        # ====================================================

        main = tk.Frame(root, bg="#f4f4f4")
        main.pack(fill="both", expand=True)

        # ====================================================
        # LEFT PANEL
        # ====================================================

        left = tk.Frame(main, bg="#f4f4f4")
        left.pack(side="left", fill="y", padx=20)

        tk.Label(
            left,
            text="Inputs",
            font=("Segoe UI", 16, "bold"),
            bg="#f4f4f4"
        ).pack(pady=10)

        self.entries = {}
        fields = [
            ("Mass of block A (kg)", "50"),
            ("Incline angle θ (deg)", "25"),
            ("Coefficient of friction μs", "0.30")
        ]

        for label, default in fields:

            frame = tk.Frame(left, bg="#f4f4f4")
            frame.pack(anchor="w", pady=8)

            tk.Label(
                frame,
                text=label,
                font=("Segoe UI", 11),
                bg="#f4f4f4"
            ).pack(anchor="w")

            entry = tk.Entry(
                frame,
                width=25,
                font=("Consolas", 12)
            )

            entry.insert(0, default)

            entry.pack()

            self.entries[label] = entry

        # ====================================================
        # BUTTONS
        # ====================================================

        btn_frame = tk.Frame(left, bg="#f4f4f4")
        btn_frame.pack(pady=15)

        solve_btn = tk.Button(
            btn_frame,
            text="Solve",
            command=self.solve,
            bg="#0078D7",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            width=14
        )

        solve_btn.grid(row=0, column=0, padx=5)

        clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_output,
            bg="#666666",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            width=14
        )

        clear_btn.grid(row=0, column=1, padx=5)

        # ====================================================
        # OUTPUT
        # ====================================================

        self.output = tk.Text(
            left,
            width=50,
            height=24,
            font=("Consolas", 11),
            bg="white"
        )

        self.output.pack(pady=10)

        # ====================================================
        # RIGHT PANEL
        # ====================================================

        right = tk.Frame(main, bg="#f4f4f4")
        right.pack(side="right", fill="both", expand=True)

        tk.Label(
            right,
            text="Dynamic Free Body Diagram",
            font=("Segoe UI", 16, "bold"),
            bg="#f4f4f4"
        ).pack(pady=10)

        self.canvas = tk.Canvas(
            right,
            width=760,
            height=650,
            bg="white"
        )

        self.canvas.pack(padx=10, pady=10)

        self.draw_fbd(50, 25, 0.30)

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

            # ------------------------------------------------
            # INPUTS
            # ------------------------------------------------

            mA = parse_expr(
                self.entries["Mass of block A (kg)"].get(),
                "50"
            )

            theta = parse_expr(
                self.entries["Incline angle θ (deg)"].get(),
                "25"
            )
            
            # Convert negative angles to positive

            theta = sp.Abs(theta)       

            mu = parse_expr(
                self.entries["Coefficient of friction μs"].get(),
                "0.30"
            )

            # Standard gravity

            g = sp.Float(9.81)
            
            
            # ------------------------------------------------
            # LOGICAL CHECKS
            # ------------------------------------------------
            # ------------------------------------------------
            # MASS CHECK
            # ------------------------------------------------

            if mA.is_number:

                mA_val = float(sp.N(mA))

                if mA_val <= 0:

                    raise ValueError(
                        "Mass of block A must be greater than 0."
                    )
            # ------------------------------------------------
            # COEFFICIENT OF FRICTION CHECK
            # ------------------------------------------------

            if mu.is_number:

                mu_val = float(sp.N(mu))

                if mu_val < 0 or mu_val > 1:

                    raise ValueError(
                        "Coefficient of friction must satisfy 0 ≤ μs ≤ 1"
                    )

            # ------------------------------------------------
            # ANGLE CHECK
            # ------------------------------------------------

            if theta.is_number:

                angle_val = float(sp.N(theta))

                if angle_val > 90:

                    raise ValueError(
                        "Angle must satisfy 0 ≤ θ ≤ 90°"
                    )

            # ------------------------------------------------
            # CALCULATIONS
            # ------------------------------------------------

            theta_rad = sp.pi * theta / 180

            # =====================================================
            # SPECIAL CASE : θ = 90°
            # =====================================================

            theta_is_90 = False

            if theta.is_number:

                if abs(float(sp.N(theta)) - 90) < 1e-10:
                    theta_is_90 = True

            # =====================================================
            # NORMAL REACTION
            # =====================================================

            if theta_is_90:

                N = sp.Integer(0)

            else:

                N = sp.simplify(
                    mA * g * sp.cos(theta_rad)
                )

            # =====================================================
            # FRICTION FORCE
            # =====================================================

            if theta_is_90:

                F = sp.Integer(0)

            else:

                F = sp.simplify(mu * N)

            # =====================================================
            # TENSIONS
            # =====================================================

            T1 = sp.simplify(
                (
                    mA * g * sp.sin(theta_rad)
                    - F
                ) / 2
            )

            T2 = sp.simplify(
                (
                    mA * g * sp.sin(theta_rad)
                    + F
                ) / 2
            )

            # =====================================================
            # CORRESPONDING HANGING MASSES
            # =====================================================

            mB_min = sp.simplify(3 * T1 / g)

            mB_max = sp.simplify(3 * T2 / g)

            # =====================================================
            # PHYSICAL CONSTRAINTS
            # =====================================================

            if mB_min.is_number:

                if float(sp.N(mB_min)) < 0:
                    mB_min = sp.Integer(0)

            else:

                mB_min = sp.Max(0, mB_min)

            if mB_max.is_number:

                if float(sp.N(mB_max)) < 0:
                    mB_max = sp.Integer(0)

            else:

                mB_max = sp.Max(0, mB_max)

            # =====================================================
            # ENSURE LOWER ≤ UPPER
            # =====================================================

            if (
                mB_min.is_number
                and mB_max.is_number
            ):

                if float(sp.N(mB_min)) > float(sp.N(mB_max)):

                    mB_min, mB_max = mB_max, mB_min
            # ------------------------------------------------
            # OUTPUT
            # ------------------------------------------------

            self.output.delete("1.0", tk.END)

            self.output.insert(
                tk.END,
                "====================================\n"
            )

            self.output.insert(
                tk.END,
                "RESULTS\n"
            )

            self.output.insert(
                tk.END,
                "====================================\n\n"
            )

            self.output.insert(
                tk.END,
                f"Normal Reaction:\n"
            )

            self.output.insert(
                tk.END,
                f"N = {sp.simplify(N)} N\n\n"
            )

            self.output.insert(
                tk.END,
                f"Maximum Static Friction:\n"
            )

            self.output.insert(
                tk.END,
                f"F = {sp.simplify(F)} N\n\n"
            )

            self.output.insert(
                tk.END,
                "CASE 1 : Impending DOWN incline\n"
            )

            self.output.insert(
                tk.END,
                f"T = {sp.simplify(T1)} N\n"
            )

            self.output.insert(
                tk.END,
                f"mB_min = {sp.simplify(mB_min)} kg\n\n"
            )

            self.output.insert(
                tk.END,
                "CASE 2 : Impending UP incline\n"
            )

            self.output.insert(
                tk.END,
                f"T = {sp.simplify(T2)} N\n"
            )

            self.output.insert(
                tk.END,
                f"mB_max = {sp.simplify(mB_max)} kg\n\n"
            )

            self.output.insert(
                tk.END,
                "------------------------------------\n"
            )

            self.output.insert(
                tk.END,
                "EQUILIBRIUM RANGE\n"
            )

            self.output.insert(
                tk.END,
                "------------------------------------\n\n"
            )

            self.output.insert(
                tk.END,
                f"{sp.simplify(mB_min)} ≤ mB ≤ {sp.simplify(mB_max)}\n"
            )

            # Numeric display

            if mB_min.is_number and mB_max.is_number:

                self.output.insert(
                    tk.END,
                    "\nApproximate Values:\n"
                )

                self.output.insert(
                    tk.END,
                    f"{float(sp.N(mB_min)):.4f}"
                    f" ≤ mB ≤ "
                    f"{float(sp.N(mB_max)):.4f} kg\n"
                )

            # ------------------------------------------------
            # UPDATE FBD
            # ------------------------------------------------

            self.draw_fbd(mA, theta, mu)

        except Exception as e:

            messagebox.showerror(
                "Error",
                str(e)
            )

    def draw_fbd(self, mA, theta, mu):

        c = self.canvas

        c.delete("all")

        # ----------------------------------------------------
        # NUMERIC ANGLE FOR DRAWING
        # ----------------------------------------------------

        try:
            angle = float(sp.N(theta))
        except:
            angle = 25

        rad = math.radians(angle)

        # ----------------------------------------------------
        # INCLINE
        # ----------------------------------------------------

        base_x = 120
        base_y = 540

        length = 450

        end_x = (
            base_x
            + length * math.cos(rad)
        )

        end_y = (
            base_y
            - length * math.sin(rad)
        )

        c.create_line(
            base_x,
            base_y,
            end_x,
            end_y,
            width=5
        )

        c.create_line(
            base_x,
            base_y,
            end_x,
            base_y,
            width=3
        )

        # ----------------------------------------------------
        # BLOCK A
        # ----------------------------------------------------

        block_dist = 170

        bx = (
            base_x
            + block_dist * math.cos(rad)
        )

        by = (
            base_y
            - block_dist * math.sin(rad)
        )

        size = 60

        c.create_rectangle(
            bx - size/2,
            by - size/2,
            bx + size/2,
            by + size/2,
            fill="#4DA6FF",
            width=2
        )

        c.create_text(
            bx,
            by,
            text="A",
            font=("Segoe UI", 18, "bold")
        )

        # ----------------------------------------------------
        # WEIGHT
        # ----------------------------------------------------

        c.create_line(
            bx,
            by,
            bx,
            by + 120,
            arrow=tk.LAST,
            width=3,
            fill="red"
        )

        c.create_text(
            bx + 40,
            by + 70,
            text="mAg",
            fill="red",
            font=("Segoe UI", 11, "bold")
        )

        # ----------------------------------------------------
        # NORMAL
        # ----------------------------------------------------

        nx = bx - 90 * math.sin(rad)
        ny = by - 90 * math.cos(rad)

        c.create_line(
            bx,
            by,
            nx,
            ny,
            arrow=tk.LAST,
            width=3,
            fill="green"
        )

        c.create_text(
            nx - 15,
            ny - 15,
            text="N",
            fill="green",
            font=("Segoe UI", 11, "bold")
        )

        # ----------------------------------------------------
        # FRICTION
        # ----------------------------------------------------

        fx = bx - 110 * math.cos(rad)
        fy = by + 110 * math.sin(rad)

        c.create_line(
            bx,
            by,
            fx,
            fy,
            arrow=tk.LAST,
            width=3,
            fill="orange"
        )

        c.create_text(
            fx - 15,
            fy + 20,
            text="F",
            fill="orange",
            font=("Segoe UI", 11, "bold")
        )

        # ====================================================
        # PULLEY SYSTEM
        # ====================================================

        rope_w = 3

        # ----------------------------------------------------
        # SMALL MOVABLE PULLEY CONNECTED TO BLOCK A
        # ----------------------------------------------------

        p1x = bx + 62 * math.cos(rad)
        p1y = by - 62 * math.sin(rad)

        r1 = 15

        c.create_line(
            bx + 25*math.cos(rad),
            by - 25*math.sin(rad),
            p1x,
            p1y,
            width=5
        )

        c.create_oval(
            p1x-r1, p1y-r1,
            p1x+r1, p1y+r1,
            fill="#D6ECFF",
            width=3
        )

        # ----------------------------------------------------
        # FIXED PULLEY ATTACHED TO INCLINE
        # ----------------------------------------------------

        p2x = end_x - 40
        p2y = end_y + 20

        r2 = 24

        # support stand
        c.create_rectangle(
            p2x-12,
            p2y+20,
            p2x+12,
            p2y+65,
            fill="#BFBFBF",
            outline=""
        )

        # platform
        c.create_rectangle(
            p2x-70,
            p2y+65,
            p2x+70,
            p2y+80,
            fill="#CFCFCF",
            outline=""
        )

        c.create_oval(
            p2x-r2, p2y-r2,
            p2x+r2, p2y+r2,
            fill="#D6ECFF",
            width=3
        )

        # ----------------------------------------------------
        # RIGHT FIXED DOUBLE PULLEY
        # ----------------------------------------------------

        p3x = 650
        p3y = 120

        c.create_rectangle(
            p3x-18,
            p3y-75,
            p3x+18,
            p3y+70,
            fill="#D3D3D3",
            width=2
        )

        # ceiling
        c.create_rectangle(
            p3x-55,
            p3y-95,
            p3x+55,
            p3y-75,
            fill="#D9CEC7",
            outline=""
        )

        # upper pulley
        c.create_oval(
            p3x-r2,
            p3y-r2,
            p3x+r2,
            p3y+r2,
            fill="#D6ECFF",
            width=3
        )

        # lower pulley
        c.create_oval(
            p3x-r2,
            p3y+50-r2,
            p3x+r2,
            p3y+50+r2,
            fill="#D6ECFF",
            width=3
        )

        # ----------------------------------------------------
        # LOWER MOVABLE PULLEY CONNECTED TO B
        # ----------------------------------------------------

        p4x = 650
        p4y = 350

        c.create_oval(
            p4x-r2,
            p4y-r2,
            p4x+r2,
            p4y+r2,
            fill="#D6ECFF",
            width=3
        )

        # connector to block
        c.create_line(
            p4x,
            p4y+r2,
            p4x,
            p4y+35,
            width=5
        )

        # block B
        c.create_rectangle(
            p4x-38,
            p4y+35,
            p4x+38,
            p4y+125,
            fill="#E8A07C",
            width=2
        )

        c.create_text(
            p4x,
            p4y+80,
            text="B",
            font=("Segoe UI", 18, "bold")
        )

        # ====================================================
        # ROPE PATH
        # ====================================================

        # two rope segments from movable pulley at A
        c.create_line(
            p1x+8,
            p1y-10,
            p2x-18,
            p2y-18,
            width=rope_w
        )

        c.create_line(
            p1x+8,
            p1y+10,
            p2x-18,
            p2y+18,
            width=rope_w
        )

        # horizontal rope to top-right pulley
        c.create_line(
            p2x+r2,
            p2y,
            p3x-r2,
            p3y,
            width=rope_w
        )

        # FIRST vertical segment
        c.create_line(
            p3x,
            p3y+74,
            p4x-r2,
            p4y,
            width=rope_w
        )

        # rope under movable pulley
        c.create_arc(
            p4x-r2,
            p4y-r2,
            p4x+r2,
            p4y+r2,
            start=180,
            extent=180,
            style=tk.ARC,
            width=rope_w
        )

        # SECOND vertical segment
        c.create_line(
            p4x+r2,
            p4y,
            p3x+r2,
            p3y+50,
            width=rope_w
        )

        # THIRD supporting rope segment
        c.create_line(
            p3x,
            p3y+50+r2,
            p4x,
            p4y-r2,
            width=rope_w
        )

        # ====================================================
        # TENSION FORCE ON BLOCK A
        # ====================================================

        tx = bx + 130 * math.cos(rad)
        ty = by - 130 * math.sin(rad)

        c.create_line(
            bx,
            by,
            tx,
            ty,
            arrow=tk.LAST,
            width=4,
            fill="purple"
        )

        c.create_text(
            tx + 18,
            ty - 10,
            text="2T",
            fill="purple",
            font=("Segoe UI", 11, "bold")
        )

        # ----------------------------------------------------
        # ANGLE MARKER
        # ----------------------------------------------------

        c.create_arc(
            base_x - 10,
            base_y - 60,
            base_x + 80,
            base_y + 30,
            start=0,
            extent=-angle,
            style=tk.ARC,
            width=2
        )

        c.create_text(
            base_x + 70,
            base_y - 20,
            text=f"θ = {theta}",
            font=("Segoe UI", 11, "bold")
        )

        # ----------------------------------------------------
        # INFO PANEL
        # ----------------------------------------------------

        c.create_rectangle(
            500,
            500,
            740,
            620,
            fill="#F5F5F5",
            width=2
        )

        c.create_text(
            620,
            525,
            text="INPUT PARAMETERS",
            font=("Segoe UI", 13, "bold")
        )

        c.create_text(
            620,
            555,
            text=f"mA = {mA} kg",
            font=("Consolas", 12)
        )

        c.create_text(
            620,
            580,
            text=f"θ = {theta}",
            font=("Consolas", 12)
        )

        c.create_text(
            620,
            605,
            text=f"μs = {mu}",
            font=("Consolas", 12)
        )

# ============================================================
# RUN APPLICATION
# ============================================================

root = tk.Tk()

app = PulleySolver(root)

root.mainloop()
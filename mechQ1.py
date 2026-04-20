import sympy as sp

try:
    # -------- INPUT HANDLING --------
    def parse_input(val, default):
        if val.strip() == "":
            return sp.sympify(default)
        return sp.sympify(val)

    print("---- FORCE INPUT (Press Enter for default) ----")

    F1 = parse_input(input("Force at B [default = 10]: "), 10)
    angle1 = parse_input(input("Angle at B (deg) [default = -90]: "), -90)

    F2 = parse_input(input("\nForce at C [default = 3.2]: "), 3.2)
    angle2 = parse_input(input("Angle at C (deg) [default = 60]: "), 60)

    F3 = parse_input(input("\nForce at E [default = 4.8]: "), 4.8)
    angle3 = parse_input(input("Angle at E (deg) [default = -90]: "), -90)

    # Convert angles to radians
    angle1 = angle1 * sp.pi / 180
    angle2 = angle2 * sp.pi / 180
    angle3 = angle3 * sp.pi / 180

    print("\n---- POINT COORDINATES (Press Enter for default) ----")

    xB = parse_input(input("xB [default = 1.2]: "), 1.2)
    yB = parse_input(input("yB [default = 0]: "), 0)

    xC = parse_input(input("\nxC [default = 1.719]: "), 1.719)
    yC = parse_input(input("yC [default = -0.3]: "), -0.3)

    xE = parse_input(input("\nxE [default = 3.139]: "), 3.139)
    yE = parse_input(input("yE [default = -0.6]: "), -0.6)

    # -------- FORCE COMPONENTS --------
    Fx1 = F1 * sp.cos(angle1)
    Fy1 = F1 * sp.sin(angle1)

    Fx2 = F2 * sp.cos(angle2)
    Fy2 = F2 * sp.sin(angle2)

    Fx3 = F3 * sp.cos(angle3)
    Fy3 = F3 * sp.sin(angle3)

    # -------- RESULTANT FORCE --------
    Rx = Fx1 + Fx2 + Fx3
    Ry = Fy1 + Fy2 + Fy3
    R = sp.sqrt(Rx**2 + Ry**2)

    # -------- MOMENTS ABOUT A (origin) --------
    MB = xB * Fy1 - yB * Fx1
    MC = xC * Fy2 - yC * Fx2
    ME = xE * Fy3 - yE * Fx3

    M_total = MB + MC + ME

    # -------- X-INTERCEPT --------
    x_intercept = sp.simplify(M_total / Ry)

    # -------- OUTPUT FORMAT --------
    def format_output(expr):
        if expr.free_symbols:   # symbolic present
            return sp.simplify(expr)
        else:                   # numeric → decimal
            return round(float(expr.evalf()), 4)

    # -------- RESULTS --------
    print("\n----- RESULTS -----")
    print("Rx =", format_output(Rx), "kN")
    print("Ry =", format_output(Ry), "kN")
    print("Resultant Force =", format_output(R), "kN")
    print("Moment about A =", format_output(M_total), "kN·m")
    print("x-intercept =", format_output(x_intercept), "m")

except Exception as e:
    print("Error:", e)
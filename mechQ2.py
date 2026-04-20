import sympy as sp

try:
    # -------- HELPERS --------
    def parse_input(val, default):
        if val.strip() == "":
            return sp.sympify(default)
        return sp.sympify(val)

    def is_numeric(x):
        return not bool(x.free_symbols)

    def get_input(prompt, default, condition=None, error_msg="Invalid input"):
        while True:
            try:
                val = parse_input(input(prompt), default)

                if condition and is_numeric(val) and not condition(val):
                    raise ValueError(error_msg)

                return val

            except Exception:
                print("⚠️", error_msg, "- Please try again.\n")

    print("---- INPUT (Press Enter for default) ----")

    # -------- INPUTS WITH VALIDATION --------
    P_val = get_input(
        "Applied force P (N) [default = 40]: ",
        40,
        lambda x: x > 0,
        "Applied force must be > 0"
    )

    mass = get_input(
        "Mass of door (kg) [default = 40]: ",
        40,
        lambda x: x > 0,
        "Mass must be > 0"
    )

    AO = get_input(
        "Distance AO (m) [default = 0.550]: ",
        0.550,
        lambda x: x > 0,
        "AO must be > 0"
    )

    OB = get_input(
        "Distance OB (m) [default = 0.175]: ",
        0.175,
        lambda x: x > 0,
        "OB must be > 0"
    )

    AB = get_input(
        "Distance AB (m) [default = 0.600]: ",
        0.600,
        lambda x: x > 0,
        "AB must be > 0"
    )

    theta = get_input(
        "Angle at B from vertical (deg) [default = 30]: ",
        30
    )

    dist_P = get_input(
        "Distance of P from hinge O (m) [default = 1.125]: ",
        1.125,
        lambda x: x > 0,
        "Distance must be > 0"
    )

    offset = get_input(
        "Offset of CG (m) [default = 0.0375]: ",
        0.0375,
        lambda x: x >= 0,
        "Offset cannot be negative"
    )

    g = 9.81

    # -------- LAW OF COSINES --------
    cos_a = (AO**2 + AB**2 - OB**2) / (2 * AO * AB)

    if is_numeric(cos_a) and (cos_a < -1 or cos_a > 1):
        raise ValueError("Invalid triangle: AO, AB, OB do not form a triangle")

    a = sp.acos(cos_a)

    # Convert theta
    theta = theta * sp.pi / 180

    # -------- FORCE CALCULATION --------
    numerator = P_val * dist_P + mass * g * AO * sp.cos(theta - a)
    denominator = 2 * AO * sp.sin(a)

    if is_numeric(denominator) and denominator == 0:
        raise ZeroDivisionError("Invalid geometry: sin(a) = 0")

    F = numerator / denominator

    # -------- OUTPUT --------
    def format_output(expr):
        if expr.free_symbols:
            return sp.simplify(expr)
        else:
            return round(float(expr.evalf()), 2)

    print("\n----- RESULT -----")
    print("Force F (in each piston) =", format_output(F)+1, "N")

except ValueError as e:
    print("Input Error:", e)

except ZeroDivisionError as e:
    print("Math Error:", e)

except Exception as e:
    print("Unexpected Error:", e)
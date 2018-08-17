
import sympy as sp

def solve_tmstp():
    tau_h = sp.Symbol('tau_h')
    tau_u = sp.Symbol('tau_u')
    tau_x = sp.Symbol('tau_x')
    tau_f = sp.Symbol('tau_f')
    tau_r = sp.Symbol('tau_r')
    r = sp.Symbol('r')
    big_u = sp.Symbol('big_u')
    big_j = sp.Symbol('big_j')
    t = sp.Symbol('t')

    u = sp.Function('u')
    x = sp.Function('x')
    h = sp.Function('h')

    C1 = sp.Symbol('C1')
    u_eq = -tau_u*sp.Derivative(u(t), t) + (big_u - u(t))/tau_f + big_u*(1 - u(t))*r
    # x_eq = -tau_x*sp.Derivative(x(t), t) + (1 - x(t))/tau_r - u(t)*x(t)*r
    
    u_eqsol = sp.dsolve(u_eq, u(t))
    print u_eqsol

    x_eq = -tau_x*sp.Derivative(x(t), t) + (1 - x(t))/tau_r - ((big_u*r*tau_f + big_u + sp.exp(-(C1*big_u*r*tau_f + C1 + big_u*r*t + t/tau_f)/tau_u))/(big_u*r*tau_f + 1))*x(t)*r
    x_sol = sp.dsolve(x_eq, x(t))
    return x_sol

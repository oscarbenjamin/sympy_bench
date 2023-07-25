#!/usr/bin/env python
"""
Test examples of things that are slow in SymPy.

List of reasons that things are slow:

1.  Many Matrix methods should use DomainMatrix methods.
2.  DomainMatrix could use python_flint for some of its operations.
3.  Symbolic elimination can be extremely slow and produce very complicated
    output.
4.  Rational function fields like QQ(x) are inefficient (due to gcd)
5.  Polynomial gcd is slow particularly for multivariate polynomials when the
    domain is not ZZ or QQ.
6.  Pivoting could be used to improve performance of row operations (e.g. by
    pivoting to select small elements).
7.  Symbols that only appear to negative powers could be used as polynomial
    generators e.g. using the domain QQ[1/x].
8.  Laurent polynomials could be used instead of rational functions e.g.
    QQ[x,1/x] rather than QQ(x).
9.  Fraction free methods for matrix inverse
    (https://github.com/sympy/sympy/issues/21834)
10. The EX domain could be used less by improving domain detection.
11. The Matrix.jordan_form method should be changed to use the same techniques
    as Matrix.eigenvects().
12. Poly functions like cancel use Expr.expand when they could convert to Poly
    and have the Poly representation perform the expansion.
13. The poly function is more efficient than Poly for converting from Expr but
    is rarely used.
14. Solving polynomial systems of equations should factorise the equations
    first.
15. When solving a polynomial system any linear equations should be solved
    first to reduce the size of the polynomial system.
16. The solve function does a lot of unnecessary checking. It should not
    naively substitute solutions into the equation and should instead note what
    things need to be checked. This is often a cause of solve being slow.
17. Factorisation over algebraic number fields is slow. Better algorithms are
    possible and there is an old PR for this.
18. In exprtools `elif factors in (None, S.One)` tries to sympify dicts which
    is slow.
19. dup_degree of a zero polynomial returns -oo. Comparing this with an int is
    slow.
20. A sin/cos domain should be implemented. This could be based on ZZ_I[z,1/z]
    if Laurent polynomial rings were added.
21. There should be methods for solving linear systems that do not try to hard
    to detect division by zero.
22. The roots function should not expand root expressions. The expansion can be
    slow and it also makes more complicated expressions.
32. Often trying to find all complex solutions to an equation is slow but
    finding only real solutions would be faster. This is particularly true for
    polynomials or equations that reduce to polynomials.
33. There should be a default limit on what primitive_element and minpoly
    attempt to do https://github.com/sympy/sympy/pull/21479
34. Using solve with rational=True should not attempt to convert float
    exponents into rationals.
"""
from functools import wraps

inf = float('inf')


def example_decorator():
    """
    Make a decorator to collect named examples of a kind of object.
    """
    examples_dict = {}
    def decorator(func):
        examples_dict[func.__name__] = func
        return func
    return decorator, examples_dict


problems = {}


def fast_time_with(func):
    """
    Mark an example to be timed that completes reasonably quickly
    (e.g. < 1 second).
    """
    time_seconds = 1
    def decorator(example_func):
        name = example_func.__name__ + '_' + func.__name__
        problems[name] = (example_func, func, time_seconds)
        return example_func
    return decorator


def slow_time_with(func, *, time_seconds):
    """
    Mark an example to be timed that is slow and takes around time_seconds to
    complete.
    """
    def decorator(example_func):
        name = example_func.__name__ + '_' + func.__name__
        problems[name] = (example_func, func, time_seconds)
        return example_func
    return decorator


def hangs_time_with(func):
    """
    Mark an example that just hangs.
    """
    def decorator(example_func):
        name = example_func.__name__ + '_' + func.__name__
        problems[name] = (example_func, func, inf)
        return example_func
    return decorator


function_example, function_examples = example_decorator()
expression_example, expression_examples = example_decorator()
subs_example, subs_examples = example_decorator()
matrix_example, matrix_examples = example_decorator()
integral_example, integral_examples = example_decorator()
system_example, system_examples = example_decorator()
linear_system_example, linear_system_examples = example_decorator()
poly_system_example, poly_system_examples = example_decorator()
equation_example, equation_examples = example_decorator()
poly_example, poly_examples = example_decorator()
poly_pair_example, poly_pair_examples = example_decorator()
rational_function_example, rational_function_examples = example_decorator()
ode_example, ode_examples = example_decorator()
ode_system_example, ode_system_examples = example_decorator()


def pinv_sympy(matrix):
    return matrix.pinv()


def inv_sympy(matrix):
    return matrix.inv()


def jnf_sympy(matrix):
    return matrix.jordan_form()


def matrix_exp_sympy(matrix):
    from sympy import Dummy
    t = Dummy('t')
    return (t*matrix).exp()


def det_sympy(matrix):
    return matrix.det()


def eigenvals_sympy(matrix):
    return matrix.eigenvals()


def charpoly_sympy(matrix):
    return matrix.charpoly()


def integrate_sympy(expr):
    return expr.doit()


def call_function(func):
    return func()


def nonlinsolve_sympy(eqs_vars):
    from sympy import nonlinsolve
    eqs, vars = eqs_vars
    return nonlinsolve(eqs, vars)


def solve_sympy(eqs_vars):
    from sympy import solve
    eqs, vars = eqs_vars
    return solve(eqs, vars)


def solveset_sympy(eq_sym):
    from sympy import solveset
    eq, sym = eq_sym
    return solveset(eq, sym)


def solveset_real_sympy(eq_sym):
    from sympy import solveset, S
    eq, sym = eq_sym
    return solveset(eq, sym, domain=S.Reals)


def linsolve_sympy(eqs_vars):
    from sympy import linsolve
    eqs, vars = eqs_vars
    return linsolve(eqs, vars)


def dsolve_sympy(eq_sym):
    from sympy import dsolve
    eq, sym = eq_sym
    return dsolve(eq, sym)


def cancel_sympy(expr):
    from sympy import cancel
    return cancel(expr)


def simplify_sympy(expr):
    from sympy import simplify
    return simplify(expr)


def poly_cancel_sympy(expr):
    """Can be a lot faster than cancel_sympy."""
    from sympy import symbols, poly
    n, d = expr.together().as_numer_denom()

    z = symbols('z')

    nr = poly(n, z).rep.rep[0]
    dr = poly(d, z).rep.rep[0]

    R = nr.ring.to_domain().unify(dr.ring.to_domain())

    nrp = R.convert(nr)
    drp = R.convert(dr)

    nrc, drc = nrp.cancel(drp)

    expr = nrc.as_expr() / drc.as_expr()

    return(expr)


def polydiv_sympy(poly_pair):
    p1, p2 = poly_pair
    from sympy import div
    return div(p1, p2)


def trigsimp_sympy(expr):
    return expr.trigsimp()


def factor_deep_sympy(expr):
    return expr.factor(deep=True)


def minpoly_sympy(expr):
    from sympy import minpoly
    return minpoly(expr)


def evalf_sympy(expr):
    return expr.evalf()


def subs_sympy(expr_rep):
    expr, rep = expr_rep
    return expr.subs(rep)


@matrix_example
@fast_time_with(pinv_sympy)
def pinv_numeric():
    """
    This example computes the pseudoinverse of a numeric matrix and is
    acceptably fast but could be faster.

    The matrix is:

    >>> from examples import pinv_numeric
    >>> my_matrix = pinv_numeric()
    >>> my_matrix
    Matrix([
    [1, 2, 3, 4, 0, 0, 0, 0, 0,  0],
    [1, 2, 0, 0, 0, 6, 0, 0, 0,  0],
    [0, 0, 3, 4, 5, 0, 0, 0, 0,  0],
    [0, 0, 0, 0, 5, 6, 0, 0, 0,  0],
    [1, 0, 0, 4, 0, 0, 7, 0, 0,  0],
    [1, 0, 0, 0, 0, 0, 0, 0, 9,  0],
    [0, 0, 0, 4, 0, 0, 0, 8, 0,  0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]])
    >>> my_matrix.pinv()
    Matrix([
    [  98288073/2801071636,   90097515/2801071636, -31905657/2801071636, -40096215/2801071636,   626616/63660719,  7881448/700267909,   -4697676/700267909,    0],
    [ 117872155/1400535818,   95344257/1400535818, -23932459/1400535818, -46460357/1400535818, -1022958/63660719, -1162874/700267909,   -7143462/700267909,    0],
    [ 121525551/1400535818,   19383993/1400535818,  64335015/1400535818, -37806543/1400535818, -1823028/63660719,  -614652/700267909,  -14575395/700267909,    0],
    [   50117108/700267909,     7030588/700267909,   27618332/700267909,  -15468188/700267909,  1722096/63660719,  -927936/700267909,  62710785/2801071636,    0],
    [-166151825/2801071636, -185812255/2801071636, 254580065/2801071636, 234919635/2801071636,  -283860/63660719,  1111140/700267909,   -3796920/700267909,    0],
    [    5437134/700267909,    67888716/700267909,  -23859684/700267909,   38591898/700267909,   236550/63660719,  -925950/700267909,    3164100/700267909,    0],
    [  -11690413/254642876,    -2630999/254642876,   -5324515/254642876,    3734899/254642876,  8020817/63660719,    -54152/63660719,     -753417/63660719,    0],
    [  -25058554/700267909,    -3515294/700267909,  -13809166/700267909,    7734094/700267909,  -861048/63660719,   463968/700267909, 159389281/1400535818,    0],
    [ -10920897/2801071636,  -10010835/2801071636,   3545073/2801071636,   4455135/2801071636,   -69624/63660719, 76931829/700267909,     521964/700267909,    0],
    [                    0,                     0,                    0,                    0,                 0,                  0,                    0, 1/10]])

    This could be made faster by:

    1. Using DomainMatrix for the internal RREF over ZZ/QQ.
    2. Using python_flint for DomainMatrix.
    """
    from sympy import Matrix
    my_matrix = Matrix([
    [1, 2, 3, 4, 0, 0, 0, 0, 0, 0],
    [1, 2, 0, 0, 0, 6, 0, 0, 0, 0],
    [0, 0, 3, 4, 5, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 6, 0, 0, 0, 0],
    [1, 0, 0, 4, 0, 0, 7, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 9, 0],
    [0, 0, 0, 4, 0, 0, 0, 8, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 10]])

    return my_matrix


@matrix_example
@hangs_time_with(pinv_sympy)
def pinv_symbolic():
    """
    This example computes the pseudoinverse of a symbolic matrix and is slow.

    The matrix is:

    >>> from examples import pinv_symbolic
    >>> my_matrix = pinv_symbolic()
    >>> my_matrix
    Matrix([
    [p1, p2, p3, p4,  0,  0,  0,  0,  0,   0],
    [p1, p2,  0,  0,  0, p6,  0,  0,  0,   0],
    [ 0,  0, p3, p4, p5,  0,  0,  0,  0,   0],
    [ 0,  0,  0,  0, p5, p6,  0,  0,  0,   0],
    [p1,  0,  0, p4,  0,  0, p7,  0,  0,   0],
    [p1,  0,  0,  0,  0,  0,  0,  0, p9,   0],
    [ 0,  0,  0, p4,  0,  0,  0, p8,  0,   0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, p10]])

    >>> my_matrix.pinv() # doctest: +SKIP
    <the output would be huge>

    https://github.com/sympy/sympy/issues/25032

    This example is slow because of testing invertibility and simplifying to
    avoid division by zero.
    """
    from sympy import symbols, Matrix
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = symbols('p1:11')

    my_matrix = Matrix([
    [p1, p2, p3, p4,  0,  0,  0,  0,  0,   0],
    [p1, p2,  0,  0,  0, p6,  0,  0,  0,   0],
    [ 0,  0, p3, p4, p5,  0,  0,  0,  0,   0],
    [ 0,  0,  0,  0, p5, p6,  0,  0,  0,   0],
    [p1,  0,  0, p4,  0,  0, p7,  0,  0,   0],
    [p1,  0,  0,  0,  0,  0,  0,  0, p9,   0],
    [ 0,  0,  0, p4,  0,  0,  0, p8,  0,   0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, p10]])

    return my_matrix


@matrix_example
@hangs_time_with(inv_sympy)
def inv_symbolic():
    """
    This example computes the inverse of a symbolic matrix and is slow.

    https://github.com/sympy/sympy/issues/25009

    >>> from examples import inv_symbolic
    >>> my_matrix = inv_symbolic()
    >>> my_matrix
    Matrix([
    [1/y + 1/w + 1/t + 1/s + 1/p,         0,            -1/w,         0,         0,      -1/t,  1],
    [                          0, 1/u + 1/q,            -1/q,         0,         0,         0,  0],
    [                       -1/w,      -1/q, 1/x + 1/w + 1/q,      -1/x,         0,         0, -1],
    [                          0,         0,            -1/x, 1/x + 1/r,      -1/r,         0,  0],
    [                          0,         0,               0,      -1/r, 1/v + 1/r,      -1/v,  0],
    [                       -1/t,         0,               0,         0,      -1/v, 1/v + 1/t,  0],
    [                          1,         0,              -1,         0,         0,         0,  0]])

    >>> my_matrix.inv() # doctest: +SKIP

    This example is slow but also using DomainMatrix is slow. That is because
    DomainMatrix can be inefficient with rational functions.

    This could be made faster by:

    1. Using DomainMatrix instead of Matrix.
    2. Speeding up rational function fields like QQ(x).
    3. Speeding up polynomial gcd.
    4. Using pivoting to improve performance (pivot top-left to bottom right).
    5. Treating symbols that only appear to negative powers as polynomial
    generators e.g. using the domain QQ[1/x].
    6. Using Laurent polynomials instead of rational functions QQ[x, 1/x].

    """
    from sympy import symbols, Matrix
    p, q, r, s, t, u, v, w, x, y = symbols('p, q, r, s, t, u, v, w, x, y')

    X = Matrix([
        [1/y + 1/w + 1/t + 1/s + 1/p, 0, -1/w, 0, 0, -1/t, 1],
        [0, 1/u + 1/q, -1/q, 0, 0, 0, 0],
        [-1/w, -1/q, 1/w + 1/q + 1/x, -1/x, 0, 0, -1],
        [0, 0, -1/x, 1/r + 1/x, -1/r, 0, 0],
        [0, 0, 0, -1/r, 1/v + 1/r, -1/v, 0],
        [-1/t, 0, 0, 0, -1/v, 1/v + 1/t, 0],
        [1, 0, -1, 0, 0, 0, 0]
    ])

    return X


@matrix_example
@hangs_time_with(jnf_sympy)
def jnf_numeric():
    """
    https://github.com/sympy/sympy/issues/24942

    This Jordan normal form calculation is slow.

    >>> from examples import jnf_numeric
    >>> A = jnf_numeric()
    >>> A
    Matrix([
    [-3,  1,  2],
    [ 1, -1,  0],
    [ 1,  0, -2]])
    >>> A.jordan_form() # doctest: +SKIP

    This is slow because it does not use the same sort of routine as for
    eigenvects. A matrix of rationals like this should be straight-forward with
    DomainMatrix.
    """
    from sympy import Matrix

    A = Matrix([[-3, 1, 2], [1, -1, 0], [1, 0, -2]])

    return A


@matrix_example
@hangs_time_with(jnf_sympy)
@hangs_time_with(matrix_exp_sympy)
def jnf_symbolic():
    """
    https://github.com/sympy/sympy/issues/21867

    >>> from examples import jnf_symbolic
    >>> A = jnf_symbolic()
    >>> A
    Matrix([
    [-k_12 - k_e,  k_21, ka],
    [       k_12, -k_21,  0],
    [        -ka,     0,  0]])
    >>> A.jordan_form() # doctest: +SKIP
    """
    from sympy import symbols, Matrix
    k_12, k_21, k_e, ka, D = symbols('k_12, k_21, k_e, ka, D')
    A = Matrix([
    [-k_12 - k_e,  k_21, ka],
    [       k_12, -k_21,  0],
    [        -ka,     0,  0]])
    return A



@integral_example
@slow_time_with(integrate_sympy, time_seconds=100)
def integral_slow():
    """
    https://github.com/sympy/sympy/issues/24952

    >>> from examples import integral_slow
    >>> I = integral_slow()
    >>> I
    Integral((2*(sqrt(5) + 3)*exp(s*(3/2 - sqrt(5)/2))/((1 + sqrt(5))*(5 + 3*sqrt(5))) + 2*exp(s*(sqrt(5)/2 + 3/2))/((-1 + sqrt(5))*(2/(1 + sqrt(5)) + 2/(-1 + sqrt(5)))))*(cos(2*pi*s) + 1), (s, 0, t))
    >>> I.doit() # doctest: +SKIP

    This example is slow because heurisch is slow when solving linear systems
    of equations over EX domain. Also meijerg is slow.

    Takes about 100 seconds.
    """
    from sympy import Symbol, Matrix, Integral, exp, cos, pi

    s = Symbol('s')
    t = Symbol('t')

    A = Matrix([[-2, 1],[1,-1]])
    B = exp(-A*s)*Matrix([[1+cos(2*pi*s)],[0]])
    C = B[0]

    return Integral(C, (s, 0, t))


@system_example
@hangs_time_with(nonlinsolve_sympy)
def solve_poly_slow():
    """
    https://github.com/sympy/sympy/issues/24868

    This system of polynomial equations should be something that can be solved
    easily by solve or nonlinsolve. It is very slow because solve_poly_system
    uses a slow and ineffective technique for getting solutions from a Lex
    basis.

    >>> from examples import solve_poly_slow
    >>> eqs, syms = solve_poly_slow()
    >>> for eq in eqs: print(eq)
    Eq((x + 1)/sqrt((x + 1)**2 + (y - 1)**2 + (z + 1)**2), 2*l*x)
    Eq((y - 1)/sqrt((x + 1)**2 + (y - 1)**2 + (z + 1)**2), 2*l*y)
    Eq((z + 1)/sqrt((x + 1)**2 + (y - 1)**2 + (z + 1)**2), 2*l*z)
    Eq(x**2 + y**2 + z**2 - 9, 0)
    >>> syms
    [x, y, z, l]

    >>> from sympy import nonlinsolve
    >>> nonlinsolve(eqs, syms) # doctest: +SKIP

    Not sure how long this takes.
    """
    from sympy import symbols, sqrt, Eq
    x, y, z = symbols('x y z')
    l = symbols('l') # "lambda"
    g = (x**2) + (y**2) + (z**2) - 9
    f = sqrt(((x+1) ** 2) + ((y-1) ** 2) + ((z+1)**2))

    eqs = [
        Eq(f.diff(x), l * g.diff(x)),
        Eq(f.diff(y), l * g.diff(y)),
        Eq(f.diff(z), l * g.diff(z)), Eq(g, 0)
    ]

    return eqs, [x, y, z, l]


@poly_system_example
@slow_time_with(nonlinsolve_sympy, time_seconds=100)
@slow_time_with(solve_sympy, time_seconds=25)
def poly_system_QQ():
    """
    https://github.com/sympy/sympy/issues/23889

    >>> from examples import poly_system_QQ
    >>> eqs, syms = poly_system_QQ()
    >>> for eq in eqs: print(eq)
    327600995*x**2 - 37869137*x + 1809975124*y**2 - 9998905626
    895613949*x**2 - 273830224*x*y + 530506983*y**2 - 10000000000
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP

    This would be faster if it returned a result with RootOf instead of
    radicals.
    """
    from sympy import symbols
    x, y = symbols('x, y')
    eqs = [
         327600995*x**2 - 37869137*x + 1809975124*y**2 - 9998905626,
         895613949*x**2 - 273830224*x*y + 530506983*y**2 - 10000000000,
    ]
    return eqs, [x, y]


@poly_system_example
@hangs_time_with(nonlinsolve_sympy)
@hangs_time_with(solve_sympy)
def poly_system_QQ_deg164_simple():
    """
    https://github.com/sympy/sympy/issues/22839

    This is a simple polynomial system but general approaches might be slow due
    to the high degree. For example fglm is slow in this case.

    >>> from examples import poly_system_QQ_deg164_simple
    >>> eqs, syms = poly_system_QQ_deg164_simple()
    >>> for eq in eqs: print(eq)
    x*y**5 - 67
    x*y**164 - 70
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP
    """
    from sympy import symbols
    x, y = symbols('x, y')
    eqs = [x*y**5-67, x*y**164-70]
    return eqs, [x, y]


@poly_system_example
@hangs_time_with(nonlinsolve_sympy)
def poly_system_QQ_factorisable():
    """
    https://github.com/sympy/sympy/issues/21907

    >>> from examples import poly_system_QQ_factorisable
    >>> eqs, syms = poly_system_QQ_factorisable()
    >>> for eq in eqs: print(eq)
    6*x*(4*x**2 + y**6)
    18*y**2*(x**2*y**3 - 1)
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP

    It should be easy to solve this by factorising the equations and then using
    Groebner bases or resultants. It goes wrong because complicated radical
    formulae are solved from one and substituted into the other and then roots
    tries to simplify them. It would also be good to have a function that
    focusses on only finding real solutions of polynomial systems. In this case
    there is only one real root (0, 0) but there are 18 complicated complex
    roots that can be expressed in terms of RootOf for a 12 degree polynomial.
    """
    from sympy import symbols
    x, y = symbols('x, y')
    eqs = [
        6*x*(4*x**2 + y**6),
        18*y**2*(x**2*y**3 - 1)
    ]
    return eqs, [x, y]


@poly_system_example
@hangs_time_with(nonlinsolve_sympy)
@hangs_time_with(solve_sympy)
def solve_poly_system_slow():
    """
    https://github.com/sympy/sympy/issues/24500

    This system of polynomial equations could be solved efficiently by using a
    better strategy than just passing everything to groebner.

    The equations could be factorised first and then most subproblems would be
    linear.

    >>> from examples import solve_poly_system_slow
    >>> eqs, syms = solve_poly_system_slow()
    >>> for eq in eqs: print(eq)
    x*(x - 1)*(c1 + c2*y*z - c2*y - c2*z - e1 - e2*y*z + e2*y + e2*z - g1*y - g2*z - s1*y)
    y*(y - 1)*(c3 + g1*x - g1 + s1*x)
    -z*(z - 1)*(-c5 + s2 + s3)
    >>> syms
    [x, y, z]
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP
    """
    from sympy import symbols
    x, y, z, e1, e2, c1, c2, c3, c5, s1, s2, s3, g1, g2 = symbols("x y z e1 e2 c1 c2 c3 c5 s1 s2 s3 g1 g2")

    fx = x*(x - 1)*(c1 + c2*y*z - c2*y - c2*z - e1 - e2*y*z + e2*y + e2*z - g1*y - g2*z - s1*y)
    fy = y*(y - 1)*(c3 + g1*x - g1 + s1*x)
    fz = -z*(z - 1)*(-c5 + s2 + s3)

    return [fx, fy, fz], [x, y, z]


def _solve_nonlinear_slow():
    """
    https://github.com/sympy/sympy/issues/24824
    """
    # Not sure this one is worth including


@integral_example
@slow_time_with(integrate_sympy, time_seconds=100)
def integral_heurisch_slow():
    """
    https://github.com/sympy/sympy/issues/24821

    This gives a wrong answer and takes a long time. The slow part is heurisch
    which does not find an answer anyway.

    Takes about 100 seconds.
    """
    from sympy import Symbol, Integral, cos, sin, oo
    x = Symbol("x")
    J = Integral(x**2 * (cos(x)/x**2 - sin(x)/x**3)**3, (x, 0, oo))
    return J


@linear_system_example
@slow_time_with(linsolve_sympy, time_seconds=75)
def linear_system_mechanics():
    """
    https://github.com/sympy/sympy/issues/24679

    (Bigger examples can be found in the issue.)

    This is a linear system of equations that is slow to solve. It is slow
    because of simplification that is indirectly used to test for
    invertibility.
    """
    from sympy import symbols, Matrix, exp, cos, sin
    a0, a1, a2, a3, a4, q0, q1 = symbols('a0 a1 a2 a3 a4 q0 q1')
    equations = [
        a0 + a3 + q0*(a0*exp(a2) + a1*exp(a0) + a1*sin(a2) + a2*sin(a4)
            + a4*cos(a2) + a4) + q1*(a0*exp(a2) + a1*exp(a1) + a1*exp(a3)
            + a2*cos(a4) + a2 + a3*cos(a3)),
        a3 + a4 + q0*(a1*exp(a3) + a2 + a3*exp(a4) + a3*cos(a4) + a4*exp(a0)
            + a4*exp(a4)) + q1*(a1*exp(a0) + a1*sin(a1) + a2*exp(a2) + a2
            + a3*cos(a3) + a4*exp(a0)),
    ]
    return equations, [q0, q1]


@linear_system_example
@fast_time_with(linsolve_sympy)
def linear_system_ZZ_symbols_example():
    """
    https://github.com/sympy/sympy/issues/24135

    This example is solves in an acceptable amount of time but is very slow if
    the symbol J is replaced by I (sqrt(-1)). See
    linear_system_ZZ_I_symbols_slow_example for the slow version.

    >>> from examples import linear_system_ZZ_symbols_example
    >>> eqs, syms = linear_system_ZZ_symbols_example()
    >>> for eq in eqs: print(eq)
    Eq(I1*(J*L1*w - J/(C1*w)) + I2*J/(C1*w), U)
    Eq(I2*(J*L2*w - J/(C2*w) - J/(C1*w)) + I3*(J*M23*w + J/(C2*w)) + I1*J/(C1*w), 0)
    Eq(I2*(J*M23*w + J/(C2*w)) + I3*(J*L3*w - J/(C3*w) - J/(C2*w)) + I4*J/(C3*w), 0)
    Eq(I4*RL + I3*J/(C3*w) - I4*J/(C3*w), 0)
    >>> syms
    [I1, I2, I3, I4]
    >>> from sympy import solve
    >>> solve(eqs, vars) # doctest: +SKIP
    """
    from sympy import symbols, Eq

    I1, I2, I3, I4 = symbols('I1 I2 I3 I4')
    L1, L2, L3, L4 = symbols('L1 L2 L3 L4')
    C1, C2, C3, C4 = symbols('C1 C2 C3 C4')
    I1, I2, I3, I4 = symbols('I1 I2 I3 I4')
    M23, U, w, p, RL, J = symbols('M23 U w p RL J')

    eqs = [
        Eq(I1*(J*L1*w - J/(C1*w)) + I2*J/(C1*w), U),
        Eq(I2*(J*L2*w - J/(C2*w) - J/(C1*w)) + I3*(J*M23*w + J/(C2*w)) + I1*J/(C1*w), 0),
        Eq(I2*(J*M23*w + J/(C2*w)) + I3*(J*L3*w - J/(C3*w) - J/(C2*w)) + I4*J/(C3*w), 0),
        Eq(I4*RL + I3*J/(C3*w) - I4*J/(C3*w), 0),
    ]
    return eqs, [I1, I2, I3, I4]


@linear_system_example
@hangs_time_with(linsolve_sympy)
def linear_system_ZZ_I_symbols_slow_example():
    """
    https://github.com/sympy/sympy/issues/24135

    This is the slow version of linear_system_ZZ_symbols_example It is slow
    because having I = sqrt(-1) in the equations causes the domain to become
    ZZ_I and then polynomial gcd over ZZ_I is slow due to using the dense
    representation. It could also be possible to avoid polynomial GCD by using
    division free methods.

    >>> from examples import linear_system_ZZ_I_symbols_slow_example
    >>> eqs, syms = linear_system_ZZ_I_symbols_slow_example()
    >>> for eq in eqs: print(eq)
    Eq(I1*(I*L1*w - I/(C1*w)) + I*I2/(C1*w), U)
    Eq(I2*(I*L2*w - I/(C2*w) - I/(C1*w)) + I3*(I*M23*w + I/(C2*w)) + I*I1/(C1*w), 0)
    Eq(I2*(I*M23*w + I/(C2*w)) + I3*(I*L3*w - I/(C3*w) - I/(C2*w)) + I*I4/(C3*w), 0)
    Eq(I4*RL + I*I3/(C3*w) - I*I4/(C3*w), 0)
    >>> syms
    [I1, I2, I3, I4]
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP
    """
    from sympy import symbols, I
    eqs, syms = linear_system_ZZ_symbols_example()
    rep = {symbols('J'): I}
    eqs = [eq.subs(rep) for eq in eqs]
    return eqs, syms


@linear_system_example
@hangs_time_with(solve_sympy)
@fast_time_with(nonlinsolve_sympy)
def linear_system_over_determined_surds():
    """
    https://github.com/sympy/sympy/discussions/22848
    https://github.com/sympy/sympy/issues/22852

    This is a fairly simple linear system involving many surds. It is slow in
    linsolve because attempting to construct an algebraic extension field that
    can contain all of these surds is slow. A domain that can do arithmetic
    with many surds is needed.

    >>> from examples import linear_system_over_determined_surds
    >>> eqs, syms = linear_system_over_determined_surds()
    >>> for eq in eqs: print(eq)
    -sqrt(6)*x1/210 + sqrt(33)*x2/2310 - sqrt(33)*x4/13860 + sqrt(4290)*x5/8316
    sqrt(11)*(x2 - x4)/660
    sqrt(11)*(-x2 + x4)/660
    sqrt(11)*x2/660 + sqrt(10)*x3/90 + 7*sqrt(11)*x4/1980 - sqrt(1430)*x5/5940
    -sqrt(6)*x1/210 + 13*sqrt(33)*x2/13860 - 2*sqrt(33)*x4/3465 + sqrt(4290)*x5/8316
    sqrt(33)*(x2 - x4)/990
    sqrt(33)*(-x2 + x4)/1980
    -sqrt(10)*x0/90 - sqrt(11)*x2/495 + sqrt(10)*x3/90 + sqrt(11)*x4/495
    -sqrt(10)*x0/90 + 2*sqrt(11)*x2/495 + sqrt(10)*x3/90 - 2*sqrt(11)*x4/495
    -sqrt(6)*x1/210 - 2*sqrt(33)*x2/3465 + 13*sqrt(33)*x4/13860 + sqrt(4290)*x5/8316
    sqrt(10)*x0/90 - sqrt(2)*x1/105 + 17*sqrt(11)*x2/3465 + sqrt(11)*x4/1155 + sqrt(1430)*x5/13860
    sqrt(10)*x0/90 + 2*sqrt(11)*x2/495 - sqrt(10)*x3/90 - 2*sqrt(11)*x4/495
    >>> syms
    (x0, x1, x2, x3, x4, x5)
    >>> from sympy import linsolve
    >>> linsolve(eqs, syms) # doctest: +SKIP
    """
    from sympy import symbols, sqrt
    x0, x1, x2, x3, x4, x5 = variables = symbols('x0, x1, x2, x3, x4, x5')
    constraints = [
        -sqrt(6)*x1/210 + sqrt(33)*x2/2310 - sqrt(33)*x4/13860 + sqrt(4290)*x5/8316,
        sqrt(11)*(x2 - x4)/660,
        sqrt(11)*(-x2 + x4)/660,
        sqrt(11)*x2/660 + sqrt(10)*x3/90 + 7*sqrt(11)*x4/1980 - sqrt(1430)*x5/5940,
        -sqrt(6)*x1/210 + 13*sqrt(33)*x2/13860 - 2*sqrt(33)*x4/3465 + sqrt(4290)*x5/8316,
        sqrt(33)*(x2 - x4)/990,
        sqrt(33)*(-x2 + x4)/1980,
        -sqrt(10)*x0/90 - sqrt(11)*x2/495 + sqrt(10)*x3/90 + sqrt(11)*x4/495,
        -sqrt(10)*x0/90 + 2*sqrt(11)*x2/495 + sqrt(10)*x3/90 - 2*sqrt(11)*x4/495,
        -sqrt(6)*x1/210 - 2*sqrt(33)*x2/3465 + 13*sqrt(33)*x4/13860 + sqrt(4290)*x5/8316,
        sqrt(10)*x0/90 - sqrt(2)*x1/105 + 17*sqrt(11)*x2/3465 + sqrt(11)*x4/1155 + sqrt(1430)*x5/13860,
        sqrt(10)*x0/90 + 2*sqrt(11)*x2/495 - sqrt(10)*x3/90 - 2*sqrt(11)*x4/495,
    ]
    return constraints, variables


@linear_system_example
@slow_time_with(linsolve_sympy, time_seconds=250)
def linear_system_sincos_derivs_example():
    """
    https://github.com/sympy/sympy/issues/23965

    This is a 3x3 system of linear equations but with many symbols and sin, cos
    in the coefficients. It is slow because the EX domain is used but it should
    be possible to use a polynomial ring and also division free methods. A
    special domain for sin and cos needs to be implemented.

    >>> from examples import linear_system_sincos_derivs_example
    >>> eqs, syms = linear_system_sincos_derivs_example()
    >>> eqs[0] # doctest: +NORMALIZE_WHITESPACE
    -L0**2*m0*sin(theta0(t))**2*Derivative(theta0(t), (t, 2)) -
    L0**2*m0*cos(theta0(t))**2*Derivative(theta0(t), (t, 2)) -
    L0**2*m1*sin(theta0(t))**2*Derivative(theta0(t), (t, 2)) -
    L0**2*m1*cos(theta0(t))**2*Derivative(theta0(t), (t, 2)) -
    L0**2*m2*sin(theta0(t))**2*Derivative(theta0(t), (t, 2)) -
    L0**2*m2*cos(theta0(t))**2*Derivative(theta0(t), (t, 2)) -
    L0*L1*m1*sin(theta0(t))*sin(theta1(t))*Derivative(theta1(t), (t, 2)) -
    L0*L1*m1*sin(theta0(t))*cos(theta1(t))*Derivative(theta1(t), t)**2 +
    L0*L1*m1*sin(theta1(t))*cos(theta0(t))*Derivative(theta1(t), t)**2 -
    L0*L1*m1*cos(theta0(t))*cos(theta1(t))*Derivative(theta1(t), (t, 2)) -
    L0*L1*m2*sin(theta0(t))*sin(theta1(t))*Derivative(theta1(t), (t, 2)) -
    L0*L1*m2*sin(theta0(t))*cos(theta1(t))*Derivative(theta1(t), t)**2 +
    L0*L1*m2*sin(theta1(t))*cos(theta0(t))*Derivative(theta1(t), t)**2 -
    L0*L1*m2*cos(theta0(t))*cos(theta1(t))*Derivative(theta1(t), (t, 2)) -
    L0*L2*m2*sin(theta0(t))*sin(theta2(t))*Derivative(theta2(t), (t, 2)) -
    L0*L2*m2*sin(theta0(t))*cos(theta2(t))*Derivative(theta2(t), t)**2 +
    L0*L2*m2*sin(theta2(t))*cos(theta0(t))*Derivative(theta2(t), t)**2 -
    L0*L2*m2*cos(theta0(t))*cos(theta2(t))*Derivative(theta2(t), (t, 2)) +
    L0*g*m0*cos(theta0(t)) + L0*g*m1*cos(theta0(t)) + L0*g*m2*cos(theta0(t))
    >>> syms
    [Derivative(theta0(t), (t, 2)), Derivative(theta1(t), (t, 2)), Derivative(theta2(t), (t, 2))]
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP
    """
    import sympy as smp

    # N = 2 takes ~2 seconds (already way too slow)
    # N = 3 takes 275 seconds
    # The original was N = 4
    N = 3

    t, g = smp.symbols('t g')
    ms = smp.symbols(f'm0:{N}')
    Ls = smp.symbols(f'L0:{N}')
    thetas = smp.symbols(f'theta0:{N}', cls=smp.Function)
    thetas = [theta(t) for theta in thetas]
    thetas_d = [smp.diff(theta) for theta in thetas]
    thetas_dd = [smp.diff(theta_d) for theta_d in thetas_d]

    xs = smp.symbols(f'x0:{N}', cls=smp.Function)
    ys = smp.symbols(f'y0:{N}', cls=smp.Function)

    def get_xn_yn():
        xs = []
        ys = []
        xn = Ls[0]*smp.cos(thetas[0])
        yn = -Ls[0]*smp.sin(thetas[0])
        xs.append(xn)
        ys.append(yn)

        for i in range(1,N):
            xn = xn + Ls[i]*smp.cos(thetas[i])
            yn = yn - Ls[i]*smp.sin(thetas[i])
            xs.append(xn)
            ys.append(yn)
        return xs, ys

    xs, ys = get_xn_yn()

    # Original problem used 1/2 rather than S(1)/2
    T = sum([smp.S(1)/2 * m * (smp.diff(x, t)**2 + smp.diff(y, t)**2) for (m,x,y) in zip(ms, xs, ys)])
    V = sum([m*g*y for (m,y) in zip(ms, ys)])
    L=T-V

    LEs = [smp.diff(L, the) - smp.diff(smp.diff(L, the_d), t) for (the, the_d) in zip(thetas, thetas_d)]
    LEs = [LE.expand() for LE in LEs]

    return LEs, thetas_dd


@linear_system_example
@hangs_time_with(solve_sympy)
def linear_system_sincos_derivs_example_2():
    """
    https://github.com/sympy/sympy/issues/23377

    An example of a Lagrangian problem. Solving for the equations of motion is
    slow. It is a linear system of equations with many sin and cos functions
    and also undefined functions and derivatives. A special domain is needed
    for sin and cos. The presence of mixed undefined functions tends to confuse
    construct_domain leading to the EX domain being used.

    >>> from examples import linear_system_sincos_derivs_example_2
    >>> eqs, syms = linear_system_sincos_derivs_example_2()
    >>> solve(eqs, syms) # doctest: +SKIP

    """
    from sympy import symbols, Function, sin, cos
    t, g = symbols('t g')
    m1, m2 = symbols('m1 m2')
    L1, L2 = symbols('L1, L2')
    the1, the2, phi1, phi2 = symbols(r't_1, t_2, p_1, p_2', cls=Function)
    t1 = the1(t)
    t2 = the2(t)
    p1 = phi1(t)
    p2 = phi2(t)
    t1d = t1.diff(t)
    t2d = t2.diff(t)
    p1d = p1.diff(t)
    p2d = p2.diff(t)
    t1dd = t1d.diff(t)
    t2dd = t2d.diff(t)
    p1dd = p1d.diff(t)
    p2dd = p2d.diff(t)

    eqs = [
        -L1*g*m1*sin(t1) - L1*g*m2*sin(t1) - L1*(L1*m1*t1dd + L1*m2*t1dd + L2*m2*p1d*p2d*sin(t2)*cos(t1)*cos(p1 - p2) - L2*m2*p1d*t2d*sin(p1 - p2)*cos(t1)*cos(t2) - L2*m2*p2d**2*sin(t2)*cos(t1)*cos(p1 - p2) - L2*m2*p2d*t1d*sin(t1)*sin(t2)*sin(p1 - p2) + 2*L2*m2*p2d*t2d*sin(p1 - p2)*cos(t1)*cos(t2) + L2*m2*p2dd*sin(t2)*sin(p1 - p2)*cos(t1) - L2*m2*t1d*t2d*sin(t1)*cos(t2)*cos(p1 - p2) + L2*m2*t1d*t2d*sin(t2)*cos(t1) + L2*m2*t2d**2*sin(t1)*cos(t2) - L2*m2*t2d**2*sin(t2)*cos(t1)*cos(p1 - p2) + L2*m2*t2dd*sin(t1)*sin(t2) + L2*m2*t2dd*cos(t1)*cos(t2)*cos(p1 - p2)) + m1*(2*L1**2*t1d**2*sin(t1)*cos(t1) + (-L1*p1d*sin(p1)*sin(t1) + L1*t1d*cos(p1)*cos(t1))*(-2*L1*p1d*sin(p1)*cos(t1) - 2*L1*t1d*sin(t1)*cos(p1)) + (L1*p1d*sin(t1)*cos(p1) + L1*t1d*sin(p1)*cos(t1))*(2*L1*p1d*cos(p1)*cos(t1) - 2*L1*t1d*sin(p1)*sin(t1)))/2 + m2*(2*L1*t1d*(L1*t1d*sin(t1) + L2*t2d*sin(t2))*cos(t1) + (-2*L1*p1d*sin(p1)*cos(t1) - 2*L1*t1d*sin(t1)*cos(p1))*(-L1*p1d*sin(p1)*sin(t1) + L1*t1d*cos(p1)*cos(t1) - L2*p2d*sin(p2)*sin(t2) + L2*t2d*cos(p2)*cos(t2)) + (2*L1*p1d*cos(p1)*cos(t1) - 2*L1*t1d*sin(p1)*sin(t1))*(L1*p1d*sin(t1)*cos(p1) + L1*t1d*sin(p1)*cos(t1) + L2*p2d*sin(t2)*cos(p2) + L2*t2d*sin(p2)*cos(t2)))/2,
        -L2*g*m2*sin(t2) - L2*m2*(-L1*p1d**2*sin(t1)*cos(t2)*cos(p1 - p2) + L1*p1d*p2d*sin(t1)*cos(t2)*cos(p1 - p2) - 2*L1*p1d*t1d*sin(p1 - p2)*cos(t1)*cos(t2) + L1*p1d*t2d*sin(t1)*sin(t2)*sin(p1 - p2) - L1*p1dd*sin(t1)*sin(p1 - p2)*cos(t2) + L1*p2d*t1d*sin(p1 - p2)*cos(t1)*cos(t2) - L1*t1d**2*sin(t1)*cos(t2)*cos(p1 - p2) + L1*t1d**2*sin(t2)*cos(t1) + L1*t1d*t2d*sin(t1)*cos(t2) - L1*t1d*t2d*sin(t2)*cos(t1)*cos(p1 - p2) + L1*t1dd*sin(t1)*sin(t2) + L1*t1dd*cos(t1)*cos(t2)*cos(p1 - p2) + L2*t2dd) + m2*(2*L2*t2d*(L1*t1d*sin(t1) + L2*t2d*sin(t2))*cos(t2) + (-2*L2*p2d*sin(p2)*cos(t2) - 2*L2*t2d*sin(t2)*cos(p2))*(-L1*p1d*sin(p1)*sin(t1) + L1*t1d*cos(p1)*cos(t1) - L2*p2d*sin(p2)*sin(t2) + L2*t2d*cos(p2)*cos(t2)) + (2*L2*p2d*cos(p2)*cos(t2) - 2*L2*t2d*sin(p2)*sin(t2))*(L1*p1d*sin(t1)*cos(p1) + L1*t1d*sin(p1)*cos(t1) + L2*p2d*sin(t2)*cos(p2) + L2*t2d*sin(p2)*cos(t2)))/2,
        -L1*(2*L1*m1*p1d*t1d*sin(t1)*cos(t1) + L1*m1*p1dd*sin(t1)**2 + 2*L1*m2*p1d*t1d*sin(t1)*cos(t1) + L1*m2*p1dd*sin(t1)**2 - L2*m2*p1d*p2d*sin(t1)*sin(t2)*sin(p1 - p2) - L2*m2*p1d*t2d*sin(t1)*cos(t2)*cos(p1 - p2) + L2*m2*p2d**2*sin(t1)*sin(t2)*sin(p1 - p2) + L2*m2*p2d*t1d*sin(t2)*cos(t1)*cos(p1 - p2) + 2*L2*m2*p2d*t2d*sin(t1)*cos(t2)*cos(p1 - p2) + L2*m2*p2dd*sin(t1)*sin(t2)*cos(p1 - p2) - L2*m2*t1d*t2d*sin(p1 - p2)*cos(t1)*cos(t2) + L2*m2*t2d**2*sin(t1)*sin(t2)*sin(p1 - p2) - L2*m2*t2dd*sin(t1)*sin(p1 - p2)*cos(t2)) + m1*((-2*L1*p1d*sin(p1)*sin(t1) + 2*L1*t1d*cos(p1)*cos(t1))*(L1*p1d*sin(t1)*cos(p1) + L1*t1d*sin(p1)*cos(t1)) + (-L1*p1d*sin(p1)*sin(t1) + L1*t1d*cos(p1)*cos(t1))*(-2*L1*p1d*sin(t1)*cos(p1) - 2*L1*t1d*sin(p1)*cos(t1)))/2 + m2*((-2*L1*p1d*sin(p1)*sin(t1) + 2*L1*t1d*cos(p1)*cos(t1))*(L1*p1d*sin(t1)*cos(p1) + L1*t1d*sin(p1)*cos(t1) + L2*p2d*sin(t2)*cos(p2) + L2*t2d*sin(p2)*cos(t2)) + (-2*L1*p1d*sin(t1)*cos(p1) - 2*L1*t1d*sin(p1)*cos(t1))*(-L1*p1d*sin(p1)*sin(t1) + L1*t1d*cos(p1)*cos(t1) - L2*p2d*sin(p2)*sin(t2) + L2*t2d*cos(p2)*cos(t2)))/2,
        -L2*m2*(-L1*p1d**2*sin(t1)*sin(t2)*sin(p1 - p2) + L1*p1d*p2d*sin(t1)*sin(t2)*sin(p1 - p2) + 2*L1*p1d*t1d*sin(t2)*cos(t1)*cos(p1 - p2) + L1*p1d*t2d*sin(t1)*cos(t2)*cos(p1 - p2) + L1*p1dd*sin(t1)*sin(t2)*cos(p1 - p2) - L1*p2d*t1d*sin(t2)*cos(t1)*cos(p1 - p2) - L1*t1d**2*sin(t1)*sin(t2)*sin(p1 - p2) + L1*t1d*t2d*sin(p1 - p2)*cos(t1)*cos(t2) + L1*t1dd*sin(t2)*sin(p1 - p2)*cos(t1) + 2*L2*p2d*t2d*sin(t2)*cos(t2) + L2*p2dd*sin(t2)**2) + m2*((-2*L2*p2d*sin(p2)*sin(t2) + 2*L2*t2d*cos(p2)*cos(t2))*(L1*p1d*sin(t1)*cos(p1) + L1*t1d*sin(p1)*cos(t1) + L2*p2d*sin(t2)*cos(p2) + L2*t2d*sin(p2)*cos(t2)) + (-2*L2*p2d*sin(t2)*cos(p2) - 2*L2*t2d*sin(p2)*cos(t2))*(-L1*p1d*sin(p1)*sin(t1) + L1*t1d*cos(p1)*cos(t1) - L2*p2d*sin(p2)*sin(t2) + L2*t2d*cos(p2)*cos(t2)))/2,
    ]

    return eqs, [t1dd, t2dd, p1dd, p2dd]


@linear_system_example
@hangs_time_with(linsolve_sympy)
def linear_system_2x2_func_derivs():
    """
    https://github.com/sympy/sympy/issues/22237

    >>> from examples import linear_system_2x2_func_derivs
    >>> eqs, syms = linear_system_2x2_func_derivs()
    >>> for eq in eqs: print(eq)
    K_t*V(t)/R - M*g*l*theta(t) + (-I_m/r + M*l)*Derivative(x(t), (t, 2)) + (K_t*K_v/R + b)*Derivative(theta(t), t) + (-K_t*K_v/(R*r) - b/r)*Derivative(x(t), t) + (I_m + J + M*l**2)*Derivative(theta(t), (t, 2))
    -K_t*V(t)/(R*r) + (-I_m/r + M*l)*Derivative(theta(t), (t, 2)) + (I/r**2 + I_m/r**2 + M + m)*Derivative(x(t), (t, 2)) - (K_t*K_v/R + b)*Derivative(theta(t), t)/r - (-K_t*K_v/(R*r) - b/r)*Derivative(x(t), t)/r
    >>> print(syms)
    [Derivative(theta(t), (t, 2)), Derivative(x(t), (t, 2))]

    This is a linear system of 2 equations for 2 unknowns. Because the
    coefficients mix functions and derivatives the EX domain gets used which is
    slow due to using cancel. Switching the order of the unknowns means that it
    gets solved quickly.
    """
    from sympy import symbols, Function, Derivative

    r, l, M, m, I, I_m, J = symbols('r l M m I I_m J')
    L, R, g, K_t, K_v, b, t = symbols('L R g K_t K_v b t')

    x, theta, V, i, tau_m, omega_m = symbols('x theta V i tau_m omega_m', cls=Function)

    eqs = [
        (K_t*V(t)/R - M*g*l*theta(t) + (-I_m/r + M*l)*Derivative(x(t), (t, 2))
         + (K_t*K_v/R + b)*Derivative(theta(t), t)
         + (-K_t*K_v/(R*r) - b/r)*Derivative(x(t), t)
         + (I_m + J + M*l**2)*Derivative(theta(t), (t, 2))),
        (-K_t*V(t)/(R*r) + (-I_m/r + M*l)*Derivative(theta(t), (t, 2))
         + (I/r**2 + I_m/r**2 + M + m)*Derivative(x(t), (t, 2))
         - (K_t*K_v/R + b)*Derivative(theta(t), t)/r
         - (-K_t*K_v/(R*r) - b/r)*Derivative(x(t), t)/r),
    ]
    return eqs, [theta(t).diff(t, 2), x(t).diff(t, 2)]


@linear_system_example
@slow_time_with(solve_sympy, time_seconds=25)
def linsys_poly_neg_powers():
    """
    https://github.com/sympy/sympy/issues/21694

    >>> from examples import linsys_poly_neg_powers
    >>> eqs, syms = linsys_poly_neg_powers()
    >>> for eq in eqs: print(eq)
    -pi**2*A_mn*m**2/l**2 + A_mn*n**2*nu/(2*R**2) - A_mn*n**2/(2*R**2) + pi*B_mn*m*n*nu/(2*R*l) + pi*B_mn*m*n/(2*R*l) + pi*C_mn*m*nu/(R*l) + D_mn/C
    pi*A_mn*m*n*nu/(2*R*l) + pi*A_mn*m*n/(2*R*l) + pi**2*B_mn*m**2*nu/(2*l**2) - pi**2*B_mn*m**2/(2*l**2) - B_mn*n**2/R**2 - pi**2*C_mn*h**2*m**2*n/(12*R**2*l**2) - C_mn*n/R**2 - C_mn*h**2*n**3/(12*R**4) + E_mn/C
    -pi*A_mn*m*nu/(R*l) + pi**2*B_mn*h**2*m**2*n/(12*R**2*l**2) + B_mn*n/R**2 + B_mn*h**2*n**3/(12*R**4) + pi**4*C_mn*h**2*m**4/(12*l**4) + pi**2*C_mn*h**2*m**2*n**2/(6*R**2*l**2) + C_mn/R**2 + C_mn*h**2*n**4/(12*R**4) - F_mn/C
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP

    This is a system of linear equations where the coefficients are effectively
    polynomials but some symbols appear as negative powers. This causes the
    domain to be a rational function field which then leads to slow gcd
    calculations. This could instead be handled by either using a Laurent
    polynomial ring or treating the denominator symbols as inverse generators.
    Division free methods could then be used to solve the system. Alternatively
    polynomial gcd should be made faster.
    """
    from sympy import symbols, pi
    ux,uy,uz,A_mn,B_mn,C_mn,n,x,y,a,m,b,C,nu,k1,k2,h,X,Y,Z,D_mn,E_mn,F_mn,E,l,beta,R,beta_0 = symbols("ux,uy,uz,A_mn,B_mn,C_mn,n,x,y,a,m,b,C,nu,k1,k2,h,X,Y,Z,D_mn,E_mn,F_mn,E,l,beta,R,beta_0")
    eq1 = -pi**2*A_mn*m**2/l**2 + A_mn*n**2*nu/(2*R**2) - A_mn*n**2/(2*R**2) + pi*B_mn*m*n*nu/(2*R*l) + pi*B_mn*m*n/(2*R*l) + pi*C_mn*m*nu/(R*l) + D_mn/C
    eq2 = pi*A_mn*m*n*nu/(2*R*l) + pi*A_mn*m*n/(2*R*l) + pi**2*B_mn*m**2*nu/(2*l**2) - pi**2*B_mn*m**2/(2*l**2) - B_mn*n**2/R**2 - pi**2*C_mn*h**2*m**2*n/(12*R**2*l**2) - C_mn*n/R**2 - C_mn*h**2*n**3/(12*R**4) + E_mn/C
    eq3 = -pi*A_mn*m*nu/(R*l) + pi**2*B_mn*h**2*m**2*n/(12*R**2*l**2) + B_mn*n/R**2 + B_mn*h**2*n**3/(12*R**4) + pi**4*C_mn*h**2*m**4/(12*l**4) + pi**2*C_mn*h**2*m**2*n**2/(6*R**2*l**2) + C_mn/R**2 + C_mn*h**2*n**4/(12*R**4) - F_mn/C
    return [eq1, eq2, eq3], [A_mn, B_mn, C_mn]



@poly_system_example
@fast_time_with(solve_sympy)
def poly_system_2x2_example():
    """
    https://github.com/sympy/sympy/issues/24060

    >>> from examples import poly_system_2x2_example
    >>> eqs, syms = poly_system_2x2_example()
    >>> for eq in eqs: print(eq)
    Eq(x*y**90, 33)
    Eq(x*y**92, 66)
    >>> from sympy import solve
    >>> solve(eqs, syms)
    [(33/35184372088832, -sqrt(2)), (33/35184372088832, sqrt(2))]

    This is equivalent to nonlin_to_poly_example except it is solved much
    faster.
    """
    from sympy import symbols, Eq
    x, y = symbols('x, y', real=True)
    eq1 = Eq(x*y**90, 33)
    eq2 = Eq(x*y**92, 66)
    return [eq1, eq2], [x, y]


@system_example
@hangs_time_with(solve_sympy)
def nonlin_to_poly_example():
    """
    https://github.com/sympy/sympy/issues/24060

    >>> from examples import nonlin_to_poly_example
    >>> eqs, syms = nonlin_to_poly_example()
    >>> for eq in eqs: print(eq)
    Eq(a*exp(-90*b), 33)
    Eq(a*exp(-92*b), 66)
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP

    This example is very slow in solve because of slow checking. It should be
    possible to avoid doing the checking though.
    """
    from sympy import symbols, Eq, exp
    a, b = symbols('a, b', real=True)
    eq1 = Eq(a*exp(-b*90),33)
    eq2 = Eq(a*exp(-b*92),66)

    return [eq1, eq2], [a, b]


@system_example
@slow_time_with(solve_sympy, time_seconds=120)
def spherical_sincos_to_poly_example():
    """
    https://github.com/sympy/sympy/issues/23138

    >>> from examples import spherical_sincos_to_poly_example
    >>> eqs, syms = spherical_sincos_to_poly_example()
    >>> for eq in eqs: print(eq)
    r*sin(t)*cos(p) - x
    r*sin(p)*sin(t) - y
    r*cos(t) - z
    >>> syms
    [r, t, p]
    >>> solve(eqs, syms) # doctest: +SKIP

    This should be easy to solve by transforming to a polynomial system.

    See spherical_sincos_as_poly_example for the version as a polynomial
    system.
    """
    from sympy import symbols, sin, cos
    x, y, z, t, p = symbols(r"x y z t p", real=True)
    r = symbols("r", real=True, positive=True)
    eq = [r*cos(p)*sin(t) - x, r*sin(p)*sin(t) - y, r*cos(t) - z]
    return eq, [r, t, p]


@system_example
@fast_time_with(solve_sympy)
def spherical_sincos_as_poly_example():
    """
    https://github.com/sympy/sympy/issues/23138

    See spherical_sincos_to_poly_example for the version with trig functions.

    >>> from examples import spherical_sincos_as_poly_example
    >>> eqs, syms = spherical_sincos_as_poly_example()
    >>> for eq in eqs: print(eq)
    cp*r*st - x
    r*sp*st - y
    ct*r - z
    ct**2 + st**2 - 1
    cp**2 + sp**2 - 1
    >>> syms
    [r, st, ct, sp, cp]
    >>> solve(eqs, syms) # doctest: +SKIP
    """
    from sympy import symbols, sin, cos
    x, y, z, st, ct, sp, cp = symbols(r"x y z st, ct, sp, cp", real=True)
    r = symbols("r", real=True, positive=True)
    eq = [r*cp*st - x, r*sp*st - y, r*ct - z, st**2 + ct**2 - 1, sp**2 + cp**2 - 1]
    return eq, [r, st, ct, sp, cp]


@system_example
@slow_time_with(nonlinsolve_sympy, time_seconds=170)
def system_algebraic_ratfunc():
    """
    https://github.com/sympy/sympy/issues/23007

    It should be possible to solve this much more quickly. The system should be
    transformed to polynomial and an extension field should be used for the
    algebraic number expressions.

    >>> from examples import system_algebraic_ratfunc
    >>> eqs, syms = system_algebraic_ratfunc()
    >>> for eq in eqs: print(eq)
    -y - (-x + cos(2*pi/7))**2/(-4*y + 4*z) + sin(2*pi/7)
    -y - (-x - cos(3*pi/7))**2/(-4*y + 4*z) + sin(3*pi/7)
    -y - (-x - cos(pi/7))**2/(-4*y + 4*z) - sin(pi/7)
    >>> syms
    [x, y, z]
    >>> from sympy import nonlinsolve
    >>> nonlinsolve(eqs, syms) # doctest: +SKIP
    """
    from sympy import sin, cos, pi, symbols
    x, y, z = symbols('x, y, z')
    eqs = [
        sin(2*pi/7) - y - (-x + cos(2*pi/7))**2/(4*z - 4*y),
        sin(3*pi/7) - y -(-x - cos(3*pi/7))**2/(4*z - 4*y),
        -sin(pi/7) - y - (-x - cos(pi/7))**2/(4*z - 4*y),
    ]
    return eqs, [x, y, z]


@system_example
@slow_time_with(solve_sympy, time_seconds=200)
def log_polysys():
    """
    https://github.com/sympy/sympy/issues/23662

    This example should be something that solve can handle efficiently. Firstly
    solve does not recognise that all of the logs are basically removable
    (maybe we need to be careful about factors of 2*pi*I). Then it should be
    possible for solve to handle the remaining polynomial system efficiently
    but it does not. See poly_sys_parametric for the version of this system
    without the log functions.

    >>> from examples import log_polysys
    >>> eqs, syms = log_polysys()
    >>> for eq in eqs: print(eq)
    Eq(w, -x - y - z + 1.0)
    Eq(log(a*e*w**2), log(x*z))
    Eq(log(b*w*x), log(y*z))
    Eq(log(c*y), log(d*w))
    >>> syms
    [x, y, z, w]
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP
    """
    from sympy import symbols, Eq, log
    x, y, z, w = symbols('x, y, z, w')
    a, b, c, d, e = symbols('a, b, c, d, e')
    variables = [x, y, z, w]
    eqs = [
        Eq(w, -x - y - z + 1.0),
        Eq(log(a*e*w**2), log(x*z)),
        Eq(log(b*w*x), log(y*z)),
        Eq(log(c*y), log(d*w)),
    ]
    return eqs, variables


@poly_system_example
@slow_time_with(solve_sympy, time_seconds=100)
def poly_sys_parametric():
    """
    https://github.com/sympy/sympy/issues/23662

    Like log_polysys but without the logs. If solve was cleverer about
    factorising the Groebner basis then this would not be slow.

    >>> from examples import log_polysys
    >>> eqs, syms = poly_sys_parametric()
    >>> for eq in eqs: print(eq)
    Eq(w, -x - y - z + 1.0)
    Eq(a*e*w**2, x*z)
    Eq(b*w*x, y*z)
    Eq(c*y, d*w)
    >>> syms
    [x, y, z, w]
    >>> from sympy import solve
    >>> solve(eqs, syms) # doctest: +SKIP
    """
    from sympy import symbols, Eq
    x, y, z, w = symbols('x, y, z, w')
    a, b, c, d, e = symbols('a, b, c, d, e')
    variables = [x, y, z, w]
    eqs = [
        Eq(w, -x - y - z + 1.0),
        Eq(a*e*w**2, x*z),
        Eq(b*w*x, y*z),
        Eq(c*y, d*w),
    ]
    return eqs, variables


def _linear_system_sincos():
    """
    https://github.com/sympy/sympy/issues/24126
    """
    # not sure whether to include this as it is quite long



@function_example
@slow_time_with(call_function, time_seconds=50)
def build_cospi257_example():
    """
    https://github.com/sympy/sympy/issues/24565

    This example is slow because there are many calls to evalf and evalf is
    slow for large numerc expressions with many repeating subexpressions.
    """
    from sympy import sqrt, S

    def build():
        def f1(a, b):
            return (a + sqrt(a**2 + b))/2, (a - sqrt(a**2 + b))/2

        def f2(a, b):
            return (a - sqrt(a**2 + b))/2

        t1, t2 = f1(-1, 256)
        z1, z3 = f1(t1, 64)
        z2, z4 = f1(t2, 64)
        y1, y5 = f1(z1, 4*(5 + t1 + 2*z1))
        y6, y2 = f1(z2, 4*(5 + t2 + 2*z2))
        y3, y7 = f1(z3, 4*(5 + t1 + 2*z3))
        y8, y4 = f1(z4, 4*(5 + t2 + 2*z4))

        x1, x9 = f1(y1, -4*(t1 + y1 + y3 + 2*y6))
        x2, x10 = f1(y2, -4*(t2 + y2 + y4 + 2*y7))
        x3, x11 = f1(y3, -4*(t1 + y3 + y5 + 2*y8))
        x4, x12 = f1(y4, -4*(t2 + y4 + y6 + 2*y1))
        x5, x13 = f1(y5, -4*(t1 + y5 + y7 + 2*y2))
        x6, x14 = f1(y6, -4*(t2 + y6 + y8 + 2*y3))
        x15, x7 = f1(y7, -4*(t1 + y7 + y1 + 2*y4))
        x8, x16 = f1(y8, -4*(t2 + y8 + y2 + 2*y5))

        v1 = f2(x1, -4*(x1 + x2 + x3 + x6))
        v2 = f2(x2, -4*(x2 + x3 + x4 + x7))
        v3 = f2(x8, -4*(x8 + x9 + x10 + x13))
        v4 = f2(x9, -4*(x9 + x10 + x11 + x14))
        v5 = f2(x10, -4*(x10 + x11 + x12 + x15))
        v6 = f2(x16, -4*(x16 + x1 + x2 + x5))
        u1 = -f2(-v1, -4*(v2 + v3))
        u2 = -f2(-v4, -4*(v5 + v6))
        w1 = -2*f2(-u1, -4*u2)

        return sqrt(sqrt(2)*sqrt(w1 + 4)/8 + S.Half)

    return build


@function_example
@hangs_time_with(call_function)
def build_rational_power():
    """
    https://github.com/sympy/sympy/issues/23399

    This attempts to compute a rational number to the power of a rational
    number but resolving this explicitly leads to an enormous numerator that
    cannot possibly be represented. There should be some fallback here to
    prevent explicit evaluation.

        x = Rational(-111503130647661, 10000000000000000)
        y = Rational(100514145917401, 250000000000000)
        x**y
    """
    from sympy import Rational

    def build():
        x = Rational(-111503130647661, 10000000000000000)
        y = Rational(100514145917401, 250000000000000)
        return x**y

    return build


@function_example
@hangs_time_with(call_function)
def build_exp_large_log():
    """
    https://github.com/sympy/sympy/issues/22248

    Evaluating this expression attempts to generate an enormous integer:

    exp(-672373333333333 + 279040000000000*log(572373333333333))
    """
    from sympy import exp, log

    def build():
        return exp(-672373333333333 + 279040000000000*log(572373333333333))

    return build


@function_example
@hangs_time_with(call_function)
def build_large_float_from_string():
    """
    https://github.com/sympy/sympy/issues/21860

    Creating this float is very slow:

        Float('-1.23516983720720999822e+472802')
    """
    def build():
        return Float('-1.23516983720720999822e+472802')

    return build


@expression_example
@slow_time_with(evalf_sympy, time_seconds=10)
def cospi257_example():
    """
    Same expression as build_cospi257_example

    Here it is returned so that other operations can be timed with it.

    >>> from examples import cospi257_example
    >>> expr = cospi257_example() # doctest: +SKIP
    >>> expr.count_ops() # doctest: +SKIP
    16088
    >>> expr.evalf() # doctest: +SKIP
    0.999925286669733
    """
    builder = build_cospi257_example()
    return builder()


@expression_example
@slow_time_with(evalf_sympy, time_seconds=40)
def expression_wallis_product():
    """
    https://github.com/sympy/sympy/issues/20461

    >>> from examples import expression_wallis_product
    >>> zero = expression_wallis_product()
    >>> zero
    -pi/2 + Product(4*n**2/(4*n**2 - 1), (n, 1, oo))
    >>> zero.evalf() # doctest: +SKIP

    Numerically evaluating the product is slow and since the expression is
    actually equal to zero it will use high precision for the product. The
    result also comes out incorrectly as nonzero.
    """
    from sympy import Product, symbols, pi, oo
    n = symbols('n')
    zero = Product(4*n**2/(4*n**2 - 1), (n, 1, oo)) - pi/2
    return zero


@expression_example
@hangs_time_with(simplify_sympy)
def expression_cos_poly():
    """
    https://github.com/sympy/sympy/issues/21787

    >>> from examples import expression_cos_poly
    >>> e = expression_cos_poly()
    >>> e
    40*(x + 5)**3*cos(10*(x + 5)**4)
    >>> e.simplify() # doctest: +SKIP
    """
    from sympy import symbols, cos
    x = symbols('x')
    return 40*(x + 5)**3*cos(10*(x + 5)**4)


@subs_example
@fast_time_with(subs_sympy)
def substitute_poly_coefficients_large():
    """
    >>> from examples import substitute_poly_coefficients_large
    >>> expr, rep = substitute_poly_coefficients_large()
    >>> s = str(expr)
    >>> print(s[:50], '...') # doctest: +SKIP
    uk_12*x**2*z + uk_14*x*y*z + uk_15*x*z**2 + uk_17* ...
    >>> print(str(rep)[:50], '...') # doctest: +SKIP
    {uk_0: 1, uk_1: 0, uk_2: 0, uk_3: 0, uk_4: 0, uk_5 ...
    >>> print(len(s), len(rep)) # doctest: +SKIP
    3031 84
    >>> expr.subs(rep) # doctest: +SKIP
    0
    """
    from sympy import symbols

    x, y, z = symbols('x, y, z')
    [uk_0, uk_1, uk_2, uk_3, uk_4, uk_5, uk_6, uk_7, uk_8, uk_9, uk_10, uk_11,
     uk_12, uk_13, uk_14, uk_15, uk_16, uk_17, uk_18, uk_19, uk_20, uk_21,
     uk_22, uk_23, uk_24, uk_25, uk_26, uk_27, uk_28, uk_29, uk_30, uk_31,
     uk_32, uk_33, uk_34, uk_35, uk_36, uk_37, uk_38, uk_39, uk_40, uk_41,
     uk_42, uk_43, uk_44, uk_45, uk_46, uk_47, uk_48, uk_49, uk_50, uk_51,
     uk_52, uk_53, uk_54, uk_55, uk_56, uk_57, uk_58, uk_59, uk_60, uk_61,
     uk_62, uk_63, uk_64, uk_65, uk_66, uk_67, uk_68, uk_69, uk_70, uk_71,
     uk_72, uk_73, uk_74, uk_75, uk_76, uk_77, uk_78, uk_79, uk_80, uk_81,
     uk_82, uk_83 ] = uk_vs = symbols('uk_:84')

    eqt = (uk_12*x**2*z + uk_14*x*y*z + uk_15*x*z**2 + uk_17*y**2*z + uk_18*y*z**2
        + uk_19*z**3 + uk_22*x**3*z + uk_24*x**2*y*z + uk_25*x**2*z**2 +
        uk_27*x*y**2*z + uk_28*x*y*z**2 + uk_29*x*z**3 - 5*uk_3*x - uk_3*y +
        uk_3*z + uk_31*y**3*z + uk_32*y**2*z**2 + uk_33*y*z**3 + uk_34*z**4 +
        uk_37*x**4*z + uk_39*x**3*y*z + uk_40*x**3*z**2 + uk_42*x**2*y**2*z +
        uk_43*x**2*y*z**2 + uk_44*x**2*z**3 + uk_46*x*y**3*z +
        uk_47*x*y**2*z**2 + uk_48*x*y*z**3 + uk_49*x*z**4 + uk_51*y**4*z +
        uk_52*y**3*z**2 + uk_53*y**2*z**3 + uk_54*y*z**4 + uk_55*z**5 +
        uk_58*x**5*z + uk_6*x*z + uk_60*x**4*y*z + uk_61*x**4*z**2 +
        uk_63*x**3*y**2*z + uk_64*x**3*y*z**2 + uk_65*x**3*z**3 +
        uk_67*x**2*y**3*z + uk_68*x**2*y**2*z**2 + uk_69*x**2*y*z**3 +
        uk_70*x**2*z**4 + uk_72*x*y**4*z + uk_73*x*y**3*z**2 +
        uk_74*x*y**2*z**3 + uk_75*x*y*z**4 + uk_76*x*z**5 + uk_78*y**5*z +
        uk_79*y**4*z**2 + uk_8*y*z + uk_80*y**3*z**3 + uk_81*y**2*z**4 +
        uk_82*y*z**5 + uk_83*z**6 + uk_9*z**2 + x**6*(-5*uk_58 - 25*uk_61 -
        125*uk_65 - 625*uk_70 - 3125*uk_76 - 15625*uk_83) + x**5*y*(-uk_58
        - 5*uk_60 - 10*uk_61 - 25*uk_64 - 75*uk_65 - 125*uk_69 -
        500*uk_70 - 625*uk_75 - 3125*uk_76 - 3125*uk_82 - 18750*uk_83)
        + x**5*(-5*uk_37 - 25*uk_40 - 125*uk_44 - 625*uk_49 - 3125*uk_55) +
        x**4*y**2*(-uk_60 - uk_61 - 5*uk_63 - 10*uk_64 - 15*uk_65 -
        25*uk_68 - 75*uk_69 - 150*uk_70 - 125*uk_74 - 500*uk_75 -
        1250*uk_76 - 625*uk_81 - 3125*uk_82 - 9375*uk_83) +
        x**4*y*(-uk_37 - 5*uk_39 - 10*uk_40 - 25*uk_43 - 75*uk_44 -
        125*uk_48 - 500*uk_49 - 625*uk_54 - 3125*uk_55) +
        x**4*(-5*uk_22 - 25*uk_25 - 125*uk_29 - 625*uk_34) +
        x**3*y**3*(-uk_63 - uk_64 - uk_65 - 5*uk_67 - 10*uk_68 - 15*uk_69 -
        20*uk_70 - 25*uk_73 - 75*uk_74 - 150*uk_75 - 250*uk_76 -
        125*uk_80 - 500*uk_81 - 1250*uk_82 - 2500*uk_83) + x**3*y**2*(-uk_39 -
        uk_40 - 5*uk_42 - 10*uk_43 - 15*uk_44 - 25*uk_47 - 75*uk_48 -
        150*uk_49 - 125*uk_53 - 500*uk_54 - 1250*uk_55) + x**3*y*(-uk_22 -
        5*uk_24 - 10*uk_25 - 25*uk_28 - 75*uk_29 - 125*uk_33 - 500*uk_34) +
        x**3*(-5*uk_12 - 25*uk_15 - 125*uk_19) + x**2*y**4*(-uk_67 - uk_68 -
        uk_69 - uk_70 - 5*uk_72 - 10*uk_73 - 15*uk_74 - 20*uk_75 - 25*uk_76
        - 25*uk_79 - 75*uk_80 - 150*uk_81 - 250*uk_82 - 375*uk_83) +
        x**2*y**3*(-uk_42 - uk_43 - uk_44 - 5*uk_46 - 10*uk_47
        - 15*uk_48 - 20*uk_49 - 25*uk_52 - 75*uk_53 - 150*uk_54 - 250*uk_55) +
        x**2*y**2*(-uk_24 - uk_25 - 5*uk_27 - 10*uk_28 - 15*uk_29 - 25*uk_32 -
        75*uk_33 - 150*uk_34) + x**2*y*(-uk_12 - 5*uk_14 - 10*uk_15 - 25*uk_18
        - 75*uk_19) + x**2*(-5*uk_6 - 25*uk_9) + x*y**5*(-uk_72 - uk_73 - uk_74
        - uk_75 - uk_76 - 5*uk_78 - 10*uk_79 - 15*uk_80 - 20*uk_81 - 25*uk_82 -
        30*uk_83) + x*y**4*(-uk_46 - uk_47 - uk_48
        - uk_49 - 5*uk_51 - 10*uk_52 - 15*uk_53
        - 20*uk_54 - 25*uk_55) + x*y**3*(-uk_27
        - uk_28 - uk_29 - 5*uk_31 - 10*uk_32 - 15*uk_33 - 20*uk_34) +
        x*y**2*(-uk_14 - uk_15 - 5*uk_17 - 10*uk_18 - 15*uk_19) + x*y*(-uk_6 -
        5*uk_8 - 10*uk_9) + y**6*(-uk_78 - uk_79 - uk_80 - uk_81 - uk_82 -
        uk_83) + y**5*(-uk_51 - uk_52 - uk_53 - uk_54 - uk_55) + y**4*(-uk_31 -
        uk_32 - uk_33 - uk_34) + y**3*(-uk_17 - uk_18 - uk_19) + y**2*(-uk_8 -
        uk_9))

    rep = {uk_i: 0 for uk_i in uk_vs}
    rep[uk_vs[0]] = 1

    return eqt, rep


@rational_function_example
@slow_time_with(poly_cancel_sympy, time_seconds=20)
@hangs_time_with(cancel_sympy)
def multi_ratfunc_example():
    """
    https://github.com/sympy/sympy/issues/24537

    >>> from examples import multi_ratfunc_example
    >>> expr = multi_ratfunc_example()
    >>> expr.is_rational_function()
    True
    >>> len(str(expr))
    3690
    >>> expr.cancel() # doctest: +SKIP

    Calling cancel on this expression is slow because it tries to expand
    everything using Expr rather than converting to poly first.
    """
    from sympy import symbols

    C_a, C_A, C_L, C_s, C_c, C_C, C_B = symbols('C_a C_A C_L C_s C_c C_C C_B')
    G_m1, G_m2, G_m3, G_m4 = symbols('G_m1 G_m2 G_m3 G_m4')
    R_A, R_B, R_C, R_L = symbols('R_A R_B R_C R_L')
    s = symbols('s')

    expr = (C_a*G_m1*R_A*R_L*s*(G_m3*R_C/(s*(C_C*R_C + C_s*R_C) + 1) +
        1)/(s**2*(C_A*C_L*R_A*R_L + C_A*C_a*R_A*R_L + C_A*C_s*R_A*R_L +
        C_L*C_a*R_A*R_L + C_a**2*R_A*R_L + C_a*C_s*R_A*R_L) + s*(C_A*R_A +
        C_L*R_L + C_a*R_A + C_a*R_L + C_s*R_L) + 1) -
        G_m1*G_m2*G_m4*R_A*R_B*R_L*(G_m3*R_C/(s*(C_C*R_C + C_s*R_C) + 1) +
        1)/(s**3*(C_A*C_B*C_L*R_A*R_B*R_L + C_A*C_B*C_a*R_A*R_B*R_L +
        C_A*C_B*C_s*R_A*R_B*R_L + C_B*C_L*C_a*R_A*R_B*R_L +
        C_B*C_a**2*R_A*R_B*R_L + C_B*C_a*C_s*R_A*R_B*R_L) +
        s**2*(C_A*C_B*R_A*R_B + C_A*C_L*R_A*R_L + C_A*C_a*R_A*R_L +
        C_A*C_s*R_A*R_L + C_B*C_L*R_B*R_L + C_B*C_a*R_A*R_B + C_B*C_a*R_B*R_L
        + C_B*C_s*R_B*R_L + C_L*C_a*R_A*R_L + C_a**2*R_A*R_L +
        C_a*C_s*R_A*R_L) + s*(C_A*R_A + C_B*R_B + C_L*R_L + C_a*R_A + C_a*R_L
        + C_s*R_L) + 1))/(-C_a**2*G_m3*R_A*R_C*R_L*s**2/((s*(C_C*R_C +
        C_s*R_C) + 1)*(s**2*(C_A*C_L*R_A*R_L + C_A*C_a*R_A*R_L +
        C_A*C_s*R_A*R_L + C_L*C_a*R_A*R_L + C_a**2*R_A*R_L + C_a*C_s*R_A*R_L)
        + s*(C_A*R_A + C_L*R_L + C_a*R_A + C_a*R_L + C_s*R_L) + 1)) -
        C_a**2*R_A*R_L*s**2/(s**2*(C_A*C_L*R_A*R_L + C_A*C_a*R_A*R_L +
        C_A*C_s*R_A*R_L + C_L*C_a*R_A*R_L + C_a**2*R_A*R_L + C_a*C_s*R_A*R_L)
        + s*(C_A*R_A + C_L*R_L + C_a*R_A + C_a*R_L + C_s*R_L) + 1) +
        C_a*G_m2*G_m3*G_m4*R_A*R_B*R_C*R_L*s/((s*(C_C*R_C + C_s*R_C) +
        1)*(s**3*(C_A*C_B*C_L*R_A*R_B*R_L + C_A*C_B*C_a*R_A*R_B*R_L +
        C_A*C_B*C_s*R_A*R_B*R_L + C_B*C_L*C_a*R_A*R_B*R_L +
        C_B*C_a**2*R_A*R_B*R_L + C_B*C_a*C_s*R_A*R_B*R_L) +
        s**2*(C_A*C_B*R_A*R_B + C_A*C_L*R_A*R_L + C_A*C_a*R_A*R_L +
        C_A*C_s*R_A*R_L + C_B*C_L*R_B*R_L + C_B*C_a*R_A*R_B + C_B*C_a*R_B*R_L
        + C_B*C_s*R_B*R_L + C_L*C_a*R_A*R_L + C_a**2*R_A*R_L +
        C_a*C_s*R_A*R_L) + s*(C_A*R_A + C_B*R_B + C_L*R_L + C_a*R_A + C_a*R_L
        + C_s*R_L) + 1)) +
        C_a*G_m2*G_m4*R_A*R_B*R_L*s/(s**3*(C_A*C_B*C_L*R_A*R_B*R_L +
        C_A*C_B*C_a*R_A*R_B*R_L + C_A*C_B*C_s*R_A*R_B*R_L +
        C_B*C_L*C_a*R_A*R_B*R_L + C_B*C_a**2*R_A*R_B*R_L +
        C_B*C_a*C_s*R_A*R_B*R_L) + s**2*(C_A*C_B*R_A*R_B + C_A*C_L*R_A*R_L +
        C_A*C_a*R_A*R_L + C_A*C_s*R_A*R_L + C_B*C_L*R_B*R_L + C_B*C_a*R_A*R_B
        + C_B*C_a*R_B*R_L + C_B*C_s*R_B*R_L + C_L*C_a*R_A*R_L + C_a**2*R_A*R_L
        + C_a*C_s*R_A*R_L) + s*(C_A*R_A + C_B*R_B + C_L*R_L + C_a*R_A +
        C_a*R_L + C_s*R_L) + 1) - C_s**2*G_m2*R_A*R_C*R_L*s**2/((s*(C_A*R_A +
        C_a*R_A) + 1)*(s**2*(C_C*C_L*R_C*R_L + C_C*C_a*R_C*R_L +
        C_C*C_s*R_C*R_L + C_L*C_s*R_C*R_L + C_a*C_s*R_C*R_L + C_s**2*R_C*R_L)
        + s*(C_C*R_C + C_L*R_L + C_a*R_L + C_s*R_C + C_s*R_L) + 1)) -
        C_s**2*R_C*R_L*s**2/(s**2*(C_C*C_L*R_C*R_L + C_C*C_a*R_C*R_L +
        C_C*C_s*R_C*R_L + C_L*C_s*R_C*R_L + C_a*C_s*R_C*R_L + C_s**2*R_C*R_L)
        + s*(C_C*R_C + C_L*R_L + C_a*R_L + C_s*R_C + C_s*R_L) + 1) +
        C_s*G_m2*G_m3*G_m4*R_A*R_B*R_C*R_L*s/((s*(C_A*R_A + C_a*R_A) +
        1)*(s**3*(C_B*C_C*C_L*R_B*R_C*R_L + C_B*C_C*C_a*R_B*R_C*R_L +
        C_B*C_C*C_s*R_B*R_C*R_L + C_B*C_L*C_s*R_B*R_C*R_L +
        C_B*C_a*C_s*R_B*R_C*R_L + C_B*C_s**2*R_B*R_C*R_L) +
        s**2*(C_B*C_C*R_B*R_C + C_B*C_L*R_B*R_L + C_B*C_a*R_B*R_L +
        C_B*C_s*R_B*R_C + C_B*C_s*R_B*R_L + C_C*C_L*R_C*R_L + C_C*C_a*R_C*R_L
        + C_C*C_s*R_C*R_L + C_L*C_s*R_C*R_L + C_a*C_s*R_C*R_L +
        C_s**2*R_C*R_L) + s*(C_B*R_B + C_C*R_C + C_L*R_L + C_a*R_L + C_s*R_C +
        C_s*R_L) + 1)) +
        C_s*G_m3*G_m4*R_B*R_C*R_L*s/(s**3*(C_B*C_C*C_L*R_B*R_C*R_L +
        C_B*C_C*C_a*R_B*R_C*R_L + C_B*C_C*C_s*R_B*R_C*R_L +
        C_B*C_L*C_s*R_B*R_C*R_L + C_B*C_a*C_s*R_B*R_C*R_L +
        C_B*C_s**2*R_B*R_C*R_L) + s**2*(C_B*C_C*R_B*R_C + C_B*C_L*R_B*R_L +
        C_B*C_a*R_B*R_L + C_B*C_s*R_B*R_C + C_B*C_s*R_B*R_L + C_C*C_L*R_C*R_L
        + C_C*C_a*R_C*R_L + C_C*C_s*R_C*R_L + C_L*C_s*R_C*R_L +
        C_a*C_s*R_C*R_L + C_s**2*R_C*R_L) + s*(C_B*R_B + C_C*R_C + C_L*R_L +
        C_a*R_L + C_s*R_C + C_s*R_L) + 1) + G_m2*G_m3*R_A*R_C/((s*(C_A*R_A +
        C_a*R_A) + 1)*(s*(C_C*R_C + C_s*R_C) + 1)) + G_m2*R_A/(s*(C_A*R_A +
        C_a*R_A) + 1) + G_m3*R_C/(s*(C_C*R_C + C_s*R_C) + 1) + 1)

    return expr


@poly_example
@hangs_time_with(solveset_sympy)
def poly_deg12_ZZ():
    """
    https://github.com/sympy/sympy/issues/23211

    >>> from examples import poly_deg12_ZZ
    >>> p, x = poly_deg12_ZZ()
    >>> p
    x**12 - 12*x**10 + 60*x**8 - 160*x**6 - 192*x**4 + 1536*x**2 + 64
    >>> from sympy import solveset
    >>> solveset(p, x) # doctest: +SKIP

    This is slow because it is possible to decompose the polynomial into parts
    of degree less than 4:

    >>> from sympy import decompose
    >>> decompose(p)
    [x**3 + 12*x**2 - 384*x + 64, x**2 - 4*x, x**2]

    It is therefore possible to express the roots in radicals but the resulting
    expressions are huge. It would be better to use RootOf.
    """
    from sympy import symbols
    x = symbols('x')
    p = x**12 - 12*x**10 + 60*x**8 - 160*x**6 - 192*x**4 + 1536*x**2 + 64
    return p, x


@poly_example
@hangs_time_with(solveset_sympy)
def poly_deg4_ZZ_I():
    """
    https://github.com/sympy/sympy/issues/23163

    >>> from examples import poly_deg4_ZZ_I
    >>> p, x = poly_deg4_ZZ_I()
    >>> p
    -I*x**4 - 2*x**3 - 2*I*x**3 - 2*x + 2*I*x + I
    >>> solveset(p, x) # doctest: +SKIP

    This is slow because it uses radical formulae. It would be better just to
    use RootOf:

    >>> from sympy import symbols, resultant, Poly, I, minpoly
    >>> y = symbols('y')
    >>> p2 = resultant(p.subs(I, y), minpoly(I, y), y)
    >>> p2
    x**8 + 4*x**7 + 8*x**6 - 4*x**5 - 2*x**4 - 4*x**3 + 8*x**2 + 4*x + 1
    >>> Poly(p2).all_roots() # doctest: +SKIP

    More is needed to determine which roots of p2 are roots of p1.
    """
    from sympy import symbols, I
    x = symbols('x')
    p = -I*x**4 - 2*x**3 - 2*I*x**3 - 2*x + 2*I*x + I
    return p, x


@poly_pair_example
@fast_time_with(polydiv_sympy)
def poly_deg2016_div():
    """
    https://github.com/sympy/sympy/issues/21760

    >>> from examples import poly_deg2016_div
    >>> p1, p2 = poly_deg2016_div()
    >>> p1
    x**2016 - x**2015 + x**1008 + x**1003 + 1
    >>> p2
    x - 1
    >>> from sympy import div
    >>> div(p1, p2) # doctest: +SKIP
    """
    from sympy import symbols
    x = symbols('x')
    p1 = x**2016 - x**2015 + x**1008 + x**1003 + 1
    p2 = x - 1
    return p1, p2


@equation_example
@hangs_time_with(solve_sympy)
def equation_rational_power():
    """
    https://github.com/sympy/sympy/issues/22151

    >>> from examples import equation_rational_power
    >>> eq, sym = equation_rational_power()
    >>> eq
    -L + M**(71/13)/Z**(14/13)
    >>> sym
    M
    >>> from sympy import solve
    >>> solve(eq, sym) # doctest: +SKIP

    This equation has 71 solutions and solve is slow at checking them. It is
    faster with check=False but it should not be necessary to do any checking
    for this particular equation.
    """
    from sympy import symbols, Rational
    L = symbols('L')
    M = symbols('M')
    Z = symbols('Z')
    expr = M**(Rational(71,13)) * Z**(-Rational(14,13)) - L
    return (expr, M)


@equation_example
@hangs_time_with(solveset_real_sympy)
def equation_sin_tan():
    """
    https://github.com/sympy/sympy/issues/23163

    This equation should not be hard to solve:

    >>> from examples import equation_sin_tan
    >>> eq, x = equation_sin_tan()
    >>> eq
    Eq(sin(x) + tan(x), 1)
    >>> from sympy import solveset, S
    >>> solveset(eq, domain=S.Reals) # doctest: +SKIP

    Internally this hangs because solveset tries to solve a quartic with large
    radical formulae (see poly_deg4_ZZ_I).
    """
    from sympy import symbols, Eq, solveset, S, sin, tan
    x = symbols('x')
    eq = Eq(sin(x) + tan(x), 1)
    return eq, x


@equation_example
@hangs_time_with(solve_sympy)
def equation_large_power():
    """
    https://github.com/sympy/sympy/issues/21771

    >>> from examples import equation_large_power
    >>> eq, sym = equation_large_power()
    >>> eq
    Eq((59.0 - 59.0/(y/2 + 1)**16)/(sqrt((y/2 + 1)**2) - 1) + 1000/(y/2 + 1)**16, 779.92)
    >>> from sympy import solve
    >>> solve(eq, sym) # doctest: +SKIP

    This is slow because it generates a large polynomial and then tries to find
    all complex roots of the polynomial. The user probably only wants real
    roots and those can be found much faster. Also the root isolation code can
    be made faster.
    """
    from sympy import symbols, Eq, sqrt
    y = symbols('y')
    eq = Eq(59.0*(1 - 1/(y/2 + 1)**16)/(sqrt((y/2 + 1)**2) - 1) + 1000/(y/2 + 1)**16, 779.92)
    return eq, y


@equation_example
@hangs_time_with(solve_sympy)
def equation_float_exponents():
    """
    https://github.com/sympy/sympy/issues/21445

    >>> from examples import equation_float_exponents
    >>> eq, sym = equation_float_exponents()
    >>> eq
    [x - 15 + 24.74*exp(-0.282833333333333*I*pi) + 12*exp(0.795166666666667*I*pi)]
    >>> from sympy import solve
    >>> solve(eq, sym) # doctest: +SKIP
    """
    from sympy import symbols, exp, I, pi
    x = symbols('x')
    eq = x - 15 + 24.74*exp(-0.282833333333333*I*pi) + 12*exp(0.795166666666667*I*pi)
    return [eq], x # hangs if the equation is in a list.


@integral_example
@hangs_time_with(integrate_sympy)
def integrate_ode_example():
    """
    https://github.com/sympy/sympy/issues/24518

    >>> from examples import integrate_ode_example
    >>> i = integrate_ode_example()
    >>> i
    Integral(exp(-a*b*(e - t))*sin(c*t)*sin(d*(-e + t)), (t, 0, e))
    >>> i.doit() # doctest: +SKIP
    """
    from sympy import symbols, Integral, exp, sin
    a, b, c, d, e, t = symbols('a, b, c, d, e, t')
    i = Integral(exp(-a*b*(e - t))*sin(c*t)*sin(d*(t - e)), (t, 0, e))
    return i


@integral_example
@hangs_time_with(integrate_sympy)
def integrate_double_integral_expsincos():
    """
    https://github.com/sympy/sympy/issues/21649

    >>> from examples import integrate_double_integral_expsincos
    >>> i = integrate_double_integral_expsincos()
    >>> i
    Integral(90*sqrt(2)*exp(-16200*(t - pi/4)**2/pi**2)*Integral(2078326.11800561*sqrt(2)*(4.99987500312492e-5*v**2*sin(t)*cos(t) - exp(-pi**2/16200))**2*exp(-(v - 200)**2/2)/sqrt(pi), (v, -oo, oo))/pi**1.5, (t, -oo, oo))
    >>> i.doit() # doctest: +SKIP
    """
    from sympy import symbols, exp, sin, cos, Integral, sqrt, pi, oo
    t, v = symbols('t, v')
    i = Integral(90*sqrt(2)*exp(-16200*(t - pi/4)**2/pi**2)*Integral(2078326.11800561*sqrt(2)*(4.99987500312492e-5*v**2*sin(t)*cos(t) - exp(-pi**2/16200))**2*exp(-(v - 200)**2/2)/sqrt(pi), (v, -oo, oo))/pi**1.5, (t, -oo, oo))
    return i


@integral_example
@hangs_time_with(integrate_sympy)
def integrate_heaviside_example():
    """
    https://github.com/sympy/sympy/issues/24546
    """
    from sympy import symbols, Heaviside, Integral
    a, x = symbols('a, x')
    return Integral(Heaviside(x - a), (x, 0, 1))


@integral_example
@hangs_time_with(integrate_sympy)
def integrate_heurisch_toolarge_example():
    """
    https://github.com/sympy/sympy/issues/24025

    >>> from examples import integrate_heurisch_toolarge_example
    >>> i = integrate_heurisch_toolarge_example()
    >>> i
    Integral(t*exp(2*t)*log(t)/(t**2*exp(2*t) + 1) + (t**2*exp(t)*log(t)/(t**2*exp(2*t) + 1) + t*log(t)*atan(t*exp(t)) + sqrt(t*exp(t)*log(t))*exp(-t)*acos(exp(t)*log(t))/2 - sqrt(t*exp(t)*log(t))*log(t)/sqrt(-exp(2*t)*log(t)**2 + 1))*exp(t) + exp(t)*log(t)*atan(t*exp(t)) + sqrt(t*exp(t)*log(t))*acos(exp(t)*log(t))/(2*t) + (t*exp(t)*atan(t*exp(t)) + sqrt(t*exp(t)*log(t))*acos(exp(t)*log(t))/(2*log(t)) - sqrt(t*exp(t)*log(t))*exp(t)/sqrt(-exp(2*t)*log(t)**2 + 1))/t, (t, 1, 5))
    >>> i.doit() # doctest: +SKIP

    This is slow because it generates an enormous set of monomials. In
    principle heurisch might be able to solve this but the intermediate
    problems generated are far too large. A limit should be set to give up.
    """
    from sympy import symbols, exp, log, sqrt, atan, acos, Integral
    t = symbols('t')
    f = (t*exp(2*t)*log(t)/(t**2*exp(2*t) + 1) +
         (t**2*exp(t)*log(t)/(t**2*exp(2*t) + 1) + t*log(t)*atan(t*exp(t)) +
          sqrt(t*exp(t)*log(t))*exp(-t)*acos(exp(t)*log(t))/2
          - sqrt(t*exp(t)*log(t))*log(t)/sqrt(-exp(2*t)*log(t)**2 + 1))*exp(t)
         + exp(t)*log(t)*atan(t*exp(t)) +
         sqrt(t*exp(t)*log(t))*acos(exp(t)*log(t))/(2*t) +
         (t*exp(t)*atan(t*exp(t)) +
          sqrt(t*exp(t)*log(t))*acos(exp(t)*log(t))/(2*log(t)) -
          sqrt(t*exp(t)*log(t))*exp(t)/sqrt(-exp(2*t)*log(t)**2 + 1))/t)
    i = Integral(f, (t, 1, 5))
    return i


@integral_example
@slow_time_with(integrate_sympy, time_seconds=600)  # Not sure about 600...
def integrate_heurisch_slow():
    """
    https://github.com/sympy/sympy/issues/23948

    This is just a large problem that is slow in heurisch.

    >>> from examples import integrate_heurisch_slow
    >>> i = integrate_heurisch_slow()
    >>> i # doctest: +NORMALIZE_WHITESPACE
    Integral((x**4 + x**3*exp(x) + (-6*x**2*exp(x) - 6*x**2)*exp(exp(x)) +
    ((2*x*exp(x) + 2*x)*exp(2*exp(x)) + (-6*x**2*exp(2*x) + 6*x**2 + (-6*x**3 +
    6*x)*exp(x))*exp(exp(x)))*log(2*x + 2*exp(x)) + (2*x*exp(2*x) - 2*x + (2*x**2 -
    2)*exp(x))*exp(2*exp(x))*log(2*x + 2*exp(x))**2)/(x**4 + x**3*exp(x)), x)
    >>> i.doit() # doctest: +SKIP
    """
    from sympy import symbols, exp, log, sqrt, atan, acos, Integral
    x = symbols('x')
    integrand = (
        ((2*x*exp(x)**2+(2*x**2-2)*exp(x)-2*x)*exp(exp(x))**2*log(2*exp(x)+2*x)**2
        +((2*exp(x)*x+2*x)*exp(exp(x))**2
        +(-6*exp(x)**2*x**2+(-6*x**3+6*x)*exp(x)+6*x**2)*exp(exp(x)))*log(2*exp(x)+2*x)
        +(-6*exp(x)*x**2-6*x**2)*exp(exp(x))+exp(x)*x**3+x**4)
        /
        (exp(x)*x**3+x**4)
    )
    return Integral(integrand, x)


@integral_example
@slow_time_with(integrate_sympy, time_seconds=600) # Not sure about 600...
def integrate_heurisch_slow2():
    """
    https://github.com/sympy/sympy/issues/23949

    This is just a large problem that is slow in heurisch.

    >>> from examples import integrate_heurisch_slow2
    >>> i = integrate_heurisch_slow2()
    >>> i # doctest: +NORMALIZE_WHITESPACE
    Integral((((-x**2 - 12*x - 36)*exp(2) + (-2*x**4 - 24*x**3 - 73*x**2 - 12*x
    - 36)*exp(x**2))*exp(exp(5)) + (-x*exp(exp(5))*log(x) + (x**2 + 13*x +
    42)*exp(exp(5)))*exp(log(x)/(x + 6)))/(x**2 + 12*x + 36), x)
    >>> i.doit() # doctest: +SKIP
    """
    from sympy import symbols, exp, log, sqrt, atan, acos, Integral
    x = symbols('x')
    integrand = (
        ((-x*exp(exp(5))*log(x)+(x**2+13*x+42)*exp(exp(5)))*exp(log(x)/(6+x))
         +((-2*x**4-24*x**3-73*x**2-12*x-36)*exp(x**2)
         +(-x**2-12*x-36)*exp(2))*exp(exp(5)))
        /(x**2+12*x+36)
    )
    return Integral(integrand, x)


@integral_example
@hangs_time_with(integrate_sympy)
def integrate_heurisch_radical_trig_slow():
    """
    https://github.com/sympy/sympy/issues/23690

    This example leads to large subproblems in heurisch and is slow.

    >>> from examples import integrate_heurisch_slow
    >>> i = integrate_heurisch_radical_trig_slow()
    >>> i
    Integral((a**2 - 4*cos(x)**2)**(3/4)*sin(2*x), x)
    >>> i.doit() # doctest: +SKIP
    """
    from sympy import symbols, cos, sin, Integral, S
    x, a = symbols('x, a')
    i = Integral((a**2-4*cos(x)**2)**(S(3)/4)*sin(2*x), x)
    return i


@integral_example
@slow_time_with(integrate_sympy, time_seconds=150)
def integrate_rational_function():
    """
    https://github.com/sympy/sympy/issues/23605

    This example is a rational function but for some reason goes through risch.
    It is slow in risch.

    >>> from examples import integrate_rational_function
    >>> i = integrate_rational_function()
    >>> i
    Integral(-T*(R/(-b + nu) - Derivative(a(T), T)/(b*nu + nu**2)) + p, nu)
    >>> i.doit() # doctest: +SKIP
    """
    from sympy import symbols, Function, Integral, diff

    p,T,nu = symbols("p T nu", positive=True)
    R,b    = symbols("R b"   , positive=True)
    a      = Function("a"    , positive=True)(T)

    srk_p = R*T/(nu-b) - a/(nu**2 + b*nu)
    integrand = p - T*diff(srk_p, T)

    return Integral(integrand, nu)


@matrix_example
@hangs_time_with(det_sympy)
@hangs_time_with(eigenvals_sympy)
@slow_time_with(charpoly_sympy, time_seconds=5)
def matrix_symbolic_4x4_functions_slow():
    """
    https://github.com/sympy/sympy/issues/23408

    This example is a 4x4 matrix with symbolic functions in the entries.
    Computing det or eigenvals is slow even though charpoly is reasonably fast.

    >>> from examples import matrix_symbolic_4x4_functions_slow
    >>> M = matrix_symbolic_4x4_functions_slow()
    >>> M
    Matrix([
    [       -mu + M_z(x) + (k_x**2 + k_y**2)/(2*m), alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x),                                          Delta,                                               0],
    [alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),         -mu - M_z(x) + (k_x**2 + k_y**2)/(2*m),                                              0,                                           Delta],
    [                                        Delta,                                              0,          mu + M_z(x) - (k_x**2 + k_y**2)/(2*m), -alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x)],
    [                                            0,                                          Delta, -alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),           mu - M_z(x) - (k_x**2 + k_y**2)/(2*m)]])
    >>> M.det() # doctest: +SKIP
    >>> M.eigenvals() # doctest: +SKIP
    >>> M.charpoly() # doctest: +SKIP
    """
    from sympy import symbols, Function, Matrix, I
    (k_x, k_y, k_z, m, alpha, B, x, mu, Delta) = symbols('k_x k_y k_z m alpha B x mu Delta', real=True)
    (M_x, M_y, M_z) = symbols('M_x M_y M_z', real=True, cls=Function)
    M = Matrix([
    [       -mu + M_z(x) + (k_x**2 + k_y**2)/(2*m), alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x),                                          Delta,                                               0],
    [alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),         -mu - M_z(x) + (k_x**2 + k_y**2)/(2*m),                                              0,                                           Delta],
    [                                        Delta,                                              0,          mu + M_z(x) - (k_x**2 + k_y**2)/(2*m), -alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x)],
    [                                            0,                                          Delta, -alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),           mu - M_z(x) - (k_x**2 + k_y**2)/(2*m)]])
    return M


@matrix_example
@slow_time_with(det_sympy, time_seconds=20)
@slow_time_with(eigenvals_sympy, time_seconds=10)
@slow_time_with(charpoly_sympy, time_seconds=5)
def matrix_symbolic_4x4_functions_fast():
    """
    https://github.com/sympy/sympy/issues/23408

    This example is a 4x4 matrix with symbolic functions in the entries.
    Computing det or eigenvals is slow even though charpoly is reasonably fast.

    >>> from examples import matrix_symbolic_4x4_functions_fast
    >>> M = matrix_symbolic_4x4_functions_fast()
    >>> M
    Matrix([
    [                k_x**2 + k_y**2 - mu + M_z(x), alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x),                                          Delta,                                               0],
    [alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),                  k_x**2 + k_y**2 - mu - M_z(x),                                              0,                                           Delta],
    [                                        Delta,                                              0,                 -k_x**2 - k_y**2 + mu + M_z(x), -alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x)],
    [                                            0,                                          Delta, -alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),                  -k_x**2 - k_y**2 + mu - M_z(x)]])
    >>> M.det() # doctest: +SKIP
    >>> M.eigenvals() # doctest: +SKIP
    >>> M.charpoly() # doctest: +SKIP
    """
    from sympy import symbols, Function, Matrix, I
    (k_x, k_y, k_z, m, alpha, B, x, mu, Delta) = symbols('k_x k_y k_z m alpha B x mu Delta', real=True)
    (M_x, M_y, M_z) = symbols('M_x M_y M_z', real=True, cls=Function)
    M = Matrix([
    [                k_x**2 + k_y**2 - mu + M_z(x), alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x),                                          Delta,                                               0],
    [alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),                  k_x**2 + k_y**2 - mu - M_z(x),                                              0,                                           Delta],
    [                                        Delta,                                              0,                 -k_x**2 - k_y**2 + mu + M_z(x), -alpha*(-I*k_x - k_y) - I*(B + M_y(x)) + M_x(x)],
    [                                            0,                                          Delta, -alpha*(I*k_x - k_y) + I*(B + M_y(x)) + M_x(x),                  -k_x**2 - k_y**2 + mu - M_z(x)]])
    return M


@matrix_example
@hangs_time_with(charpoly_sympy)
def matrix_4x4_sincos_symbolic():
    """
    https://github.com/sympy/sympy/issues/22685

    >>> from examples import matrix_4x4_sincos_symbolic
    >>> M = matrix_4x4_sincos_symbolic()
    >>> M
    Matrix([
    [C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) + m0 + m1*cos(c*kz) + m2*(cos(a*kx) + cos(a*ky)),                                                                                               0,                                                                     t*sin(a*kx) + I*t*sin(a*ky),                gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) - I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz)],
    [                                                                                              0, C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) + m0 + m1*cos(c*kz) + m2*(cos(a*kx) + cos(a*ky)),                gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) + I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz),                                                                    -t*sin(a*kx) + I*t*sin(a*ky)],
    [                                                                    t*sin(a*kx) - I*t*sin(a*ky),                gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) - I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz), C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) - m0 - m1*cos(c*kz) - m2*(cos(a*kx) + cos(a*ky)),                                                                                               0],
    [               gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) + I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz),                                                                    -t*sin(a*kx) - I*t*sin(a*ky),                                                                                               0, C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) - m0 - m1*cos(c*kz) - m2*(cos(a*kx) + cos(a*ky))]])
    >>> M.charpoly() # doctest: +SKIP
    >>> M.eigenvals() # doctest: +SKIP

    The matrix should have only two eigenvalues as the characteristic
    polynomial factorises. This example needs a sin/cos domain. Already
    DomainMatrix can compute the characteristic polynomial much faster than
    Matrix.charpoly can. However correctly factorising the characteristic
    polynomial requires recognising that trig functions can be simplified.
    """
    from sympy import Matrix, I, cos, sin, symbols, nsimplify
    C0, C1, C2, a, c, kx, ky, kz, m0, m1, m2, t, gamma = symbols('C0 C1 C2 a c kx ky kz m0 m1 m2 t gamma', real=True)
    H = Matrix([
    [C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) + m0 + m1*cos(c*kz) + m2*(cos(a*kx) + cos(a*ky)),                                                                                               0,                                                                     t*sin(a*kx) + I*t*sin(a*ky),                gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) - I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz)],
    [                                                                                              0, C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) + m0 + m1*cos(c*kz) + m2*(cos(a*kx) + cos(a*ky)),                gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) + I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz),                                                                    -t*sin(a*kx) + I*t*sin(a*ky)],
    [                                                                    t*sin(a*kx) - I*t*sin(a*ky),                gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) - I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz), C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) - m0 - m1*cos(c*kz) - m2*(cos(a*kx) + cos(a*ky)),                                                                                               0],
    [               gamma*(-cos(a*kx) + cos(a*ky))*sin(c*kz) + I*gamma*sin(a*kx)*sin(a*ky)*sin(c*kz),                                                                    -t*sin(a*kx) - I*t*sin(a*ky),                                                                                               0, C0 - C1*cos(c*kz) - C2*(cos(a*kx) + cos(a*ky)) - m0 - m1*cos(c*kz) - m2*(cos(a*kx) + cos(a*ky))]])
    return H



@expression_example
@slow_time_with(trigsimp_sympy, time_seconds=20)
def expression_trig_Ipi_example():
    """
    https://github.com/sympy/sympy/issues/23983

    >>> from examples import expression_trig_Ipi_example
    >>> expr = expression_trig_Ipi_example()
    >>> expr
    -sin(x - 3)/cos(x - 3) + 3*cos(3*x + 1)/sin(3*x + 1) - 2/x + 2/(x*(log(x**2) + I*pi))
    >>> expr.trigsimp() # doctest: +SKIP

    This example is slow because various poly operations are slow over
    algebraic number fields e.g. factorisation in ZZ_I is slow.
    """
    from sympy import symbols, sin, cos, log, I, pi
    x = symbols('x')
    e = -sin(x - 3)/cos(x - 3) + 3*cos(3*x + 1)/sin(3*x + 1) - 2/x + 2/(x*(log(x**2) + I*pi))
    return e


@expression_example
@hangs_time_with(factor_deep_sympy)
def expression_exp_poly():
    """
    https://github.com/sympy/sympy/issues/23766

    A possible fix for this can be found in the issue.

    >>> from examples import expression_exp_poly
    >>> expr = expression_exp_poly()
    >>> expr
    exp((x + 1)**25 + 1)
    >>> expr.factor(deep=True) # doctest: +SKIP

    This is very particular issue that factor deep=True ends up trying to make
    a large dense polynomial.
    """
    from sympy import E, symbols
    x = symbols('x')
    return E ** ((x+1)**25+1)


@expression_example
@hangs_time_with(minpoly_sympy)
def expression_QQIcbrt3():
    """
    https://github.com/sympy/sympy/issues/22400

    >>> from examples import expression_QQIcbrt3
    >>> e = expression_QQIcbrt3()
    >>> e
    2**(1/3) + 3**(1/3) + 5**(1/3)*(-1/2 + sqrt(3)*I/2)
    >>> from sympy import minpoly
    >>> minpoly(e) # doctest: +SKIP

    Computing the minpoly here is slow. There are a bunch of techniques that
    could be used to speed it up. Ultimately the slow part is factorising large
    polynomials but there could be ways to avoid this such as by deflating the
    polynomial first. Partly it is slow because minpoly expands the expression
    leading to an Add with more terms. Maybe it would be better not to expand
    the expression.
    """
    from sympy import sqrt, root, I
    e = root(2,3)+root(3,3)+(-1+I*sqrt(3))/2*root(5,3)
    return e


@expression_example
@hangs_time_with(minpoly_sympy)
def expression_sqrts_I():
    """
    https://github.com/sympy/sympy/issues/21528

    >>> from examples import expression_sqrts_I
    >>> e = expression_sqrts_I()
    >>> e
    -sqrt(11) + sqrt(10) + 2*sqrt(7) - sqrt(2)*(4 + I) + I
    >>> from sympy import minpoly
    >>> minpoly(e) # doctest: +SKIP

    The algorithm used for computing minimal polynomials could be improved. The
    slow part is factorising large polynomials over QQ so factorisation could
    also be improved.
    """
    from sympy import sqrt, I
    e = -sqrt(11) + sqrt(10) + 2*sqrt(7) - sqrt(2)*(4 + I) + I
    return e


@expression_example
@slow_time_with(minpoly_sympy, time_seconds=20)
@fast_time_with(evalf_sympy)
def expression_rootof_poly():
    """
    https://github.com/sympy/sympy/issues/23677

    More examples in the issue.

    Computing the minimal polynomial of this expression is slow.
    """
    from sympy import symbols, CRootOf
    x = symbols('x')
    p = 4000000*x**3 - 239960000*x**2 + 4782399900*x - 31663998001
    r1 = CRootOf(p, 0)
    r2 = CRootOf(p, 1)
    e = (7680000000000000000*r1**4*r2**4
         - 614323200000000000000*r1**4*r2**3
         + 18458112576000000000000*r1**4*r2**2
         - 246896663036160000000000*r1**4*r2
         + 1240473830323209600000000*r1**4
         - 614323200000000000000*r1**3*r2**4
         - 1476464424954240000000000*r1**3*r2**2
         - 99225501687553535904000000*r1**3
         + 18458112576000000000000*r1**2*r2**4
         - 1476464424954240000000000*r1**2*r2**3
         - 593391458458356671712000000*r1**2*r2
         + 2981354896834339226880720000*r1**2
         - 246896663036160000000000*r1*r2**4
         - 593391458458356671712000000*r1*r2**2
         - 39878756418031796275267195200*r1
         + 1240473830323209600000000*r2**4
         - 99225501687553535904000000*r2**3
         + 2981354896834339226880720000*r2**2
         - 39878756418031796275267195200*r2
         + 200361370275616536577343808012
    )
    return e


@ode_example
@slow_time_with(dsolve_sympy, time_seconds=50)
def ode_linear_slow():
    """
    https://github.com/sympy/sympy/issues/23700

    >>> from examples import ode_linear_slow
    >>> ode, sym = ode_linear_slow()
    >>> ode
    Eq(J_m*L_a*Derivative(w_m(t), (t, 2))/K_t + (D_m*R_a/K_t + K_b)*w_m(t) + (D_m*L_a + J_m*R_a)*Derivative(w_m(t), t)/K_t, e_a(t))
    >>> from sympy import dsolve
    >>> dsolve(ode, sym) # doctest: +SKIP
    """
    from sympy import symbols, Function, Derivative, Eq
    t = symbols("t")
    e_a = Function("e_a")
    w_m = Function("w_m")
    w = w_m(t)
    dw = w.diff(t)
    d2w = dw.diff(t)
    J_m, R_a, K_b, K_t, D_m, L_a = symbols("J_m R_a K_b K_t D_m L_a", real=True, positive=True)

    eq = Eq(J_m*L_a*d2w/K_t + (D_m*R_a/K_t + K_b)*w + (D_m*L_a + J_m*R_a)*dw/K_t, e_a(t))

    return eq, w_m(t)


@ode_system_example
@hangs_time_with(dsolve_sympy)
def ode_system_matrix_exponential():
    """
    https://github.com/sympy/sympy/issues/21867

    >>> from examples import ode_system_matrix_exponential
    >>> eqs, syms = ode_system_matrix_exponential()
    >>> for eq in eqs: print(eq)
    Eq(Derivative(x(t), t), k_21*y(t) + ka*z(t) - (k_12 + k_e)*x(t))
    Eq(Derivative(y(t), t), k_12*x(t) - k_21*y(t))
    Eq(Derivative(z(t), t), -ka*x(t))
    >>> from sympy import dsolve
    >>> dsolve(eqs, syms) # doctest: +SKIP

    This is slow due to slow calculation of the matrix exponential. The matrix
    exponential in turn is slow due to slow calculation of Jordan normal form.

    See jnf_symbolic for the matrix used here.
    """
    from sympy import symbols, Function, Eq, Derivative
    t, k_12, k_21, k_e, ka, D = symbols('t, k_12, k_21, k_e, ka, D')
    x, y, z = symbols('x, y, z', cls=Function)

    eq1 = Eq(Derivative(x(t), t), z(t)*ka + y(t)*k_21 - x(t)*(k_e + k_12))
    eq2 = Eq(Derivative(y(t), t), x(t)*k_12 - y(t)*k_21)
    eq3 = Eq(Derivative(z(t), t), -x(t)*ka)
    eqs = [eq1, eq2, eq3]
    syms = [x(t), y(t), z(t)]
    return eqs, syms


def run_and_time(queue, name, setup, func, timeout):
    """
    Run the example with timeout.

    Push the name and time to queue for the parent process.
    """
    import signal
    import time

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError

    try:
        arg = setup()

        signal.signal(signal.SIGALRM, handler)
        try:
            signal.alarm(timeout)
            start = time.time()
            func(arg)
            end = time.time()
            time_seconds = end - start
        except TimeoutError:
            time_seconds = f"timeout ({timeout} seconds))"
    except BaseException as e:
        print(e)
        time_seconds = f"error ({e.__class__.__name__})"

    queue.put((name, time_seconds))


def run_timings(problems, timeout):
    """
    Run the examples and time them.
    """
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    queue = multiprocessing.Queue()

    for name, (setup, func, time_seconds) in problems.items():
        print(f"Running {name}")
        args=(queue, name, setup, func, timeout)
        p = multiprocessing.Process(target=run_and_time, args=args)
        p.start()
        print(*queue.get(timeout=2*timeout))
        p.join(timeout)


def main(args):
    if args.doctest:
        import doctest
        doctest.testmod()
        return

    if args.list:
        print("Available problems:")
        for name, (setup, func, time_seconds) in problems.items():
            print(name, time_seconds)
        return

    if args.name:
        problems_used = {args.name: problems[args.name]}
    else:
        problems_used = problems

    run_timings(problems_used, args.timeout)


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Run examples.')
    parser.add_argument('--doctest', action='store_true', help='Run doctests')
    parser.add_argument('--slow', action='store_true', help='Run slow examples')
    parser.add_argument('--timeout', type=int, default=10, help='Timeout for each example')
    parser.add_argument('--name', type=str, help='Name of example to run')
    parser.add_argument('--list', action='store_true', help='List examples')
    args = parser.parse_args(sys.argv[1:])
    main(args)

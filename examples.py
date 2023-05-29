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
matrix_example, matrix_examples = example_decorator()
integral_example, integral_examples = example_decorator()
system_example, system_examples = example_decorator()
linear_system_example, linear_system_examples = example_decorator()
poly_system_example, poly_system_examples = example_decorator()
rational_function_example, rational_function_examples = example_decorator()


def pinv_sympy(matrix):
    return matrix.pinv()


def inv_sympy(matrix):
    return matrix.inv()


def jnf_sympy(matrix):
    return matrix.jordan_form()


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


def linsolve_sympy(eqs_vars):
    from sympy import linsolve
    eqs, vars = eqs_vars
    return linsolve(eqs, vars)


def cancel_sympy(expr):
    from sympy import cancel
    return cancel(expr)


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


def trigsimp_sympy(expr):
    return expr.trigsimp()


def factor_deep_sympy(expr):
    return expr.factor(deep=True)


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

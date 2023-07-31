#!/usr/bin/env python3

import time
import sys
import signal
import argparse
import dataclasses
import math
from sympy import *
from sympy.polys.matrices import DomainMatrix


def main(*args):
    """Main program."""
    parser = argparse.ArgumentParser(description='Time rref methods.')
    parser.add_argument('--timeout', type=float, default=1,
                        help='timeout in seconds (default: 1.0) (0 for no timeout)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='do not print progress')
    parser.add_argument('--operation', '-o', type=str, default='rref',
                        help='operation to time (default: rref)')
    parser.add_argument('--exclude-methods', type=str, default='',
                        help='comma-separated list of methods to exclude')
    parser.add_argument('--n', type=str, default='1,2,3,4,5,6,7,8,9,10',
                        help='main parameter values')
    parser.add_argument('--repeats', type=int, default=1,
                        help='number of times to repeat each timing (default: 1)')
    parser.add_argument('--matrices', type=str, default='ZZnn_d',
                        help='matrices to time (default: ZZnn_d)')
    parser.add_argument('--density', type=float, default=1.0,
                        help='density of sparse matrices (default: 1.0)')
    parser.add_argument('--bits', type=int, default=6,
                        help='bitsize for integers (default: 6)')
    parser.add_argument('--dbits', type=int, default=2,
                        help='bitsize for denominators (default: 2)')
    parser.add_argument('--size', type=int, default=10,
                        help='size of square matrices (default: 10)')
    parser.add_argument('--aspect', type=float, default=1.0,
                        help='width/height ratio for wide/tall matrices (default: 1.0)')
    parser.add_argument('--show', action='store_true',
                        help='show available matrices types and methods')
    parser.add_argument('--plot', action='store_true',
                        help='plot results')
    parser.add_argument('--doctest', action='store_true',
                        help='run doctests and exit')

    args = parser.parse_args(args)

    if args.doctest:
        import doctest
        doctest.testmod()
        return

    matrices = args.matrices

    if matrices not in matrix_types:
        print(f'Error: unknown matrix type: {matrices}')
        print(f'Known matrix types: {", ".join(matrix_types)}')
        return

    if args.show:
        print()
        print('Matrix classes:')
        print()
        for m, mf in matrix_types.items():
            print(f'{m}:\n    {mf.__doc__}')
        print()
        for op, op_dict in operations.items():
            print()
            print(f'Operations for {op}:')
            for dom, methods_dict in op_dict.items():
                print(f'Methods for {dom}:')
                print()
                for m, mf in methods_dict.items():
                    print(f'{m}:\n    {mf.__doc__}')
                print()
            print()
        return

    matrices_f = matrix_types[matrices]
    domain = matrices_f.domain

    operation = args.operation
    methods = operations[operation][domain]

    if args.timeout == 0:
        args.timeout = None

    args.n = [int(s) for s in args.n.split(',')]

    if args.exclude_methods:
        for m in args.exclude_methods.split(','):
            if m not in methods:
                print(f'Error: unknown method: {m}')
                print(f'Known methods: {", ".join(methods)}')
                return
            methods.pop(m)

    opts = Options(
        timeout=args.timeout,
        quiet=args.quiet,
        operation=args.operation,
        n=args.n,
        size=args.size,
        density=args.density,
        bits=args.bits,
        dbits=args.dbits,
        aspect=args.aspect,
        repeats=args.repeats,
    )

    print('Matrices:', matrices)
    print('Domain:', domain)
    print('Methods:', ', '.join(methods.keys()))
    print(f'Parameters:, {params_to_str(matrices_f, opts)}')
    print(f'Timing with n={opts.n}')
    print()
    print(matrices_f.__doc__)
    print()
    for m, mf in methods.items():
        print(f'{m}: {mf.__doc__}')
    print()
    print('Timing...')
    print()
    times, table, rownames, colnames = make_table(methods, matrices_f, opts)
    print()
    print('Finished!')
    print()
    print(table)
    print()

    if args.plot:
        plot_timings(matrices, times, rownames, colnames, opts)


@dataclasses.dataclass
class Options:
    """Options for timings."""
    timeout: float = 1.0
    repeats: int = 1
    quiet: bool = False
    operation: str = 'rref'
    n: str = '1,2,3,4,5,6,7,8,9,10'
    size: int = 10
    density: float = 1.0
    bits: int = 6
    dbits: int = 2
    matrices: str = 'ZZnn_d'
    aspect: float = 1.0


def make_options(opts=None, **kwargs):
    """Make an Options object from keyword arguments."""
    if opts is None:
        return Options(**kwargs)
    elif isinstance(opts, Options):
        return dataclasses.replace(opts, **kwargs)
    else:
        raise TypeError('opts must be None or an Options object')


def format_time_secs(t, tmax=None):
    """Format a time in seconds with appropriate units."""

    if t is None:
        if tmax is None:
            return 'timeout'
        else:
            return f'>{format_time_secs(tmax).strip()}'

    if t < 1e-6:
        t = t*1e9
        unit = 'ns'
    elif t < 1e-3:
        t = t*1e6
        unit = 'us'
    elif t < 1:
        t = t*1e3
        unit = 'ms'
    else:
        unit = 's'

    return f'{t:5.1f} {unit}'


class TimeoutError(Exception):
    pass


def time_func(func, timeout=None):
    """Time a function call returning time in seconds."""
    if timeout is None:
        start = time.time()
        result = func()
        tsecs = time.time() - start
        return result, tsecs

    def handler(signum, frame):
        raise TimeoutError('timeout')

    old_handler = None

    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(math.ceil(timeout))
        start = time.time()
        result = func()
        tsecs = time.time() - start
        # alarm needs an int so we round up but then check here afterwards
        if tsecs > timeout:
            raise TimeoutError('timeout')
    except TimeoutError:
        result = tsecs = None
    finally:
        signal.alarm(0)
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)

    return result, tsecs


def make_table(methods, matrices, opts=Options()):
    """Make a table of timings for the given methods and matrices."""
    if opts.quiet:
        vprint = lambda *args, **kwargs: None
    else:
        vprint = print

    names = []
    times = []

    for n in opts.n:
        name, M = matrices(n, opts=opts)
        M = [M] + [matrices(n, opts=opts)[1] for i in range(opts.repeats-1)]
        names.append(name)
        vprint(f'{name}:')
        times_M = []
        result = None

        try:
            for n, (m, func) in enumerate(methods.items()):
                vprint(f'{m} ', end='', flush=True)

                if times and times[-1][n] is None:
                    times_M.append(None)
                    vprint('skipped')
                    continue

                total = 0
                for i in range(opts.repeats):
                    result_m, t = time_func(lambda: func(M[i]), timeout=opts.timeout)

                    vprint(f'{format_time_secs(t, opts.timeout).strip()} ', end='', flush=True)
                    if t is None:
                        total = None
                        break
                    total += t
                vprint()

                if total is not None:
                    times_M.append(total / opts.repeats)
                else:
                    times_M.append(None)

                if result is None:
                    result = result_m
                elif result_m is not None and result != result_m:
                    vprint(f'Error: {name} {func.__name__} result mismatch')
                    breakpoint()

            times.append(times_M)
            vprint()

            if all(t is None for t in times_M):
                vprint('Quitting: all methods timed out')
                break

        except KeyboardInterrupt:
            vprint('Quitting: interrupted')
            names.pop()
            break

    times_str = [[format_time_secs(t, opts.timeout) for t in times_M] for times_M in times]

    rownames = names
    colnames = list(methods.keys())
    table = TableForm(times_str, headings=[rownames, colnames])
    return times, table, rownames, colnames


def params_to_str(matrices, opts):
    """Convert matrices and options to a string."""
    if isinstance(matrices, str):
        matrix_type = matrix_types[matrices]
    else:
        matrix_type = matrices
    params = ', '.join(f'{p}={getattr(opts,p)}' for p in matrix_type.parameters)
    return params


def plot_timings(matrices, times, rownames, colnames, opts=Options()):
    """Plot timings."""
    import matplotlib.pyplot as plt
    import numpy as np

    if opts.timeout:
        times = [[t if t is not None else opts.timeout for t in ts] for ts in times]

    matrix_type = matrix_types[matrices]
    params = params_to_str(matrix_type, opts)
    p = matrix_type.n
    title = f'{opts.operation} {matrices} times  vs {p} for {params}'

    times = np.array(times)

    fig, ax = plt.subplots()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(p)
    ax.set_ylabel('time (s)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True)

    n = opts.n[:times.shape[0]]

    for i, colname in enumerate(colnames):
        ax.plot(n, times[:,i], marker='x', label=colname)

    ax.hlines(opts.timeout, 0, n[-1], linestyles='dashed', label='timeout')

    ax.legend()

    ax.set_xticks(n)
    ax.set_xticklabels(rownames, rotation=45)

    plt.savefig('rref_timings.svg')
    plt.show()


# --------------------------------------------------------------------------- #
#                      Matrix types                                           #
# --------------------------------------------------------------------------- #


matrix_types = {}


def matrix_type(domain, n, parameters):
    """Decorator to register a matrix type."""
    def decorator(func):
        matrix_types[func.__name__] = func
        func.domain = domain
        func.n = n
        func.parameters = parameters
        return func
    return decorator


def ZZmn_d(m, n, bits, seed=None):
    """Random dense m x n matrix over ZZ with bitsize ``bits``.

    >>> ZZmn_d(2, 3, 6, seed=0)
    Matrix([
    [45, -14,  34],
    [50, -10, -58]])
    """
    maxn = 2**bits - 1
    return randMatrix(m, n, -maxn, maxn, seed=seed)


def ZZmn_s(m, n, bits, density, seed=None):
    """Random sparse m x n matrix over ZZ with bitsize ``bits``.

    ``density`` is the average number of non-zero entries per row.

    >>> ZZmn_s(3, 5, 6, 1, seed=0)
    Matrix([
    [0,   0,   0,  0, 0],
    [0, -10,   0,  0, 0],
    [0,   0, -58, 50, 0]])
    """
    maxn = 2**bits - 1
    percent = min(100, 100*density/n)
    return randMatrix(m, n, -maxn, maxn, percent=percent, seed=seed)


def QQmn_d(m, n, nbits, dbits, seed=None):
    """Random dense m x n matrix over QQ with bitsize ``bits``.

    >>> QQmn_d(2, 3, 6, 2, seed=0)
    Matrix([
    [45/2,    -7,  34],
    [  25, -10/3, -29]])
    """
    maxn = 2**nbits - 1
    maxd = 2**dbits - 1
    M = randMatrix(m, n, -maxn, maxn, seed=seed)
    Md = randMatrix(m, n, 1, maxd, seed=seed)
    for i, j in Md.todok():
        M[i, j] /= Md[i, j]
    return M


def QQmn_s(m, n, nbits, dbits, density, seed=None):
    """Random sparse m x n matrix over QQ with bitsize ``bits``.

    ``nbits`` is the bitsize of the numerators and ``dbits`` is the bitsize of
    the denominators.

    ``density`` is the average number of non-zero entries per row.

    >>> QQmn_s(3, 5, nbits=6, dbits=6, density=1, seed=0)
    Matrix([
    [0,      0,      0,     0, 0],
    [0, -10/49,      0,     0, 0],
    [0,      0, -58/25, 10/11, 0]])
    """
    maxn = 2**nbits - 1
    maxd = 2**dbits - 1
    percent = min(100, 100*density/n)
    M = randMatrix(m, n, -maxn, maxn, percent=percent, seed=seed)
    dok = M.todok()
    Md = randMatrix(len(dok), 1, 1, maxd, seed=seed)
    for n, ((i, j), Mij) in enumerate(dok.items()):
        M[i, j] = Mij / Md[n, 0]
    return M


@matrix_type('ZZ', 'size (n x n)',  ['bits'])
def ZZnn_d(n, seed=None, **kwargs):
    """Random dense square n x n matrix over ZZ.

    --bits is the maximum bitsize of the integers

    >>> name, M = ZZnn_d(5, bits=6, seed=0)
    >>> name
    '5x5'
    >>> M
    Matrix([
    [ 45, -14,  34,  50, -10],
    [-58, -30,  60,   2,  -1],
    [-12,  54,  37,  43, -25],
    [ 60,  -2, -18,  11,  51],
    [ 53, -36,   1, -46, -27]])
    """
    opts = make_options(**kwargs)
    return f'{n}x{n}', ZZmn_d(n, n, opts.bits, seed=seed)


@matrix_type('ZZ', 'size (n x n)',  ['bits', 'density'])
def ZZnn_s(n, seed=None, **kwargs):
    """Random sparse square n x n matrix over ZZ.

    --density is the average number of non-zero entries per row.
    --bits is the maximum bitsize of the integers

    >>> name, M = ZZnn_s(5, bits=6, density=1, seed=0)
    >>> name
    '5x5'
    >>> M
    Matrix([
    [0, -12,  0,  0, 0],
    [0,   0,  0, 54, 0],
    [0,   0, 60, -1, 0],
    [0,   0,  0,  0, 0],
    [0,   0,  0,  0, 2]])
    """
    opts = make_options(**kwargs)
    return f'{n}x{n}', ZZmn_s(n, n, opts.bits, opts.density, seed=seed)


@matrix_type('ZZ', 'size (n x (n+1))',  ['bits'])
def ZZnn1_d(n, seed=None, **kwargs):
    """Random dense square n x (n+1) matrix over ZZ.

    --bits is the maximum bitsize of the integers

    >>> name, M = ZZnn1_d(5, bits=6, seed=0)
    >>> name
    '5x6'
    >>> M
    Matrix([
    [ 45, -14,  34,  50, -10, -58],
    [-30,  60,   2,  -1, -12,  54],
    [ 37,  43, -25,  60,  -2, -18],
    [ 11,  51,  53, -36,   1, -46],
    [-27, -46,  33, -51,  16,  39]])
    """
    opts = make_options(**kwargs)
    return f'{n}x{n+1}', ZZmn_d(n, n+1, opts.bits, seed=seed)


@matrix_type('ZZ', 'size (n x (n+1))',  ['bits', 'density'])
def ZZnn1_s(n, seed=None, **kwargs):
    """Random sparse square n x (n+1) matrix over ZZ.

    --density is the average number of non-zero entries per row.
    --bits is the maximum bitsize of the integers

    >>> name, M = ZZnn1_s(5, bits=6, seed=0)
    >>> name
    '5x6'
    >>> M
    Matrix([
    [  0,  0, 0,   0, 0, 0],
    [  0,  0, 0,   0, 0, 0],
    [-30, -1, 0,   0, 0, 0],
    [  0,  0, 0,   0, 0, 0],
    [ 60,  0, 0, -58, 2, 0]])
    """
    opts = make_options(**kwargs)
    return f'{n}x{n+1}', ZZmn_s(n, n+1, opts.bits, opts.density, seed=seed)


@matrix_type('ZZ', 'size (n x (a * n))',  ['bits', 'aspect'])
def ZZw_d(n, seed=None, **kwargs):
    """Random dense wide n x m matrix over ZZ.

    --bits is the maximum bitsize of the integers
    --aspect is the ratio of m/n

    >>> name, M = ZZw_d(3, bits=6, aspect=2, seed=0)
    >>> name
    '3x6'
    >>> M
    Matrix([
    [ 45, -14,  34, 50, -10, -58],
    [-30,  60,   2, -1, -12,  54],
    [ 37,  43, -25, 60,  -2, -18]])
    """
    opts = make_options(**kwargs)
    m = int(n*opts.aspect)
    return f'{n}x{m}', ZZmn_d(n, m, opts.bits, seed=seed)


@matrix_type('ZZ', 'density', ['bits', 'aspect', 'size'])
def ZZw_s_d(n, seed=None, **kwargs):
    """Random sparse wide matrix over ZZ with n non-zero entries per row.

    --bits is the maximum bitsize of the integers
    --aspect is the ratio of width/height
    --size is the number of rows

    >>> name, M = ZZw_s_d(3, bits=6, aspect=2, size=3, seed=0)
    >>> name
    'd=3'
    >>> M
    Matrix([
    [  0,  2, 0, 0, 0, 0],
    [  0,  0, 0, 0, 0, 0],
    [-30, 60, 0, 0, 0, 0]])
    """
    opts = make_options(**kwargs)
    d = n
    n = opts.size
    m = int(n*opts.aspect)
    return f'd={d}', ZZmn_s(n, m, opts.bits, d, seed=seed)


@matrix_type('ZZ', 'size (n x (a * n))',  ['bits', 'aspect', 'density'])
def ZZw_s(n, seed=None, **kwargs):
    """Random sparse wide n x m matrix over ZZ.

    --density is the average number of non-zero entries per row.
    --bits is the maximum bitsize of the integers
    --aspect is the ratio of m/n

    >>> name, M = ZZw_s(3, bits=6, aspect=2, density=2, seed=0)
    >>> name
    '3x6'
    >>> M
    Matrix([
    [  0, 37,   0, 0, 43, 0],
    [  0, 60, -25, 0,  0, 0],
    [-12, 54,   0, 0,  0, 0]])
    """
    opts = make_options(**kwargs)
    m = int(n*opts.aspect)
    return f'{n}x{m}', ZZmn_s(n, m, opts.bits, opts.density, seed=seed)


@matrix_type('ZZ', 'density',  ['bits', 'size'])
def ZZd_d(n, seed=None, **kwargs):
    """Random sparse square matrix over ZZ with n non-zero entries per row.

    --bits is the maximum bitsize of the integers
    --size is the number of rows and columns

    >>> name, M = ZZd_d(2, bits=6, size=5, seed=0)
    >>> name
    'd=2'
    >>> M
    Matrix([
    [  0,  53,  0,   0,   0],
    [  0,   0,  0, -36, -46],
    [  0, -51, 11,  51,   0],
    [-46,   1,  0,   0,  33],
    [  0,   0,  0,   0, -27]])
    """
    opts = make_options(**kwargs)
    d = n
    n = opts.size
    return f'd={d}', ZZmn_s(n, n, opts.bits, d, seed=seed)


@matrix_type('ZZ', 'size ((a * n) x n)',  ['bits', 'aspect'])
def ZZt_d(n, seed=None, **kwargs):
    """Random tall dense m x n matrix over ZZ.

    --bits is the maximum bitsize of the integers
    --aspect is the ratio of m/n

    >>> name, M = ZZt_d(3, bits=6, aspect=2, seed=0)
    >>> name
    '6x3'
    >>> M
    Matrix([
    [ 45, -14,  34],
    [ 50, -10, -58],
    [-30,  60,   2],
    [ -1, -12,  54],
    [ 37,  43, -25],
    [ 60,  -2, -18]])
    """
    opts = make_options(**kwargs)
    m = int(n*opts.aspect)
    return f'{m}x{n}', ZZmn_d(m, n, opts.bits, seed=seed)


@matrix_type('ZZ', 'size ((a * n) x n)',  ['bits', 'aspect', 'density'])
def ZZt_s(n, seed=None, **kwargs):
    """Random tall sparse m x n matrix over ZZ.

    --density is the average number of non-zero entries per row.
    --bits is the maximum bitsize of the integers
    --aspect is the ratio of m/n

    >>> name, M = ZZt_s(3, bits=6, aspect=2, density=1, seed=0)
    >>> name
    '6x3'
    >>> M
    Matrix([
    [  0, 37,   0],
    [  0, 43,   0],
    [  0, 60, -25],
    [  0,  0,   0],
    [-12, 54,   0],
    [  0,  0,   0]])
    """
    opts = make_options(**kwargs)
    m = int(n*opts.aspect)
    return f'{m}x{n}', ZZmn_s(m, n, opts.bits, opts.density, seed=seed)


@matrix_type('ZZ', 'bits', ['size'])
def ZZmm_b(n, seed=None, **kwargs):
    """Random dense square m x m matrix with n bits over ZZ.

    --size is the dimension of the matrix

    >>> name, M = ZZmm_b(20, size=4, seed=0)
    >>> name
    '20 bits'
    >>> M
    Matrix([
    [ 722305, -240658,  540970, 818401],
    [-166573, -963674, -505588, 976151],
    [  23645,  -29511, -199367, 877101],
    [ 595169,  691752, -412483, 981048]])
    """
    opts = make_options(**kwargs)
    return f'{n} bits', ZZmn_d(opts.size, opts.size, n, seed=seed)


@matrix_type('QQ', 'size (n x n)', ['bits', 'dbits'])
def QQnn_d(n, seed=None, **kwargs):
    """Random dense square n x n matrix over QQ.

    --bits is the bitsize of the numerators
    --dbits is the bitsize of the denominators

    >>> name, M = QQnn_d(5, bits=6, seed=0)
    >>> name
    '5x5'
    >>> M
    Matrix([
    [45/2,  -7,   34,   25, -10/3],
    [ -29, -15,   30,    1,  -1/2],
    [  -4,  54, 37/3,   43, -25/2],
    [  60,  -2,   -6, 11/2,    17],
    [53/3, -12,    1,  -23,   -27]])
    """
    opts = make_options(**kwargs)
    return f'{n}x{n}', QQmn_d(n, n, opts.bits, opts.dbits, seed=seed)


@matrix_type('QQ', 'size (n x n)', ['bits', 'dbits', 'density'])
def QQnn_s(n, seed=None, **kwargs):
    """Random sparse n x n matrix over QQ.

    --bits is the bitsize of the numerators
    --dbits is the bitsize of the denominators
    --density is the average number of non-zero entries per row.

    >>> name, M = QQnn_s(5, bits=6, dbits=2, density=1, seed=0)
    >>> name
    '5x5'
    >>> M
    Matrix([
    [0, -6,  0,    0, 0],
    [0,  0,  0,   18, 0],
    [0,  0, 30, -1/2, 0],
    [0,  0,  0,    0, 0],
    [0,  0,  0,    0, 2]])
    """
    opts = make_options(**kwargs)
    return f'{n}x{n}', QQmn_s(n, n, opts.bits, opts.dbits, opts.density, seed=seed)


@matrix_type('QQ', 'density',  ['bits', 'size', 'dbits'])
def QQd_d(n, seed=None, **kwargs):
    """Random sparse square matrix over QQ with n non-zero entries per row.

    --bits is the bitsize of the numerators
    --dbits is the bitsize of the denominators
    --size is the number of rows and columns

    >>> name, M = QQd_d(2, bits=6, dbits=2, size=5, seed=0)
    >>> name
    'd=2'
    >>> M
    Matrix([
    [  0, 53/2,    0,    0,     0],
    [  0,    0,    0,  -12,   -23],
    [  0,  -51, 11/2, 51/2,     0],
    [-23,  1/2,    0,    0,  33/2],
    [  0,    0,    0,    0, -27/2]])
    """
    opts = make_options(**kwargs)
    d = n
    n = opts.size
    return f'd={d}', QQmn_s(n, n, opts.bits, opts.dbits, d, seed=seed)


@matrix_type('QQ', 'denom-bits',  ['bits', 'size'])
def QQd_b(n, seed=None, **kwargs):
    """Random dense square matrix over QQ with n bits in the denominators.

    --bits is the bitsize of the numerators
    --size is the number of rows and columns

    >>> name, M = QQd_b(2, bits=6, size=5, seed=0)
    >>> name
    '2 bits'
    >>> M
    Matrix([
    [45/2,  -7,   34,   25, -10/3],
    [ -29, -15,   30,    1,  -1/2],
    [  -4,  54, 37/3,   43, -25/2],
    [  60,  -2,   -6, 11/2,    17],
    [53/3, -12,    1,  -23,   -27]])
    """
    opts = make_options(**kwargs)
    dbits = n
    n = opts.size
    return f'{dbits} bits', QQmn_d(n, n, opts.bits, dbits, seed=seed)


@matrix_type('QQ', 'size (n x (a * m))', ['bits', 'aspect'])
def QQw_d(n, seed=None, **kwargs):
    """Random dense wide n x m matrix over QQ.

    --bits is the bitsize of the numerators
    --dbits is the bitsize of the denominators
    --aspect is the ratio of m/n

    >>> name, M = QQw_d(3, bits=6, aspect=2, seed=0)
    >>> name
    '3x6'
    >>> M
    Matrix([
    [45/2, -7,    34,   25, -10/3, -29],
    [ -15, 30,     1, -1/2,    -4,  54],
    [37/3, 43, -25/2,   60,    -2,  -6]])
    """
    opts = make_options(**kwargs)
    m = int(n*opts.aspect)
    return f'{n}x{m}', QQmn_d(n, m, opts.bits, opts.dbits, seed=seed)


@matrix_type('QQ', 'size (n x (a * m))', ['bits', 'aspect'])
def QQt_d(n, seed=None, **kwargs):
    """Random tall dense m x n matrix over QQ.

    --bits is the bitsize of the numerators
    --dbits is the bitsize of the denominators
    --aspect is the ratio of m/n

    >>> name, M = QQt_d(3, bits=6, aspect=2, seed=0)
    >>> name
    '6x3'
    >>> M
    Matrix([
    [45/2,    -7,    34],
    [  25, -10/3,   -29],
    [ -15,    30,     1],
    [-1/2,    -4,    54],
    [37/3,    43, -25/2],
    [  60,    -2,    -6]])
    """
    opts = make_options(**kwargs)
    m = int(n*opts.aspect)
    return f'{m}x{n}', QQmn_d(m, n, opts.bits, opts.dbits, seed=seed)


@matrix_type('ZZ', 'size (n x n)', ['bits', 'density'])
def ZZband_s(n, seed=None, **kwargs):
    """Random sparse banded n x n matrix over ZZ.

    --bits is the maximum bitsize of the numerator/denominator
    --density is the number of bands

    >>> name, M = ZZband_s(5, bits=6, density=2, seed=0)
    >>> name
    '5x5'
    >>> M
    Matrix([
    [45, -14,   0,   0,  0],
    [ 0,  34,  50,   0,  0],
    [ 0,   0, -10, -58,  0],
    [ 0,   0,   0, -30, 60],
    [ 0,   0,   0,   0,  2]])
    """
    opts = make_options(**kwargs)
    d = int(opts.density)
    entries = ZZmn_d(n, d, opts.bits, seed=seed)
    M = zeros(n)
    jmin = (d-1)//2
    jmax = d - jmin
    for i in range(n):
        M[i, max(0, i-jmin):min(n, i+jmax)] = entries[i, :min(n, i+jmax)-max(0, i-jmin)]
    return f'{n}x{n}', M


# --------------------------------------------------------------------------- #
#                      Matrix .rref() methods.                                #
# --------------------------------------------------------------------------- #


# The PR changes which methods use which algorithms so to time them correctly we
# need to check for the new method.
with_PR_rref = hasattr(DomainMatrix, 'rref_gj_div')


def rref_Matrix(M):
    """Matrix .rref() method."""
    M_rref, _ = M.rref()
    return M_rref


if not with_PR_rref:


    def rref_DMd_QQ(M):
        """DM dense .rref() over QQ"""
        M_rref, _ = M.to_DM(QQ).to_dense().rref()
        return M_rref.to_Matrix()


    def rref_DMs_QQ(M):
        """DM sparse .rref() over QQ"""
        M_rref, _ = M.to_DM(QQ).rref()
        return M_rref.to_Matrix()


    def rref_DMd_ZZ1(M):
        """DM dense .rref_den() over ZZ, divide Matrix"""
        M_rref, den, _ = M.to_DM(ZZ).to_dense().rref_den()
        return M_rref.to_Matrix()/den


    def rref_DMd_ZZ2(M):
        """DM dense .rref_den() over ZZ, divide DM"""
        M_rref, den, _ = M.to_DM(ZZ).to_dense().rref_den()
        return (M_rref/den).to_Matrix()


    def rref_DMs_ZZ1(M):
        """DM sparse .rref_den() over ZZ, divide Matrix"""
        M_rref, den, _ = M.to_DM(ZZ).rref_den()
        return M_rref.to_Matrix()/den


    def rref_DMs_ZZ2(M):
        """DM sparse .rref_den() over ZZ, divide DM"""
        M_rref, den, _ = M.to_DM(ZZ).rref_den()
        return (M_rref/den).to_Matrix()


    def rref_DMd_QQc(M):
        """DM clear_denoms, dense .rref_den() over QQ."""
        d, Mq = M.to_DM(QQ).to_dense().clear_denoms()
        M_rref_q, den, _ = Mq.rref_den()
        return M_rref_q.to_Matrix()/den


    def rref_DMs_QQc(M):
        """DM clear_denoms, sparse .rref_den() over QQ."""
        _, Mz = M.to_DM(QQ).clear_denoms()
        M_rref_z, den, _ = Mz.rref_den()
        return M_rref_z.to_Matrix()/den


    def rref_DMd_ZZc(M):
        """DM clear_denoms, dense .rref_den() over ZZ."""
        _, Mz = M.to_DM(QQ).clear_denoms(convert=True)
        M_rref_z, den, _ = Mz.to_dense().rref_den()
        return (M_rref_z/den).to_Matrix()


    def rref_DMs_ZZc(M):
        """DM clear_denoms, sparse .rref_den() over ZZ."""
        _, Mz = M.to_DM(QQ).clear_denoms()
        M_rref_z, den, _ = Mz.rref_den()
        return M_rref_z.to_Matrix()/den


    def rref_DMd_QQff(M):
        """DM fraction_free, dense .rref_den() over QQ."""
        Mq = M.to_DM(QQ).to_dense()
        M_rref_q, den, _ = Mq.rref_den()
        return M_rref_q.to_Matrix()/den


    def rref_DMs_QQff(M):
        """DM fraction_free, sparse .rref_den() over QQ."""
        Mq = M.to_DM(QQ)
        M_rref_q, den, _ = Mq.rref_den()
        return M_rref_q.to_Matrix()/den


    def rref_DMs_ZZ_hybrid(M):
        """DM ZZ hybrid, use rref over QQ or rref_den over ZZ."""
        Mz = M.to_DM(ZZ)

        method = _rref_ZZ_fastest_method(Mz)

        if method == 'rref_QQ':
            # Use Gauss-Jordan over QQ for sparse matrices.
            M_rref_q, _ = Mz.convert_to(QQ).rref()
            return M_rref_q.to_Matrix()

        elif method == 'rref_den_ZZ':
            # Otherwise use fraction-free Gauss-Jordan over ZZ.
            M_rref_z, den, _ = Mz.rref_den()
            return M_rref_z.to_Matrix() / den

        else:
            assert False


    def rref_DMs_QQ_hybrid(M):
        """DM QQ hybrid, use rref, clear denominators, or use rref_den over QQ."""
        Mq = M.to_DM(QQ)

        method = _rref_QQ_fastest_method(Mq)

        if method == 'rref_QQ':
            # Sparse or has large denominators, use Gauss-Jordan over QQ.
            M_rref_q, _ = Mq.rref()
            return M_rref_q.to_Matrix()

        elif method == 'rref_ZZ_clear_denoms':
            # Clear small denominators and use fraction-free Gauss-Jordan over ZZ.
            _, Mz = Mq.clear_denoms(convert=True)
            M_rref_z, den, _ = Mz.rref_den()
            return M_rref_z.to_Matrix() / den

        elif method == 'rref_den_QQ':
            # Use fraction-free Gauss-Jordan over QQ.
            M_rref_q, den, _ = Mq.rref_den()
            return M_rref_q.to_Matrix() / den

        else:
            assert False


else:
    # With the PR that adds hybrid rref to DM.

    def rref_DMd_QQ(M):
        """DM dense .rref() over QQ"""
        M_rref, _ = M.to_DM(QQ).to_dense().rref_gj_div()
        return M_rref.to_Matrix()


    def rref_DMs_QQ(M):
        """DM sparse .rref() over QQ"""
        M_rref, _ = M.to_DM(QQ).rref_gj_div()
        return M_rref.to_Matrix()


    def rref_DMd_ZZ1(M):
        """DM dense .rref_den() over ZZ, divide Matrix"""
        M_rref, den, _ = M.to_DM(ZZ).to_dense().rref_den_gj_ff()
        return M_rref.to_Matrix()/den


    def rref_DMd_ZZ2(M):
        """DM dense .rref_den() over ZZ, divide DM"""
        M_rref, den, _ = M.to_DM(ZZ).to_dense().rref_den_gj_ff()
        return (M_rref/den).to_Matrix()


    def rref_DMs_ZZ1(M):
        """DM sparse .rref_den() over ZZ, divide Matrix"""
        M_rref, den, _ = M.to_DM(ZZ).rref_den_gj_ff()
        return M_rref.to_Matrix()/den


    def rref_DMs_ZZ2(M):
        """DM sparse .rref_den() over ZZ, divide DM"""
        M_rref, den, _ = M.to_DM(ZZ).rref_den_gj_ff()
        return (M_rref/den).to_Matrix()


    def rref_DMd_QQc(M):
        """DM clear_denoms, dense .rref_den() over QQ."""
        d, Mq = M.to_DM(QQ).to_dense().clear_denoms()
        M_rref_q, den, _ = Mq.rref_den_gj_ff()
        return M_rref_q.to_Matrix()/den


    def rref_DMs_QQc(M):
        """DM clear_denoms, sparse .rref_den() over QQ."""
        _, Mz = M.to_DM(QQ).clear_denoms()
        M_rref_z, den, _ = Mz.rref_den_gj_ff()
        return M_rref_z.to_Matrix()/den


    def rref_DMd_ZZc(M):
        """DM clear_denoms, dense .rref_den() over ZZ."""
        _, Mz = M.to_DM(QQ).clear_denoms(convert=True)
        M_rref_z, den, _ = Mz.to_dense().rref_den_gj_ff()
        return (M_rref_z/den).to_Matrix()


    def rref_DMs_ZZc(M):
        """DM clear_denoms, sparse .rref_den() over ZZ."""
        _, Mz = M.to_DM(QQ).clear_denoms()
        M_rref_z, den, _ = Mz.rref_den_gj_ff()
        return M_rref_z.to_Matrix()/den


    def rref_DMd_QQff(M):
        """DM fraction_free, dense .rref_den() over QQ."""
        Mq = M.to_DM(QQ).to_dense()
        M_rref_q, den, _ = Mq.rref_den_gj_ff()
        return M_rref_q.to_Matrix()/den


    def rref_DMs_QQff(M):
        """DM fraction_free, sparse .rref_den() over QQ."""
        Mq = M.to_DM(QQ)
        M_rref_q, den, _ = Mq.rref_den_gj_ff()
        return M_rref_q.to_Matrix()/den


    def rref_DMs_QQ_hybrid(M):
        """DM sparse .rref() over QQ, best algorithm."""
        return M.to_DM(QQ).rref()[0].to_Matrix()


    def rref_DMs_ZZ_hybrid(M):
        """DM sparse .rref_den(), best method."""
        M_rref_z, den, _ = M.to_DM(ZZ).rref_den(keep_domain=False)
        return M_rref_z.to_Matrix()/den


def _rref_ZZ_fastest_method(Mz):
    """Return True if rref_QQ should be used."""
    # In the extreme of very sparse matrices and low bit counts it is faster to
    # use Gauss-Jordan elimination over QQ rather than fraction-free
    # Gauss-Jordan over ZZ. In the opposite extreme of dense matrices and high
    # bit counts it is faster to use fraction-free Gauss-Jordan over ZZ. These
    # two extreme cases need to be handled differently because they lead to
    # different asymptotic complexities. In between these two extremes we need
    # a threshold for deciding which method to use. This threshold is
    # determined empirically by timing the two methods with random matrices.

    # Empirically determined parameter.
    PARAM = 10000

    # First compute the density. This is the average number of non-zero entries
    # per row but only counting rows that have at least one non-zero entry
    # since RREF can ignore fully zero rows.
    rows_nz = Mz.rep.values()
    nrows = len(rows_nz)
    ncols = Mz.shape[1]
    density = sum(map(len, rows_nz))/nrows if nrows else 0

    # For small matrices use QQ if more than half the entries are zero.
    if nrows < 10:
        if density < ncols/2:
            return 'rref_QQ'
        else:
            return 'rref_den_ZZ'

    # These are just shortcuts for the formula below.
    if density < 5:
        return 'rref_QQ'
    elif density > 5 + PARAM/nrows:
        return 'rref_den_ZZ'

    elements, _ = Mz.to_flat_nz()
    bits = max([e.bit_length() for e in elements], default=1)
    wideness = max(1, 2/3*ncols/nrows) # 1 unless ncols > 1.5*nrows
    max_density = (5 + PARAM/(nrows*bits**2)) * wideness

    if density < max_density:
        return 'rref_QQ'
    else:
        return 'rref_den_ZZ'


def _rref_QQ_fastest_method(Mq):
    """Return True if rref_den should be used."""
    # The same sorts of considerations apply here as in the case of ZZ. Here
    # though a new more significant consideration is what sort of denominators
    # we have and what to do with them so we focus on that firstly.

    # First compute the density. This is the average number of non-zero entries
    # per row but only counting rows that have at least one non-zero entry
    # since RREF can ignore fully zero rows.
    rows_nz = Mq.rep.values()
    nrows = len(rows_nz)
    ncols = Mq.shape[1]
    density = sum(map(len, rows_nz))/nrows if nrows else 0

    # For sparse matrices use Gauss-Jordan elimination over QQ regardless.
    if density < min(5, ncols/2):
        return 'rref_QQ'

    # Compare the bit-length of the lcm of the denominators to the bit length
    # of the numerators.
    #
    # The threshold here is empirical. It does sort of make sense though
    # because using rref with QQ for a pure integer matrix is often about 4x
    # slower than rref_den over ZZ due to gcd overhead vs fraction-free
    # division. Here we prefer rref over QQ if clearing denominators would
    # result in a numerator matrix having 5x the bit size of the current
    # numerators.
    elements, _ = Mq.to_flat_nz()
    numer_bits = max([e.numerator.bit_length() for e in elements], default=1)

    denom_lcm = ZZ.one
    for e in elements:
        denom_lcm = ZZ.lcm(denom_lcm, e.denominator)
        if denom_lcm.bit_length() > 5*numer_bits:
            return 'rref_QQ'

    if denom_lcm.bit_length() < 50:
        # This is probably very common. A bit length of 50 is large compared to
        # fractions like 1/2 or 1/3 often seen in user inputs.
        return 'rref_ZZ_clear_denoms'
    else:
        # The lcm is large, but not too large compared to the bit length of the
        # numerators. Using rref_den is often about the same speed as clearing
        # denominators but there is an intermediate regime where rref_den is a
        # lot faster.
        return 'rref_den_QQ'


rref_methods_zz = {
    'Matrix': rref_Matrix,
    'DMd_QQ': rref_DMd_QQ,
    #'DMd_ZZ1': rref_DMd_ZZ1,
    'DMd_ZZ2': rref_DMd_ZZ2,
    'DMs_QQ': rref_DMs_QQ,
    #'DMs_ZZ1': rref_DMs_ZZ1,
    'DMs_ZZ2': rref_DMs_ZZ2,
    'DMs_ZZ_hybrid': rref_DMs_ZZ_hybrid,
}


rref_methods_qq = {
    'Matrix': rref_Matrix,
    'DMd_QQ': rref_DMd_QQ,
    'DMd_ZZc': rref_DMd_ZZc,
    'DMd_QQc': rref_DMd_QQc,
    'DMd_QQff': rref_DMd_QQff,
    'DMs_QQ': rref_DMs_QQ,
    'DMs_ZZc': rref_DMs_ZZc,
    'DMs_QQc': rref_DMs_QQc,
    'DMs_QQff': rref_DMs_QQff,
    'DMs_QQ_hybrid': rref_DMs_QQ_hybrid,
}


rref_methods = {
    'ZZ': rref_methods_zz,
    'QQ': rref_methods_qq,
}


# --------------------------------------------------------------------------- #
#                      Matrix .charpoly() methods.                            #
# --------------------------------------------------------------------------- #


def charpoly_Matrix(M):
    """Matrix .charpoly() method."""
    return M.charpoly()


def charpoly_DMd(M):
    """DM dense .charpoly()"""
    dM = M.to_DM().to_dense()
    p = dM.charpoly()
    l = Dummy('lambda')
    return PurePoly(p, l, domain=dM.domain)


def charpoly_DMs(M):
    """DM sparse .charpoly()"""
    dM = M.to_DM()
    p = dM.charpoly()
    l = Dummy('lambda')
    return PurePoly(p, l, domain=dM.domain)


def charpoly_DMd_base(M):
    """DM dense .charpoly(), base method"""
    dM = M.to_DM().to_dense()
    p = dM.charpoly_base()
    l = Dummy('lambda')
    return PurePoly(p, l, domain=dM.domain)


def charpoly_DMc(M):
    """Compute charpoly, clearing denominators."""
    dM = M.to_DM()
    K = dM.domain
    d, dM = dM.clear_denoms(convert=True)
    d = getattr(d, 'element', d)
    p = dM.charpoly()
    l = Dummy('lambda')
    pp = PurePoly(p, l, domain=dM.domain)
    return pp.transform(Poly(l, l), Poly(K(1)/d, l))


charpoly_methods_domain = {
    'Matrix': charpoly_Matrix,
    'DMd': charpoly_DMd,
    'DMs': charpoly_DMs,
    'DMd_base': charpoly_DMd_base,
}


charpoly_methods_field = {
    'DMc': charpoly_DMc,
}


charpoly_methods = {
    'ZZ': charpoly_methods_domain,
    'QQ': charpoly_methods_domain | charpoly_methods_field,
}


# --------------------------------------------------------------------------- #
#                      Operations                                             #
# --------------------------------------------------------------------------- #


operations = {
    'rref': rref_methods,
    'charpoly': charpoly_methods,
}


if __name__ == '__main__':
    main(*sys.argv[1:])

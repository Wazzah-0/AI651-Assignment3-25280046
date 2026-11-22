"""AI651 PA3 Q1 verification. Run: python q1_verify.py"""
from fractions import Fraction
import numpy as np

K = 8


def neighbourhoods(kmax=K):
    return [(nm, nz, n-nm-nz)
            for n in range(1, kmax+1)
            for nm in range(n+1)
            for nz in range(n-nm+1)]

NBRS = neighbourhoods()


def q1a():
    print('\nQ1(a): affine-message collision')
    seen = {}
    for counts in NBRS:
        key = (sum(counts), counts[2] - counts[0])
        if key in seen and seen[key][1] != counts[1]:
            a = seen[key]
            print(f'  {a} versus {counts}: both (n, S)={key}, different steady counts')
            assert all(w*key[1]+c*key[0] == w*key[1]+c*key[0]
                       for w,c in [(1,0),(-2.5,3),(0.7,-1.3)])
            return
        seen.setdefault(key, counts)
    raise AssertionError('No witness found')


def q1b():
    print('\nQ1(b): base-(K+1) encoding and NumPy verification')
    B = K + 1
    counts = np.asarray(NBRS, dtype=np.int64)
    messages = np.array([1, B, B**2], dtype=np.int64)
    encoded = counts @ messages
    decoded = np.column_stack((encoded%B, (encoded//B)%B, encoded//B**2))
    assert np.unique(encoded).size == len(counts) == 164
    assert np.array_equal(decoded, counts)
    print(f'  phi(-1), phi(0), phi(+1) = {tuple(messages)}')
    print(f'  All {len(counts)} allowed triples encoded uniquely and decoded correctly.')

    full = np.array([(a,b,c) for a in range(K+1)
                    for b in range(K+1) for c in range(K+1)], dtype=np.int64)
    assert np.unique(full @ messages).size == (K+1)**3
    print(f'  Stronger check: all {len(full)} coordinate-bounded triples separate.')
    assert np.dot([K,0,0], [1,K,K*K]) == np.dot([0,1,0], [1,K,K*K])
    print(f'  Base K={K} has collision (K,0,0) vs (0,1,0).')

    nonsteady_messages = np.array([1,0,1], dtype=np.int64)
    assert np.array_equal(counts @ nonsteady_messages, counts[:,0]+counts[:,2])
    print('  Non-steady count: phi(-1)=1, phi(0)=0, phi(+1)=1.')


def aggregate(counts, kind):
    if kind == 'Sum':
        return counts
    if kind == 'Mean':
        return tuple(Fraction(c, sum(counts)) for c in counts)
    if kind == 'Max':
        return tuple(int(c > 0) for c in counts)
    raise ValueError(kind)


def q1c():
    print('\nQ1(c): exhaustive aggregation/task table')
    tasks = {
        'count steady': lambda x: x[1],
        'more than half slowing': lambda x: 2*x[0] > sum(x),
        'any slowing': lambda x: x[0] > 0,
    }
    expected = {
        'count steady': {'Sum':True,'Mean':False,'Max':False},
        'more than half slowing': {'Sum':True,'Mean':True,'Max':False},
        'any slowing': {'Sum':True,'Mean':True,'Max':True},
    }
    for task_name, task in tasks.items():
        for kind in ('Sum','Mean','Max'):
            seen = {}
            collision = None
            for counts in NBRS:
                key = aggregate(counts, kind)
                answer = task(counts)
                if key in seen and seen[key][1] != answer:
                    collision = (seen[key], (counts, answer), key)
                    break
                seen.setdefault(key, (counts, answer))
            possible = collision is None
            assert possible == expected[task_name][kind]
            if possible:
                print(f'  {task_name:25} {kind:4} YES')
            else:
                print(f'  {task_name:25} {kind:4} NO: {collision}')


if __name__ == '__main__':
    q1a()
    q1b()
    q1c()
    print('\nAll Q1 checks passed.')

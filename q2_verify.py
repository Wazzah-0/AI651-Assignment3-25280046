"""AI651 PA3 Q2 verification + heatmaps. Run: python q2_verify.py"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 40
CENTRE = 20
STEPS = 20
RNG = np.random.default_rng(0)


def shift_back(x):
    """Given motion rule x_i(t+1) = x_(i+1)(t)."""
    return np.roll(x, -1)


def shift_fwd(x):
    """Reversed motion rule x_i(t+1) = x_(i-1)(t)."""
    return np.roll(x, +1)


def symmetric_layer(x,a,b):
    return a*x + b*(np.roll(x,+1)+np.roll(x,-1))


def q2a_fit():
    print('\nQ2(a)i: fit symmetric layer')
    x = np.zeros(N)
    x[CENTRE] = 1
    target = shift_back(x)
    A = np.column_stack((x, np.roll(x,+1)+np.roll(x,-1)))
    a,b = np.linalg.lstsq(A,target,rcond=None)[0]
    sse = np.sum((A @ np.array([a,b]) - target)**2)
    assert np.isclose(a,0) and np.isclose(b,.5) and np.isclose(sse,.5)
    print(f'  SSE = a² + (b-1)² + b²; a={a:g}, b={b:g}, minimum SSE={sse:g}')
    return a,b


def circ_spread(v):
    theta = np.arange(N) * 2*np.pi/N
    weight = v/v.sum()
    R = np.hypot(np.sum(weight*np.cos(theta)), np.sum(weight*np.sin(theta)))
    return np.sqrt(-2*np.log(R))*N/(2*np.pi)


def q2a_rollout(a,b):
    print('\nQ2(a)ii: 20-step smooth bump rollout')
    idx = np.arange(N)
    x0 = np.exp(-((idx-CENTRE)**2)/8)
    pred = np.zeros((STEPS+1,N))
    truth = np.zeros_like(pred)
    pred[0] = truth[0] = x0
    for t in range(STEPS):
        pred[t+1] = symmetric_layer(pred[t],a,b)
        truth[t+1] = shift_back(truth[t])
    print(f'  Final prediction: peak={pred[-1].max():.4f} at {pred[-1].argmax()}, spread={circ_spread(pred[-1]):.3f}')
    print(f'  Final truth: peak={truth[-1].max():.4f} at {truth[-1].argmax()}, spread={circ_spread(truth[-1]):.3f}')
    print(f'  Mass: initial={x0.sum():.6f}; prediction={pred[-1].sum():.6f}; truth={truth[-1].sum():.6f}')
    assert np.isclose(pred[-1].sum(),x0.sum()) and np.isclose(truth[-1].sum(),x0.sum())
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    vmax = max(pred.max(),truth.max())
    for ax, arr, title in zip(axes,(pred,truth),('Fitted symmetric layer','True backward motion')):
        im = ax.imshow(arr,aspect='auto',origin='upper',cmap='magma',vmin=0,vmax=vmax,
                       extent=[-.5,N-.5,STEPS+.5,-.5])
        ax.set(xlabel='Sensor index', ylabel='Time step', title=title)
    fig.colorbar(im,ax=axes,label='Reading',shrink=.9)
    out = Path(__file__).resolve().parent/'q2a_bump_heatmaps.png'
    fig.savefig(out,dpi=160)
    plt.close(fig)
    print(f'  Saved {out}')


def q2b():
    print('\nQ2(b): directed weights and cross-correlation kernel')
    a,wm,wp = 0,0,1
    for _ in range(200):
        x = RNG.standard_normal(N)
        result = a*x + wm*np.roll(x,+1) + wp*np.roll(x,-1)
        assert np.allclose(result,shift_back(x))
    print('  a=0, w_minus=0, w_plus=1; kernel [0,0,1]; circular padding. OK')


def history_layer(x_t,x_prev):
    return np.roll(x_t,+1) + np.roll(x_t,-1) - x_prev


def q2c():
    print('\nQ2(c): history-based symmetric layer')
    for direction,step,reverse in (
        ('backward',shift_back,shift_fwd),
        ('forward',shift_fwd,shift_back),
    ):
        for _ in range(500):
            x_t = RNG.standard_normal(N)
            x_prev = reverse(x_t)
            assert np.allclose(history_layer(x_t,x_prev),step(x_t))
        print(f'  {direction} motion: 500/500 exact')
    for _ in range(500):
        left, right = RNG.standard_normal((2,N))
        x_prev = left+right
        x_t = shift_back(left)+shift_fwd(right)
        target = shift_back(shift_back(left))+shift_fwd(shift_fwd(right))
        assert np.allclose(history_layer(x_t,x_prev),target)
    print('  Superposed left and right movers: 500/500 exact')
    print('  a=0, b=1, c=-1; x_i(t+1)+x_i(t-1)=x_(i-1)(t)+x_(i+1)(t).')


if __name__ == '__main__':
    a,b = q2a_fit()
    q2a_rollout(a,b)
    q2b()
    q2c()
    print('\nAll Q2 checks passed.')

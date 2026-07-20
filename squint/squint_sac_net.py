"""Flax networks for the Squint-style SAC+C51 trainer (arXiv 2602.21203).

  - ONE shared CNN encoder (16x16: Conv4x4/s2 -> Conv4x4/s1 -> flatten, repr 1024). Trained by the
    critic loss; the actor consumes DETACHED encoder features (standard SAC shared-encoder trick).
  - Actor & Critic each own a SEPARATE Projection: rgb Linear(1024->50)+LN+Tanh, state
    Linear(n->256)+LN+ReLU, concat -> 306.
  - Actor: 3x [Linear(256)+LN+ReLU] -> (mean, log_std); log_std tanh-squashed to [-5, 2]; squashed
    Gaussian. Action space is [-1,1]^6 so action = tanh(x) (scale 1, bias 0).
  - Critic: C51 distributional, ensemble of num_q=2; Projection -> +action -> 3x [Linear(512)+LN+ReLU]
    -> Linear(101 atoms); support linspace(-20, 20, 101).

Init: orthogonal (Conv gain sqrt(2), Linear gain 1), bias 0.
"""
import jax
import jax.numpy as jp
from flax import linen

REPR_DIM = 1024
# C51 support: 101 atoms over [-20,20], sized for the NORMALIZED reward (SquintEnv ~1/step ->
# discounted return ~10 at gamma 0.9). A much wider support would leave the achievable return in a
# tiny slice of the atoms, with spacing too coarse to resolve the per-step lift advantage (~0.6).
# dz = 0.4.
NUM_ATOMS = 101
V_MIN, V_MAX = -20.0, 20.0
NUM_Q = 2
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

_ORTH_RELU = linen.initializers.orthogonal(jp.sqrt(2.0))   # Conv/ReLU layers
_ORTH_1 = linen.initializers.orthogonal(1.0)               # Linear heads
_ZEROS = linen.initializers.zeros


class CNNEncoder(linen.Module):
    """16x16 RGB -> 1024 features."""
    @linen.compact
    def __call__(self, pix):                                 # pix: (...,16,16,C) in [0,1]
        x = pix - 0.5                                        # center: env gives [0,1]
        # padding='VALID' (PyTorch-style no pad): 16 ->7 ->4 -> 4x4x64 = 1024 (flax defaults to
        # SAME, which would silently give 4096).
        x = linen.relu(linen.Conv(32, (4, 4), (2, 2), padding="VALID",
                                  kernel_init=_ORTH_RELU, bias_init=_ZEROS)(x))
        x = linen.relu(linen.Conv(64, (4, 4), (1, 1), padding="VALID",
                                  kernel_init=_ORTH_RELU, bias_init=_ZEROS)(x))
        return x.reshape(x.shape[:-3] + (-1,))               # flatten -> 4x4x64 = 1024


class Projection(linen.Module):
    """rgb(1024->50,LN,Tanh) || state(n->256,LN,ReLU) -> 306."""
    @linen.compact
    def __call__(self, rgb_feat, state):
        r = linen.Dense(50, kernel_init=_ORTH_1, bias_init=_ZEROS)(rgb_feat)
        r = jp.tanh(linen.LayerNorm()(r))
        s = linen.Dense(256, kernel_init=_ORTH_1, bias_init=_ZEROS)(state)
        s = linen.relu(linen.LayerNorm()(s))
        return jp.concatenate([r, s], axis=-1)               # 306


class Actor(linen.Module):
    """Shared-encoder features + proprio -> squashed-Gaussian (mean, log_std)."""
    action_size: int

    @linen.compact
    def __call__(self, rgb_feat, state):
        x = Projection()(rgb_feat, state)
        for _ in range(3):  # ALL Linear layers use orthogonal gain 1.0 (only Conv gets sqrt2)
            x = linen.relu(linen.LayerNorm()(linen.Dense(256, kernel_init=_ORTH_1, bias_init=_ZEROS)(x)))
        mean = linen.Dense(self.action_size, kernel_init=_ORTH_1, bias_init=_ZEROS)(x)
        log_std = linen.Dense(self.action_size, kernel_init=_ORTH_1, bias_init=_ZEROS)(x)
        log_std = jp.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return mean, log_std


def sample_action(mean, log_std, key):
    """Squashed-Gaussian sample + tanh-corrected log-prob (action space [-1,1], scale 1 bias 0)."""
    std = jp.exp(log_std)
    x = mean + std * jax.random.normal(key, mean.shape)      # rsample
    y = jp.tanh(x)
    # log N(x) - sum log(1 - tanh(x)^2 + 1e-6)
    log_prob = -0.5 * (((x - mean) / (std + 1e-8)) ** 2 + 2 * jp.log(std + 1e-8) + jp.log(2 * jp.pi))
    log_prob = log_prob - jp.log(1.0 - y ** 2 + 1e-6)
    return y, log_prob.sum(-1, keepdims=True)


class QMLP(linen.Module):
    """One C51 Q-MLP (no projection): (proj||action) -> 3x[Dense512+LN+ReLU] -> 101 atom logits."""
    @linen.compact
    def __call__(self, x):
        for _ in range(3):  # gain 1.0, not sqrt2
            x = linen.relu(linen.LayerNorm()(linen.Dense(512, kernel_init=_ORTH_1, bias_init=_ZEROS)(x)))
        return linen.Dense(NUM_ATOMS, kernel_init=_ORTH_1, bias_init=_ZEROS)(x)   # atom logits


class Critic(linen.Module):
    """ONE shared Projection -> concat action -> vmap NUM_Q Q-MLPs over it."""
    @linen.compact
    def __call__(self, rgb_feat, state, action):
        proj = Projection()(rgb_feat, state)                 # shared across the Q-ensemble
        x = jp.concatenate([proj, action], axis=-1)
        VmapQ = linen.vmap(QMLP, variable_axes={"params": 0}, split_rngs={"params": True},
                           in_axes=None, out_axes=0, axis_size=NUM_Q)
        return VmapQ()(x)                                    # (NUM_Q, ..., NUM_ATOMS)


Q_SUPPORT = jp.linspace(V_MIN, V_MAX, NUM_ATOMS)


def expected_q(logits):
    """(NUM_Q, batch, atoms) logits -> (NUM_Q, batch) expected Q-values."""
    return jp.sum(jax.nn.softmax(logits, -1) * Q_SUPPORT, -1)

# simulation.py
# Continuous Hopfield-like dream simulation with emergent collapse:
# - pattern depths (attractor strengths)
# - noise-effectiveness that decreases as coherence increases (noise-fatigue)
# - no manual "countdown" — collapse emerges from dynamics

import numpy as np

def shannon_entropy(probs, eps=1e-12):
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log2(probs))

def softmax(x, temp=1.0):
    ex = np.exp((x - np.max(x)) / temp)
    return ex / np.sum(ex)

class DreamSimulator:
    def __init__(self, N=120, P=6, emotion_frac=0.15, deep_index=0, deep_strength=2.5, seed=42):
        """
        N: number of nodes
        P: number of stored patterns
        emotion_frac: fraction of nodes treated as 'emotion' nodes
        deep_index: index of pattern to make deeper (self-attractor)
        deep_strength: multiplicative strength of deep attractor (s_deep)
        """
        np.random.seed(seed)
        self.N = N
        self.P = P
        self.emotion_frac = emotion_frac
        self.emotion_nodes = np.arange(int(N * emotion_frac))
        # create patterns (binary -1,+1) with small overlap structure
        patterns = np.random.choice([-1, 1], size=(P, N))
        for i in range(1, P):
            patterns[i] = np.where(np.random.rand(N) < 0.85, patterns[i], patterns[i-1])
        self.patterns = patterns
        # pattern depths (attractor strengths) - baseline 1.0
        depths = np.ones(P)
        depths[deep_index] = deep_strength  # make one attractor deeper
        self.depths = depths
        # Hebbian-like weighted matrix with pattern depths
        W = np.zeros((N, N))
        for mu in range(P):
            W += depths[mu] * np.outer(patterns[mu], patterns[mu])
        W = W / N
        np.fill_diagonal(W, 0.0)
        self.W = W

    def energy(self, state):
        return -0.5 * state.T @ self.W @ state

    def narrative_probs(self, state, temp=0.5):
        dots = self.patterns @ state
        return softmax(dots, temp=temp)

    def emotional_entropy(self, state, bins=12):
        acts = state[self.emotion_nodes]
        hist, _ = np.histogram(acts, bins=bins, range=(-1,1), density=True)
        probs = hist / np.sum(hist) if np.sum(hist) > 0 else np.ones_like(hist)/len(hist)
        return shannon_entropy(probs)

    def _coherence(self, state):
        # coherence = mean absolute activation (0..1)
        return float(np.mean(np.abs(state)))

    def run(self,
            base_noise=0.14,
            lucid=False,
            lucid_self_strength=1.8,
            update_tau=0.25,
            timesteps=600,
            stop_eps=1e-4,
            noise_coherence_k=12.0,
            noise_coherence_mid=0.25):
        """
        base_noise: base Gaussian noise scale
        noise_coherence_k, noise_coherence_mid: parameters for sigmoid gating of noise by coherence
        stop_eps: stop when TDE < stop_eps (emergent collapse)
        """
        state = np.tanh(0.05 * np.random.randn(self.N))
        em_ent = []
        nar_ent = []
        TDE = []
        energy = []
        sims = []
        for t in range(timesteps):
            # coherence and dynamic noise effectiveness (noise-fatigue)
            C = self._coherence(state)  # between 0 and 1
            # sigmoid gate: near 0 coherence -> gate ~1; high coherence -> gate ~0
            gate = 1.0 / (1.0 + np.exp(noise_coherence_k * (C - noise_coherence_mid)))
            noise_effective = base_noise * gate
            noise = np.random.normal(scale=noise_effective, size=self.N)
            # drive from weights
            input_drive = self.W @ state
            dx = -state + input_drive + noise
            if lucid:
                # stabilizing self-attractor (in addition to deep pattern)
                self_vector = np.mean(self.patterns, axis=0)
                dx += lucid_self_strength * self_vector
            # small nonlinear competition: saturating update
            state = state + update_tau * dx
            state = np.tanh(state)
            # measurements
            e_ent = self.emotional_entropy(state)
            dots = self.patterns @ state
            probs = softmax(dots, temp=0.45)
            n_ent = shannon_entropy(probs)
            tde = 0.6 * e_ent + 0.4 * n_ent
            en = self.energy(state)
            # record
            em_ent.append(e_ent); nar_ent.append(n_ent); TDE.append(tde); energy.append(en); sims.append(dots.copy())
            # emergent stop: when TDE is extremely small (network locked to a single attractor)
            if tde < stop_eps:
                break
        import numpy as _np
        return {
            'em_ent': _np.array(em_ent),
            'nar_ent': _np.array(nar_ent),
            'TDE': _np.array(TDE),
            'energy': _np.array(energy),
            'sims': _np.array(sims),
            'final_coherence': self._coherence(state),
            'timesteps': len(TDE)
        }

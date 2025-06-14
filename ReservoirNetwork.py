import numpy as np
from   scipy.sparse import random as sparse_random, csr_matrix
from   scipy.special import expit
from   joblib import Parallel, delayed
from   scipy.sparse.linalg import eigs

class ReservoirNetwork:
    """
    Class implementing a reservoir network with history-dependent weights and state updates.
    """

    def __init__(
        self,
        Fs                   = 1000,
        fmin                 = 5,
        fstep                = 1,                         # Optimized parameter
        sigma                = 0.005378072060537638,      # Optimized parameter
        sparsity             = 0.35379850103623395,       # Optimized parameter
        spectral_radius      = 0.4264161247207072,        # Optimized parameter
        base_geometric_ratio = 0.9300355782205447,        # Optimized parameter
        random_state         = None,
    ):

        self.Fs              = Fs
        self.fmin            = fmin
        self.fmax            = Fs/2                       # By default, set fmax to Nyquist frequency
        self.fstep           = fstep
        self.sigma           = sigma
        self.sparsity        = sparsity
        self.spectral_radius = spectral_radius
        self.base_geometric_ratio = base_geometric_ratio
        self.random_state    = random_state
        
        # History weights
        self.history_weights, self.frange = self.generate_history_weights()
        self.K = np.size(self.history_weights['w_t_minus_1'])

        # Get omega and beta parameters
        a1         = self.history_weights["w_t_minus_1"]
        a2         = self.history_weights["w_t_minus_2"]
        dt         = 1/self.Fs
        self.omega = np.sqrt((a1 + a2 - 1)/(a2 * dt**2))
        self.beta  = -(a1 + 2*a2)/(2*a2*dt)

        # Initialize random seed
        self.rng = np.random.default_rng(self.random_state)
        
        # Initialize internal states
        self.x_t         = np.zeros(self.K)
        self.x_t_minus_1 = np.zeros(self.K)
        self.x_t_minus_2 = np.zeros(self.K)
        self.A_t         = np.zeros(self.K)
        self.phi_t       = np.zeros(self.K)

        # Input weight matrix W_in (K x 1)
        self.W_in  = np.ones((self.K, 1))

        # # Generate sparse random recurrent weight matrix W_res (K x K)
        self.W_res = self.generate_reservoir_weights()
    
    def generate_reservoir_weights(self):
        """
        Generate a sparse random recurrent weight matrix W_res and scale it to the desired spectral radius.
        """

        rng = self.rng
        
        # 1) build a sparse random matrix (CSR) directly
        W_res = sparse_random(
            self.K, self.K,
            density=self.sparsity,
            data_rvs=lambda n: rng.uniform(-1, 0, size=n),
            format='csr',
            random_state=self.random_state
        )

        # 2) zero out the diagonal in-place
        W_res.setdiag(0)

        # 3) compute the leading eigenvalue on the sparse matrix
        vals = eigs(W_res, k=1, which='LM', return_eigenvectors=False)
        max_eig = np.abs(vals[0])
        if max_eig > 0:
            # scale the sparse matrix without densifying
            W_res *= (self.spectral_radius / max_eig)

        return W_res

    def generate_history_weights(self):
        """
        Create the 'history_weights': w_t_minus_1 and w_t_minus_2 while ensuring:
        1) The roots of the characteristic equation remain within the unit circle.
        2) w_t_minus_1 > 0.
        """
        
        frange = np.arange(self.fmin, self.fmax+1, self.fstep)        
        Fs     = self.Fs
        r      = self.base_geometric_ratio
    
        # Compute initial weights
        w_t_minus_1 = 2 * r * np.cos(2 * np.pi * frange / Fs)
        w_t_minus_2 = (-r**2) * np.ones_like(w_t_minus_1)

        # Apply stability constraints
        discriminant      = w_t_minus_1**2 + 4*w_t_minus_2 + 0j
        sqrt_discriminant = np.sqrt(discriminant)
        z1   = (w_t_minus_1 + sqrt_discriminant)/2
        z2   = (w_t_minus_1 - sqrt_discriminant)/2

        r1 = np.abs(z1)
        r2 = np.abs(z2)
        
        # Ensure |r1| < 1 and |r2| < 1, and w_t_minus_1 > 0
        final_valid = (r1 < 1) & (r2 < 1) & (w_t_minus_1 > 0)
    
        # Apply final constraints
        w_t_minus_1 = w_t_minus_1[final_valid]
        w_t_minus_2 = w_t_minus_2[final_valid]
        frange = frange[final_valid]
        print('fmin=',np.min(frange),', fmax=',np.max(frange), ', fstep=', frange[1]-frange[0], ', N nodes=', np.size(frange))
    
        history_weights = {
            'w_t_minus_1': w_t_minus_1,
            'w_t_minus_2': w_t_minus_2
        }
    
        return history_weights, frange
        
    def reset_states(self):
        """
        Reset the internal states (x_t, x_t_minus_1, x_t_minus_2) of the network to zero.
        """
        self.x_t         = np.zeros(self.K)
        self.x_t_minus_1 = np.zeros(self.K)
        self.x_t_minus_2 = np.zeros(self.K)

    def update(self, u_t):
        """
        Update the reservoir state using the current input u_t, applying noise, activation functions, and history updates.
        """
        noise = self.rng.normal(0, self.sigma, size=self.K)
    
        pre_activation = (
            self.W_in.dot(u_t).flatten()                                   # input drive
            + self.history_weights['w_t_minus_1'] * self.x_t_minus_1       # 1-step history
            + self.history_weights['w_t_minus_2'] * self.x_t_minus_2       # 2-step history
            + self.W_res.dot(expit(np.cos(self.phi_t)))                    # recurrent drive
            + noise                                                        # noise
        )

        self.x_t = pre_activation
    
        # Vectorized amplitude and phase computation
        vt = (self.x_t - self.x_t_minus_1) * self.Fs  # Compute velocity
        self.A_t   = np.sqrt(self.x_t**2 + ((vt + self.beta * self.x_t) / self.omega) ** 2)
        self.phi_t = np.arctan2((vt + self.beta * self.x_t), (self.omega * self.x_t))
    
        # Update history
        self.x_t_minus_2[:] = self.x_t_minus_1
        self.x_t_minus_1[:] = self.x_t

    def collect_states(self, input_sequence):
        """
        Collect and return the network states, amplitudes, and phases for each time step in the input sequence.
        """
        T = input_sequence.shape[0]  # Number of time steps
        K = self.K                   # Number of reservoir neurons
    
        self.reset_states()
        
        states = np.zeros((T, K))
        amplitudes = np.zeros((T, K))
        phases = np.zeros((T, K))
    
        for t in range(T):
            u_t = input_sequence[t]  # Current input
            self.update(u_t)
            states[t] = self.x_t
            amplitudes[t] = self.A_t
            phases[t] = self.phi_t
    
        return states, amplitudes, phases

def _process_sequence(res_net, seq):
    # reshape & run the reservoir
    seq = seq.reshape(-1, 1)
    _, amp, _ = res_net.collect_states(seq)
    return np.mean(amp**2, axis=0)
    
def extract_reservoir_features(res_net, sequences, n_jobs=-1):
    """
    Parallel feature extraction using a module‐level helper to reduce pickling overhead.
    """
    features = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_sequence)(res_net, seq) for seq in sequences
    )
    return np.vstack(features)

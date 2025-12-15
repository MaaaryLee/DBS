"""
BGN environment that integrates with MATLAB Online.
This version can work with pre-computed data or trigger MATLAB Online simulations.
"""

import gymnasium as gym
import numpy as np
import scipy.io
import antropy
import os
import json
try:
    # When imported as `core.BGN_MC_Online`
    from core.matlab_online_workflow import MATLABOnlineBridge
except Exception:
    # When executed with `python core/BGN_MC_Online.py` (core/ on sys.path)
    from matlab_online_workflow import MATLABOnlineBridge

class BGN_MC_Online(gym.Env):
    def __init__(self, tmax=1000, pd=True, use_matlab_online=False):
        if pd: 
            self.pd = 1
        else: 
            self.pd = 0
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.action_space = gym.spaces.Box(low=np.array([-1.0, -1.0], dtype=np.float32), high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

        self.tmax = tmax
        self.past_sgis = 0
        self.current_step = 0
        self.max_steps = tmax // 100
        self.use_matlab_online = use_matlab_online
        
        # Initialize MATLAB Online bridge if needed
        if use_matlab_online:
            self.bridge = MATLABOnlineBridge()
        
        # Load pre-computed data as fallback
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
            vars_path = os.path.join(repo_root, "matlab", "bgn_vars.mat")
            self.bgn_data = scipy.io.loadmat(vars_path)
            self.sgis = self.bgn_data['sgis']
            self.vgi = self.bgn_data['vgi']
            self.vth = self.bgn_data['vth']
            self.vsn = self.bgn_data['vsn']
            self.vge = self.bgn_data['vge']
            # Some exports may omit Istim; keep it optional.
            Istim = self.bgn_data.get("Istim", np.array([]))
            self.Istim = np.asarray(Istim).flatten()
            self.has_precomputed_data = True
        except:
            self.has_precomputed_data = False
            print("⚠️  No pre-computed data found. MATLAB Online mode required.")
        
        # Constants from original implementation
        self.sgis_min = 1082.0999226306508
        self.sgis_max = 3506.499645178415
        self.sd_min = 0.07085760036983213
        self.sd_max = 0.2060141359064851
        self.A_min = 0.0050925265308826816
        self.A_max = 0.042526051971444476
        self.M_min = 0.002065105439508318
        self.M_max = 0.04131553203125676
        self.C_min = 0.10438751644244766
        self.C_max = 0.993985189386497
        self.Pb_min = 4547158.333768198
        self.Pb_max = 5801474.044129377
        self.SampEn_min = 0.001344649634141824
        self.SampEn_max = 0.005891566645584552

    def reset(self, pd=True, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.past_sgis = 0.5
        
        # Get initial observation
        observation, reward, terminated, truncated, info = self.step()
        self.past_sgis = info['r3']
        return observation, info

    def step(self, action=np.array([-1.0, -1.0]), sim_time=10000):
        action = np.clip(action, -1, 1)
        freq = 185 * ((action[0] + 1)/2)
        amp = 5000 * ((action[1] + 1)/2)
        
        self.current_step += 1
        terminated = 1 if self.current_step >= self.max_steps else 0
        
        if self.use_matlab_online:
            # Use MATLAB Online for simulation
            return self._step_with_matlab_online(freq, amp, sim_time, terminated)
        else:
            # Use pre-computed data
            return self._step_with_precomputed_data(freq, amp, sim_time, terminated)
    
    def _step_with_matlab_online(self, freq, amp, sim_time, terminated):
        """Step using MATLAB Online simulation."""
        # Prepare parameters for MATLAB Online
        self.bridge.prepare_simulation_params(freq, amp, self.pd, self.tmax)
        
        # Check if we have fresh results from MATLAB Online
        results_file = os.path.join(self.bridge.data_dir, "simulation_results.mat")
        
        if os.path.exists(results_file):
            # Load fresh results from MATLAB Online
            results = self.bridge.load_simulation_results()
            if results is not None:
                sgis = results['sgis']
                # Prefer signals from MATLAB Online if present; otherwise fall back to precomputed (if available).
                vgi = results.get('vgi', getattr(self, "vgi", None))
                vsn = results.get('vsn', getattr(self, "vsn", None))
            else:
                # Fallback to precomputed data
                sgis = getattr(self, "sgis", None)
                vgi = getattr(self, "vgi", None)
                vsn = getattr(self, "vsn", None)
        else:
            # No fresh results, use precomputed data
            if self.has_precomputed_data:
                sgis = self.sgis
                vgi = self.vgi
                vsn = self.vsn
            else:
                # Create dummy data if no precomputed data available
                sgis = np.random.randn(10, 1000) * 0.1
                vgi = np.random.randn(10, 1000) * 0.1
                vsn = np.random.randn(10, 1000) * 0.1

        # If MATLAB Online results are partial and no precomputed fallback exists, avoid crashes.
        if sgis is None:
            sgis = np.random.randn(10, 1000) * 0.1
        if vgi is None:
            vgi = np.random.randn(10, 1000) * 0.1
        if vsn is None:
            vsn = np.random.randn(10, 1000) * 0.1
        
        return self._calculate_reward_and_observation(sgis, vgi, vsn, freq, amp, terminated)
    
    def _step_with_precomputed_data(self, freq, amp, sim_time, terminated):
        """Step using pre-computed data."""
        if not self.has_precomputed_data:
            raise RuntimeError("No pre-computed data available and MATLAB Online not enabled")
        
        sgis = self.sgis
        vgi = self.vgi
        vsn = self.vsn
        
        return self._calculate_reward_and_observation(sgis, vgi, vsn, freq, amp, terminated)
    
    def _calculate_reward_and_observation(self, sgis, vgi, vsn, freq, amp, terminated):
        """Calculate reward and observation from simulation data."""
        # Calculate SGi normalization (key biomarker)
        current_sgis_norm = (np.sum(np.average(np.abs(np.fft.fft(sgis)), axis=0)[1:20]) - self.sgis_min)/(self.sgis_max-self.sgis_min)
        
        # Calculate reward components
        r1 = ((self.past_sgis - current_sgis_norm)-0)/(1-0)
        r2_max = 1
        theta = 0.85
        r2 = (theta * (freq/185) + (1-theta) * amp/5000)/r2_max
        r3 = current_sgis_norm
        r4 = 1 if r1==0 and r2==0 and r3==0 else 0

        self.past_sgis = current_sgis_norm

        # Calculate final reward
        epsilon = 0.68
        reward = epsilon * -r3 + (1-epsilon) * -r2

        # Calculate observation components
        if hasattr(self, "Istim") and len(self.Istim) > 0:
            i = len(self.Istim) - 1
        else:
            i = sgis.shape[1] - 1

        # Standard deviation of SGi
        sd = (np.average(np.std(sgis, axis=1)) - self.sd_min)/(self.sd_max-self.sd_min)

        # Amplitude of SGi
        A = np.average(np.var(sgis, axis=1))
        A_norm = (A-self.A_min)/(self.A_max-self.A_min)

        # Mobility of SGi
        M = np.average(np.sqrt(np.var(np.diff(sgis), axis=1)/A))
        M_norm = (M-self.M_min)/(self.M_max-self.M_min)

        # Complexity of SGi
        C = np.average(np.sqrt(np.var(np.diff(np.diff(sgis)), axis=1)/A) / M)
        C_norm = (C-self.C_min)/(self.C_max-self.C_min)

        # P-beta power
        Pb = (np.sum(np.average(np.abs(np.fft.fft(vgi[:, max(0, i-10000):i])) / 0.1, axis=0)[12:31]) - self.Pb_min)/(self.Pb_max - self.Pb_min)

        # Sample entropy
        try:
            SampEn = (np.average([antropy.sample_entropy(vsn[n][max(0, i-10000):i]) for n in range(10)]) - self.SampEn_min)/(self.SampEn_max - self.SampEn_min)
        except:
            SampEn = 0.5  # Default value if calculation fails

        observation = np.array((sd, A_norm, M_norm, C_norm, Pb, SampEn))

        return observation, reward, bool(terminated), False, {'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4}
    
    def trigger_matlab_online_simulation(self, freq, amp):
        """Manually trigger a MATLAB Online simulation."""
        if not self.use_matlab_online:
            print("⚠️  MATLAB Online mode not enabled. Enable with use_matlab_online=True")
            return False
        
        print(f"🚀 Triggering MATLAB Online simulation: freq={freq}Hz, amp={amp}mA")
        self.bridge.prepare_simulation_params(freq, amp, self.pd, self.tmax)
        print("📋 Next steps:")
        print("1. Upload matlab_data/simulation_params.json to MATLAB Online")
        print("2. Run the simulation in MATLAB Online")
        print("3. Download simulation_results.mat")
        print("4. Place it in matlab_data/ directory")
        return True

def test_online_environment():
    """Test the MATLAB Online environment."""
    print("=== Testing MATLAB Online Environment ===")
    
    # Test with pre-computed data first
    print("\n1. Testing with pre-computed data...")
    env = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=False)
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    # Test a few steps
    for i in range(3):
        action = np.array([0.5, 0.5])  # Medium DBS
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, terminated={terminated}")
    
    # Test MATLAB Online mode
    print("\n2. Testing MATLAB Online mode...")
    env_online = BGN_MC_Online(tmax=1100, pd=True, use_matlab_online=True)
    
    # Trigger a simulation
    env_online.trigger_matlab_online_simulation(130, 2500)
    
    print("\n✅ Environment tests completed!")

if __name__ == "__main__":
    test_online_environment()

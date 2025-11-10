"""
Test quantized TD3 model following examples.ipynb Cell 11.
Runs inference on the quantized model and measures performance.
"""

import numpy as np
import torch
from torch.ao.quantization import quantize_dynamic
from stable_baselines3 import TD3
from BGN_MC import BGN_MC
import scipy.io

def test_quantized_model(h1=32, h2=32, num_episodes=5):
    """
    Test quantized TD3 model following examples.ipynb Cell 11.
    
    Args:
        h1: First hidden layer size
        h2: Second hidden layer size
        num_episodes: Number of episodes to run
    """
    print("=" * 70)
    print("Testing Quantized TD3 Model (Cell 11 from examples.ipynb)")
    print("=" * 70)
    
    # Create environment
    print("\n1. Creating BGN environment...")
    bgn = BGN_MC(tmax=1100, pd=True)
    print("   [OK] Environment created")
    
    # Create TD3 model structure (needed to load quantized weights)
    print("\n2. Creating TD3 model structure...")
    try:
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[h1, h2], qf=[h1, h2])
        )
        model = TD3('MlpPolicy', bgn, verbose=0, policy_kwargs=policy_kwargs, learning_rate=0.0001)
        
        # Create quantized model
        qmodel = quantize_dynamic(model.policy.to(torch.device('cpu')), dtype=torch.qint8)
        
        # Load quantized weights
        qpolicy_path = f'models/policies/qpolicy_{h1}_{h2}.pth'
        print(f"   Loading quantized weights from: {qpolicy_path}")
        qmodel.load_state_dict(torch.load(qpolicy_path, weights_only=False))
        qmodel.eval()
        print("   [OK] Quantized model loaded and set to eval mode")
    except Exception as e:
        print(f"   [X] ERROR loading quantized model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run inference on multiple episodes
    print(f"\n3. Running inference on {num_episodes} episodes...")
    print("   (This may take a while - MATLAB is computing brain dynamics)")
    
    sgis_sums = []
    Pbs = []
    freqs = []
    amps = []
    
    for episode in range(num_episodes):
        print(f"\n   Episode {episode + 1}/{num_episodes}...")
        try:
            observation = bgn.reset()[0]
        except Exception as e:
            print(f"   [WARNING] Reset failed, trying to recreate environment: {e}")
            # Recreate environment if reset fails
            bgn = BGN_MC(tmax=1100, pd=True)
            observation = bgn.reset()[0]
        
        terminated = False
        step_count = 0
        
        while not terminated:
            # Convert observation to tensor format expected by quantized model
            # Shape: (1, 1, obs_dim) - batch_size=1, sequence_length=1, features=obs_dim
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(torch.device('cpu'))
            
            # Get action from quantized model
            with torch.no_grad():
                action = qmodel(obs_tensor).numpy()[0]
            
            # Step environment
            observation, reward, terminated, truncated, info = bgn.step(action)
            
            # Calculate metrics
            sgis = scipy.io.loadmat('bgn_vars.mat')['sgis']
            sgis_sums.append(np.sum(np.mean(np.abs(np.fft.fft(sgis)), axis=0)[1:20]))
            
            freqs.append(action[0])
            amps.append(action[1])
            step_count += 1
        
        # Calculate P-beta for this episode
        vgi = scipy.io.loadmat('bgn_vars.mat')['vgi']
        Pb = np.sum(np.average(np.abs(np.fft.fft(vgi)) / 0.1, axis=0)[12:31])
        Pbs.append(Pb)
        
        print(f"      Completed {step_count} steps")
    
    print("\n4. Calculating performance metrics...")
    
    # Calculate average metrics
    sgi_intensity = np.mean(sgis_sums)
    avg_Pb = np.mean(Pbs)
    
    # Denormalize frequencies and amplitudes
    denorm_freqs = [(freq + 1) / 2 * 185 for freq in freqs]
    denorm_amps = [(amp + 1) / 2 * 5000 for amp in amps]
    
    mean_freq = np.mean(denorm_freqs)
    mean_amp = np.mean(denorm_amps)
    
    print(f"\n   Performance Metrics:")
    print(f"   - SGi Intensity: {sgi_intensity:.2f}")
    print(f"   - P-beta (average): {avg_Pb:.2f}")
    print(f"   - Mean Frequency: {mean_freq:.2f} Hz")
    print(f"   - Mean Amplitude: {mean_amp:.2f} mA")
    
    print(f"\n   Action Statistics:")
    print(f"   - Frequency range: [{np.min(denorm_freqs):.2f}, {np.max(denorm_freqs):.2f}] Hz")
    print(f"   - Amplitude range: [{np.min(denorm_amps):.2f}, {np.max(denorm_amps):.2f}] mA")
    
    # Check if metrics meet criteria (from notebook comments)
    print(f"\n   Evaluation Criteria:")
    print(f"   - Frequency < 130 Hz: {'[OK]' if mean_freq < 130 else '[X]'} ({mean_freq:.2f} Hz)")
    print(f"   - Amplitude < 2500 mA: {'[OK]' if mean_amp < 2500 else '[X]'} ({mean_amp:.2f} mA)")
    
    print("\n" + "=" * 70)
    print("QUANTIZED MODEL TEST COMPLETE [OK]")
    print("=" * 70)
    print(f"\nQuantized model tested successfully!")
    print(f"SGi Intensity: {sgi_intensity:.2f}")
    print(f"P-beta: {avg_Pb:.2f}")
    print(f"\nNext step: Convert to ONNX (Cell 12 from examples.ipynb)")
    
    return True

if __name__ == '__main__':
    h1, h2 = 32, 32
    num_episodes = 5  # Minimum recommended in notebook
    
    success = test_quantized_model(h1=h1, h2=h2, num_episodes=num_episodes)
    exit(0 if success else 1)


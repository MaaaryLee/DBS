"""
Test quantized TD3 actor variants.
Runs inference on the quantized module and measures performance.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.io
import torch

# Importing quantize_model registers TD3Actor for torch.load()
import quantize_model  # noqa: F401
from BGN_MC import BGN_MC


_VARIANT_SUFFIX = {
    "fp32": "actor_fp32_{h1}_{h2}.pt",
    "dynamic_int8": "actor_int8_dynamic_{h1}_{h2}.pt",
    "static_int8": "actor_int8_static_{h1}_{h2}.pt",
}


def _load_actor(method: str, h1: int, h2: int) -> torch.nn.Module:
    if method not in _VARIANT_SUFFIX:
        raise ValueError(f"Unknown quantization method '{method}'")

    path = Path("models/policies") / _VARIANT_SUFFIX[method].format(h1=h1, h2=h2)
    if not path.exists():
        raise FileNotFoundError(f"Quantized actor not found: {path}")

    actor = torch.load(path, map_location="cpu")
    actor.eval()
    return actor


def test_quantized_model(h1: int = 32, h2: int = 32, num_episodes: int = 5, method: str = "static_int8"):
    """
    Test a quantized TD3 actor module.

    Args:
        h1: First hidden layer size
        h2: Second hidden layer size
        num_episodes: Number of episodes to run
        method: One of {"fp32", "dynamic_int8", "static_int8"}
    """
    print("=" * 70)
    print(f"Testing TD3 Actor ({method})")
    print("=" * 70)

    print("\n1. Creating BGN environment...")
    bgn = BGN_MC(tmax=1100, pd=True)
    print("   [OK] Environment created")

    print("\n2. Loading actor checkpoint...")
    try:
        actor = _load_actor(method, h1, h2)
        print("   [OK] Actor loaded and set to eval mode")
    except Exception as exc:
        print(f"   [X] Failed to load actor: {exc}")
        return False

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
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0)

            with torch.no_grad():
                action = actor(obs_tensor).numpy()[0]

            observation, reward, terminated, truncated, info = bgn.step(action)

            sgis = scipy.io.loadmat('bgn_vars.mat')['sgis']
            sgis_sums.append(np.sum(np.mean(np.abs(np.fft.fft(sgis)), axis=0)[1:20]))

            freqs.append(action[0])
            amps.append(action[1])
            step_count += 1

        vgi = scipy.io.loadmat('bgn_vars.mat')['vgi']
        Pb = np.sum(np.average(np.abs(np.fft.fft(vgi)) / 0.1, axis=0)[12:31])
        Pbs.append(Pb)

        print(f"      Completed {step_count} steps")

    print("\n4. Calculating performance metrics...")

    sgi_intensity = np.mean(sgis_sums)
    avg_Pb = np.mean(Pbs)

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

    print(f"\n   Evaluation Criteria:")
    print(f"   - Frequency < 130 Hz: {'[OK]' if mean_freq < 130 else '[X]'} ({mean_freq:.2f} Hz)")
    print(f"   - Amplitude < 2500 mA: {'[OK]' if mean_amp < 2500 else '[X]'} ({mean_amp:.2f} mA)")

    print("\n" + "=" * 70)
    print("TD3 ACTOR TEST COMPLETE [OK]")
    print("=" * 70)
    print(f"\nActor ({method}) tested successfully!")
    print(f"SGi Intensity: {sgi_intensity:.2f}")
    print(f"P-beta: {avg_Pb:.2f}")
    print(f"\nNext step: Convert to ONNX (Cell 12 from examples.ipynb)")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TD3 actor evaluation with quantized checkpoints")
    parser.add_argument('--variant', choices=list(_VARIANT_SUFFIX.keys()), default='static_int8')
    parser.add_argument('--episodes', type=int, default=5)
    args = parser.parse_args()

    h1, h2 = 32, 32
    success = test_quantized_model(h1=h1, h2=h2, num_episodes=args.episodes, method=args.variant)
    exit(0 if success else 1)
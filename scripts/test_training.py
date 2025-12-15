"""
Test script to verify TD3 training works with MATLAB engine.
This runs a minimal training session to verify the pipeline works.
"""

import numpy as np
from BGN_MC import BGN_MC
from stable_baselines3 import TD3
import torch
import os
import tempfile
import shutil

def test_training_pipeline():
    """Test TD3 training with MATLAB engine."""
    print("=" * 70)
    print("Testing TD3 Training Pipeline with MATLAB Engine")
    print("=" * 70)
    
    try:
        # Create temporary directories for test
        test_models_dir = 'test_models/TD3_test'
        test_logdir = 'test_logs'
        
        if os.path.exists(test_models_dir):
            shutil.rmtree(test_models_dir)
        if os.path.exists(test_logdir):
            shutil.rmtree(test_logdir)
        
        os.makedirs(test_models_dir, exist_ok=True)
        os.makedirs(test_logdir, exist_ok=True)
        
        print("\n1. Creating BGN environment...")
        env = BGN_MC(tmax=1100, pd=True, mode='hvgi_sgi')
        print("   [OK] Environment created")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        
        print("\n2. Testing environment reset...")
        obs, info = env.reset()
        print(f"   [OK] Reset successful")
        print(f"   Observation shape: {obs.shape}")
        
        print("\n3. Creating TD3 model...")
        h1, h2 = 22, 22
        policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[h1, h2], qf=[h1, h2])
        )
        
        model = TD3(
            'MlpPolicy',
            env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=test_logdir,
            learning_rate=0.0001
        )
        print("   [OK] TD3 model created")
        
        print("\n4. Testing model prediction (before training)...")
        action, _ = model.predict(obs, deterministic=True)
        print(f"   [OK] Prediction successful")
        print(f"   Action: {action}")
        print(f"   Action shape: {action.shape}")
        
        print("\n5. Running minimal training (50 timesteps)...")
        print("   (This may take a minute - MATLAB is computing brain dynamics)")
        model.learn(
            total_timesteps=50,
            reset_num_timesteps=True,
            tb_log_name='TD3_test'
        )
        print("   [OK] Training completed successfully")
        
        print("\n6. Testing model prediction (after training)...")
        obs_new, _ = env.reset()
        action_new, _ = model.predict(obs_new, deterministic=True)
        print(f"   [OK] Prediction after training successful")
        print(f"   Action: {action_new}")
        
        print("\n7. Testing model save/load...")
        model_path = f'{test_models_dir}/test_model.zip'
        model.save(model_path)
        print(f"   [OK] Model saved to {model_path}")
        
        # Test loading
        model_loaded = TD3.load(model_path, env=env)
        action_loaded, _ = model_loaded.predict(obs_new, deterministic=True)
        print(f"   [OK] Model loaded successfully")
        print(f"   Loaded model action: {action_loaded}")
        
        print("\n8. Testing full episode with trained model...")
        obs, _ = env.reset()
        terminated = False
        step_count = 0
        total_reward = 0
        
        while not terminated and step_count < 5:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
        
        print(f"   [OK] Episode simulation successful")
        print(f"   Steps: {step_count}, Total reward: {total_reward:.4f}")
        
        # Cleanup
        print("\n9. Cleaning up test files...")
        shutil.rmtree(test_models_dir)
        shutil.rmtree(test_logdir)
        print("   [OK] Cleanup complete")
        
        print("\n" + "=" * 70)
        print("TRAINING PIPELINE TEST: PASSED [OK]")
        print("=" * 70)
        print("\nThe training pipeline works correctly with MATLAB Engine!")
        print("You can now run full training with: python training.py")
        return True
        
    except Exception as e:
        print(f"\n[X] ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Make sure MATLAB Engine is installed and working")
        print("2. Verify BGN environment works: python test_bgn_environment.py")
        print("3. Check that stable-baselines3 is installed correctly")
        return False

if __name__ == '__main__':
    success = test_training_pipeline()
    exit(0 if success else 1)


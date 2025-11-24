"""
Test Cell 1 from examples.ipynb exactly as written
"""

from BGN_MC import *

# set up a BGN simulation that will run for 1100ms. pd=True
# signifies that the brain will be simulated in the Parkinsonian
# state. 
bgn = BGN_MC(tmax=1100, pd=True)

# this .reset() function will initialize all neuron readings, 
# and is set to run for 100ms. During this time, no DBS actions
# will be applied to the simulated brain. 
obs = bgn.reset()[0]
print(f"Reset successful! Observation shape: {obs.shape}")
print(f"Observation values: {obs}")

# run .step() in a while loop until our simulation is complete.
# .step() is set to run for 100ms, and reports back the 
# observation vector, the reward incurred by the agent if it 
# decided to take an action, a truncated variable which is
# functionally meaningless to us, and an info variable, which 
# may be helpful for debugging. Note that no arguments are being
# passed to the .step() function, and so no DBS will be applied. 
terminated = 0
step_count = 0
while terminated != 1 and step_count < 5:  # Limit to 5 steps for testing
    observation, reward, terminated, truncated, info = bgn.step()
    step_count += 1
    print(f"Step {step_count}: reward={reward:.4f}, terminated={terminated}")

print(f"\n[SUCCESS] Cell 1 code executed successfully!")
print(f"Completed {step_count} steps before termination check")


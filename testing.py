from Matlab_Impl import BGN_MC
from stable_baselines3 import TD3

model = TD3.load('models/TD3/SB3_TD3_5000', env=BGN_MC.BGN_MC(1400))
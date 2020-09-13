from osim.env import L2M2019Env
import numpy as np

env = L2M2019Env(visualize=False)
observation = env.reset(obs_as_dict=False)  # If set to false, returns (339,) array
print(np.shape(observation))

""" 
Observation:
 
 get_observation(self)
 |      ## Values in the observation vector
 |      # 'vtgt_field': vtgt vectors in body frame (2*11*11 = 242 values)
 |      # 'pelvis': height, pitch, roll, 6 vel (9 values)
 |      # for each 'r_leg' and 'l_leg' (*2)
 |      #   'ground_reaction_forces' (3 values)
 |      #   'joint' (4 values)
 |      #   'd_joint' (4 values)
 |      #   for each of the eleven muscles (*11)
 |      #       normalized 'f', 'l', 'v' (3 values)
 |      # 242 + 9 + 2*(3 + 4 + 4 + 11*3) = 339

"""
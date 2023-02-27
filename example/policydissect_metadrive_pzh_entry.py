import numpy as np
import time
from policydissect.policydissect import do_policy_dissection
from policydissect.metadrive.metadrive_env import SafeMetaDriveEnv
from policydissect.utils.policy import ppo_inference_tf
import os
import pickle
import pickle

if __name__ == "__main__":


    with open("pzh data.pkl", "rb") as f:
        collected_episodes = pickle.load(f)

    pd_ret = do_policy_dissection(collected_episodes, save_figure=False)
    # with open("{}.pkl".format("metadrive_ret"), "wb+") as file:
    #     pickle.dump(pd_ret, file)

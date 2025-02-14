import os

from ddpg_per import DDPG
from envs import TradingEnv

if __name__ == "__main__":

    # disable GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # specify what to test
    delta_action_test = False
    bartlett_action_test = False

    # specify weights file to load
    tag = "6"
    
     # set init_ttm, spread, and other parameters according to the env that the model is trained
    env_test = TradingEnv(continuous_action_flag=True, sabr_flag=False, dg_random_seed=None, spread=0.01, num_contract=1, init_ttm=10, trade_freq=0.2, num_sim=10000)
    ddpg_test = DDPG(env_test)

    print("\n\n***")
    if delta_action_test:
        print("Testing delta actions.")
    else:
        print("Testing agent actions.")
        if tag == "":
            print("tesing the model saved at the end of the training.")
        else:
            print("Testing model saved at " + tag + "K episode.")
        ddpg_test.load(tag=tag)

    ddpg_test.test(10000, delta_flag=delta_action_test, bartlett_flag=bartlett_action_test)

    # for i in range(1, 51):
    #     tag = str(i)
    #     print("****** ", tag)
    #     ddpg_test.load(tag=tag)
    #     ddpg_test.test(3001, delta_flag=delta_action_test)

    ddpg_test.plot()
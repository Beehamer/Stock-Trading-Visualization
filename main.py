import gym
import sys
from pathlib import Path

from argparse import ArgumentParser
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

from argparse import ArgumentParser

import pandas as pd

def cmd_parse():
    parser = ArgumentParser()

    parser.add_argument("--model",
                        help="The RL farmework/policy/model to use. Default is PPO2",
                        dest="model", default="PPO2")
    parser.add_argument("--training_set_size", type=int, 
                        help="The training set size (int) to use for training before the agent starts trading. Default is 200",
                        dest="training_set_size", default="200")
    parser.add_argument("--ticker", 
                        help="Ticker symbol for the data to be trained on and traded with. Default is MSFT",
                        dest="ticker", default="MSFT")
    parser.add_argument("--look_back_days", type=int,
                        help="Number (int) of look back days before making trading decision. Default is 5 days",
                        dest="look_back_days", default="40")
    parser.add_argument("--output_file",
                        help="Name of output file for the rewards and the total net_worth. Default is output.csv",
                        dest="output_file", default="output.csv")
    return parser


def main():

    cmd_parser = cmd_parse()
    options = cmd_parser.parse_args()

    ## Get the Stock Ticker data ##
    # print("The Stock ticker used here is ", options.ticker)

    file = Path("./data/" + options.ticker + ".csv")
    if file.is_file():
        df = pd.read_csv('./data/'+ options.ticker + '.csv')
        df = df.sort_values('Date')
        print("Loading ticker data from: " + "./data/" + options.ticker + ".csv")
    else:
        print("Data file for ticker does not exist. Please download data first to ./data/" + options.ticker + ".csv")


    ## Get the training set size ##
    print("The options.training_set_size is ", options.training_set_size)
        
    ## Get the number of look back days ##
    print("The options.look-back-days here is: ", options.look_back_days)

    ## Get the model we are using to train the agent ##
    print("The model to train the agent here is: ", options.model)

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df, options.look_back_days, options.training_set_size, options.output_file)])

    if options.model == "PPO2":
        model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=options.training_set_size)
    
    obs = env.reset()
    for i in range(options.training_set_size, len(df['Date'])):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render(title= options.ticker)
    env.close()

if __name__ == "__main__":
    main()

import inspect

# Unused System

# Unused stable_baselines3

from CybORG import CybORG
from CybORG.Agents import B_lineAgent

from CybORG.Agents.rl import RLagent
from CybORG.Agents.Wrappers import ChallengeWrapper

from evaluation import training_main as evaluate
from evaluation import graph_evaluation as graph
from graph import generate_graph


def wrap(env):
    return ChallengeWrapper('Blue', env)


def setup_cyborg(turns_per_game):
    red_agent = B_lineAgent  # Red agent we are training against

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
    env = wrap(cyborg)

    return red_agent, env


if __name__ == "__main__":
    total_timesteps = 100000
    graph_interval = 1000
    rounds = 100


    model_name = "modified_BlueTableWrapper"
    graph_name = 'graph_data_' + f"{model_name}"+".csv"


    red_agent, env = setup_cyborg(graph_interval)
    model = RLagent(env=env, agent_type="PPO", verbose=0) # Creates initial model - training from scratch
    # model = RLagent(env=env, agent_type="PPO",model_name="INSERT MODEL NAME HERE") # Trains an existing model

    print(f"Training {model_name} against a red {red_agent.__name__} for {rounds} rounds of {graph_interval} timesteps")
    print("\n" + "*" * 50)


    # MAIN TRG ALGORITHM
    timesteps_passed = 0

    for period in range(int(total_timesteps / graph_interval)):
        model.train(timesteps=graph_interval, log_name=f"{model_name}_X_{red_agent.__name__}_Round_")
        timesteps_passed += graph_interval

        #print("trained for", timesteps_passed)

        # Save the model
        model.save(f"{model_name}")

        graph(model_name, graph_name, timesteps_passed)

        if timesteps_passed % int(total_timesteps / 10) == 0:
            percent_done = round((timesteps_passed / total_timesteps) * 100)
            generate_graph(graph_name,period)
            print(f"{percent_done}% done")
            #evaluate(model_name=model_name)


    print(f"Trained {model_name}  against a red {red_agent.__name__} for {rounds} rounds of {total_timesteps} timesteps")




# Trash (or treasure?)

# def wrap(env):
#   return OpenAIGymWrapper('Blue', EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))

#    for red_agent in red_agents: # If multiple agents
#        for RL_algo in RL_algos: # If multiple agents"""

# steps = round(timesteps / 1000, 2)  # gives number of thousand steps -- for pretty output

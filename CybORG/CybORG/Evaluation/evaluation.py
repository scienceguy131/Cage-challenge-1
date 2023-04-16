import inspect
import time
from statistics import mean, stdev
import csv

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents.Wrappers.GarrettWrapper import CompetitiveWrapper
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from CybORG.Agents.rl import RLagent

MAX_EPS = 10
agent_name = 'Blue'


def wrap(env):
    return ChallengeWrapper('Blue', env)


def graph_evaluation(model_name,graph_name, timesteps_trained):
    # Setup agent
    agent = RLagent(env=None, agent_type="PPO", model_name=model_name)
    #print(f'Evaluating agent {agent.__class__.__name__}\n\tIf this is incorrect please update the code to load in your agent')

    graph_data = open(graph_name, 'a',newline='')
    graph_writer = csv.writer(graph_data)
    #graph_writer.writerow(['timesteps','reward'])

    #file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + f"/{model_name}/" + f'testingOutput.txt'
    #print(f'\tSaving evaluation results to {file_name}')

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path,'sim', agents={'Red': B_lineAgent})
    wrapped_cyborg = wrap(cyborg)

    mean_reward, std_reward = evaluate_policy(model=agent, env=wrapped_cyborg, n_eval_episodes=10)

    # Print the results
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    graph_writer.writerow([timesteps_trained, mean_reward])
    #print("*row added at", timesteps_trained, "steps")


    graph_data.close()
    """for num_steps in [50]:
        for red_agent in [B_lineAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg, num_steps)

            observation = wrapped_cyborg.reset()
            action_space = wrapped_cyborg.get_action_space(agent_name)

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()

            graph_writer.writerow([timesteps_trained, mean(total_reward)])
            print("*row added at", timesteps_trained, "steps")
    graph_data.close()"""



def training_main(model_name):
    # Setup agent
    agent = RLagent(env=None, agent_type="PPO", model_name=model_name)
    #print(f'Evaluating agent {agent.__class__.__name__}\n\tIf this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '\Evaluation\evals\\' + f"{model_name}" + time.strftime("%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'-Saving evaluation results to {file_name}')


    # Evaluation log setup
    cyborg_version = '1.2'
    scenario = 'Scenario1b'

    name = "Al"
    team = "SuperNerds"
    name_of_agent = model_name

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{cyborg_version}, {scenario}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")


    # print(f'\tUsing CybORG v{cyborg_version}, {scenario}\n*')

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            action_space = wrapped_cyborg.get_action_space(agent_name)

            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')



            with open(file_name, 'a+') as data:
                data.write(
                    f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
    print("*")

"""
if __name__ == "__main__":
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    # ask for a name
    name = input('Name: ')
    # ask for a team
    team = input("Team: ")
    # ask for a name for the agent
    name_of_agent = input("Name of technique: ")

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    # Change this line to load your agent
    agent = RLagent(env=None, agent_type="PPO",
                    model_name="Chad")  # PPO against red B_lineAgent for 0.1 million training eps_1

    print(f'Using agent {agent.__class__.__name__}, if this is incorrect please update the code to load in your agent')

    file_name = str(inspect.getfile(CybORG))[:-10] + '/Evaluation/' + time.strftime(
        "%Y%m%d_%H%M%S") + f'_{agent.__class__.__name__}.txt'
    print(f'Saving evaluation results to {file_name}')
    with open(file_name, 'a+') as data:
        data.write(f'CybORG v{1.0}, {scenario}\n')
        data.write(f'author: {name}, team: {team}, technique: {name_of_agent}\n')
        data.write(f"wrappers: {wrap_line}\n")

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    print(f'using CybORG v{cyborg_version}, {scenario}\n')
    for num_steps in [30, 50, 100]:
        for red_agent in [B_lineAgent, RedMeanderAgent, SleepAgent]:

            cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
            wrapped_cyborg = wrap(cyborg)

            observation = wrapped_cyborg.reset()
            # observation = cyborg.reset().observation

            action_space = wrapped_cyborg.get_action_space(agent_name)
            # action_space = cyborg.get_action_space(agent_name)
            total_reward = []
            actions = []
            for i in range(MAX_EPS):
                r = []
                a = []
                # cyborg.env.env.tracker.render()
                for j in range(num_steps):
                    action = agent.get_action(observation, action_space)
                    observation, rew, done, info = wrapped_cyborg.step(action)
                    # result = cyborg.step(agent_name, action)
                    r.append(rew)
                    # r.append(result.reward)
                    a.append((str(cyborg.get_last_action('Blue')), str(cyborg.get_last_action('Red'))))
                total_reward.append(sum(r))
                actions.append(a)
                # observation = cyborg.reset().observation
                observation = wrapped_cyborg.reset()
            print(
                f'Average reward for red agent {red_agent.__name__} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}')
            with open(file_name, 'a+') as data:
                data.write(
                    f'steps: {num_steps}, adversary: {red_agent.__name__}, mean: {mean(total_reward)}, standard deviation {stdev(total_reward)}\n')
                for act, sum_rew in zip(actions, total_reward):
                    data.write(f'actions: {act}, total reward: {sum_rew}\n')
"""
# observation = cyborg.reset().observation
# action_space = cyborg.get_action_space(agent_name)

# def wrap(env):
#    return OpenAIGymWrapper(agent_name, EnumActionWrapper(FixedFlatWrapper(ReduceActionSpaceWrapper(env))))

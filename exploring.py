import inspect
import random
from pprint import pprint
from CybORG import CybORG

path = str(inspect.getfile(CybORG))
path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

env = CybORG(path, 'sim')
results = env.reset(agent='Red')
blue_obs = env.get_observation('Blue')

params = ['Interface', 'Processes','Sessions','System info','User Info'] #files
users = list(blue_obs.keys())[1:]
print(users)

for param in params:
    print(f"*****{param}*****")
    for user in users:
        print(f"***USER:{user}***")
        pprint(blue_obs[user][param])
        print('*')
    print('*' * 50 + "\n")


#info_required = {'Test_Host': {'User_info': 'All',
#                               'System_info': 'All',
#                              'Processes': 'All',
#                             'Files': ['/root', '/bin', '/sbin', '/etc', '/home', '/usr/sbin/', '/usr/bin/']}}
#state = env.get_true_state(info_required)

#print(state)
#obs = results.observation
#pprint(obs['User0'])

#blue_obs = env.get_observation('Blue')

#pprint((blue_obs['User0']))
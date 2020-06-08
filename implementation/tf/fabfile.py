#!/usr/bin/env python3
from fabric import Connection
from fabric import task
import os
import time
from pprint import pprint

from runner import *

password = os.environ['FAB_PWD']
user = os.environ['FAB_USR']

servers = ['u-pl{}.ms.mff.cuni.cz'.format(i) for i in range(1,21)]

concurrent = 1

_, parameters = load_params('params.yaml')
# parameters = list(parameters)[:3]

server2task = dict(((s, []) for s in servers))
for i, params in enumerate(parameters):
    server2task[servers[i%len(servers)]].append(params)


@task
def run(ctx):
    for server, tasks in server2task.items():
        counter = 5
        while counter > 0:
            try:
                c = Connection(server, user=user, connect_kwargs={'password':password})
                make_command = lambda x, lst: ' && '.join(['python run.py \'{}\' &> /dev/null'.format(str(p).replace('\'','"')) for p in lst[x::concurrent]])
                for z in range(concurrent):
                    command = make_command(z, tasks)
                    c.run('. .bash_profile && cd tfdiag && {}'.format(command), disown=True)
                    print(command)
                counter=0
                print('... run')
            except Exception as e:
                print('oops, error')
                print(e)
                counter = counter - 1
                if counter == 0:
                    raise TimeoutError('ERROR: cannot connect to server {}'.format(server))
                time.sleep(2)
    print('DONE')



#!/usr/bin/env python3
from fabric import Connection
from fabric import task
import os
import time
from pprint import pprint

from runner import *

password = os.environ['FAB_PWD']
user = os.environ['FAB_USR']

servers = ['u-pl{}.ms.mff.cuni.cz'.format(i) for i in range(1,3)]


_, parameters = load_params('params.yaml')
parameters = list(parameters)[:3]

server2task = dict(((s, []) for s in servers))
for i, params in enumerate(parameters):
    server2task[servers[i%len(servers)]].append(params)


@task
def run(ctx):
    for server, tasks in server2task.items():
        counter = 2
        while counter > 0:
            try:
                c = Connection(server, user=user, connect_kwargs={'password':password})
                command = ' && '.join(['python run.py \'{}\' &> /dev/null'.format(str(p).replace('\'','"')) for p in tasks])
                print(command)
                c.run('. .bash_profile && cd tfdiag && {}'.format(command), disown=True)
                counter=0
                print('... run')
            except Exception as e:
                print('oops, error')
                print(e)
                counter = counter - 1
                if counter == 0:
                    raise TimeoutError('ERROR: cannot connect to server {}'.format(server))
                time.sleep(1)
    print('DONE')



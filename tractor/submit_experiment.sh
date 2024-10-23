#!/bin/bash

rez env simpleFarm-popart tractor -- \
    farmexec --prio=high --name=do_not_kill_experiment_$1 --prod=mvg --requirements=cuda16G --rezPackages='git' bash /s/apps/users/almarouka/line-detection/repo/tractor/experiments/$1.sh


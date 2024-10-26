#!/bin/bash

rez env simpleFarm-popart tractor -- \
    farmexec --prio=medium --name=do_not_kill_experiment_$1 --prod=mvg --requirements=cuda20G --rezPackages='git' bash /s/apps/users/almarouka/line-detection/repo/tractor/experiments/$1.sh


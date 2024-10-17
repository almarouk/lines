#!/bin/bash

rez env simpleFarm-popart tractor -- \
    farmexec --name=do_not_kill_test_experiment --prod=mvg --requirements=cuda16G bash /s/apps/users/almarouka/line-detection/repo/experiments/test.sh

#  --prio=high
#!/bin/bash

rez env simpleFarm-popart tractor -- \
    farmexec --prio=high --name=do_not_kill_test__$1 --prod=mvg --requirements=cuda16G bash /s/apps/users/almarouka/line-detection/repo/tractor/tests/$1.sh

    
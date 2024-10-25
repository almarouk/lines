#!/bin/bash

rez env simpleFarm-popart tractor -- \
    farmexec --prio=medium --name=do_not_kill_test_generic --prod=mvg --requirements=cuda20G bash /s/apps/users/almarouka/line-detection/repo/tractor/tests/generic.sh "$1" "$2" "$3"

    
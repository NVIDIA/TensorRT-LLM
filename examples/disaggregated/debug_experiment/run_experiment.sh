#!/bin/bash

echo current time: $(date)

./kill.sh

./run_good.sh

./kill.sh

./run_bad.sh
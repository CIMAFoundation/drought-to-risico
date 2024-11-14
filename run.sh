#!/bin/bash

export PYTHONPATH=$PYTHONPATH:'./src'
.venv/bin/python3 src/run.py

rsync -avr  output/ root@192.168.0.13:/share/archivio/experience/data/MeteoModels/CHIQUITANIA_FUEL_MAP_ML
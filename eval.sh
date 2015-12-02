#!/bin/bash

ulimit -S -n 1000000
cd Rouge
java -jar rouge2.0.jar
cd ..

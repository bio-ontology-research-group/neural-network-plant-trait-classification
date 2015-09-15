#!/bin/bash

if [ ! -f results ]; then 
    mkdir results
fi

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu,floatX=float32 python cnnWTest.py | tee ./results/allResults.txt
sleep 5 

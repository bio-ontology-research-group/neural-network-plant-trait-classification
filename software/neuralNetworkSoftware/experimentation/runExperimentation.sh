#!/bin/bash

if [ ! -f results ]; then
    mkdir results
fi


THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu,floatX=float32 python ../symmetry/cnnWTest.py | tee ./results/symmetry.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../stamenNo/cnnWTest.py | tee ./results/stamenNo.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../petalNo/cnnWTest.py | tee ./results/petalNo.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../lifeForm/cnnWTest.py | tee ./results/lifeForm.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../leafVenation/cnnWTest.py | tee ./results/leafVenation.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../leafStructure/cnnWTest.py | tee ./results/leafStructure.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../leafMargin/cnnWTest.py | tee ./results/leafMargin.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../leafForm/cnnWTest.py | tee ./results/leafForm.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../importantDiagnostics/cnnWTest.py | tee ./results/importantDiagnostics.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../fruitType/cnnWTest.py | tee ./results/fruitType.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../fruitSurface/cnnWTest.py | tee ./results/fruitSurface.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../fruitPericarp/cnnWTest.py | tee ./results/fruitPericarp.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../fruitDehiscence/cnnWTest.py | tee ./results/fruitDehiscence.txt

THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../floweringAndNotFlowering/cnnWTest.py | tee ./results/floweringAndNotFlowering.txt


THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu, floatX=float32 python ../colours/cnnWTest.py | tee ./results/colours.txt



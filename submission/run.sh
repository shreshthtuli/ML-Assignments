#!/bin/bash
if [ $1 = 1 ]
then
    python ./Naive_Bayes.py $2 $3 $4
    exit
elif [ $1 = 2 ]
then
    python ./Support_Vector_Machine.py $2 $3 $4 $5
    exit
fi
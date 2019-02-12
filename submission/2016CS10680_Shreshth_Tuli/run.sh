#!/bin/bash
if [ $1 = 1 ]
then
    python ./Linear_Regression.py $2 $3 $4 $5
    exit
elif [ $1 = 2 ]
then
    python ./Locally_Weighted_Linear_Regression.py $2 $3 $4
    exit
elif [ $1 = 3 ]
then
    python ./Logistic_Regression.py $2 $3
    exit
elif [ $1 = 4 ]
then
    python ./Gaussian_Discriminant_Analysis.py $2 $3 $4
    exit
fi
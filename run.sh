#!/bin/bash

PYTHON=/home/$USER/python2.7/bin/python

labels_count=(600 700 800 900 1000 2500 5000 10000)
topwords_count=(7000 8000 9000 10000 15000 20000 50000)

ALL=2

ALLBESTLCOUNT=600
ALLBESTWCOUNT=8000

ALLBESTLCOUNT=50000
ALLBESTWCOUNT=100000

function run()
{
    echo "-------------------running-----------------------"
    $PYTHON gen_train_data.py $ALLBESTLCOUNT > out 2>&1
    $PYTHON preprocess.py $ALLBESTWCOUNT > out 2>&1
    echo "-------label: $ALLBESTLCOUNT----topwords: $ALLBESTWCOUNT-------"
    rate=`$PYTHON classify.py 0`
    $PYTHON tags_feature.py
    $PYTHON gen_results.py
    echo "------------correct rate: $rate------------------"

    echo "-----------------end running---------------------"
}

function run_test()
{
    echo "-------------------testing-----------------------"
    for lcount in ${labels_count[@]}
        do
            $PYTHON gen_train_data.py $lcount > out 2>&1
            for wcount in ${topwords_count[@]}
            do
                correct_rate=`for_test_ALL $wcount`
                echo "---labels count: $lcount---topwords count: $wcount---coreect rate: $correct_rate---"
            done
        done
    echo "-----------------end testing---------------------"
}

function for_test_ALL()
{
    wcount=$1

    $PYTHON preprocess.py $wcount > out 2>&1
    rate=`$PYTHON classify.py 1`
    echo $rate

    return 0
}

run_test
#run

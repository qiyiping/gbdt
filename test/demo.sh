#!/bin/sh

# train/test files are generated by the following function:
# t = 1.0 / (1.0 + exp(-x - 2y + z))

mkdir -p ../data
python build_data.py > ../data/train.txt
python build_data.py > ../data/test.txt

feature_size=3
train_file="../data/train.txt"
max_depth=4
iterations=30
shrinkage=0.1
feature_ratio=1.0
data_ratio=1.0
debug="true"
min_leaf_size=0
# options: "SquaredError(ls), LAD(lad), LogLoss(logloss)"
loss="LogLoss"
metric="auc"
num_of_threads=4
model="../data/train.txt.model"
input="../data/test.txt"
test_file="${input}"
classification="false"
if [ "$loss" == "LogLoss" ]; then
    classification="true"
fi

echo -------------------
echo start training
echo -------------------
cmd="../src/cpp/gbdt_train --feature_size ${feature_size} \
                           --train_file ${train_file} \
                           --max_depth ${max_depth} \
                           --iterations ${iterations} \
                           --shrinkage ${shrinkage} \
                           --feature_ratio ${feature_ratio} \
                           --data_ratio ${data_ratio} \
                           --debug ${debug} \
                           --min_leaf_size ${min_leaf_size} \
                           --loss ${loss} \
                           --threads ${num_of_threads}"

time $cmd

echo -------------------
echo start predicting
echo -------------------
cmd="../src/cpp/gbdt_predict --model ${model} \
                             --feature_size ${feature_size} \
                             --input ${input} \
                             --metric ${metric} \
                             --classification ${classification}"
time $cmd

#-*- coding:utf-8 -*-

import os
import re
import sys
import random

import numpy as np

from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn import preprocessing
from sklearn import linear_model

from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

is_test = False

global_tags =dict()
tags_dataset_filename = "./data/tags.data"

train_dataset_filename = "./middle/train.data"
test_dataset_filename = "./middle/test.data"
predict_dataset_filename = "./middle/predict.data"
results_output_filename = "./middle/predict.out"

def load_dataset_from_file(input_file_name):
    dataset = dict()
    dataset["labels"] = list()
    dataset["samples"] = list()
    dataset["others"] = list()
    dataset["segments"] = list()
    dataset["cuts"] = list()
    with open(input_file_name, "r") as f:
        for line in f:
            try:
                items = line.strip().decode("utf-8").split("\t")
                if len(items) == 2:
                    label = items[0]
                    sample = items[1]
                    dataset["labels"].append(label)
                    dataset["samples"].append(sample)
                else:
                    other = items[0:4]
                    segment = items[-2]
                    cut = items[-1]
                    dataset["others"].append(other)
                    dataset["segments"].append(segment)
                    dataset["cuts"].append(cut)
            except Exception, e:
                continue

    return dataset

def load_global_tags(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            good_tag_name = items[3]
            bad_tag_name = items[9]
            good_tag_id = items[0]
            bad_tag_id = items[6]
            global_tags[good_tag_name] = dict({"id":good_tag_id, "attr":1})
            global_tags[bad_tag_name] = dict({"id":bad_tag_id, "attr":-1})

def predict_data_to_list(predict_data):
    predict_list = list()
    others = predict_data["others"]
    cuts = predict_data["cuts"]
    for index in range(len(others)):
        tmp_list = others[index] 
        tmp_list.append(list(cuts[index]))
        predict_list.append(tmp_list)

    return predict_list

def calculate_correct_rate(classify, test_array, test_data, test_label):
    predict_array = classify.predict(test_array)
    correct_rate = np.mean(predict_array == np.array(test_label))

    print correct_rate

def generate_predict_datasets(classify, predict_array, predict_data, outfile_file_name, no_proba = False):
    index = 0
    predict_count = len(predict_array)
    global global_tags
    with open(outfile_file_name, "a") as f:
        for index, line in enumerate(predict_data):
            array = predict_array[index]
            result = classify.predict(array.reshape(1, -1))
            if no_proba == True:
                result_array = classify.decision_function(array.reshape(1, -1))
                tag_score = result_array.shape[1]
            else:
                result_array = classify.predict_log_proba(array.reshape(1, -1))
                #tag_score = round(result_array.max() - result_array.mean(), 4)
                tag_score = round(result_array.std(), 4)
            tag = global_tags[result[0]]
            string = "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (line[0], line[1], line[2], line[3], tag["id"], tag["attr"], tag_score, result[0], "".join(line[4]))
            f.write(string.encode("utf-8"))

def bayesp(predict_vectorizer, train_array, test_array, test_samples, test_labels, predict_samples, predict_data_list):
    global is_test
    classify = naive_bayes.MultinomialNB(alpha = 0.0005)
    classify = classify.fit(train_array, train_labels)
    calculate_correct_rate(classify, test_array, test_samples, test_labels)

    if is_test:
        return
    block_size = 10000
    loop_count = ((len(predict_samples) + block_size) / block_size)
    if os.path.isfile(results_output_filename):
        os.remove(results_output_filename)
    for offset in range(loop_count):
        predict_samples_block = predict_samples[offset * block_size : (offset + 1) * block_size]
        predict_data_block = predict_data_list[offset * block_size : (offset + 1) * block_size]
        predict_array = predict_vectorizer.fit_transform(predict_samples_block).toarray()
        generate_predict_datasets(classify, predict_array, predict_data_block, results_output_filename)

def svmp(predict_vectorizer, train_array, test_array, test_samples, test_labels, predict_samples, predict_data_list):
    scaled_train_array = preprocessing.scale(train_array)
    scaled_test_array = preprocessing.scale(test_array)

    classify = svm.SVC(C = 50, decision_function_shape = "ovr", cache_size = 20000)
    classify = classify.fit(scaled_train_array, train_labels)
    calculate_correct_rate(classify, scaled_test_array, test_samples, test_labels)
    return

    block_size = 10000
    loop_count = ((len(predict_samples) + block_size) / block_size)
    if os.path.isfile(results_output_filename):
        os.remove(results_output_filename)
    for offset in range(loop_count):
        predict_samples_block = predict_samples[offset * block_size : (offset + 1) * block_size]
        predict_data_block = predict_data_list[offset * block_size : (offset + 1) * block_size]
        predict_array = predict_vectorizer.fit_transform(predict_samples_block).toarray()
        scaled_predict_array = preprocessing.scale(predict_array)
        generate_predict_datasets(classify, scaled_predict_array, predict_data_block, results_output_filename, no_proba = True)

def countVectorizer(train_samples, test_samples):
    #sklearn fit and transform
    train_vectorizer = CountVectorizer(dtype = "float64")
    train_array = train_vectorizer.fit_transform(train_samples).toarray()
    test_vectorizer = CountVectorizer(vocabulary = train_vectorizer.vocabulary_, dtype = "float64")
    test_array = test_vectorizer.fit_transform(test_samples).toarray()
    predict_vectorizer = CountVectorizer(vocabulary = train_vectorizer.vocabulary_, dtype = "float64")

    return train_array, test_array, predict_vectorizer

def tfidfVectorizer(train_samples, test_samples):
    train_vectorizer = TfidfVectorizer()
    train_array = train_vectorizer.fit_transform(train_samples).toarray()
    test_vectorizer = TfidfVectorizer(vocabulary = train_vectorizer.vocabulary_)
    test_array = test_vectorizer.fit_transform(test_samples).toarray()
    predict_vectorizer = TfidfVectorizer(vocabulary = train_vectorizer.vocabulary_)

    return train_array, test_array, predict_vectorizer

if __name__ == "__main__":
    if len(sys.argv) == 2:
        is_test = bool(int(sys.argv[1]))
    #load some needed data for the project
    load_global_tags(tags_dataset_filename)
    train_data = load_dataset_from_file(train_dataset_filename)
    test_data = load_dataset_from_file(test_dataset_filename)
    predict_data = load_dataset_from_file(predict_dataset_filename)

    #get the needed samples and labels
    train_samples = train_data["samples"]
    train_labels = train_data["labels"]
    test_samples = test_data["samples"]
    test_labels = test_data["labels"]
    predict_samples = predict_data["segments"]

    predict_data_list = predict_data_to_list(predict_data)

    #train_array, test_array, predict_vectorizer = countVectorizer(train_samples, test_samples)
    train_array, test_array, predict_vectorizer = tfidfVectorizer(train_samples, test_samples)
    
    #svmp(predict_vectorizer, train_array, test_array, test_samples, test_labels, predict_samples, predict_data_list)
    bayesp(predict_vectorizer, train_array, test_array, test_samples, test_labels, predict_samples, predict_data_list)

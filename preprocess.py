#-*- coding:utf-8 -*-

import os
import re
import sys
import random

import jieba
import jieba.analyse
import numpy as np

#topk = 7000
topk = 500

stop_words = set()
global_words = dict()
global_predict_words = dict()
tags_words = dict()

#input file name
jieba_userdict_filename = "./data/user.dict"
jieba_stopwords_filename = "./data/stop_words.dict"
test_dataset_filename = "./data/test.data"
predict_dataset_filename = "./data/predict.data"
tags_words_filename = "./data/tags.words"

train_dataset_filename = "./middle/train.txt"

train_data_filename = "./middle/train.data"
test_data_filename = "./middle/test.data"

predict_data_filename = "./middle/predict.data"
words_frequent_filename = "./middle/words.fre"

special_words = [u"床", u"水", u"车", u"称", u"笔"]

#jieba.enable_parallel(4)
jieba.load_userdict(jieba_userdict_filename)
#jieba.analyse.set_stop_words(jieba_stopwords_filename)

def load_stop_words(input_file_name):
    with open(input_file_name) as f:
        for word in f:
            stop_words.add(word.decode("utf-8").strip())

def load_tags_words(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            tag_name = items[0]
            tags_words[tag_name] = items[1:]

def prepare_origin_datasets(input_file_name, is_test = False):
    origin_data = dict()
    origin_data["ori"] = dict()
    origin_data["ori"]["samples"] = list()
    origin_data["ori"]["labels"] = list()
    with open(input_file_name, "r") as infile:
        for line in infile:
            data_list = line.strip().decode("utf-8").split("\t")
            if len(data_list) != 2: continue
            comment = data_list[1]
            label = data_list[0];
            origin_data["ori"]["labels"].append(label)
            if is_test == True:
                origin_data["ori"]["samples"].append(comment)
            else:
                if label in tags_words.keys():
                    tag_feature = " ".join(tags_words[label])
                else:
                    tag_feature = label
                #origin_data["ori"]["samples"].append(tag_feature + "," + comment)
                origin_data["ori"]["samples"].append(label + "," + comment)

    return origin_data

def prepare_train_datasets(input_file_name):
    return prepare_origin_datasets(input_file_name)

def prepare_test_datasets(input_file_name):
    return prepare_origin_datasets(input_file_name, is_test = True)

def jieba_word_segment(datasets, is_predict = False):
    text = []
    global global_words
    global special_words
    global global_predict_words
    for line in datasets:
        oneline = ""
        seg_list = jieba.cut(line)
        #seg_list = jieba.analyse.extract_tags(line, topK = 10)
        one_line_words = dict()
        for word in seg_list:
            if word not in stop_words:
                if word not in one_line_words.keys():
                    one_line_words[word] = 0
                one_line_words[word] += 1
        middle = sorted(one_line_words.items(), key=lambda d:d[1], reverse = True)
        for item in middle[:20]:
            oneline += " " + item[0]
            if (len(item[0]) < 2) and (item[0] not in special_words) or (item[0].isdigit()):
                continue
            if is_predict == True:
                if item[0] not in global_predict_words.keys():
                    global_predict_words[item[0]] = 0
                global_predict_words[item[0]] += 1
                continue
            if item[0] not in global_words.keys():
                global_words[item[0]] = 0
            global_words[item[0]] += 1
        text.append(oneline)

    return text

def word_segment_predict_data(predict_data):
    for key in predict_data.keys():
        for index in range(len(predict_data[key])):
            for comments in predict_data[key][index]["cut"]:
                sub_comment = comments["sc"]
                tmp_comment = []
                tmp_comment.append(sub_comment)
                seg_comment = jieba_word_segment(tmp_comment, is_predict = True)
                if "seg" not in predict_data[key][index].keys():
                    predict_data[key][index]["seg"] = []
                predict_data[key][index]["seg"].append(seg_comment)

    return predict_data

def word_segment_train_data(train_data):
    samples = train_data["ori"]["samples"]
    train_data["seg"] = dict()
    train_data["seg"]["samples"] = list()
    train_data["seg"]["labels"] = list(train_data["ori"]["labels"])

    for index, sample in enumerate(samples):
        tmp_comment = []
        tmp_comment.append(sample)
        seg_comment = jieba_word_segment(tmp_comment)
        train_data["seg"]["samples"].append(seg_comment)

    return train_data

def prepare_predict_datasets(infile):
    predict_data = {}

    with open(infile, "r") as inf:
        for line in inf:
            hotel_comment = {}
            line_list = line.strip().decode("utf-8").split("\t")
            if len(line_list) != 3:
                continue
            hotel_id = int(line_list[0])
            hotel_comment["cid"] = line_list[1]
            hotel_comment["ori"] = line_list[2]
            if hotel_id not in predict_data.keys():
                predict_data[hotel_id] = []
            predict_data[hotel_id].append(hotel_comment)
        return predict_data

def prepare_predict_comments(predict_data, is_cut_comment = False):
    for key in predict_data.keys():
        for index in range(len(predict_data[key])):
            full_comment = predict_data[key][index]["ori"]
            comment_index = predict_data[key][index]["cid"]
            if is_cut_comment == False:
                tag_comment["po"] = (0, len(full_comment))
                tag_comment["sc"] = full_comment
                if "cut" in predict_data[key][index].keys():
                    predict_data[key][index]["cut"].append(tag_comment)
                else:
                    predict_data[key][index]["cut"] = []
                    predict_data[key][index]["cut"].append(tag_comment)
            else:
                tmp = re.split(u' |…|,|\.|\!|\?|！|？|，|。|；', full_comment)
                #tmp = re.split(u' |…|,|\.|\!|\?|！|？|，|。|～|（|）|、|；', full_comment)
                #tmp = re.split(u" | |\.|\!|\?|！|？|。|～", full_comment)
                begin_pos = 0
                end_pos = 0
                next_item_used = False
                for ii in range(len(tmp)):
                    #don't make segment repeated
                    if next_item_used:
                        continue
                    next_item_used = False

                    tag_comment = {}
                    sub_comment = tmp[ii]

                    begin_pos = end_pos
                    end_pos = begin_pos + len(sub_comment)
                    if len(sub_comment) <= 4:
                        if ii + 1 < len(tmp):
                            sub_comment += "," + tmp[ii + 1]
                            end_pos = begin_pos + len(sub_comment)
                            next_item_used = True
                    tag_comment["po"] = (begin_pos, end_pos, comment_index)
                    tag_comment["sc"] = sub_comment
                    if "cut" not in predict_data[key][index].keys():
                        predict_data[key][index]["cut"] = []
                    predict_data[key][index]["cut"].append(tag_comment)

    return predict_data

def predict_data_dict_to_list(predict_data, is_cut_comment = False):
    predict_list = list()
    for hotel_id in clean_predict_data.keys():
            for index in range(len(predict_data[hotel_id])):
                seg_full_comment = predict_data[hotel_id][index]["seg"]
                cut_full_comment = predict_data[hotel_id][index]["cut"]
                for ii, seg_comment in enumerate(seg_full_comment):
                    start_pos = cut_full_comment[ii]["po"][0]
                    end_pos = cut_full_comment[ii]["po"][1]
                    comment_id = cut_full_comment[ii]["po"][2]
                    cut_comment = cut_full_comment[ii]["sc"]
                    tmp_list = list((str(hotel_id), str(comment_id), str(start_pos), str(end_pos), seg_comment, cut_comment))
                    predict_list.append(tmp_list)

    return predict_list

def remove_none_train_text(train_data):
    seg_samples = train_data["seg"]["samples"]
    for ii in range(len(seg_samples) -1, -1, -1):
        if is_text_none(seg_samples[ii]):
            del train_data["seg"]["samples"][ii]
            del train_data["seg"]["labels"][ii]
            del train_data["ori"]["samples"][ii]
            del train_data["ori"]["labels"][ii]

    return train_data

def is_text_none(text):
    global special_words
    if (len(text) <= 3) and (text not in special_words):
        return True
    else:
        return False

def remove_none_predict_text(predict_data):
    for key in predict_data.keys():
        for index in range(len(predict_data[key])):
            seg_full_comment = list(predict_data[key][index]["seg"])
            cut_full_comment = list(predict_data[key][index]["cut"])
            for ii in range(len(seg_full_comment) -1, -1, -1):
                if is_text_none(seg_full_comment[ii]):
                    del predict_data[key][index]["seg"][ii]
                    del predict_data[key][index]["cut"][ii]

    return predict_data

def words_frequent_statistics():
    tmp_global_words = dict()
    for word in global_words.keys():
        if word not in tmp_global_words.keys():
            tmp_global_words[word] = 0
        tmp_global_words[word] += global_words[word]
    for word in global_predict_words.keys():
        if word not in tmp_global_words.keys():
            tmp_global_words[word] = 0
        tmp_global_words[word] += global_predict_words[word]
    with open(words_frequent_filename, "w") as f:
        for word in tmp_global_words.keys():
            f.write("%s\t%s\n" % (word.encode("utf-8"), tmp_global_words[word]))

def segment_generate_global_words(train_data, test_data, predict_data, topK = 500):
    global topk
    global global_words
    global global_predict_words

    tmp_train_data = word_segment_train_data(train_data)
    tmp_test_data = word_segment_train_data(test_data)
    tmp_predict_data = word_segment_predict_data(predict_data)

    middle = sorted(global_words.items(), key=lambda d:d[1], reverse = True)
    global_words = middle[:topK]

    tmp_global_words = dict()
    #tuple to dict
    for item in global_words:
        if item[0] not in tmp_global_words.keys():
            tmp_global_words[item[0]] = item[1]
    global_words = dict(tmp_global_words)
    words_frequent_statistics()

    print "train words: %d, all words: %d, topK words: %d" % (len(middle), len(global_words), topk)

    return tmp_train_data, tmp_test_data, tmp_predict_data

def remove_unused_words(line):
    global global_words
    newline = ""
    items = line.split(" ")
    for word in items:
        if word in global_words.keys():
            newline += " " + word

    return newline

def remove_unused_train_words(train_data):
    for index, seg_comment in enumerate(train_data["seg"]["samples"]):
        seg_comment = " ".join(seg_comment)
        res_comment = remove_unused_words(seg_comment)
        train_data["seg"]["samples"][index] = res_comment

    return train_data

def remove_unused_predict_words(predict_data):
    for key in predict_data.keys():
        for index in range(len(predict_data[key])):
            for ii, comment in enumerate(predict_data[key][index]["seg"]):
                comment = " ".join(comment)
                res_comment = remove_unused_words(comment)
                predict_data[key][index]["seg"][ii] = res_comment

    return predict_data

def get_predict_samples(clean_predict_data):
    predict_samples = list()
    for key in clean_predict_data.keys():
        for index in range(len(predict_data[key])):
            for ii, comment in enumerate(predict_data[key][index]["seg"]):
                predict_samples.append(comment)

    return predict_samples

def remove_unused_data(train_data, test_data, predict_data):
    tmp_train_data = remove_unused_train_words(train_data)
    tmp_test_data = remove_unused_train_words(test_data)
    tmp_predict_data = remove_unused_predict_words(predict_data)

    #remove some less info text and label
    clean_train_data = remove_none_train_text(tmp_train_data)
    clean_test_data = remove_none_train_text(tmp_test_data)
    clean_predict_data = remove_none_predict_text(tmp_predict_data)

    return clean_train_data, test_data, clean_predict_data

def dump_data_to_file(output_file_name, samples, labels = None):
    with open(output_file_name, "w") as f:
        if labels == None:
            for sample in samples:
                f.write("%s\n" % "\t".join(sample).encode("utf-8"))
        else:
            for index, sample in enumerate(samples):
                f.write("%s\t%s\n" % (labels[index].encode("utf-8"), sample.encode("utf-8")))
        

if __name__ == "__main__":
    if len(sys.argv) == 2:
        topk = int(sys.argv[1])
    #load some needed data for the project
    load_stop_words(jieba_stopwords_filename)
    load_tags_words(tags_words_filename)
    train_data = prepare_train_datasets(train_dataset_filename)
    test_data = prepare_test_datasets(test_dataset_filename)
    predict_data = prepare_predict_datasets(predict_dataset_filename)
    predict_data = prepare_predict_comments(predict_data, is_cut_comment = True)

    #word segment for all datasets
    train_data, test_data, predict_data = segment_generate_global_words(train_data, test_data, predict_data, topK = topk)
    #do data washing, remove unused words, empty comment and so on
    clean_train_data, clean_test_data, clean_predict_data = remove_unused_data(train_data, test_data, predict_data)

    #get the needed samples and labels
    clean_train_samples = clean_train_data["seg"]["samples"]
    clean_train_labels = clean_train_data["seg"]["labels"]
    clean_test_samples = clean_test_data["seg"]["samples"]
    clean_test_labels = clean_test_data["seg"]["labels"]
    clean_predict_samples = get_predict_samples(clean_predict_data)

    predict_data_list = predict_data_dict_to_list(clean_predict_data)

    dump_data_to_file(train_data_filename, clean_train_samples, clean_train_labels)
    dump_data_to_file(test_data_filename, clean_test_samples, clean_test_labels)
    dump_data_to_file(predict_data_filename, predict_data_list)

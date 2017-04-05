#-*- coding:utf-8 -*-

import sys
import math
from snownlp import SnowNLP

average_score = 0.0

global_tags_attr_dict = dict()
global_tags_feas_dict = dict()
global_predict_list = list()
user_defined_comment_dict = dict()

final_hotel_tags_dict = dict()
tags_words = dict()

tags_words_filename = "./data/tags.words"
user_defined_comment_filename = "./data/user.def"
comments_filename = "./data/unknown_comment.data"

tags_fea_filename = "./middle/tags.fea"
predict_filename = "./middle/predict.out"

clean_results_filename = "./result/predict.cle"
delete_results_filename = "./result/predict.del"
tags_results_filename = "./result/tags.res"
hotels_results_filename = "./result/hotels.res"
segment_results_filename = "./result/segment.res"

def load_tags_feature(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            tag_name = items[0]
            tag_positive = float(items[1])
            tag_feas = items[2:]
            if tag_name not in global_tags_feas_dict.keys():
                global_tags_feas_dict[tag_name] = set(tag_feas)
                global_tags_attr_dict[tag_name] = tag_positive

def load_predict_file(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            global_predict_list.append(items)

def load_user_defined_comment(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            label = items[0]
            comment = items[1]
            if comment not in user_defined_comment_dict.keys():
                user_defined_comment_dict[comment] = label

def load_tags_words(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            tag_name = items[0]
            tag_fea = "|".join(items[1:]).replace(" ", "").split("#")
            noun_list = tag_fea[0].split("|")
            adj_list = tag_fea[1].split("|")
            tags_words[tag_name] = dict()
            tags_words[tag_name]["noun"] = noun_list
            tags_words[tag_name]["adj"] = adj_list

def is_delete_current_comment(comment, tag_name, tag_attr, tag_score):
    global average_score
    flag = True
    positive = float(SnowNLP(comment).sentiments)
    tag_positive = global_tags_attr_dict[tag_name]
    #if (positive >= 0.5009 and tag_attr == 1) or (positive <= 0.4990 and tag_attr == -1):
    #    flag = False
    #if (positive >= 0.5009 and tag_positive >= 0.5009) or (positive <= 0.4990 and tag_positive <= 0.49990):
    #    flag = False
    if (positive >= 0.6 and tag_attr == 1) or (positive <= 0.4 and tag_attr == -1):
        flag = False
    if (positive >= 0.6 and tag_positive >= 0.6) or (positive <= 0.4 and tag_positive <= 0.4):
        flag = False

    if (tag_score >= (average_score * 1.0)) and flag == True:
        flag = False
    #if tag_score >= average_score:
    #    flag = False
    ##some comment can't match a label, we set it
    #if comment in user_defined_comment_dict.keys():
    #    flag = False

    return flag, round(positive, 4)

def calculate_similarity(tag_name, tag_feas, comment):
    global tags_words
    match_count = 0

    tag_words = tags_words[tag_name]
    noun_list = tag_words["noun"]
    adj_list = tag_words["adj"]
    for index, word in enumerate(noun_list):
        if len(word) == 0:
            continue
        if word in comment:
            match_count += 50
    for index, word in enumerate(adj_list):
        if len(word) == 0:
            continue
        if word in comment:
            match_count += 5

    return match_count

def gen_clean_results(output_file_name, delete_file_name):
    global global_tags_feas_dict
    global global_predict_list
    global final_hotel_tags_dict

    odelete = open(delete_file_name, "w")
    with open(output_file_name, "w") as f:
        for line in global_predict_list:
            try:
                hotel_id = line[0]
                comment_id = line[1]
                start_pos = line[2]
                end_pos = line[3]
                tag_id = line[4]
                tag_attr = int(line[5])
                tag_score = float(line[6])
                tag_name = line[7]
                comment = line[8]
                is_delete, positive =  is_delete_current_comment(comment, tag_name, tag_attr, tag_score)
                if is_delete == True:
                    odelete.write("%s\n" % "\t".join(line).encode("utf-8"))
                    continue
                tag_feas = global_tags_feas_dict[tag_name]
                similarity = calculate_similarity(tag_name, tag_feas, comment)
                #if similarity < 5:
                #    odelete.write("%s\n" % "\t".join(line).encode("utf-8"))
                #    continue
                if comment in user_defined_comment_dict.keys():
                    line[7] = user_defined_comment_dict[comment]
                    similarity = 10
                f.write("%s\t%s\n" % ("\t".join(line).encode("utf-8"), similarity))
                if hotel_id not in final_hotel_tags_dict.keys():
                    final_hotel_tags_dict[hotel_id] = dict()
                if tag_name not in final_hotel_tags_dict[hotel_id].keys():
                    final_hotel_tags_dict[hotel_id][tag_name] = 0
                final_hotel_tags_dict[hotel_id][tag_name] += 1
            except Exception, e:
                print "-----------%s------------" % e
                continue
    odelete.close()

def make_sum_for_tags(input_file_name):
    tmp_dict = dict()
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            key = "%s_%s_%s_%s" % (items[0], items[4], items[7], items[5])
            if key not in tmp_dict.keys():
                tmp_dict[key] = dict()
            if "count" not in tmp_dict[key].keys():
                tmp_dict[key]["count"] = 0
            if "simil" not in tmp_dict[key].keys():
                tmp_dict[key]["simil"] = 0
            if "score" not in tmp_dict[key].keys():
                tmp_dict[key]["score"] = 0.0
            tmp_dict[key]["count"] += 1
            tmp_dict[key]["simil"] += int(items[3])
            tmp_dict[key]["score"] += float(items[9])

    return tmp_dict

def calculate_final_score(tmp_dict):
    clean_data_dict = dict()
    tags_df_dict = dict()
    for key in tmp_dict.keys():
        key_info = key.split("_")
        hotel_id = key_info[0]
        tag_id = key_info[1]
        tag_count = tmp_dict[key]["count"]
        if hotel_id not in clean_data_dict.keys():
            clean_data_dict[hotel_id] = dict()
        if tag_id not in clean_data_dict[hotel_id].keys():
            clean_data_dict[hotel_id][tag_id] = 0
        if tag_id not in tags_df_dict.keys():
            tags_df_dict[tag_id] = 0.0
        clean_data_dict[hotel_id][tag_id] = tag_count
    for tag_id in tags_df_dict.keys():
        tag_df = 0
        for hotel_id in clean_data_dict.keys():
            if tag_id in clean_data_dict[hotel_id].keys():
                tag_df += 1
        tags_df_dict[tag_id] = tag_df
    
    doc_count = len(tmp_dict.keys())
    for key in tmp_dict.keys():
        tag_id = key.split("_")[1]
        tag_tf = tmp_dict[key]["count"]
        tag_simil = tmp_dict[key]["simil"]
        tag_score = tmp_dict[key]["score"]
        tag_df = tags_df_dict[tag_id]
        tag_tfidf = round(tag_tf * math.log(doc_count / tag_df), 4)
        tmp_dict[key]["tfidf"] = tag_tfidf
        tmp_dict[key]["fscore"] = round(tag_tfidf * ((tag_simil + 1) * 1.0 / tag_tf) * ((tag_score + 1) * 1.0 / tag_tf) / 100, 4)

    return tmp_dict

def get_needed_tags(score_dict, good = 20, bad = 10):
    tags_results = dict()
    tmp_dict = dict()
    for key in score_dict.keys():
        key_info = key.split("_")
        hotel_id = key_info[0]
        tag_id = key_info[1]
        tag_name = key_info[2]
        tag_attr = key_info[3]
        tag_count = score_dict[key]["count"]
        if hotel_id not in tmp_dict.keys():
            tmp_dict[hotel_id] = dict()
        if tag_attr not in tmp_dict[hotel_id].keys():
            tmp_dict[hotel_id][tag_attr] = dict()
        tag_key = "%s_%s_%s" % (tag_id, tag_name, tag_count)
        tmp_dict[hotel_id][tag_attr][tag_key] = score_dict[key]["fscore"]

    for hotel_id in tmp_dict.keys():
        bad_tag = tuple()
        good_tag = tuple()
        for attr in tmp_dict[hotel_id].keys():
            if int(attr) < 0:
                bad_tag = sorted(tmp_dict[hotel_id][attr].items(), key=lambda v:v[1], reverse=True)[:10]
            if int(attr) > 0:
                good_tag = sorted(tmp_dict[hotel_id][attr].items(), key=lambda v:v[1], reverse=True)[:20]
        if hotel_id not in tags_results.keys():
            tags_results[hotel_id] = list()
        for tag_key in bad_tag:
            items = tag_key[0].split("_")
            tag_id = items[0]
            tag_name = items[1]
            tag_count = items[2]
            tag_attr = -1
            tag_fscore = tag_key[1]
            tags_results[hotel_id].append((tag_id, tag_name, tag_count, tag_attr, tag_fscore))
        for tag_key in good_tag:
            items = tag_key[0].split("_")
            tag_id = items[0]
            tag_name = items[1]
            tag_count = items[2]
            tag_attr = 1
            tag_fscore = tag_key[1]
            tags_results[hotel_id].append((tag_id, tag_name, tag_count, tag_attr, tag_fscore))

    return tags_results

def gen_tags_results(input_file_name, output_file_name):
    tmp_dict = make_sum_for_tags(input_file_name)
    fscore_dict = calculate_final_score(tmp_dict)
    tags_results = get_needed_tags(fscore_dict, good = 20, bad = 10)

    with open(output_file_name, "w") as f:
        for hotel_id in tags_results.keys():
            for line in tags_results[hotel_id]:
                string = "%s\t%s\t%s\t%s\t%s\t%s\n" % (hotel_id, line[0], line[1], line[2], line[3], line[4])
                f.write("%s" % string.encode("utf-8"))

def gen_hotels_results(input_file_name, output_file_name):
    tmp_dict = dict()
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            if len(items) != 3:
                continue
            hotel_id = items[0]
            if hotel_id not in tmp_dict.keys():
                tmp_dict[hotel_id] = 0
            tmp_dict[hotel_id] += 1
    with open(output_file_name, "w") as f:
        for hotel_id in tmp_dict.keys():
            f.write("%s\t%s\n" % (hotel_id, tmp_dict[hotel_id]))

def gen_segment_results(input_file_name, output_file_name):
    pass

def gen_results():
    gen_clean_results(clean_results_filename, delete_results_filename)
    gen_tags_results(clean_results_filename, tags_results_filename)
    #gen_hotels_results(comments_filename, hotels_results_filename)
    #gen_segment_results(predict_filename, segment_results_filename)

def get_average_score(input_file_name):
    count = 0
    value = 0.0
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            value += float(items[6])
            count += 1

    return value / count

if __name__ == "__main__":
    load_tags_words(tags_words_filename)
    load_tags_feature(tags_fea_filename)
    load_predict_file(predict_filename)
    average_score = get_average_score(predict_filename)
    load_user_defined_comment(user_defined_comment_filename)
    gen_results()

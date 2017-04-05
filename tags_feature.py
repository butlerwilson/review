#-*- coding:utf-8 -*-

import jieba
from jieba import posseg
from snownlp import SnowNLP

jieba_userdict_filename = "./data/user.dict"
user_tags_filename = "./data/tags.data"
similarity_filename = "./data/similarity.dict"
tags_seg_filename = "./middle/tags.fea"

tags_set = set()
similarity_dict = dict()

#jieba.load_userdict(jieba_userdict_filename)

def load_global_tags(input_file_name):
    index = [3, 4, 9, 10]
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split("\t")
            for i, item in enumerate(items):
                try:
                    if i in index:
                        tags_set.add(items[i])
                except Exception, e:
                    continue

def load_similarity_dict(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            items = line.strip().decode("utf-8").split(" ")
            if items[0] not in similarity_dict.keys():
                similarity_dict[items[0]] = list()
            similarity_dict[items[0]] = list(items[1:])

def jieba_segment_2_outfile(out_file_name):
    global tags_seg_set
    global similarity_dict

    with open(out_file_name, "w") as f:
        for tag in tags_set:
            tmp_list = list()
            for word, flag in posseg.cut(tag):
                #if len(tmp_list) == 0 or flag == "a" or flag == "n" or flag == "vn":
                tmp_list.append(word)
                if word in similarity_dict.keys():
                    tmp_list.extend(similarity_dict[word])
            positive_probability = SnowNLP(tag).sentiments
            f.write("%s\t%s\t%s\n" % (tag.encode("utf-8"), round(positive_probability, 4), "\t".join(set(tmp_list)).encode("utf-8")))

if __name__ == "__main__":
    load_global_tags(user_tags_filename)
    load_similarity_dict(similarity_filename)
    jieba_segment_2_outfile(tags_seg_filename)

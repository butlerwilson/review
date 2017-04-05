#-*- coding:utf-8 -*-
import sys

per_sample_count = 600
global_train_data = dict()

origin_train_filename = "./data/train.data"
output_train_filename = "./middle/train.txt"

def load_train_data(input_file_name):
    with open(input_file_name, "r") as f:
        for line in f:
            try:
                items = line.strip().decode("utf-8").split("\t")
                label = items[0] + "\t" + items[1]
                sample = items[2]
                if label not in global_train_data.keys():
                    global_train_data[label] = []
                global_train_data[label].append(sample)
            except Exception, e:
                continue

def output_train_data(output_file_name):
    count = 0
    with open(output_file_name, "w") as f:
        for label in global_train_data.keys():
            for sample in global_train_data[label][:per_sample_count]:
                f.write("%s\t%s\n" % (label.encode("utf-8"), sample.encode("utf-8")))
                count += 1
    print "train dataset length: %d, per sample count: %d, used train dataset length: %d" % (len(global_train_data.keys()), per_sample_count, count)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        per_sample_count = int(sys.argv[1])
    load_train_data(origin_train_filename)
    output_train_data(output_train_filename)

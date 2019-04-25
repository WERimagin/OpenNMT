import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from tqdm import tqdm
from collections import defaultdict
import collections
import math
from statistics import mean, median,variance,stdev
import random
import json
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("-src", type=str, default="src.txt", help="input model epoch")
parser.add_argument("-tgt", type=str, default="target.txt", help="input model epoch")
parser.add_argument("-pred1", type=str, default="pred.txt", help="input model epoch")
parser.add_argument("-pred2", type=str, default="pred.txt", help="input model epoch")
parser.add_argument("-data_rate",type=float, default=1.0)
parser.add_argument("-notsplit", action="store_true")
args = parser.parse_args()

random.seed(0)

srcs=[]
targets=[]
predicts1=[]
predicts2=[]

with open(args.src,"r")as f:
    for line in f:
        srcs.append(line.strip())

with open(args.tgt,"r")as f:
    for line in f:
        targets.append(line.strip())

with open(args.pred1,"r")as f:
    for line in f:
        predicts1.append(line.strip())

with open(args.pred2,"r")as f:
    for line in f:
        predicts2.append(line.strip())

data_size=int(len(srcs)*args.data_rate)
for i in range(data_size):
    if predicts1[i]!=predicts2[i]:
        print(srcs[i])
        print(targets[i])
        print(predicts1[i])
        print(predicts2[i])
        print()

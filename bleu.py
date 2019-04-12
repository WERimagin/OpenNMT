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
parser.add_argument("-pred", type=str, default="pred.txt", help="input model epoch")
args = parser.parse_args()

random.seed(0)

srcs=[]
targets=[]
predicts=[]

with open(args.src,"r")as f:
    for line in f:
        src.append(line.strip())

with open(args.tgt,"r")as f:
    for line in f:
        target.append(line.strip())

with open(args.pred,"r")as f:
    for line in f:
        predict.append(line.strip())

#srcs=[s.split() for s in targets]
targets=[t.split() for t in target]
predicts=[p.split() for p in predict]

target_dict=defaultdict(lambda: [])
predict_dict=defaultdict(str)
src_set=set(srcs)
for s,t,p in zip(src,targets,predicts):
    target_dict[s].append(t)
    predict_dict[s]=p

targets=[target_dict[s] for s in src_set]
predicts=[predict_dict[s] for s in src_set]

print(corpus_bleu(targets,predicts,weights=(1,0,0,0)))
print(corpus_bleu(targets,predicts,weights=(0.5,0.5,0,0)))
print(corpus_bleu(targets,predicts,weights=(0.333,0.333,0.333,0)))
print(corpus_bleu(targets,predicts,weights=(0.25,0.25,0.25,0.25)))

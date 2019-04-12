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

def compute_score(t,p):
    t_dict=collections.Counter(t)
    p_dict=collections.Counter(p)
    common=sum((t_dict & p_dict).values())
    return common

def compute_score_refs(ts,p):
    sum_dict=collections.Counter()
    p_dict=collections.Counter(p)
    for t in ts:
        t_dict=collections.Counter(t)
        sum_dict=(t_dict & p_dict) | sum_dict
    common=sum(sum_dict.values())
    return common

def ngram(words, n):
    return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]]
    

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

if True:
    target_dict=defaultdict(lambda: [])
    predict_dict=defaultdict(str)
    src_set=set(src)

    for s,t,p in zip(src,target,predict):
        target_dict[s].append(t)
        predict_dict[s]=p

    print("size:{}\n".format(len(target)))

    score_sum=0
    count_target=0
    count_predict=0
    t_list=[]
    p_list=[]
    for i,s in tqdm(enumerate(src_set)):
        t=target_dict[s]
        p=predict_dict[s]
        score=compute_score_refs(t,p)
        score_sum+=score
        c_t=min(map(len,t))
        c_p=len(p)
        p_list.append(c_t)
        count_target+=c_t
        count_predict+=c_p
        #print(score,len(p))
        #print(p)
    penalty=math.exp(1-count_target/count_predict) if count_target>count_predict else 1
    score=penalty*score_sum/count_predict
    print(score)
    print()

import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from onmt.utils.corenlp import CoreNLP

from tqdm import tqdm
from collections import defaultdict
import collections
import math
from statistics import mean, median,variance,stdev
import random
import json
import argparse




parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="data/squad-src-val-interro.txt", help="input model epoch")
parser.add_argument("--tgt", type=str, default="data/squad-tgt-val-interro.txt", help="input model epoch")
parser.add_argument("--noninterro", type=str, default="data/squad-noninterro-val-interro.txt", help="input model epoch")
parser.add_argument("--pred", type=str, default="pred.txt", help="input model epoch")
parser.add_argument("--ratio", type=float, default=1.0, help="input model epoch")
parser.add_argument("--print", action="store_true", help="input model epoch")
args = parser.parse_args()

random.seed(0)

srcs=[]
targets=[]
predicts=[]
noninterros=[]

with open(args.src,"r")as f:
    for line in f:
        srcs.append(line.strip())

with open(args.tgt,"r")as f:
    for line in f:
        targets.append(line.strip())

with open(args.pred,"r")as f:
    for line in f:
        predicts.append(line.strip())

with open(args.noninterro,"r")as f:
    for line in f:
        noninterros.append(line.strip())

data_size=int(len(srcs)*args.ratio)
srcs=srcs[0:data_size]
targets=targets[0:data_size]
predicts=predicts[0:data_size]
noninterros=noninterros[0:data_size]

#srcs=[s.split() for s in targets]
#predict
p_noninterros=[]
corenlp=CoreNLP()
for p in tqdm(predicts):
    interro,p_noninterro=corenlp.forward(p)
    p_noninterros.append(p_noninterro)
#target
t_noninterros=[t.split() for t in noninterros]

if args.print:
    for i in range(data_size):
        print(srcs[i])
        print(targets[i])
        print(" ".join(t_noninterros[i]))
        print(predicts[i])
        print(" ".join(p_noninterros[i]))
        print()


target_dict=defaultdict(lambda: [])
predict_dict=defaultdict(str)

src_set=set(srcs)
for s,t,p in zip(srcs,t_noninterros,p_noninterros):
    target_dict[s].append(t)
    predict_dict[s]=p

targets_set=[target_dict[s] for s in src_set]
predicts_set=[predict_dict[s] for s in src_set]

print(corpus_bleu(targets_set,predicts_set,weights=(1,0,0,0)))
print(corpus_bleu(targets_set,predicts_set,weights=(0.5,0.5,0,0)))
print(corpus_bleu(targets_set,predicts_set,weights=(0.333,0.333,0.333,0)))
print(corpus_bleu(targets_set,predicts_set,weights=(0.25,0.25,0.25,0.25)))

print()

####normal bleu

targets=[t.split() for t in targets]
predicts=[p.split() for p in predicts]

target_dict=defaultdict(lambda: [])
predict_dict=defaultdict(str)
src_set=set(srcs)
for s,t,p in zip(srcs,targets,predicts):
    target_dict[s].append(t)
    predict_dict[s]=p

targets=[target_dict[s] for s in src_set]
predicts=[predict_dict[s] for s in src_set]

print(corpus_bleu(targets,predicts,weights=(1,0,0,0)))
print(corpus_bleu(targets,predicts,weights=(0.5,0.5,0,0)))
print(corpus_bleu(targets,predicts,weights=(0.333,0.333,0.333,0)))
print(corpus_bleu(targets,predicts,weights=(0.25,0.25,0.25,0.25)))

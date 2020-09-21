# coding=utf-8
import os
from transformers import AlbertForTokenClassification
from transformers import BertForTokenClassification
from transformers import BertTokenizer

# 模型选择
# bert, ernie, xlnet
MODEL_CLASSES = {
    'bert': (BertForTokenClassification, BertTokenizer),
    'albert': (AlbertForTokenClassification, BertTokenizer),
}



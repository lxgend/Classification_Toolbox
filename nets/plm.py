# coding=utf-8
from transformers import AlbertForSequenceClassification
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import XLNetForSequenceClassification
from transformers import XLNetTokenizer

PATH_ALBERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/albert_zh_xxlarge_google_pt'
PATH_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/roberta_wwm_ext_zh_hit_pt'
PATH_XLNET = '/Users/lixiang/Documents/nlp_data/pretrained_model/xlnet_base_zh_hit_pt'

# 模型选择
# robert, albert, xlnet, ernie
MODEL_CLASSES = {
    'albert': (AlbertForSequenceClassification, BertTokenizer, PATH_ALBERT),  # 中文模型使用此Tokenizer
    'bert': (BertForSequenceClassification, BertTokenizer, PATH_BERT),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, PATH_XLNET),
}

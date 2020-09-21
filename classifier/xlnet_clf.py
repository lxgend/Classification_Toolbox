# coding=utf-8
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD,Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
# from transformers import AdamW
from transformers import BertForSequenceClassification, BertModel
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from parm import *

# PATH_MODEL_BERT = '/home/ubuntu/MyFiles/albert_zh_xxlarge_google_pt'
PATH_MODEL_BERT = '/home/ubuntu/MyFiles/roberta_wwm_ext_zh_hit_pt'

# PATH_MODEL_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/albert_zh_xxlarge_google_pt'
# PATH_MODEL_BERT = '/Users/lixiang/Documents/nlp_data/pretrained_model/roberta_wwm_ext_zh_hit_pt'


MAX_LEN = 128
BATCH_SIZE = 64

EPOCH = 1
LR = 1.28e-4
# LR = 2e-5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class ModelProcessor(object):
    def __init__(self, path, device):
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model = BertForSequenceClassification.from_pretrained(path, output_hidden_states=True)
        self.model.to(device)

        print(self.model)
        print(path)
        # print(self.model.dropout)
        # print(self.model.classifier)


def feature_styg(df, col1, col2):
    # df[COL_TXT] = df[col1].str.cat(df[col2])
    df[COL_TXT] = df[[col1, col2]].values.tolist()

    return df, COL_TXT


def txt2id(df, col1, col2, model):
    def use_tokenizer(row, max_seq_len=MAX_LEN):
        result = model.tokenizer.encode_plus(text=row[col1],
                                             text_pair=row[col2],
                                             add_special_tokens=True,
                                             max_length=max_seq_len,
                                             pad_to_max_length=True,
                                             return_token_type_ids=True,
                                             return_attention_mask=True)
        row['ids'] = result['input_ids']
        row['seq_ids'] = result['token_type_ids']
        row['attention_mask'] = result['attention_mask']
        return row

    df = df.apply(use_tokenizer, axis=1)

    return df, 'ids', 'seq_ids', 'attention_mask'


def build_dataloader(df, col_ids, col_seq_ids, col_masks, mode=None):
    # (batch_size, sequence_length)
    t_ids = torch.tensor(df[col_ids].values.tolist(), dtype=torch.long, device=DEVICE)
    t_seq_ids = torch.tensor(df[col_seq_ids].values.tolist(), dtype=torch.long, device=DEVICE)
    t_masks = torch.tensor(df[col_masks].values.tolist(), dtype=torch.long, device=DEVICE)
    t_labels = torch.tensor(df[COL_LB].values.tolist(), dtype=torch.long, device=DEVICE)
    t_dataset = TensorDataset(t_ids, t_seq_ids, t_masks, t_labels)

    if mode == 'predict':
        t_dataloader = DataLoader(dataset=t_dataset, batch_size=BATCH_SIZE)
    else:
        t_dataloader = DataLoader(dataset=t_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return t_dataloader


def build_sts_matrix(df, col, save_filename):
    def layer_usage(outputs, layer_no=None, pooling_stgy=None, last_4_layers=None):

        all_hidden_state = outputs[2]

        if last_4_layers:
            # last 4 encoder layers
            last_4_hidden_state_list = all_hidden_state[-5:-1]
            # list to tensor
            last_4_hidden_state = torch.stack(last_4_hidden_state_list, dim=1)  # torch.Size([bz, 4, 80, 1024])

            if last_4_layers == 'concat':
                # wrong!
                hidden_state = torch.cat(last_4_hidden_state_list, dim=1)  # torch.Size([bz, 4*80, 1024])
            # torch.Size([bz, 80, 1024])
            elif last_4_layers == 'sum':
                hidden_state = torch.sum(last_4_hidden_state, dim=1)
            elif last_4_layers == 'max':
                hidden_state = torch.max(last_4_hidden_state, dim=1)[0]
            else:
                hidden_state = torch.mean(last_4_hidden_state, dim=1)

        elif layer_no is not None:
            hidden_state = all_hidden_state[layer_no]
        else:
            # hidden state of last encoder layer
            hidden_state = outputs[0]
        if pooling_stgy == 'sum':
            sts_emb = torch.sum(hidden_state, dim=1)  # torch.Size([bz, 1024])
        else:
            sts_emb = torch.mean(hidden_state, dim=1)  # torch.Size([bz, 1024])
        return sts_emb

    sts_matrix = torch.tensor([], device=DEVICE)

    with torch.no_grad():
        from tqdm import tqdm
        for step, batch_data in enumerate(tqdm(t_dataloader)):
            batch_ids, batch_mask = batch_data

            outputs = model.language_model(input_ids=batch_ids, attention_mask=batch_mask)

            sts_emb = layer_usage(outputs=outputs)

            # L2-norm
            sts_emb_norm2 = torch.norm(sts_emb, p=2, dim=1, keepdim=True)  # torch.Size([bz, 1])
            sts_feat = torch.div(sts_emb, sts_emb_norm2)  # torch.Size([bz, 1024])

            # add to the matrix
            sts_matrix = torch.cat((sts_matrix, sts_feat), dim=0)

    sts_matrix.cpu()

    file = '%s.pt' % (save_filename)
    torch.save(sts_matrix, os.path.join(PATH_MD_TMP, file))

    print(file + ' created!')
    print(sts_matrix.shape)


def train_clf(model, device, dataloader, epoch):
    model.train()

    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {
            'params':
                [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
                0.01},
        {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0}
    ]

    # optimizer = AdamW(optimizer_grouped_parameters,
    #                   lr=2e-05)

    # optimizer = SGD(model.parameters(), lr=LR, momentum=0.8)
    # optimizer = SGD(model.parameters(), lr=LR, momentum=0.8)
    # optimizer = RMSprop(model.parameters(), lr=LR, alpha=0.9)
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=200, num_training_steps=-1)

    # 存储每一个batch的loss
    loss_collect = []

    def layer_usage(hidden_states, layer_no=None, pooling_stgy=None, last_4_layers=None):
        pass

    for epoch in range(epoch):
        for step, batch_data in enumerate(dataloader):
            # 取出4个tensor
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_ids, batch_seq_ids, batch_masks, batch_labels = batch_data

            # 打印状态
            # print('Epoch: ', epoch, '| Step: ', step, '| label: ',
            #       batch_labels)

            outputs = model(input_ids=batch_ids,
                            attention_mask=batch_masks,
                            token_type_ids=batch_seq_ids,
                            labels=batch_labels)

            _loss, _logits, hidden_states = outputs[:3]
            loss = _loss

            # concat, torch.Size([batch_size, max_len, 768 * 4]), 每个序列中，每个token，有768*4 size的tensor
            # hidden_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)

            # last_4_hidden_state_list = hidden_states[-5:-1]
            # list to tensor
            # last_4_hidden_state = torch.stack(last_4_hidden_state_list, dim=1)  # torch.Size([bz, 4, max_len, 768])
            # hidden_output = torch.mean(last_4_hidden_state, dim=1)
            # hidden_output = torch.sum(last_4_hidden_state, dim=1)

            # the second-to-last hidden_state
            # hidden_output = hidden_states[-2]

            # get CLS token feature, torch.Size([batch_size, x ])
            # cls_out = hidden_output[:, 0, :]
            # dropout_output = model.dropout(cls_out)
            # logits = model.classifier(dropout_output)

            # classifier = nn.Linear(768*4, model.num_labels)
            # classifier.to(device)
            # logits = classifier(dropout_output)

            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(logits.view(-1, model.num_labels), batch_labels.view(-1))

            # loss scaling，代替loss.backward()
            # with amp.scaled_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            loss.backward()
            # loss_collect.append(loss.item())

            print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss.item())

            optimizer.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), 'model.pt')


def dev_clf(model, device, dataloader):
    print('dev')

    import numpy as np
    from sklearn.metrics import classification_report

    model.load_state_dict(torch.load('model.pt', map_location=lambda storage, loc: storage))
    model.to(device)

    model.eval()

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='dev'):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            outputs = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=batch_labels)

            loss, logits = outputs[:2]

            logits = logits.softmax(dim=1).argmax(dim=1)
            pred_labels.append(logits.detach().cpu().numpy())
            true_labels.append(batch_labels.detach().cpu().numpy())
    # 查看各个类别的准召
    print(classification_report(np.concatenate(true_labels), np.concatenate(pred_labels)))


def predict_clf(model, device, dataloader):
    model.load_state_dict(torch.load('model.pt', map_location=lambda storage, loc: storage))
    model.to(device)

    model.eval()

    pred_labels = np.array([])
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='test'):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            outputs = model(
                batch_seqs, batch_seq_masks, batch_seq_segments, labels=batch_labels)

            loss, logits = outputs[:2]

            logits = logits.softmax(dim=1).argmax(dim=1)

            pred_labels = np.append(pred_labels, logits.detach().cpu().numpy())

    return pred_labels


def build_submission(df, y_pred):
    df['label'] = pd.Series(y_pred)

    df['label'] = df['label'].astype(int).astype(str)

    df = df.filter(items=['id', 'label'])

    df.to_json('afqmc_predict.json', orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    from data_processor.data_builder import data_loader

    # filename = 'train'
    filename = 'dev'
    # filename = 'test'
    df = data_loader(filename=filename, path=PATH_DATA_PRE)
    df[COL_LB] = df[COL_LB].astype(int)
    df[COL_CLS] = df[COL_CLS].astype(int)

    # df = df.head()

    df, col_txt = feature_styg(df, COL_ST1, COL_ST2)
    model_processor = ModelProcessor(path=PATH_MODEL_BERT, device=DEVICE)

    df, col_ids, col_seq_ids, col_masks = txt2id(df, COL_ST1, COL_ST2, model_processor)

    print('token done')

    # t_dataloader = build_dataloader(df, col_ids, col_seq_ids, col_masks, mode='predict')
    t_dataloader = build_dataloader(df, col_ids, col_seq_ids, col_masks)

    # train_clf(model=model_processor.model, device=DEVICE, dataloader=t_dataloader, epoch=EPOCH)
    dev_clf(model=model_processor.model, device=DEVICE, dataloader=t_dataloader)

    # submit
    # pred_labels = predict_clf(model=model_processor.model, device=DEVICE, dataloader=t_dataloader)
    # build_submission(df, pred_labels)

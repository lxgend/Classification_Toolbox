# coding=utf-8
import json
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from tqdm import tqdm

from data_processor.data2example import cls_data_processors
from data_processor.example2dataset import load_and_cache_examples
from nets.plm import MODEL_CLASSES


def train(args, train_dataset, model):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    print('args.train_batch_size')
    print(args.train_batch_size)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()

    # scheduler.step()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}]

    lr = 0.0001
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):

        for step, batch_data in enumerate(train_dataloader):
            # set model to training mode

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            optimizer.zero_grad()

            outputs = model(input_ids=batch_input_ids,
                            attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids,
                            labels=batch_label_ids)

            loss, scores = outputs[:2]

            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                print('epoch: {} | step: {} | loss: {}'.format(epoch, step, loss.item()))

            global_step += 1

    torch.save(model.state_dict(), 'cls_fine_tuned.pt')

    return global_step


def evaluate(args, eval_dataset, model):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    from sklearn.metrics import classification_report

    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch_data in tqdm(eval_dataloader, desc='dev'):
            model.eval()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            outputs = model(input_ids=batch_input_ids,
                            attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids,
                            labels=batch_label_ids)

            # logits: (batch_size, num_labels)
            loss, logits = outputs[:2]
            # softmax, 最里层dim归一化(num_labels层), shape不变
            # argmax, 最里层dim取最大值下标(num_labels层),得到每个example对应的pred label, (batch_size)
            predictions = logits.softmax(dim=-1).argmax(dim=-1)

            pred_labels.append(predictions.detach().cpu().numpy())
            true_labels.append(batch_label_ids.detach().cpu().numpy())
    # 查看各个类别的准召
    print(classification_report(np.concatenate(true_labels), np.concatenate(pred_labels)))


def predict(args, test_dataset, model):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

    results = []

    # from data_processor.data_example import get_entity

    with torch.no_grad():
        for step, batch_data in enumerate(tqdm(test_dataloader, desc='test')):
            model.eval()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            outputs = model(input_ids=batch_input_ids,
                            attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids,
                            labels=batch_label_ids)

            # logits: (batch_size, max_len, num_labels)
            loss, logits = outputs[:2]

            # softmax, 最里层dim归一化, shape不变, (batch_size, max_len, num_labels)
            # argmax, 最里层dim取最大值,得到 index对应label, (batch_size, max_len)
            predictions = logits.softmax(dim=1).argmax(dim=1)

            predictions = predictions.detach().cpu().numpy().tolist()
            # predictions = predictions[0][1:-1]  # [CLS]XXXX[SEP]
            #
            # label_entities = get_entity(predictions, args.id2label)
            # d = {}
            # d['id'] = step
            # # d['tag_seq'] = " ".join([args.id2label[x] for x in preds])
            # d['entities'] = label_entities
            # results.append(d)

    with open('predict_tmp.json', 'w') as writer:
        for d in results:
            writer.write(json.dumps(d) + '\n')


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_data_processor = cls_data_processors[args.task_name]()
    label_list = cls_data_processor.get_labels()

    num_labels = len(label_list)

    print('num_labels %d' % (num_labels))

    args.id2label = {i: label for i, label in enumerate(label_list)}

    model_class, tokenizer_class, model_path = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, num_labels=num_labels)
    model.to(args.device)

    if args.do_train:
        # 读数据
        train_dataset = load_and_cache_examples(args, tokenizer, cls_data_processor, data_type='train')
        print('train_dataset')
        print(len(train_dataset))

        # train
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        global_step = train(args, train_dataset, model)
        print("global_step = %s" % global_step)

    if args.do_eval:
        print('evaluate')

        eval_dataset = load_and_cache_examples(args, tokenizer, cls_data_processor, data_type='dev')

        model.load_state_dict(torch.load('cls_fine_tuned.pt', map_location=lambda storage, loc: storage))
        model.to(args.device)
        evaluate(args, eval_dataset, model)

    if args.do_test:
        print('test')
        test_dataset = load_and_cache_examples(args, tokenizer, cls_data_processor, data_type='test')

        model.load_state_dict(torch.load('cls_fine_tuned.pt', map_location=lambda storage, loc: storage))
        model.to(args.device)
        predict(args, test_dataset, model)


class Args(object):
    def __init__(self):
        self.task_name = 'tnews'
        self.data_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'data', 'tnews_public'])
        self.overwrite_cache = 1
        self.local_rank = 0
        self.n_gpu = torch.cuda.device_count()
        self.max_seq_length = 42
        self.model_type = 'bert'

        self.do_train = 1
        self.per_gpu_train_batch_size = 16
        self.num_train_epochs = 1
        self.max_steps = -1
        self.gradient_accumulation_steps = 1

        self.do_eval = 1
        self.eval_batch_size = 16

        self.do_test = 0
        self.test_batch_size = 1


if __name__ == '__main__':
    # args = get_argparse().parse_args()
    args = Args()
    main(args)

# coding=utf-8
import json
import os

import numpy as np
import torch
import torch.distributed
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from classifier.nets.plm import MODEL_CLASSES
from data_processor.data2example import clf_data_processors
from data_processor.example2dataset import load_and_cache_examples


def train(args, train_dataset, model):
    args.train_batch_size = args.per_gpu_train_batch_size

    train_sampler = RandomSampler(train_dataset if args.local_rank == -1 else DistributedSampler)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #      'weight_decay': 0.0}]
    # lr = 0.0001
    # optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.8)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    optimizer.zero_grad()
    for epoch in range(int(args.num_train_epochs)):

        for step, batch_data in enumerate(train_dataloader):
            # set model to training mode
            model.train()

            batch_data = tuple(t.to(args.device) for t in batch_data)
            batch_input_ids, batch_input_mask, batch_segment_ids, batch_label_ids = batch_data

            outputs = model(input_ids=batch_input_ids,
                            attention_mask=batch_input_mask,
                            token_type_ids=batch_segment_ids,
                            labels=batch_label_ids)

            loss, scores = outputs[:2]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if step % 5 == 0:
                print('epoch: {} | step: {} | loss: {}'.format(epoch, step, loss.item()))

            global_step += 1

    # torch.save(model.state_dict(), 'cls_fine_tuned.pt')
    if 'cuda' in str(args.device):
        torch.cuda.empty_cache()

    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

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
    # Setup CUDA, GPU & distributed training
    # would not use distributed
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu")
        args.n_gpu = torch.cuda.device_count()

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')  # one machine, multiple gpu
        args.n_gpu = 1  # one gpu for one process
    args.device = device

    # data init
    clf_data_processor = clf_data_processors[args.task_name]()
    label_list = clf_data_processor.get_labels()
    num_labels = len(label_list)
    args.id2label = {i: label for i, label in enumerate(label_list)}

    print('num_labels %d' % (num_labels))
    print('model %s' % args.model_type)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model_class, tokenizer_class, model_path = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path, num_labels=num_labels)

    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, clf_data_processor, data_type='train')
        print('train_dataset %d' % len(train_dataset))

        # train
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        global_step = train(args, train_dataset, model)
        print("global_step = %s" % global_step)

    if args.do_eval:
        print('evaluate')

        eval_dataset = load_and_cache_examples(args, tokenizer, clf_data_processor, data_type='dev')

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

        # model.load_state_dict(torch.load('cls_fine_tuned.pt', map_location=lambda storage, loc: storage))
        # model.to(args.device)
        evaluate(args, eval_dataset, model)

    if args.do_test:
        print('test')
        test_dataset = load_and_cache_examples(args, tokenizer, clf_data_processor, data_type='test')

        model.load_state_dict(torch.load('cls_fine_tuned.pt', map_location=lambda storage, loc: storage))
        model.to(args.device)
        predict(args, test_dataset, model)


class Args(object):
    def __init__(self):
        self.task_name = 'tnews'
        self.data_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'data', 'tnews_public'])
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.overwrite_cache = 1
        self.max_seq_length = 42
        self.model_type = 'albert'

        self.local_rank = -1
        self.use_cpu = 0
        self.n_gpu = torch.cuda.device_count()

        self.do_train = 1
        self.per_gpu_train_batch_size = 16
        self.num_train_epochs = 1
        self.max_steps = -1
        self.gradient_accumulation_steps = 1
        self.warmup_proportion = 0.1
        self.learning_rate = 1e-4
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0

        self.do_eval = 0
        self.eval_batch_size = 16

        self.do_test = 0
        self.test_batch_size = 1


if __name__ == '__main__':
    # args = get_argparse().parse_args()

    import time

    a = time.time()

    args = Args()
    main(args)

    print(time.time() - a)

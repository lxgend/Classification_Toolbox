# coding=utf-8
import os

import torch
from nets.plm import MODEL_CLASSES

def train(args, train_dataset, model):
    pass

def evaluate(args, eval_dataset, model):
    pass

def predict(args, test_dataset, model):
    pass


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cls_data_processor = cls_data_processors[args.task_name](args.data_dir)

    label_list = cls_data_processor.get_labels()

    num_labels = len(label_list)

    args.id2label = {i: label for i, label in enumerate(label_list)}

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_dir)
    model = model_class.from_pretrained(args.model_dir, num_labels=num_labels)
    model.to(args.device)


    if args.do_train:
        # 读数据
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, ner_data_processor, data_type='train')
        print('train_dataset')
        print(len(train_dataset))

        # train
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        global_step = train(args, train_dataset, model)
        print("global_step = %s" % global_step)


class Args(object):
    def __init__(self):
        self.task_name = 'tnews'
        self.model_dir = '/Users/lixiang/Documents/nlp_data/pretrained_model/roberta_wwm_ext_zh_hit_pt'
        self.data_dir = os.path.join(*[os.path.dirname(os.path.abspath(__file__)), 'data', 'tnews_public'])
        self.overwrite_cache = 1
        self.local_rank = 0
        self.n_gpu = torch.cuda.device_count()
        self.train_max_seq_length = 55
        self.eval_max_seq_length = 55
        self.model_type = 'bert'

        self.do_train = 1
        self.per_gpu_train_batch_size = 16
        self.num_train_epochs = 3
        self.max_steps = -1
        self.gradient_accumulation_steps = 1

        self.do_eval = 0
        self.eval_batch_size = 16

        self.do_test = 0
        self.test_batch_size = 1


if __name__ == '__main__':
    # args = get_argparse().parse_args()
    args = Args()
    main(args)

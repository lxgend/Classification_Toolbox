# coding=utf-8
'''example → feature →  tensor dataset'''
import logging
import os
from typing import List

import torch
import torch.distributed
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, :max_len]  # 尾部截断
    all_attention_mask = all_attention_mask[:, :max_len]
    all_token_type_ids = all_token_type_ids[:, :max_len]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def xlnet_collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_token_type_ids, all_lens, all_labels = map(torch.stack, zip(*batch))
    max_len = max(all_lens).item()
    all_input_ids = all_input_ids[:, -max_len:]
    all_attention_mask = all_attention_mask[:, -max_len:]  # 头部截断
    all_token_type_ids = all_token_type_ids[:, -max_len:]
    return all_input_ids, all_attention_mask, all_token_type_ids, all_labels


def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_length,
                                 task,
                                 label_list=None) -> List[InputFeatures]:
    # if output_mode is None:
    #     output_mode = cls_data_processors[task]
    #     logger.info("Using output mode %s for task %s" % (output_mode, task))

    # label2id = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            text=example.text_a,
            text_pair=example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']
        label = int(example.label)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # input_len = len(input_ids)
        # # Zero-pad up to the sequence length.
        # padding_length = max_length - len(input_ids)
        # if pad_on_left:
        #     input_ids = ([pad_token] * padding_length) + input_ids
        #     attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        #     token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        # else:
        #     input_ids = input_ids + ([pad_token] * padding_length)
        #     attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        #     token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                            max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                            max_length)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))
    return features


def load_and_cache_examples(args, tokenizer, processor, data_type='train'):
    if args.local_rank not in [-1, 0] and args.do_eval == 0:
        torch.distributed.barrier()

    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()

    # filename
    cached_features_file = os.path.join(args.data_dir, '{}_{}_{}.cache'.format(
        data_type,
        str(args.max_seq_length),
        str(args.task_name)))

    if os.path.exists(cached_features_file) and args.overwrite_cache == 0:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:

        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)

        max_length = processor.get_max_length()

        # features = convert_examples_to_features(examples,
        #                                         tokenizer,
        #                                         label_list=label_list,
        #                                         max_length=max_length,
        #                                         pad_on_left=bool(args.model_type in ['xlnet']),
        #                                         # pad on the left for xlnet
        #                                         pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        #                                         pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        #                                         )

        features = convert_examples_to_features(examples,
                                                tokenizer,
                                                label_list=label_list,
                                                max_length=max_length,
                                                task=args.task_name
                                                )
        logger.info("Saving features into cached file %s", cached_features_file)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


if __name__ == '__main__':
    pass

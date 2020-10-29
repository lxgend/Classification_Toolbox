# coding=utf-8
'''raw data on disk to three List[InputExample]'''
import csv
import json
import os
from typing import List

import numpy as np


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file, store in a list."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines


class TnewsProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_train_examples(self, data_dir) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        result = result[:32]
        return result

    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

        result = result[:64]
        return result

    def get_test_examples(self, data_dir) -> List[InputExample]:
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    # according to the data, defines ids for the labels
    def get_labels(self):
        """See base class."""
        labels = []
        for i in range(17):
            if i == 5 or i == 11:
                continue
            labels.append(str(100 + i))
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "100"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def max_length(self, data_dir):
        examples = self.get_train_examples(data_dir)
        all_lens = np.array([])
        for txt in examples:
            all_lens = np.append(all_lens, len(txt.text_a))
        all_lens = np.sort(all_lens)
        max_len = np.percentile(all_lens, 95)  # 95%分位数
        max_len = int(max_len) + 10
        # 42
        return max_len

    def get_max_length(self):
        return 42


clf_data_processors = {
    'tnews': TnewsProcessor,
}

if __name__ == '__main__':
    processor = clf_data_processors['tnews']()
    # from parm import PATH_DATA_TNEWS
    # print(processor.get_max_length(PATH_DATA_TNEWS))

# coding=utf-8
'''raw data on disk to three List[InputExample]'''
import csv
import json
from typing import List

import numpy as np
import pandas as pd

from parm import *


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
    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip the headers
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


class TnewsProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "train.csv")), "train")
        # result = result[:150]
        return result

    def get_dev_examples(self) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "dev.csv")), "dev")
        # result = result[:64]
        return result

    def get_test_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "test.csv")), "test")

    # according to the data, defines ids for the labels
    def get_labels(self):
        with open(os.path.join(self.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            labels = json.load(f)
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            text_b = None
            label = str(line[0]) if set_type != 'test' else "100"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_max_length(self):
        return 42


class TnewsProcessor_vec(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "train.csv")))
        # result = result[:150]
        return result

    def get_dev_examples(self) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "dev.csv")))
        # result = result[:64]
        return result

    def get_test_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "test.csv")))

    @classmethod
    def _read_csv(cls, input_file):
        df = pd.read_csv(input_file, dtype=object, encoding='utf-8')
        return df

    # according to the data, defines ids for the labels
    def get_labels(self):
        with open(os.path.join(self.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            labels = json.load(f)
        return labels

    def _create_examples(self, df):
        df = df.rename(columns={'token': 'text_a'})
        return df

    def get_max_length(self):
        return 42



class TnewsProcessor_vec(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "train.csv")))
        # result = result[:150]
        return result

    def get_dev_examples(self) -> List[InputExample]:
        """See base class."""
        result = self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "dev.csv")))
        # result = result[:64]
        return result

    def get_test_examples(self) -> List[InputExample]:
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "test.csv")))

    @classmethod
    def _read_csv(cls, input_file):
        df = pd.read_csv(input_file, dtype=object, encoding='utf-8')
        return df

    # according to the data, defines ids for the labels
    def get_labels(self):
        with open(os.path.join(self.data_dir, 'labels.json'), 'r', encoding='utf-8') as f:
            labels = json.load(f)
        return labels

    def _create_examples(self, df):
        df = df.rename(columns={'token': 'text_a'})
        return df

    def get_max_length(self):
        return 42


clf_data_processors = {
    'tnews': TnewsProcessor,
    'tnews_vec': TnewsProcessor_vec,
    'tnews_tfidf': TnewsProcessor_vec,
}

if __name__ == '__main__':
    processor = clf_data_processors['tnews2'](PATH_DATA_TNEWS_PRE)

    ll = processor.get_dev_examples()

    print(ll[0].text_a)

    # from parm import PATH_DATA_TNEWS
    # print(processor.get_max_length(PATH_DATA_TNEWS))

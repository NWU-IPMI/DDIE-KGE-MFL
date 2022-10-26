# -*- coding: utf-8 -*-

from transformers.data.processors.utils import InputExample, InputFeatures, DataProcessor
import os, logging

logger = logging.getLogger(__name__)


class DDIEProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'),
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "test.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir,"dev.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self, mode='ddie'):
        """See base class."""
        if mode == 'ddie':
            return ['other', 'mechanism', 'effect', 'advise', 'int']
        elif mode == 'pretraining':
            return ['negative', 'positive']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []

        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            label = line[0]
            text_a = line[1] + ' [SEP] ' + line[2] + ' [SEP] ' + line[3] + ' [SEP]'
            if i % 1000 == 0:
                logger.info(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples



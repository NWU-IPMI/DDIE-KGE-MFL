# -*- coding: utf-8 -*-
import os
import torch
from processor_ddie import DDIEProcessor
import logging
import stanza
import csv
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Union
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    dep_mask0: Optional[List[int]] = None
    dep_mask1: Optional[List[int]] = None
    dep_mask2: Optional[List[int]] = None
    dep_mask3: Optional[List[int]] = None
    drug_a_id: Optional[Union[int]] = None
    drug_b_id: Optional[Union[int]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    """
        load or cache the examples
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = DDIEProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'test' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)
    ))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(
            args.data_dir)
        features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length,
                                                label_list=label_list, stanza_path=args.stanza_path,
                                                entity_path=args.entity_path)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    drug_id = tokenizer.vocab['drug']
    drug_1_id = tokenizer.vocab['##1']
    drug_2_id = tokenizer.vocab['##2']
    all_input_ids = [f.input_ids for f in features]
    all_entity1_pos = []
    all_entity2_pos = []
    for input_ids in all_input_ids:
        entity1_pos = args.max_seq_length - 1
        entity2_pos = args.max_seq_length - 1
        for i in range(args.max_seq_length):
            if input_ids[i] == drug_id and input_ids[i + 1] == drug_1_id:
                entity1_pos = i
            if input_ids[i] == drug_id and input_ids[i + 1] == drug_2_id:
                entity2_pos = i
        all_entity1_pos.append(entity1_pos)
        all_entity2_pos.append(entity2_pos)
    assert len(all_input_ids)==len(all_entity1_pos)==len(all_entity2_pos)

    range_list = list(range(args.max_seq_length, 2 * args.max_seq_length))
    all_relative_dist1 = torch.tensor([[x - e1 for x in range_list] for e1 in all_entity1_pos], dtype=torch.long)
    all_relative_dist2 = torch.tensor([[x - e2 for x in range_list] for e2 in all_entity2_pos], dtype=torch.long)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_dep_mask0 = torch.tensor([f.dep_mask0 for f in features], dtype=torch.long)
    all_dep_mask1 = torch.tensor([f.dep_mask1 for f in features], dtype=torch.long)
    all_dep_mask2 = torch.tensor([f.dep_mask2 for f in features], dtype=torch.long)
    all_dep_mask3 = torch.tensor([f.dep_mask3 for f in features], dtype=torch.long)

    all_drug_a_ids = torch.tensor([f.drug_a_id for f in features], dtype=torch.long)
    all_drug_b_ids = torch.tensor([f.drug_b_id for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features],dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_relative_dist1, all_relative_dist2, all_dep_mask0,
                            all_dep_mask1, all_dep_mask2, all_dep_mask3, all_labels,
                            all_drug_a_ids, all_drug_b_ids)
    return dataset
def convert_examples_to_features(examples, tokenizer, max_length, label_list, stanza_path, entity_path):
    """
       Loads a data file into a list of ``InputFeatures``

       Args:
           examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
           tokenizer: Instance of a tokenizer that will tokenize the examples
           max_length: Maximum example length. Defaults to the tokenizer's max_len
           task: GLUE task
           label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
           output_mode: String indicating the output mode. Either ``regression`` or ``classification``

       Returns:
           If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
           task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
           ``InputFeatures`` which can be fed to the model.

       """
    if max_length is None:
        max_length = tokenizer.max_len
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    nlp = stanza.Pipeline('en', package='craft', dir=stanza_path)
    entity2id = {}
    with open(entity_path, 'r', encoding='utf-8') as fr:
        lines = csv.reader(fr, delimiter='\t')
        for line in lines:
            idx = line[0]
            entity = line[1]
            entity2id[entity] = idx
    for (example_index, example) in enumerate(examples):
        if example_index % 1000 == 0:
            logger.info("Writing example %d of %d ..." % (example_index, len(examples)))
        tokens_a = tokenizer.tokenize(example.text_a)
        entity_one_start = tokens_a.index("<e1>")
        entity_one_end = tokens_a.index("</e1>")
        entity_two_start = tokens_a.index("<e2>")
        entity_two_end = tokens_a.index("</e2>")
        # Replace the token
        tokens_a[entity_one_start] = "$"
        tokens_a[entity_one_end] = "$"
        tokens_a[entity_two_start] = "#"
        tokens_a[entity_two_end] = "#"
        special_tokens_count = 2
        if len(tokens_a) > max_length - special_tokens_count:
            tokens_a = tokens_a[:(max_length - special_tokens_count)]
        tokens = tokens_a
        tokens += ['[SEP]']
        tokens = ['[CLS]'] + tokens
        token_type_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        dependency_mask_0, dependency_mask_1, dependency_mask_2, dependency_mask_3 = get_dependency_parsing_tree_mask(
            example.text_a, nlp, tokenizer)
        dependency_mask0_pad = [0] * len(attention_mask)
        dependency_mask1_pad = [0] * len(attention_mask)
        dependency_mask2_pad = [0] * len(attention_mask)
        dependency_mask3_pad = [0] * len(attention_mask)

        for idx, item in enumerate(dependency_mask_0):
            if item != 0:
                dependency_mask0_pad[idx + 1] = 1
        for idx, item in enumerate(dependency_mask_1):
            if item != 0:
                dependency_mask1_pad[idx + 1] = 1
        for idx, item in enumerate(dependency_mask_2):
            if item != 0:
                dependency_mask2_pad[idx + 1] = 1
        for idx, item in enumerate(dependency_mask_3):
            if item != 0:
                dependency_mask3_pad[idx + 1] = 1

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with token type length {} vs {}".format(
            len(token_type_ids), max_length)
        label_id = int(label_map[example.label])
        split_tokens = example.text_a.split("[SEP]")
        drug_a = split_tokens[1].strip()
        drug_b = split_tokens[2].strip()
        drug_a_id = int(entity2id[drug_a])
        drug_b_id = int(entity2id[drug_b])

        if example_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("drug_a_id:%s" % str(drug_a_id))
            logger.info("drug_b_id:%s" % str(drug_b_id))
        features.append(InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                      label=label_id, dep_mask0=dependency_mask0_pad, dep_mask1=dependency_mask1_pad,
                                      dep_mask2=dependency_mask2_pad, dep_mask3=dependency_mask3_pad,
                                      drug_a_id=drug_a_id,
                                      drug_b_id=drug_b_id))
    return features


def get_dependency_parsing_tree_mask(sentence, nlp, tokenizer):
    """

    :param sentence: a sample texts
    :param nlp: the object of stanza or other nlp tool
    :param tokenizer:
    :return: the distance embedding of dependency parsing tree
    """
    sentence = sentence[:sentence.index('[SEP]')]
    sentence = sentence.replace('<e1>', '')
    sentence = sentence.replace('</e1>', '')
    sentence = sentence.replace('<e2>', '')
    sentence = sentence.replace('</e2>', '')

    documents = nlp(sentence)
    words = []
    head = []
    subj_pos = -1
    obj_pos = -1
    before_sentence_words = 0

    for sen in documents.sentences:
        before_sentence_words = len(words)
        for index, word in enumerate(sen.words):
            if word.text == 'DRUG1':  # subject
                subj_pos = index + before_sentence_words
            elif word.text == 'DRUG2':
                obj_pos = index + before_sentence_words
            if word.head == 0:
                head += [0]
            else:
                head += [word.head + before_sentence_words]
            words += [word.text]
    subj_pos = [subj_pos]
    obj_pos = [obj_pos]
    length = len(words)
    cas = None
    subj_ancestors = set(subj_pos)
    for subj in subj_pos:
        h = head[subj]
        temp = [subj]
        while h > 0:
            temp += [h - 1]
            subj_ancestors.add(h - 1)
            h = head[h - 1]
        if cas is None:
            cas = set(temp)
        else:
            cas.intersection_update(temp)
    obj_ancestors = set(obj_pos)
    for obj in obj_pos:
        h = head[obj]
        temp = [obj]
        while h > 0:
            temp += [h - 1]
            obj_ancestors.add(h - 1)
            h = head[h - 1]
        cas.intersection_update(temp)
    if len(cas) == 1:
        lca = list(cas)[0]
    else:
        children_count = {k: 0 for k in cas}
        for c in cas:
            if head[c] > 0 and head[c] - 1 in cas:
                children_count[head[c] - 1] += 1
        for c in cas:
            if children_count[c] == 0:
                lca = c
                break
    path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)
    path_nodes.add(lca)
    # compute distance to path_nodes
    distance = [-1 if i not in path_nodes else 0 for i in range(length)]

    for i in range(length):
        if distance[i] < 0:
            stack = [i]
            while stack[-1] >= 0 and stack[-1] not in path_nodes:
                stack.append(head[stack[-1]] - 1)
            if stack[-1] in path_nodes:
                for d, j in enumerate(reversed(stack)):
                    distance[j] = d
            else:
                for j in stack:
                    if j >= 0 and distance[j] < 0:
                        distance[j] = int(1e4)  # aka infinity
    dependency_mask_0 = []
    dependency_mask_1 = []
    dependency_mask_2 = []
    dependency_mask_3 = []

    for index, item in enumerate(distance):
        word_piece = tokenizer.tokenize(words[index])
        len_word_piece = len(word_piece)

        if item <= 0:
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_0 += [0]
            dependency_mask_0 += [1]
            dependency_mask_0 += [2] * (len_word_piece - 1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_0 += [0]
        else:
            dependency_mask_0 += [0] * len_word_piece

        if item <= 1:
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_1 += [0]
            dependency_mask_1 += [1]
            dependency_mask_1 += [2] * (len_word_piece - 1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_1 += [0]
        else:
            dependency_mask_1 += [0] * len_word_piece

        if item <= 2:
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_2 += [0]
            dependency_mask_2 += [1]
            dependency_mask_2 += [2] * (len_word_piece - 1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_2 += [0]
        else:
            dependency_mask_2 += [0] * len_word_piece

        if item <= 3:
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_3 += [0]
            dependency_mask_3 += [1]
            dependency_mask_3 += [2] * (len_word_piece - 1)
            if words[index] == 'DRUG1' or words[index] == 'DRUG2':
                dependency_mask_3 += [0]
        else:
            dependency_mask_3 += [0] * len_word_piece

    return dependency_mask_0, dependency_mask_1, dependency_mask_2, dependency_mask_3

import util
import numpy as np
import random
from transformers import AutoTokenizer
import os
from os.path import join
import json
import itertools
import pickle
import logging
import torch

logger = logging.getLogger(__name__)


def convert_to_torch_tensor(input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map,
                            is_training, gold_starts, gold_ends, gold_mention_cluster_map,
                            split_starts=None, split_ends=None, predictions_start=None,
                            predictions_ends=None, predictions_cluster_map=None, predictions_split_map=None):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
    sentence_len = torch.tensor(sentence_len, dtype=torch.long)
    genre = torch.tensor(genre, dtype=torch.long)
    sentence_map = torch.tensor(sentence_map, dtype=torch.long)
    is_training = torch.tensor(is_training, dtype=torch.bool)
    gold_starts = torch.tensor(gold_starts, dtype=torch.long)
    gold_ends = torch.tensor(gold_ends, dtype=torch.long)
    gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)

    if split_starts is None:
        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
           is_training, gold_starts, gold_ends, gold_mention_cluster_map,
    else:
        split_starts = torch.tensor(split_starts, dtype=torch.long)
        split_ends = torch.tensor(split_ends, dtype=torch.long)
        predictions_start = torch.tensor(predictions_start, dtype=torch.long)
        predictions_ends = torch.tensor(predictions_ends, dtype=torch.long)
        predictions_split_map = torch.tensor(predictions_split_map, dtype=torch.long)
        predictions_cluster_map = torch.tensor(predictions_cluster_map, dtype=torch.long)

        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
            is_training, split_starts, split_ends,\
            predictions_start, predictions_ends, predictions_split_map, predictions_cluster_map, \
            gold_starts, gold_ends, gold_mention_cluster_map


class CorefDataProcessor:
    def __init__(self, config, language='english'):
        self.config = config
        self.language = config['language'] or language

        self.max_seg_len = config['max_segment_len']
        self.max_training_seg = config['max_training_sentences']
        self.data_dir = config['data_dir']
        self.long_doc_strategy = config['long_doc_strategy']
        if self.long_doc_strategy not in ['keep', 'truncate', 'split', 'even-chunks']:
            raise Exception("Invalid strategy for long documnets, use either 'keep', 'truncate', 'split' or 'even-chunks'")

        # Get tensorized samples
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path):
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info('Loaded tensorized examples from cache')
        else:
            # Generate tensorized samples
            self.tensor_samples = {}
            tensorizer = Tensorizer(self.config)
            paths = {
                'trn': join(self.data_dir, f'train.{self.language}.{self.max_seg_len}.jsonlines'),
                'dev': join(self.data_dir, f'dev.{self.language}.{self.max_seg_len}.jsonlines'),
                'tst': join(self.data_dir, f'test.{self.language}.{self.max_seg_len}.jsonlines')
            }
            for split, path in paths.items():
                logger.info('Tensorizing examples from %s; results will be cached)' % path)
                is_training = (split == 'trn')
                with open(path, 'r') as f:
                    samples = [json.loads(line) for line in f.readlines()]
                tensor_samples = itertools.chain(
                    *(tensorizer.tensorize_example(sample, is_training) for sample in samples)
                )
                tensor_samples = list(tensor_samples)
                self.tensor_samples[split] = [(doc_key, convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples]
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)

    def get_tensor_examples(self):
        # For each split, return list of tensorized samples to allow variable length input (batch size = 1)
        return self.tensor_samples['trn'], self.tensor_samples['dev'], self.tensor_samples['tst']

    def get_stored_info(self):
        return self.stored_info

    def get_cache_path(self):
        name = f'cached.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}'
        name += f'.split={self.long_doc_strategy}'
        name += '.bin'
        cache_path = join(self.data_dir, name)
        return cache_path


class Tensorizer:
    def __init__(self, config):
        self.config = config
        self.long_doc_strategy = config['long_doc_strategy']
        self.tokenizer = AutoTokenizer.from_pretrained(config['bert_tokenizer_name'])

        # Will be used in evaluation
        self.stored_info = {}
        self.stored_info['tokens'] = {}  # {doc_key: ...}
        self.stored_info['subtoken_maps'] = {}  # {doc_key: ...}; mapping back to tokens
        self.stored_info['gold'] = {}  # {doc_key: ...}
        self.stored_info['genre_dict'] = {genre: idx for idx, genre in enumerate(config['genres'])}
        self.stored_info['sentence_map'] = {}

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for speaker in speakers:
            if len(speaker_dict) > self.config['max_num_speakers']:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, is_training, is_hybrid=True, split_starts=None, split_ends=None, predictions=None):
        # Mentions and clusters
        clusters = example['clusters']
        gold_mentions = sorted(tuple(mention) for mention in util.flatten(clusters))
        gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)}
        gold_mention_cluster_map = np.zeros(len(gold_mentions))  # 0: no cluster
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id + 1

        # Speakers
        speakers = example['speakers']
        speaker_dict = self._get_speaker_dict(util.flatten(speakers))

        # Sentences/segments
        sentences = example['sentences']  # Segments
        sentence_map = example['sentence_map']
        num_words = sum([len(s) for s in sentences])
        max_sentence_len = self.config['max_segment_len']
        sentence_len = np.array([len(s) for s in sentences])

        # Bert input
        input_ids, input_mask, speaker_ids = [], [], []
        for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
            while len(sent_input_ids) < max_sentence_len:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        # Keep info to store
        doc_key = example['doc_key']
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        self.stored_info['gold'][doc_key] = example['clusters']
        self.stored_info['tokens'][doc_key] = example['tokens']
        self.stored_info['sentence_map'][doc_key] = sentence_map

        # Construct example
        genre = self.stored_info['genre_dict'].get(doc_key[:2], 0)
        gold_starts, gold_ends = self._tensorize_spans(gold_mentions)
        if split_starts is None:
            example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                          gold_starts, gold_ends, gold_mention_cluster_map)
        else:
            predictions_start = []
            predictions_ends = []
            predictions_split_map = []
            predictions_cluster_map = []

            for index, split_predictions in enumerate(predictions):
                for cluster_index, cluster in enumerate(split_predictions):
                    for mention in cluster:
                        predictions_start.append(mention[0])
                        predictions_ends.append(mention[1])
                        predictions_split_map.append(index)
                        predictions_cluster_map.append(cluster_index)

            example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                              gold_starts, gold_ends, gold_mention_cluster_map, split_starts, split_ends, predictions_start,
                              predictions_ends, predictions_cluster_map, predictions_split_map)

        if (is_hybrid or is_training) and len(sentences) > self.config['max_training_sentences']:
            if self.long_doc_strategy == 'split':
                out = []
                for sentence_offset in range(0, len(sentences), self.config['max_training_sentences']):
                    out.append((
                        f'{doc_key}_{sentence_offset}',
                        self.truncate_example(*example_tensor, sentence_offset=sentence_offset)
                    ))

                    self.stored_info['subtoken_maps'][f'{doc_key}_{sentence_offset}'] = example.get('subtoken_map', None)
                    self.stored_info['gold'][f'{doc_key}_{sentence_offset}'] = example['clusters']
                    self.stored_info['tokens'][f'{doc_key}_{sentence_offset}'] = example['tokens']
                return out
            elif self.long_doc_strategy == 'truncate':
                return [(doc_key, self.truncate_example(*example_tensor))]
            elif self.long_doc_strategy == 'even-chunks':
                out = []
                max_len = self.config["max_training_sentences"]
                n = len(sentences)
                buckets = min(n, (len(sentences) + max_len - 1) // max_len)
                floor = n // buckets
                ceiling = floor + 1
                stepdown = n % buckets
                offset = 0
                for _ in range(stepdown):
                    out.append((
                        f'{doc_key}_{offset}',
                        self.truncate_example(*example_tensor, max_sentences=ceiling, sentence_offset=offset)
                    ))
                    offset += ceiling
                for _ in range(stepdown, buckets):
                    out.append((
                        f'{doc_key}_{offset}',
                        self.truncate_example(*example_tensor, max_sentences=floor, sentence_offset=offset)
                    ))
                    offset += floor
                return out
            else:
                return [(doc_key, example_tensor)]
        else:
            return [(doc_key, example_tensor)]

    def truncate_example(self, input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, is_training,
                         gold_starts, gold_ends, gold_mention_cluster_map, max_sentences=None, sentence_offset=None):

        max_sentences = self.config["max_training_sentences"] if max_sentences is None else max_sentences
        num_sentences = input_ids.shape[0]
        assert num_sentences >= max_sentences

        sent_offset = sentence_offset
        if sent_offset is None:
            sent_offset = random.randint(0, num_sentences - max_sentences)
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset: sent_offset + max_sentences].sum()

        input_ids = input_ids[sent_offset: sent_offset + max_sentences, :]
        input_mask = input_mask[sent_offset: sent_offset + max_sentences, :]
        speaker_ids = speaker_ids[sent_offset: sent_offset + max_sentences, :]
        sentence_len = sentence_len[sent_offset: sent_offset + max_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]

        return input_ids, input_mask, speaker_ids, sentence_len, genre, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map

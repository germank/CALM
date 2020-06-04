# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from metrics import compute_metrics
from calm_aux import *
import json

def log_loss(fout, calm_iterator, loss):
    sequence_id = calm_iterator.get_current_sequence_id()
    domain_id = calm_iterator.get_current_sequence_class_id()
    domain_name = calm_iterator.get_current_sequence_class_name()
    data_row = {'loss': loss, 'domain': domain_id, 'domain_name': domain_name, 'sequence': sequence_id}
    fout.write(json.dumps(data_row))
    fout.write('\n')


class CALM(object):
    def __init__(self, corpus_path, model_level, switch_frequency, nswitches,
            window_size, batch_size, cuda):
        """
        Creates an iterable object for the CALM task.
        Usage: 
            calm_it = CALM(corpus_path, ....)
            for inp,tgt in calm:
                pred = f(inp)
                loss = L(pred, tgt)
                log_loss(fout, calm_it, loss)
        Args:
            corpus_path: path to the root of a CALM corpus (e.g. data/news_dev).
            model_level: one of "char" or "word". (str)
            switch_frequency: mean sequence length in number of tokens. (int)
            nswitches: number of switches occurring within a same class. (int)
            window_size: number of consecutive tokens taken per batch. (int)
            batch_size: number of concurrent text streams. (int)
            cuda: send data to GPU (bool)
        """
        self.corpora = MultiCorpora(corpus_path, model_level)
        self.switch_frequency = switch_frequency
        self.nswitches = nswitches
        self.window_size = window_size
        self.batch_size = batch_size
        self.cuda = cuda
        self._reset_iterator()

    def _reset_iterator(self):
        self.sequence_id = -1
        self.sequence_it = get_random_alternating_iterator(
                self.corpora, self.switch_frequency, self.nswitches,
                self.window_size, self.batch_size, self.cuda)
        self._next_sequence()

    def get_vocabulary(self):
        return self.corpora.vocabulary

    def __iter__(self):
        return self

    def __len__(self):
        return (len(self.sequence_it))

    def get_current_sequence_id(self):
        return self.sequence_id

    def get_current_sequence_class_id(self):
        return self.sequence_it.get_current_index()

    def get_current_sequence_class_name(self):
        return self.sequence_it.get_current_iterator().get_name()

    def __next__(self):
        try:
            return next(self.chunks_it)
        except StopIteration:
            self._next_sequence()
            return next(self.chunks_it)

    def _next_sequence(self):
        self.current_input_sequence, self.current_target_sequence = next(self.sequence_it)
        self.sequence_id += 1
        self.chunks_it = safe_iterate_chunks(self.current_input_sequence,
                    self.current_target_sequence, self.window_size, 
                    self.batch_size)



#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    ~~~~~~~~~~~~~~~~~~~~~~~
    :author: JoeshpCheung
    :python version: 3.6
"""
from typing import Union, Dict, Any, List, Tuple

import logging
import os
import re

import torch

logger = logging.getLogger(__name__)
MODEL_STATE_PREFIX = 'model_state_epoch_'
MODEL_BEST_STATE_NAME = 'model_state_best.th'

TRAINING_STATE_PREFIX = 'training_state_epoch_'
TRAINING_BEST_STATE_NAME = 'training_state_best.th'


class Checkpointer:
    def __init__(self, save_path: str = None,
                 num_models_to_keep: int = 5) -> None:
        self._save_path = save_path
        self._num_models_to_keep = num_models_to_keep

    def save_checkpoint(self,
                        epoch: Union[int, str],
                        model_state: Dict[str, Any],
                        training_states: Dict[str, Any],
                        is_best: bool = False):

        if self._save_path is not None:
            if self._num_models_to_keep:
                model_file_path = MODEL_STATE_PREFIX + '{}.th'.format(epoch)
                model_file_full_path = os.path.join(self._save_path,
                                                    model_file_path)
                torch.save(model_state, model_file_full_path)

                train_file_path = TRAINING_STATE_PREFIX + '{}.th'.format(epoch)
                train_file_full_path = os.path.join(self._save_path,
                                                    train_file_path)
                torch.save(training_states, train_file_full_path)

            if is_best:
                torch.save(
                    model_state,
                    os.path.join(self._save_path, MODEL_BEST_STATE_NAME))
                torch.save(
                    training_states,
                    os.path.join(self._save_path, TRAINING_BEST_STATE_NAME))

            self._remove_old_state()

    def restore_last_checkpoint(self):
        model_state, training_states = None, None
        if self._save_path is not None:
            epoches = self._find_exist_epoch()
            if epoches:
                last_epoch = max(epoches)

                model_file_path = MODEL_STATE_PREFIX + '{}.th'.format(
                    last_epoch)
                model_file_full_path = os.path.join(self._save_path,
                                                    model_file_path)

                train_file_path = TRAINING_STATE_PREFIX + '{}.th'.format(
                    last_epoch)
                train_file_full_path = os.path.join(self._save_path,
                                                    train_file_path)

                model_state, training_states = self._restore(
                    model_file_full_path, train_file_full_path)

        return model_state, training_states

    def restore_best_checkpoint(self):
        model_state, training_states = None, None
        if self._save_path is not None:
            model_file_full_path = os.path.join(self._save_path,
                                                MODEL_BEST_STATE_NAME)
            train_file_full_path = os.path.join(self._save_path,
                                                TRAINING_BEST_STATE_NAME)
            model_state, training_states = self._restore(
                model_file_full_path, train_file_full_path)
        return model_state, training_states

    def _find_exist_epoch(self):
        epoches = [
            int(re.findall('\d+', file)[0])
            for file in os.listdir(self._save_path)
            if file.startswith(MODEL_STATE_PREFIX)
        ]
        return epoches

    def _remove_old_state(self):
        epoches = self._find_exist_epoch()
        remove_epoches = list(sorted(epoches))
        for epoch in remove_epoches[:-self._num_models_to_keep]:
            os.remove(
                os.path.join(self._save_path,
                             MODEL_STATE_PREFIX + '{}.th'.format(epoch)))
            os.remove(
                os.path.join(self._save_path,
                             TRAINING_STATE_PREFIX + '{}.th'.format(epoch)))

    def _restore(self, model_file_full_path, train_file_full_path):
        logger.info('restore state from {} {}'.format(model_file_full_path,
                                                     train_file_full_path))
        if os.path.exists(train_file_full_path):
            model_state = torch.load(model_file_full_path, map_location='cpu')
        else:
            model_state = None
            logger.warning('{} is not exists'.format(model_file_full_path))

        if os.path.exists(train_file_full_path):
            training_states = torch.load(os.path.join(train_file_full_path),
                                         map_location='cpu')
        else:
            training_states = None
            logger.warning('{} is not exists'.format(train_file_full_path))
        return model_state, training_states
 

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    ~~~~~~~~~~~~~~~~~~~~~~~
    :author: JoeshpCheung
    :python version: 3.6
"""
import torch
import torch.nn as nn
import logging
import os
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from sklearn.metrics import fbeta_score

from checkpointer import Checkpointer
import time
# from ipdb import set_trace
logger = logging.getLogger('train')


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_dataset: Dataset,
                 dev_dataset: Dataset = None,
                 learning_rate=0.001,
                 optimizer: str = 'adam',
                 weight_decay=None,
                 weight=None,
                 batch_size=64,
                 num_train_epochs: int = 5,
                 print_every_step=10,
                 patience=10,
                 save_path='',
                 tensorboard_path=None,
                 num_models_to_keep=5,
                 cuda=True,
                 judby_fscore=False,
                 model_name = 'None'):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() and cuda else 'cpu')
        logger.info('device type: {}'.format(self.device))
        self.model = model.to(self.device)
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay,
        self.optimizer = self.get_optimizer(optimizer)
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

        self._batch_size = batch_size
        self._num_train_epochs = num_train_epochs
        self._print_every_step = print_every_step
        self._patience = patience
        self._save_path = save_path
        self._summary_writer = SummaryWriter(
            tensorboard_path) if tensorboard_path else None

        self._epoch = 1
        self._step = 0
        self._bad_cnt = 0
        self._best_acc = 0
        self._best_fscore=0
        self._last_loss = 0

        self.model = self.model.to(self.device)
        self.criterion = self.build_criterion(weight=weight)
        self.checkpointer = Checkpointer(save_path=self._save_path,
                                         num_models_to_keep=num_models_to_keep)
        self.judby_fscore = judby_fscore
        self.model_name = model_name

    def train(self):
        tic = time.time()
        if self._save_path:
            if os.path.exists(self._save_path):
                self.restore()
                logger.info('save_path: {}'.format(self._save_path))
            else:
                os.makedirs(self._save_path, exist_ok=True)
        else:
            logger.warning('请设置save_path用于保存模型')

        for epoch in range(self._epoch, self._num_train_epochs + 1):
            tmp_tic = time.time()
            is_stop = self._run_epoch()
            tmp_toc = time.time()
            logger.info('Epoch {}, Time cost {:.4f}'.format(self._epoch-1, (tmp_toc - tmp_tic)))
            if is_stop:
                logger.info('验证集上的指标超过{}次下降，提前停止训练'.format(self._bad_cnt))
                break
        toc = time.time()
        logger.info('Finish train(), Totally Time cost {:.4f}'.format((toc - tic)))

    def _run_epoch(self):
        self.model.train()
        train_data_loader = DataLoader(self.train_dataset,
                                       batch_size=self._batch_size,
                                       shuffle=True,
                                       drop_last=True)
        losses = list()
        true_num = 0
        total_num = 0

        for step, (inputs, labels) in enumerate(train_data_loader):
            self._step += 1
            # Forward pass
            num_sample = len(labels)
            # HAN model need
            if self.model_name == 'HAN':
                self.model._init_hidden_state(num_sample)
            if self.model_name == 'HAN':
                outputs, _, _ = self.model(inputs)
            else:
                outputs = self.model(inputs)

            # print('outputs: ', len(outputs), outputs.shape)
            # print('labels: ', len(labels), labels.shape, labels)
            labels = labels.to(self.device)
            loss = self.criterion(outputs, labels)
            # print('outputs: ', outputs)
            # print('label: ', labels)
            # print(loss)

            # Backward and optimize
            self.optimizer.zero_grad()  # Clears the gradients of all optimized
            loss.backward(retain_graph=True)
            self.optimizer.step()

            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            tmp_true = (predicted == labels).sum().item()
            tmp_num = len(labels)
            true_num += tmp_true
            total_num += tmp_num
            tmp_acc = tmp_true / tmp_num

            # logger.info('Epoch {} step {} batch_size {}, Train Loss: {:.4f}, Acc:{:.4f}'.format(
            #     self._epoch, step, tmp_num, loss.item(), tmp_acc))

            del outputs, loss
            torch.cuda.empty_cache()

        logger.info('\n{}'.format('-' * 20))
        mean_loss = sum(losses) / len(losses)
        accucary = true_num / total_num

        logger.info('Epoch {}, Train Loss: {:.4f}, Acc:{:.4f}'.format(
            self._epoch, mean_loss, accucary))

        if self._summary_writer:
            self._summary_writer.add_scalar('train_acc', accucary, self._epoch)
            self._summary_writer.add_scalar('train_loss', mean_loss,
                                            self._epoch)

        is_stop, is_best = self._eval_dev_data()
        self._epoch += 1
        self.save(is_best)

        return is_stop

    def _eval_dev_data(self):
        self.model.eval()
        dev_data_loader = DataLoader(self.dev_dataset,
                                     batch_size=self._batch_size,
                                     shuffle=False,
                                     drop_last=False)
        is_stop = False
        is_best = False
        if dev_data_loader is not None:
            true_num = 0
            total_num = 0
            y_true = []
            y_pred = []
            losses = []
            total_pred = []
            total_true = []
            with torch.no_grad():
                for step, (inputs, labels) in enumerate(dev_data_loader):
                    # inputs = [input.to(self.device) for input in inputs]
                    # labels = labels.to(self.device)

                    num_sample = len(labels)
                    if self.model_name == 'HAN':
                        self.model._init_hidden_state(num_sample)
                    if self.model_name == 'HAN':
                        outputs, _, _ = self.model(inputs)
                    else:
                        outputs = self.model(inputs)

                    labels = labels.to(self.device)
                    loss = self.criterion(outputs, labels)
                    losses.append(loss.item() * len(labels))
                    _, predicted = torch.max(outputs.data, 1)
                    true_num += (predicted == labels).sum().item()
                    total_num += len(labels)
                    y_true.extend(labels.tolist())
                    y_pred.extend(predicted.tolist())
                    
                    predicted = predicted.cpu().numpy()
                    labels = labels.cpu().numpy()
                    
                    # print('predicted: ', predicted)
                    # print('labels: ', labels)
                    
                    total_pred.extend(predicted)
                    total_true.extend(labels)

            mean_loss = sum(losses) / total_num
            accucary = true_num / total_num
            
            fscore = fbeta_score(total_true, total_pred, beta=1, average='macro')

            if self._summary_writer:
                self._summary_writer.add_scalar('dev_acc', accucary,
                                                self._epoch)
                self._summary_writer.add_scalar('dev_loss', mean_loss,
                                                self._epoch)

            if self.judby_fscore:
                if fscore > self._best_fscore:
                    self._best_fscore = fscore
                    is_best = True
                    self._bad_cnt = 0
            else:
                if accucary > self._best_acc:
                    self._best_acc = accucary
                    is_best = True
                    self._bad_cnt = 0

            if mean_loss >= self._last_loss:
                self._bad_cnt += 1
            else:
                self._bad_cnt = max(self._bad_cnt - 1, 0)

            self._last_loss = mean_loss

            if (self._bad_cnt >= self._patience) and (self._patience != -1):
                is_stop = True

            logger.info('Epoch {}, Dev   Loss: {:.4f}, Acc:{:.4f}, fscore:{:.4f}, {}'.format(self._epoch, mean_loss, accucary, fscore, 'is best' if is_best else ''))


        return is_stop, is_best

    def save(self, is_best=False):
        model_state = self.model.state_dict()
        training_states = {
            'optimizer': self.optimizer.state_dict(),
            'epoch': self._epoch,
            'step': self._step,
            'bad_cnt': self._bad_cnt,
            'best_acc': self._best_acc,
            'last_loss': self._last_loss,
        }
        self.checkpointer.save_checkpoint(epoch=self._epoch,
                                          model_state=model_state,
                                          training_states=training_states,
                                          is_best=is_best)

    def restore(self, best=False):
        if best:
            model_state, training_states = self.checkpointer.restore_best_checkpoint(
            )
        else:
            model_state, training_states = self.checkpointer.restore_last_checkpoint(
            )
        if model_state is not None:
            self.model.load_state_dict(model_state)
        if training_states is not None:
            self.optimizer.load_state_dict(training_states['optimizer'])
            self._epoch = training_states['epoch']
            self._step = training_states['step']
            self._bad_cnt = training_states['bad_cnt']
            self._best_acc = training_states['best_acc']
            self._last_loss = training_states['last_loss']

    def build_criterion(self, weight=None):
        criterion = nn.CrossEntropyLoss(weight=weight)
        return criterion

    def get_optimizer(self, optimizer_type):
        from torch import optim
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=self._learning_rate)
        else:
            return optim.SGD(self.model.parameters(), lr=self._learning_rate)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : ADER
# @File         : main.py
# @Description  : main file for training ADER

import argparse
import os
import math
import tensorflow as tf
from ADER import Ader
from EWC import Ewc
from tqdm import tqdm
from util import *
import gc
import time
import metric_evaluation


def str2bool(v: str) -> bool:
    """ Convert string to boolean.
        Args:
            v (str): String.
        Returns:
            (bool): True or False.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_periods(dataset: str) -> list:
    """ Get list of periods for continue learning.
        Args:
            dataset (str): Name of dataset, in ['DIGINETICA', 'YOOCHOOSE'].
        Returns:
            periods (list): list of period in the form of [1, 2, ..., period_num].
    """
    # For continue learning: periods = [1, 2, ..., period_num]
    datafiles = os.listdir(os.path.join('..', '..', '..', 'data', dataset))
    period_num = int(len(list(filter(lambda file: file.endswith(".txt"), datafiles))))
    periods = range(1, period_num)
    #  create dictionary to save model
    for period in periods:
        if not os.path.isdir(os.path.join('model', 'period%d' % period)):
            os.makedirs(os.path.join('model', 'period%d' % period))
    return periods


def load_exemplars(exemplar_pre: dict) -> list:
    """ Load exemplar in previous cycle.
        Args:
            exemplar_pre (dict): Exemplars from previous cycle in the form of {item_id: [session, session,...], ...}
        Returns:
            exemplars (list): Exemplars list in the form of [session, session]
    """
    exemplars = []
    for item in exemplar_pre.values():
        if isinstance(item, list):
            exemplars.extend([i for i in item if i])
    return exemplars


if __name__ == '__main__':

    gc.enable()
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='DIGINETICA', type=str)  # name of dataset in ['DIGINETICA', 'YOOCHOOSE']
    parser.add_argument('--save_dir', default='ADER', type=str)  # name of dictionary save the results
    # exemplar
    parser.add_argument('--exemplar_size', default=30000, type=int)  # size of exemplars
    parser.add_argument('--lambda_', default=0.8, type=float)  # base adaptive weight
    # baseline
    parser.add_argument('--finetune', default=False, type=bool)  # use fine tuned hyper-parameter without dropout
    parser.add_argument('--dropout', default=False, type=bool)  # use dropout
    parser.add_argument('--ewc', default=False, type=bool)  # use ewc
    parser.add_argument('--joint', default=False, type=bool)  # use joint learning
    parser.add_argument('--ewc_sample_num', default=1000, type=int)  # number of exemplars to generate fisher info
    # ablation study
    parser.add_argument('--selection', default='herding', type=str)  # in ['herding', 'loss', 'random']
    parser.add_argument('--disable_distillation', default=False, type=bool)  # if true, disable knowledge distillation
    parser.add_argument('--equal_exemplar', default=False, type=bool)  # if true, save equal number of exemplars per item
    parser.add_argument('--fix_lambda', default=False, type=bool) # if true, fix the adaptive weight
    # batch size and device setup
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--test_batch', default=256, type=int)
    parser.add_argument('--device_num', default=0, type=int)
    # hyper-parameters grid search
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--stop', default=5, type=int)  # number of epoch for early stop
    # hyper-parameter fixed
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--hidden_units', default=150, type=int)
    parser.add_argument('--maxlen', default=51, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    args = parser.parse_args()

    # T1 has sequences of 50 items, while T2, T3 and T4 have sequences of 51 items (50 + target)
    if args.dataset == 'T1':
        args.maxlen == 50

    # Set path
    task_path = os.path.join('results', args.dataset + '-' + args.save_dir)
    if not os.path.isdir(task_path):
        os.makedirs(task_path)
    for iteration in range(1, 6):
        path = os.path.join(task_path, 'test_' + str(iteration))
        if not os.path.isdir(path):
            os.makedirs(path)
            metrics_path = os.path.join(path, 'metrics')
            os.makedirs(metrics_path)
            break
    os.chdir(path)

    # Record logs
    logs = open('Training_logs.txt', mode='w')
    logs.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # For reproducibility
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_num)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    # Set configurations
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    # Build model
    if args.dataset == 'DIGINETICA':
        item_num = 43136    # number of items in DIGINETICA
    elif args.dataset == 'YOOCHOOSE':
        item_num = 25958    # number of items in YOOCHOOSE
    elif (args.dataset == 'T1') or (args.dataset == 'T2') or (args.dataset == 'T3') or (args.dataset == 'T4'):
        item_num = 645974  # number of items in T1, T2, T3, T4
    else:
        raise ValueError('Invalid dataset name')

    # Disable dropout for EWC and fine-tune baseline
    args.dropout_rate = 0 if (args.ewc or args.finetune) else args.dropout_rate
    # Initialize model
    with tf.device('/gpu:%d' % args.device_num):
        model = Ader(item_num, args) if not args.ewc else Ewc(item_num, args)

    # Loop each period for continue learning
    periods = get_periods(args.dataset)
    print('Continue Learning: number of periods is %d.' % len(periods))
    logs.write('Continue Learning: number of periods is %d.\n' % len(periods))
    dataloader = DataLoader(args.dataset)
    best_epoch, item_num_prev = 0, 0
    t_start = time.time()

    step_MRR_5 = []
    step_RECALL_5 = []
    step_NDCG_5 = []
    step_ACCURACY = []
    epoch_MRR_5 = []
    epoch_RECALL_5 = []
    epoch_NDCG_5 = []
    epoch_ACCURACY = []
    test_MRR_5 = []
    test_RECALL_5 = []
    test_NDCG_5 = []
    test_ACCURACY = []
    for period in periods:

        print('Period %d:' % period)
        logs.write('Period %d:\n' % period)
        best_performance, performance = 0, 0

        # Prepare data
        # load train data
        train_sess, info = dataloader.train_loader(period - 1)
        logs.write(info + '\n')
        if args.joint and period > 1:
            for p in range(1, period):
                pre_train_sess, info = dataloader.train_loader(p-1)
                logs.write(info + '\n')
                train_sess.extend(pre_train_sess)
        train_sampler = Sampler(train_sess, args.maxlen, args.batch_size)
        valid_subseq, train_subseq = train_sampler.split_data(valid_portion=0.1, return_train=True)
        batch_num = train_sampler.batch_num()

        # load test data
        test_sess, info = dataloader.evaluate_loader(len(periods))
        logs.write(info + '\n')
        max_item = dataloader.max_item()
        # exemplar
        if period > 1 and not(args.finetune or args.dropout or args.joint):
            exemplar_data_logits = load_exemplars(fast_exemplar)
            exemplar_size = len(exemplar_data_logits)
            exemplar_subseq = np.array(exemplar_data_logits)[:, 0].tolist()
            # prepare exemplar sampler
            batch_num = train_sampler.batch_num()
            exemplar_batch = int(exemplar_size / batch_num)
            exemplar_sampler = Sampler([], args.maxlen, exemplar_batch)
            exemplar_sampler.add_exemplar(exemplar_data_logits)
        else:
            exemplar_subseq = []

        # Set loss
        if period > 1 and not (args.finetune or args.dropout or args.joint):
            # find lambda for current cycle
            if args.ewc or args.fix_lambda:
                lambda_ = args.lambda_
            else:
                train_size = train_sampler.data_size()
                lambda_ = args.lambda_ * math.sqrt((item_num_prev / max_item) * (exemplar_size / train_size))
            model.update_loss(lambda_=lambda_)
        else:
            model.set_vanilla_loss()

        # Start of the main training
        with tf.Session(config=config) as sess:

            # Initialize variables or reload from previous period
            saver = tf.train.Saver(max_to_keep=1)
            if period > 1 and not args.joint:
                saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period - 1, best_epoch))
            else:
                sess.run(tf.global_variables_initializer())

            # Train
            best_epoch = 1
            for epoch in range(1, args.num_epochs + 1):

                # train each epoch
                curr_MRR_5 = curr_RECALL_5 = curr_NDCG_5 = curr_ACCURACY = []
                for _ in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b',
                              desc='Training epoch %d/%d' % (epoch, args.num_epochs)):
                    # load train batch
                    seq, pos = train_sampler.sampler()

                    if period > 1 and not (args.finetune or args.dropout or args.joint or args.ewc):

                        # load exemplar batch
                        ex_seq, ex_pos, logits = exemplar_sampler.exemplar_sampler()
                        seq = seq + ex_seq

                        if args.disable_distillation:
                            # exemplar using one-hot label
                            sess.run(model.train_op, {model.input_seq: seq,
                                                      model.pos: pos,
                                                      model.is_training: True,
                                                      model.max_item: max_item,
                                                      model.exemplar_pos: ex_pos,
                                                      model.dropout_rate: args.dropout_rate,
                                                      model.lr: args.lr})
                        else:
                            # exemplar using logistic-matching label, knowledge distillation
                            sess.run(model.train_op, {model.input_seq: seq,
                                                      model.pos: pos,
                                                      model.is_training: True,
                                                      model.max_item: max_item,
                                                      model.exemplar_logits: logits,
                                                      model.dropout_rate: args.dropout_rate,
                                                      model.lr: args.lr})
                    else:
                        # without using exemplar, for initial cycle and baselines
                        sess.run(model.train_op, {model.input_seq: seq,
                                                  model.pos: pos,
                                                  model.is_training: True,
                                                  model.max_item: max_item,
                                                  model.dropout_rate: args.dropout_rate,
                                                  model.lr: args.lr})

                    # Step evaluation
                    step_evaluator = Evaluator(valid_subseq, True, args.maxlen, args.test_batch,
                                               max_item, 'valid', model, sess)
                    info = step_evaluator.evaluate(epoch, is_print=False)
                    step_MRR_5.append(step_evaluator.results()[0])
                    step_RECALL_5.append(step_evaluator.results()[1])
                    step_NDCG_5.append(step_evaluator.results()[2])
                    step_ACCURACY.append(step_evaluator.results()[3])

                if period > 1 and args.ewc:
                    # if use ewc, update saved variables and fisher for each epoch
                    model.variables_prev = sess.run(model.variables)
                    random_exemplar = random.sample(exemplar_subseq, min(len(exemplar_subseq), args.ewc_sample_num))
                    model.compute_fisher(sess, random_exemplar, 50, max_item)

                # Epoch evaluation
                epoch_evaluator = Evaluator(valid_subseq, True, args.maxlen, args.test_batch,
                                            max_item, 'valid', model, sess)
                info = epoch_evaluator.evaluate(epoch)
                logs.write(info + '\n')
                epoch_MRR_5.append(epoch_evaluator.results()[0])
                epoch_RECALL_5.append(epoch_evaluator.results()[1])
                epoch_NDCG_5.append(epoch_evaluator.results()[2])
                epoch_ACCURACY.append(epoch_evaluator.results()[3])

                performance = epoch_evaluator.results()[0]

                # early stop
                if best_performance > performance:
                    stop_counter += 1
                    if stop_counter >= args.stop:
                        break
                else:
                    stop_counter = 0
                    best_epoch = epoch
                    best_performance = performance
                    saver.save(sess, 'model/period%d/epoch=%d.ckpt' % (period, epoch))

            # Select exemplars
            if not (args.dropout or args.finetune or args.joint):
                exemplar_candidate = train_subseq
                exemplar_candidate.extend(valid_subseq)
                exemplar_candidate.extend(exemplar_subseq)
                exemplar = ExemplarGenerator(exemplar_candidate,
                                             args.exemplar_size, args.equal_exemplar, args.batch_size, args.maxlen,
                                             args.dropout_rate, max_item)
                if args.selection == 'herding':
                    saved_num = exemplar.herding_selection(sess, model)
                elif args.selection == 'loss':
                    saved_num = exemplar.loss_selection(sess, model)
                elif args.selection == 'random':
                    saved_num = exemplar.randomly_selection(sess, model)
                else:
                    print("Invalid exemplar selection method")
                info = 'Total saved exemplar: %d' % saved_num
                print(info)
                logs.write(info + '\n')
                fast_exemplar = exemplar.exemplars
                del exemplar

            # Save current item number for next cycle
            item_num_prev = max_item

            # If use ewc method, calculate fisher and save variable for the next sample
            if args.ewc:
                exemplar_subseq = np.array(load_exemplars(fast_exemplar))[:, 0].tolist()
                model.variables_prev = sess.run(model.variables)
                random_exemplar = random.sample(exemplar_subseq, min(len(exemplar_subseq), args.ewc_sample_num))
                model.compute_fisher(sess, random_exemplar, 50, max_item)

            # Test performance
            saver.restore(sess, 'model/period%d/epoch=%d.ckpt' % (period, best_epoch))
            test_evaluator = Evaluator(test_sess, True, args.maxlen, args.test_batch,
                                       max_item, 'test', model, sess)
            info = test_evaluator.evaluate(best_epoch)
            logs.write(info + '\n')
            test_MRR_5.append(test_evaluator.results()[0])
            test_RECALL_5.append(test_evaluator.results()[1])
            test_NDCG_5.append(test_evaluator.results()[2])
            test_ACCURACY.append(test_evaluator.results()[3])


    avg_MRR_5, avg_RECALL_5, avg_NDCG_5, avg_ACCURACY = np.array(test_MRR_5).mean(), \
                                                        np.array(test_RECALL_5).mean(), \
                                                        np.array(test_NDCG_5).mean(), \
                                                        np.array(test_ACCURACY).mean()
    info = 'Average Test Metrics: (MRR@5: %.4f, RECALL@5: %.4f, NDCG@5: %.4f, ACCURACY: %.4f)' % (avg_MRR_5,
                                                                                                  avg_RECALL_5,
                                                                                                  avg_NDCG_5,
                                                                                                  avg_ACCURACY)
    print(info)
    logs.write(info + '\n')

    print(step_MRR_5)
    print(step_RECALL_5)
    # Metrics Evaluation per STEP
    metric_evaluation.Metric_Evaluation(
        {'metric': 'MRR5', 'metric_values': step_MRR_5,
         'task': args.dataset, 'mode': 'STEP', 'set': 'validation'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'RECALL5', 'metric_values': step_RECALL_5,
         'task': args.dataset, 'mode': 'STEP', 'set': 'validation'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'NDCG5', 'metric_values': step_NDCG_5,
         'task': args.dataset, 'mode': 'STEP', 'set': 'validation'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'ACCURACY', 'metric_values': step_ACCURACY,
         'task': args.dataset, 'mode': 'STEP', 'set': 'validation'})

    # Metrics Evaluation per EPOCH
    metric_evaluation.Metric_Evaluation(
        {'metric': 'MRR5', 'metric_values': epoch_MRR_5,
         'task': args.dataset, 'mode': 'EPOCH', 'set': 'validation'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'RECALL5', 'metric_values': epoch_RECALL_5,
         'task': args.dataset, 'mode': 'EPOCH', 'set': 'validation'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'NDCG5', 'metric_values': epoch_NDCG_5,
         'task': args.dataset, 'mode': 'EPOCH', 'set': 'validation'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'ACCURACY', 'metric_values': epoch_ACCURACY,
         'task': args.dataset, 'mode': 'EPOCH', 'set': 'validation'})

    # Metrics Evaluation per PERIOD
    metric_evaluation.Metric_Evaluation(
        {'metric': 'MRR5', 'metric_values': test_MRR_5,
         'task': args.dataset, 'mode': 'PERIOD', 'set': 'test'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'RECALL5', 'metric_values': test_RECALL_5,
         'task': args.dataset, 'mode': 'PERIOD', 'set': 'test'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'NDCG5', 'metric_values': test_NDCG_5,
         'task': args.dataset, 'mode': 'PERIOD', 'set': 'test'})
    metric_evaluation.Metric_Evaluation(
        {'metric': 'ACCURACY', 'metric_values': test_ACCURACY,
         'task': args.dataset, 'mode': 'PERIOD', 'set': 'test'})


    print('Total time: %.2f minutes.' % ((time.time() - t_start) / 60.0))
    logs.write('Total time: %.2f minutes\nDone.' % ((time.time() - t_start) / 60.0))
    logs.close()
    print('Done.')

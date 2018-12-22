# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Trains a model using one or more GPUs.
"""
from multiprocessing import Process

import caffe
import numpy as np
from numpy import zeros,arange
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.pyplot import twinx
from math import ceil
import cv2


def train(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        gpus,  # list of device ids
        timing=False,  # show timing info for compute and communications
):
    # NCCL uses a uid to identify a session
    uid = caffe.NCCL.new_uid()

    caffe.init_log()
    caffe.log('Using devices %s' % str(gpus))

    procs = []
    for rank in range(len(gpus)):
        p = Process(target=solve_step,
                    args=(solver, snapshot, gpus, timing, uid, rank))
        p.daemon = True
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


def time(solver, nccl):
    fprop = []
    bprop = []
    total = caffe.Timer()
    allrd = caffe.Timer()
    for _ in range(len(solver.net.layers)):
        fprop.append(caffe.Timer())
        bprop.append(caffe.Timer())
    display = solver.param.display

    def show_time():
        if solver.iter % display == 0:
            s = '\n'
            for i in range(len(solver.net.layers)):
                s += 'forw %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % fprop[i].ms
            for i in range(len(solver.net.layers) - 1, -1, -1):
                s += 'back %3d %8s ' % (i, solver.net._layer_names[i])
                s += ': %.2f\n' % bprop[i].ms
            s += 'solver total: %.2f\n' % total.ms
            s += 'allreduce: %.2f\n' % allrd.ms
            caffe.log(s)

    solver.net.before_forward(lambda layer: fprop[layer].start())
    solver.net.after_forward(lambda layer: fprop[layer].stop())
    solver.net.before_backward(lambda layer: bprop[layer].start())
    solver.net.after_backward(lambda layer: bprop[layer].stop())
    solver.add_callback(lambda: total.start(), lambda: (total.stop(), allrd.start()))
    solver.add_callback(nccl)
    solver.add_callback(lambda: '', lambda: (allrd.stop(), show_time()))


def solve(proto, snapshot, gpus, timing, uid, rank):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        solver.restore(snapshot)

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)
    solver.step(solver.param.max_iter)

def solve_step(proto, snapshot, gpus, timing, uid, rank):
    caffe.set_mode_gpu()
    caffe.set_device(gpus[rank])
    caffe.set_solver_count(len(gpus))
    caffe.set_solver_rank(rank)
    caffe.set_multiprocess(True)

    solver = caffe.SGDSolver(proto)
    if snapshot and len(snapshot) != 0:
        solver.restore(snapshot)

    nccl = caffe.NCCL(solver, uid)
    nccl.bcast()

    if timing and rank == 0:
        time(solver, nccl)
    else:
        solver.add_callback(nccl)

    if solver.param.layer_wise_reduce:
        solver.net.after_backward(nccl)


    #solver = caffe.SGDSolver('/home/zhujiagang/temporal-segment-networks/models/ucf101/gating_three_solver.prototxt')
    #solver.restore('/home/zhujiagang/temporal-segment-networks/models/ucf101_split_1_gating_three_iter_200.solverstate')
    # 等价于solver文件中的max_iter，即最大解算次数
    niter = solver.param.max_iter
    display = solver.param.display
    test_iter = 950
    test_interval = 200
    # 初始化
    train_loss = zeros(int(ceil(niter // display)))
    test_loss = zeros(int(ceil(niter // test_interval)))
    test_acc = zeros(int(ceil(niter // test_interval)))
    # 辅助变量
    _train_loss = 0;
    _test_loss = 0;
    _accuracy = 0;
    _max_accuracy = 0;
    _max_accuracy_iter = 0;
    # 进行解算
    for it in range(niter):
        solver.step(1)
        _train_loss += solver.net.blobs['rgb_flow_gating_loss'].data
        if it % display == 0:
            train_loss[it // display] = _train_loss / display
            _train_loss = 0

        if it % test_interval == 0:
            print '\n my test, train iteration', it
            for test_it in range(test_iter):
                #print '\n my test, test iteration \n', test_it
                solver.test_nets[0].forward()
                _test_loss += solver.test_nets[0].blobs['rgb_flow_gating_loss'].data
                _accuracy += solver.test_nets[0].blobs['rgb_flow_gating_accuracy'].data
            test_loss[it / test_interval] = _test_loss / test_iter
            test_acc[it / test_interval] = _accuracy / test_iter
            if _max_accuracy < test_acc[it / test_interval]:
                _max_accuracy = test_acc[it / test_interval]
                _max_accuracy_iter = it
                solver.net.save('/home/zhujiagang/temporal-segment-networks/models/ucf101_split_1_gating_three_iter_' + str(it) + '.caffemodel')
                print '\nnewly max: _max_accuracy and _max_accuracy_iter', _max_accuracy, _max_accuracy_iter
            print '\n_max_accuracy and _max_accuracy_iter', _max_accuracy, _max_accuracy_iter
            _test_loss = 0
            _accuracy = 0

    print '\nplot the train loss and test accuracy\n'
    print '\n_max_accuracy and _max_accuracy_iter\n', _max_accuracy, _max_accuracy_iter

    _, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # train loss -> 绿色
    ax1.plot(display * arange(len(train_loss)), train_loss, 'g')
    # test loss -> 黄色
    ax1.plot(test_interval * arange(len(test_loss)), test_loss, 'y')
    # test accuracy -> 红色
    ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy')
    plt.show()

def train_this(
        solver,  # solver proto definition
        snapshot,  # solver snapshot to restore
        gpus,  # list of device ids
        timing=False,  # show timing info for compute and communications
):

    train(solver, snapshot, gpus, timing)

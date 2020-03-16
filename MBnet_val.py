# -*- coding: utf-8 -*-

"""Implementation of Mobilenet V3.
Architecture: https://arxiv.org/pdf/1905.02244.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataloader
from utils import normalize, train_in_batch, shuffle_data, augument_data
import tensorflow as tf
import copy
import numpy as np
import random

__all__ = ['mobilenet_v3_large', 'mobilenet_v3_small']

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _batch_normalization_layer(inputs, momentum=0.997, epsilon=1e-3, is_training=True, name='bn', reuse=None):
    return tf.layers.batch_normalization(inputs=inputs,
                                         momentum=momentum,
                                         epsilon=epsilon,
                                         scale=True,
                                         center=True,
                                         training=is_training,
                                         name=name,
                                         reuse=reuse)


def _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=False, strides=1, reuse=None, padding="SAME"):
    conv = tf.layers.conv2d(
        inputs=inputs, filters=filters_num,
        kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
        padding=padding, #('SAME' if strides == 1 else 'VALID'),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name,
        reuse=reuse)
    return conv


def _conv_1x1_bn(inputs, filters_num, name, use_bias=True, is_training=True, reuse=None):
    kernel_size = 1
    strides = 1
    x = _conv2d_layer(inputs, filters_num, kernel_size, name=name + "/conv", use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    return x

def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)

def _conv_bn_relu(inputs, filters_num, kernel_size, name, use_bias=True, strides=1, is_training=True, activation=relu6, reuse=None):
    x = _conv2d_layer(inputs, filters_num, kernel_size, name, use_bias=use_bias, strides=strides)
    x = _batch_normalization_layer(x, momentum=0.997, epsilon=1e-3, is_training=is_training, name=name + '/bn',
                                   reuse=reuse)
    x = activation(x)
    return x


def _dwise_conv(inputs, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1),
                padding='SAME', name='dwise_conv', use_bias=False,
                reuse=None):
    kernel_size = (k_w, k_h)
    in_channel = inputs.get_shape().as_list()[-1]
    filters = int(in_channel*depth_multiplier)
    return tf.layers.separable_conv2d(inputs, filters, kernel_size,
                                      strides=strides, padding=padding,
                                      data_format='channels_last', dilation_rate=(1, 1),
                                      depth_multiplier=depth_multiplier, activation=None,
                                      use_bias=use_bias, name=name, reuse=reuse
                                      )




def hard_swish(x, name='hard_swish'):
    with tf.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def _fully_connected_layer(inputs, units, name="fc", activation=None, use_bias=True, reuse=None):
    return tf.layers.dense(inputs, units, activation=activation, use_bias=use_bias,
                           name=name, reuse=reuse)


def _global_avg(inputs, pool_size, strides, padding='valid', name='global_avg'):
    return tf.layers.average_pooling2d(inputs, pool_size, strides,
                                       padding=padding, data_format='channels_last', name=name)


def _squeeze_excitation_layer(input, out_dim, ratio, layer_name, is_training=True, reuse=None):
    with tf.variable_scope(layer_name, reuse=reuse):
        squeeze = _global_avg(input, pool_size=input.get_shape()[1:-1], strides=1)

        excitation = _fully_connected_layer(squeeze, units=out_dim / ratio, name=layer_name + '_excitation1',
                                            reuse=reuse)
        excitation = relu6(excitation)
        excitation = _fully_connected_layer(excitation, units=out_dim, name=layer_name + '_excitation2', reuse=reuse)
        excitation = hard_sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
        scale = input * excitation
        return scale


def mobilenet_v3_block(input, k_s, expansion_ratio, output_dim, stride, name, is_training=True,
                       use_bias=True, shortcut=True, activatation="RE", ratio=16, se=False,
                       reuse=None):
    bottleneck_dim = expansion_ratio  

    with tf.variable_scope(name, reuse=reuse):
        # pw mobilenetV2
        net = _conv_1x1_bn(input, bottleneck_dim, name="pw", use_bias=use_bias)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # dw
        net = _dwise_conv(net, k_w=k_s, k_h=k_s, strides=[stride, stride], name='dw',
                          use_bias=use_bias, reuse=reuse)

        net = _batch_normalization_layer(net, momentum=0.997, epsilon=1e-3,
                                         is_training=is_training, name='dw_bn', reuse=reuse)

        if activatation == "HS":
            net = hard_swish(net)
        elif activatation == "RE":
            net = relu6(net)
        else:
            raise NotImplementedError

        # squeeze and excitation
        if se:
            channel = net.get_shape().as_list()[-1]
            net = _squeeze_excitation_layer(net, out_dim=channel, ratio=ratio, layer_name='se_block')

        # pw & linear
        net = _conv_1x1_bn(net, output_dim, name="pw_linear", use_bias=use_bias)

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            net += input
            net = tf.identity(net, name='block_output')

    return net


def mobilenet_v3_small(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
    end_points = {}
    layers = [
        [16, 16, 3, 2, "RE", True, 16],
        [16, 24, 3, 2, "RE", False, 72],
        [24, 24, 3, 1, "RE", False, 88],
        [24, 40, 5, 2, "RE", True, 96],
        [40, 40, 5, 1, "RE", True, 240],
        [40, 40, 5, 1, "RE", True, 240],
        [40, 48, 5, 1, "HS", True, 120],
        [48, 48, 5, 1, "HS", True, 144],
        [48, 96, 5, 2, "HS", True, 288],
        [96, 96, 5, 1, "HS", True, 576],
        [96, 96, 5, 1, "HS", True, 576],
    ]

    print("inputs",inputs)
    input_size = inputs.get_shape().as_list()[1:-1]
    print("input_size",input_size)

    #assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

    reduction_ratio = 4
    with tf.variable_scope('init', reuse=reuse):
        init_conv_out = _make_divisible(16 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
                          use_bias=False, strides=2, is_training=is_training, activation=hard_swish)

    with tf.variable_scope("MobilenetV3_small", reuse=reuse):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = _make_divisible(in_channels * multiplier)
            out_channels = _make_divisible(out_channels * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training, use_bias=True,
                                   shortcut=(in_channels==out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x

        conv1_in = _make_divisible(96 * multiplier)
        conv1_out = _make_divisible(576 * multiplier)
        x = _conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
                          use_bias=True, strides=1, is_training=is_training, activation=hard_swish)

        x = _squeeze_excitation_layer(x, out_dim=conv1_out, ratio=reduction_ratio, layer_name="conv1_out",
                                     is_training=is_training, reuse=None)
        end_points["conv1_out_1x1"] = x

        x = _global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
        #x = hard_swish(x)
        end_points["global_pool"] = x

    with tf.variable_scope('Logits_out', reuse=reuse):
        conv2_in = _make_divisible(576 * multiplier)
        conv2_out = _make_divisible(1280 * multiplier)
        x = _conv2d_layer(x, filters_num=conv2_out, kernel_size=1, name="conv2", use_bias=True, strides=1)
        x = hard_swish(x)
        end_points["conv2_out_1x1"] = x
        
        # triplet
        triplet_out = _make_divisible(128 * multiplier)
        triplet_y = _conv2d_layer(x, filters_num=128, kernel_size=1, name="triplet_out", use_bias=True, strides=1)
        triplet = tf.layers.flatten(triplet_y)
        triplet = tf.identity(triplet, name='Logits_out')
        end_points["triplet_out"] = triplet
        
        # softmax
        x = _conv2d_layer(x, filters_num=103, kernel_size=1, name="softmax_out", use_bias=True, strides=1)
        softmax = tf.layers.flatten(x)
        softmax = tf.identity(softmax, name='Logits_out')
        end_points["softmax_out"] = softmax

    return softmax, triplet, end_points


def mobilenet_v3_large(inputs, classes_num, multiplier=1.0, is_training=True, reuse=None):
    end_points = {}
    layers = [
        [16, 16, 3, 1, "RE", False, 16],
        [16, 24, 3, 2, "RE", False, 64],
        [24, 24, 3, 1, "RE", False, 72],
        [24, 40, 5, 2, "RE", True, 72],
        [40, 40, 5, 1, "RE", True, 120],

        [40, 40, 5, 1, "RE", True, 120],
        [40, 80, 3, 2, "HS", False, 240],
        [80, 80, 3, 1, "HS", False, 200],
        [80, 80, 3, 1, "HS", False, 184],
        [80, 80, 3, 1, "HS", False, 184],

        [80, 112, 3, 1, "HS", True, 480],
        [112, 112, 3, 1, "HS", True, 672],
        [112, 160, 5, 1, "HS", True, 672],
        [160, 160, 5, 2, "HS", True, 672],
        [160, 160, 5, 1, "HS", True, 960],
    ]

    input_size = inputs.get_shape().as_list()[1:-1]
    assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

    reduction_ratio = 4
    with tf.variable_scope('init', reuse=reuse):
        init_conv_out = _make_divisible(16 * multiplier)
        x = _conv_bn_relu(inputs, filters_num=init_conv_out, kernel_size=3, name='init',
                          use_bias=False, strides=2, is_training=is_training, activation=hard_swish)

    with tf.variable_scope("MobilenetV3_large", reuse=reuse):
        for idx, (in_channels, out_channels, kernel_size, stride, activatation, se, exp_size) in enumerate(layers):
            in_channels = _make_divisible(in_channels * multiplier)
            out_channels = _make_divisible(out_channels * multiplier)
            exp_size = _make_divisible(exp_size * multiplier)
            x = mobilenet_v3_block(x, kernel_size, exp_size, out_channels, stride,
                                   "bneck{}".format(idx), is_training=is_training, use_bias=True,
                                   shortcut=(in_channels==out_channels), activatation=activatation,
                                   ratio=reduction_ratio, se=se)
            end_points["bneck{}".format(idx)] = x

        conv1_in = _make_divisible(160 * multiplier)
        conv1_out = _make_divisible(960 * multiplier)
        x = _conv_bn_relu(x, filters_num=conv1_out, kernel_size=1, name="conv1_out",
                          use_bias=True, strides=1, is_training=is_training, activation=hard_swish)
        end_points["conv1_out_1x1"] = x

        x = _global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
        #x = hard_swish(x)
        end_points["global_pool"] = x

    with tf.variable_scope('Logits_out', reuse=reuse):
        conv2_in = _make_divisible(960 * multiplier)
        conv2_out = _make_divisible(1280 * multiplier)

        x = _conv2d_layer(x, filters_num=conv2_out, kernel_size=1, name="conv2", use_bias=True, strides=1)
        x = hard_swish(x)
        end_points["conv2_out_1x1"] = x
         
        # triplet
        triplet_out = _make_divisible(128 * multiplier)
        y = _conv2d_layer(y, filters_num=triplet_out, kernel_size=1, name="y", use_bias=True, strides=1)
        triplet = tf.layers.flatten(y)
        triplet = tf.identity(triplet, name='output')
        end_points["Logits_out"] = triplet

        # softmax
        x = _conv2d_layer(x, filters_num=classes_num, kernel_size=1, name="x", use_bias=True, strides=1)
        softmax = tf.layers.flatten(x)
        softmax = tf.identity(softmax, name='output')
        end_points["Logits_out"] = softmax

    return softmax, triplet, end_points

def semi_triplet_loss(Embedding, labels):
    labels = tf.reshape(labels,[-1])
    prediction_semi = tf.nn.l2_normalize(Embedding,axis=1)
    loss_semi = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels,prediction_semi)
    return loss_semi

def cross_entropy(prediction, labels):
    labels = tf.reshape(labels,[-1])
    sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=prediction)
    sparse = tf.reduce_mean(sparse)
    return sparse

def accuracy(prediction, labels):
	labels = tf.cast(tf.reshape(labels,[-1]),tf.int64)
	prediction = tf.argmax(prediction, 1)
	correct_prediction = tf.equal(prediction, labels)
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy
def triplet_val(ground_truth_vector, ground_truth_label, testing_vector, testing_label):
    correct = 0
    number = len(testing_vector)
    for i in range(len(testing_vector)):
        dis_index = [None]*103
        index = [i for i in range(103)]
        for j in range(len(ground_truth_vector)):
            d = np.linalg.norm(testing_vector[i] - ground_truth_vector[j])
            #dis_index.append([d,j])
            label_index=ground_truth_label[j][0]
            if(dis_index[label_index]!=None):
                if(dis_index[label_index]>d):
                    dis_index[label_index]=d
            else:
                dis_index[label_index]=d
        top_index = sorted(zip(dis_index, index))[:1]    
        for idx in top_index:
            if testing_label[i] == idx[1]:
                correct += 1
                break
    return correct / number
if __name__ == "__main__":
    ''' Original Testing '''
    print("begin ...")
    input_test = tf.zeros([8, 224, 224, 3])
    num_classes = 103
    # model, end_points = mobilenet_v3_large(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    # model, end_points = mobilenet_v3_small(input_test, num_classes, multiplier=1.0, is_training=True, reuse=None)
    print("done !")

    ''' Application '''
    data_path = 'dataset/downloads'
    tf.reset_default_graph()
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.float32, [None, 224, 224, 1])        
        y_true = tf.placeholder(tf.int32, [None, 1])
        alpha = tf.placeholder(tf.float32)
        beta = tf.placeholder(tf.float32)
    with tf.name_scope('stem'):
            
        # out_triplet,out_softmax = MobileV3(inputs)
        softmax, triplet, end_points = mobilenet_v3_small(
            inputs, # input_test
            num_classes,
            multiplier=1.0, 
            is_training=True, 
            reuse=None
        ) # model: softmax outcome
            
    #print(softmax)
    #print(y_true)
    with tf.name_scope('loss'):
        # alpha = 1 / ( (batch_size*(batch_size-1)) * (batch_size**2 - batch_size) )
        # triplet
        triplet_loss = semi_triplet_loss(triplet,y_true)
        # softmax
        softmax_loss = cross_entropy(softmax, y_true)

        loss = tf.add(tf.multiply(alpha,triplet_loss), tf.multiply(beta, softmax_loss))        
        #optim = tf.train.AdamOptimizer().minimize(loss)
        adam = tf.train.AdamOptimizer()
        optim = adam.minimize(loss)
        #trainer = tf.compat.v1.train.AdamOptimizer() 
        #optim = trainer.minimize(loss)
        # exp step size
        # global_step = tf.Variable(0, trainable=False)
        # starter_learning_rate = 0.05
        # learning_rate = tf.train.exponential_decay(starter_learning_rate,
        #                                       global_step,
        #                                       100, 0.9, staircase=True)
        # optim = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        acc=accuracy(softmax, y_true)

    epochs = 3000
    batch_size_triplet = 4
    batch_size_softmax = 64
    iteration = 10000//batch_size_softmax
    saver = tf.train.Saver()
    max_val_acc = 0
    start_triplet = 0
    max_val_triplet_acc = 0
    triplet_goal = 0.7
    softmax_goal = 0.7
    val_triplet_acc = 0
    logfile = open("log.txt", "w")

    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, "./mobile_checkpoint/320.ckpt")
        triplet_TrainData, TestData = dataloader.dataloader(data_path)
        softmax_TrainData = dataloader.triplet_to_softmax(triplet_TrainData)
        x_test,y_test,end=dataloader.GenNoBatch(TestData)
        print("Start Training")
        #######Prepare triplet validtion data####### 
        train_val_data = []
        train_val_label = []
        for i in range(len(triplet_TrainData)):
            sample_index = random.sample(range(0, len(triplet_TrainData[i])), 10) 
            for j in sample_index:
                train_val_data.append(dataloader.loaddata(triplet_TrainData[i][j][0]))
                train_val_label.append([triplet_TrainData[i][j][1]])
        for epoch in range(epochs):
            ########Training Stage#######
            if start_triplet == 0:
                if max_val_acc > softmax_goal:
                    start_triplet = 1
            if start_triplet == 1:
                if max_val_triplet_acc > triplet_goal:
                    start_triplet = 2

            #######Which Type of Data#######
            if start_triplet == 0:
                TrainData=copy.deepcopy(softmax_TrainData)
            else:
                TrainData=copy.deepcopy(triplet_TrainData)


            training_loss = 0 
            training_acc = 0
            triplet_sum = 0
            real_iteration = 0
            while(True): 
                if start_triplet == 1:
                    alpha_input = 1
                    beta_input = 0
                    x_train,y_train,end=dataloader.GenBatch(TrainData,batch_size_triplet) 
                elif start_triplet == 2:
                    alpha_input = 0.5
                    beta_input = 0.5
                    x_train,y_train,end=dataloader.GenBatch(TrainData,batch_size_triplet) 
                else:
                    alpha_input = 0
                    beta_input = 1
                    x_train,y_train=dataloader.GenRandomBatch(TrainData,batch_size_softmax)

                #######Stop Condition#######
                if start_triplet == 0:
                    if real_iteration >= iteration:
                        break
                else:
                    if end == 1:
                        break
                real_iteration += 1


                x_train = augument_data(x_train,False)
                training_loss_batch, _,training_acc_batch,pred,TL,SL = sess.run(
                    [loss,optim,acc,softmax,triplet_loss,softmax_loss],

                    feed_dict={inputs:x_train,y_true:y_train,alpha:alpha_input,beta:beta_input}
                )
                training_acc += training_acc_batch
                triplet_sum += TL
                #print(opt)
            #######Testing softmax#######
            testing_acc, test_val_vector,lr= sess.run(
                [acc,triplet,adam._lr_t],
                feed_dict={inputs:x_test,y_true:y_test,alpha:alpha_input,beta:beta_input}
            )
            ######Testing triplet########
            if start_triplet != 0:
                train_val_vector = sess.run([triplet],feed_dict={inputs:train_val_data, y_true: train_val_label})
                val_triplet_acc = triplet_val(train_val_vector[0], train_val_label, test_val_vector, y_test)
            #######Saving Condition#######
            if start_triplet != 0 and val_triplet_acc > max_val_triplet_acc :
                print("Save!!!")
                #saver.save(sess, "./mobile_checkpoint/%d.ckpt"%(epoch))

                
            if testing_acc > max_val_acc:
                max_val_acc = testing_acc
            if val_triplet_acc > max_val_triplet_acc:
                max_val_triplet_acc = val_triplet_acc

            print("epoch:%d Stage:%d train_acc:%.3f val_acc:%.3f triplet_val_acc:%.3f max_val_acc:%.3f max_val_triplet_acc:%.3f lr:%.6f"
                %(epoch, start_triplet, training_acc/real_iteration, testing_acc, 
                  val_triplet_acc, max_val_acc, max_val_triplet_acc,lr)
            )
            logfile.write("epoch:%d Stage:%d train_acc:%.3f val_acc:%.3f triplet_val_acc:%.3f max_val_acc:%.3f max_val_triplet_acc:%.3f lr:%.6f\n"
                %(epoch, start_triplet, training_acc/real_iteration, testing_acc, 
                  val_triplet_acc, max_val_acc, max_val_triplet_acc,lr)
            )

        saver.save(sess, "./mobile_checkpoint/new_mobile_val_last.ckpt")
   

# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 17:16:03 2023

@author: chenj
"""

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve,auc

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

def pares_tf(example_proto):
    dics={
        
           'RadiomicFeatures':tf.FixedLenFeature([], tf.string), 
           'DeepFeatures': tf.FixedLenFeature([], tf.string),
           'n10lDeepFeatures': tf.FixedLenFeature([], tf.string),
           'ircsnDeepFeatures':tf.FixedLenFeature([], tf.string),
           'ValidatedPhases':tf.FixedLenFeature([], tf.string),
           'Label': tf.FixedLenFeature([], tf.string)
            }
    parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)
    RadiomicsFeatures= tf.decode_raw(parsed_example['RadiomicFeatures'],out_type=tf.float32)
    RadiomicsFeatures = tf.reshape(RadiomicsFeatures, [1,102])
    DeepFeatures=tf.decode_raw(parsed_example['DeepFeatures'],out_type=tf.float32)
    DeepFeatures=tf.reshape(DeepFeatures,[1,400])
    
    n10lDeepFeatures=tf.decode_raw(parsed_example['n10lDeepFeatures'],out_type=tf.float32)
    n10lDeepFeatures=tf.reshape(n10lDeepFeatures,[1,400])
    ircsnDeepFeatures=tf.decode_raw(parsed_example['ircsnDeepFeatures'],out_type=tf.float32)
    ircsnDeepFeatures=tf.reshape(ircsnDeepFeatures,[1,400])
    
    ValidatedPhases = tf.decode_raw(parsed_example['ValidatedPhases'],out_type=tf.int32)
    ValidatedPhases = tf.reshape(ValidatedPhases, [1])
    Label = tf.decode_raw(parsed_example['Label'],out_type=tf.int32)
    Label = tf.reshape(Label, [1])
    return RadiomicsFeatures,DeepFeatures,n10lDeepFeatures,ircsnDeepFeatures,ValidatedPhases,Label

radiomics_inputs_ = tf.placeholder(tf.float32, [None,102,1])
deepfeatures_inputs_ = tf.placeholder(tf.float32,[None,400,1])

n10ldeepfeatures_inputs_ = tf.placeholder(tf.float32,[None,400,1])

ircsndeepfeatures_inputs_ = tf.placeholder(tf.float32,[None,400,1])
labels_ = tf.placeholder(tf.int32, [1])

trainset=tf.data.TFRecordDataset(filenames=['F:/DataforExtract/slowfast+i3d_5_fold_files_TOSHIBA/Trainuncombatrunning3_3.tfrecords'])
trainset=trainset.map(pares_tf)
trainset=trainset.shuffle(7000).repeat(5002).batch(1)
iterator = trainset.make_one_shot_iterator()
next_patch = iterator.get_next()

testset=tf.data.TFRecordDataset(filenames=['F:/DataforExtract/slowfast+i3d_5_fold_files_TOSHIBA/Testuncombatrunning3_3.tfrecords'])
testset=testset.map(pares_tf)
testset=testset.shuffle(7000).repeat(12).batch(1)
iterator2 = testset.make_one_shot_iterator()
next_patch2 = iterator2.get_next()


def attention(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size =inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
    
def radiomics_feature_pro(radiomics_feature_inputs_):
    
    conv1D_1=tf.layers.Conv1D(filters=1,kernel_size=5,padding='same')
    conv1D_2=tf.layers.Conv1D(filters=1,kernel_size=3,padding='valid')
    batch_N=tf.layers.BatchNormalization()
    
    max_pool_1d_1 = tf.layers.MaxPooling1D(pool_size=2,strides=2, padding='same')
    
    
    radiomics_conv1=conv1D_1(radiomics_feature_inputs_)
    radiomics_conv2=conv1D_2(radiomics_conv1)
    radiomics_BN1=batch_N(radiomics_conv2)
    radiomics_BN1=tf.nn.relu(radiomics_BN1)
    radiomics_conv3=conv1D_1(radiomics_BN1)
    radiomics_BN2=batch_N(radiomics_conv3)
    radiomics_BN2=tf.nn.relu(radiomics_BN2+radiomics_conv2)
    radiomics_pool1=max_pool_1d_1(radiomics_BN2)
    
    return radiomics_pool1

def deep_feature_pro(deep_feature_inputs_):
    conv1D_1=tf.layers.Conv1D(filters=1,kernel_size=5,padding='same')
    conv1D_2=tf.layers.Conv1D(filters=1,kernel_size=3,padding='valid')
    batch_N=tf.layers.BatchNormalization()
    
    max_pool_1d_1 = tf.layers.MaxPooling1D(pool_size=2,strides=2, padding='same')
    
    deepfeature_conv1=conv1D_1(deep_feature_inputs_)
    deepfeature_BN1=batch_N(deepfeature_conv1)
    deepfeature_BN1=tf.nn.relu(deepfeature_BN1)
    deepfeature_conv2=conv1D_1(deepfeature_BN1)
    deepfeature_BN2=batch_N(deepfeature_conv2)
    deepfeature_BN2=tf.nn.relu(deepfeature_BN2+deepfeatures_inputs_)
    deepfeature_pool1=max_pool_1d_1(deepfeature_BN2)

    deepfeature_conv3=conv1D_1(deepfeature_pool1)
    deepfeature_BN3=batch_N(deepfeature_conv3)
    deepfeature_BN3=tf.nn.relu(deepfeature_BN3)
    deepfeature_conv4=conv1D_1(deepfeature_BN3)
    deepfeature_BN4=batch_N(deepfeature_conv4)
    deepfeature_BN4=tf.nn.relu(deepfeature_BN4+deepfeature_pool1)    
    deepfeature_pool2=max_pool_1d_1(deepfeature_BN4)

    deepfeature_conv5=conv1D_1(deepfeature_pool2)
    
    deepfeature_BN5=batch_N(deepfeature_conv5)
    deepfeature_BN5=tf.nn.relu(deepfeature_BN5)
    deepfeature_conv6=conv1D_1(deepfeature_BN5)
    deepfeature_BN6=batch_N(deepfeature_conv6)
    deepfeature_BN6=tf.nn.relu(deepfeature_BN6+deepfeature_pool2)  
    deepfeature_pool3=max_pool_1d_1(deepfeature_BN6)
    
    return deepfeature_pool3

with tf.name_scope('LSTM_Classifier'):
    
    radiomics_pool1=radiomics_feature_pro(radiomics_inputs_)
    #radiomics_pool1=tf.reshape(radiomics_pool1,[1,1,50])
  
    deepfeature_pool3_o=deep_feature_pro(deepfeatures_inputs_)
    #deepfeature_pool3_o=tf.reshape(deepfeature_pool3_o,[1,1,50])#deep_feature_pro(deepfeatures_inputs_)
    deepfeature_pool3_n=deep_feature_pro(n10ldeepfeatures_inputs_)
#    deepfeature_pool3_i=deep_feature_pro(ircsndeepfeatures_inputs_)
    
    deepfeature_pool3_o=tf.reshape(deepfeature_pool3_o,[1,50])
    
    deepfeature_pool3_n=tf.reshape(deepfeature_pool3_n,[1,50])
#    deepfeature_pool3_i=tf.reshape(deepfeature_pool3_i,[1,50])
    
    pre_fusion_deep_feature=tf.concat([deepfeature_pool3_o,deepfeature_pool3_n],axis=0)
    pre_fusion_deep_feature=tf.reshape(pre_fusion_deep_feature,[1,2,50])
    deepfeature_pool3_after_fusion=attention(inputs=pre_fusion_deep_feature,attention_size=64,return_alphas=False)

    radiomics_pool1=tf.reshape(radiomics_pool1,[1,50])
    deepfeature_pool3=tf.reshape(deepfeature_pool3_after_fusion,[1,50])
    
    pre_fusion_feature=tf.concat([radiomics_pool1,deepfeature_pool3], axis=0)

    pre_fusion_feature=tf.reshape(pre_fusion_feature,[1,2,50])
    # #dropout5=0.3*radiomics_pool1+0.7*deepfeature_pool3
    # dropout5=attention(inputs=pre_fusion_feature,attention_size=128,return_alphas=False) 
#    embeding=tf.reshape(radiomics_pool1,[1,9,50])
    
    output=attention(inputs=pre_fusion_feature,attention_size=64,return_alphas=False)
    #output2=tf.reshape(output,[1,64])
    #output2=tf.reduce_mean(embeding, 1)
    
    fc5=tf.layers.dense(output,
                           units=32,
                           activation=lrelu,
                           )
    #dropout5=tf.nn.dropout(fc5,rate=0.4)
    hypothesis=tf.layers.dense(fc5,
                            units=2,
                            activation='sigmoid',
                            )
#pred_prob=tf.nn.softmax(result,axis=1)



Y=tf.one_hot(labels_, 2)

# nb_classes = 2

# W = tf.Variable(tf.random_normal([32,nb_classes]),name='weight')
# b = tf.Variable(tf.random_normal([nb_classes]),name='bias')
# #fc5_c = fc5.astype(np.float32)
# hypothesis = tf.nn.softmax(tf.nn.xw_plus_b(fc5,W,b))

#hypothesis =np.max(tf.nn.softmax(fc5_c))

loss = tf.reduce_mean(tf.reduce_sum(abs(hypothesis-Y)*abs(hypothesis-Y),axis=1))


learning_rate = tf.placeholder(tf.float32)
cost = tf.reduce_mean(loss)#tf.reduce_sum(abs(hypothesis-Y)*abs(hypothesis-Y)))
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session() as sess:

    epochs = 200
    batch_size = 1
    #lr =5e-5
    precision_final=[]
    recall_final=[]
    auc_final=[]
    for running in range(10):
        print(running)
        sess.run(tf.global_variables_initializer())
        total_batch=820

        test_batch=205
        for eps in range(epochs):
            lr=(5e-5)*((5-int(eps/100))/5)
            total_cost=0
            for ibatch in range(total_batch):
                radiomicsfeatures_train,deepfeatures_train,n10ldeepfeatures_train,ircsndeepfeatures_train,ValidatedPhases_train,label_train\
                    =sess.run(next_patch)
                
                #radiomicsfeatures_train=radiomicsfeatures_train.T
                
                radiomicsfeatures_train2=np.reshape(radiomicsfeatures_train,[1,102])
                #radiomicsfeatures_train2=radiomicsfeatures_train2.T
                radiomicsfeatures_train2=np.reshape(radiomicsfeatures_train2,[1,102,1])
                deepfeatures_train2=np.reshape(deepfeatures_train,[1,400])
                deepfeatures_train2=np.reshape(deepfeatures_train2,[1,400,1])
                
                
                n10ldeepfeatures_train2=np.reshape(n10ldeepfeatures_train,[1,400])
                n10ldeepfeatures_train2=np.reshape(n10ldeepfeatures_train2,[1,400,1])
                
                ircsndeepfeatures_train2=np.reshape(ircsndeepfeatures_train,[1,400])
                ircsndeepfeatures_train2=np.reshape(ircsndeepfeatures_train2,[1,400,1])
                
                label_train2=np.reshape(label_train,[1])

                batch_cost,_=sess.run([cost, opt], feed_dict={deepfeatures_inputs_:deepfeatures_train2,\
                                                              n10ldeepfeatures_inputs_:n10ldeepfeatures_train2,ircsndeepfeatures_inputs_:ircsndeepfeatures_train2,\
                                                                  radiomics_inputs_:radiomicsfeatures_train2,labels_: label_train2,learning_rate:lr})
                total_cost=total_cost+batch_cost
            
            
            if (eps%20)==0:    
                print("Epoch: {}/{}".format(eps+1, epochs))
                print(total_cost)
        predictresult=[]
        originalresult=[]
        possibility=[]
        for ibatch2 in range(test_batch):
            radiomicsfeatures_test,deepfeatures_test,n10ldeepfeatures_test,ircsndeepfeatures_test,ValidatedPhases_test,label_test=sess.run(next_patch2)
                    
            radiomicsfeatures_test2=np.reshape(radiomicsfeatures_test,[1,102])
            #radiomicsfeatures_test2=radiomicsfeatures_test2.T
            radiomicsfeatures_test2=np.reshape(radiomicsfeatures_test2,[1,102,1])
            deepfeatures_test2=np.reshape(deepfeatures_test,[1,400])
            #deepfeatures_test2=deepfeatures_test2.T
            deepfeatures_test2=np.reshape(deepfeatures_test2,[1,400,1])
            
            n10ldeepfeatures_test2=np.reshape(n10ldeepfeatures_test,[1,400])
            n10ldeepfeatures_test2=np.reshape(n10ldeepfeatures_test2,[1,400,1])
                
            ircsndeepfeatures_test2=np.reshape(ircsndeepfeatures_test,[1,400])
            ircsndeepfeatures_test2=np.reshape(ircsndeepfeatures_test2,[1,400,1])            
        
            label_test2=np.reshape(label_test,[1])
            hypothesis_,Y_=sess.run([hypothesis,Y], feed_dict={deepfeatures_inputs_:deepfeatures_test2,\
                                                               n10ldeepfeatures_inputs_:n10ldeepfeatures_test2,ircsndeepfeatures_inputs_:ircsndeepfeatures_test2,\
                                                                   radiomics_inputs_:radiomicsfeatures_test2,labels_:label_test2,learning_rate:lr})
            predictresult.append(np.argmax(hypothesis_))
            originalresult.append(np.argmax(Y_))
            possibility.append(hypothesis_[0,1])
        
        fpr,tpr,thresholds=roc_curve(originalresult, possibility)
        confusion_mat = confusion_matrix(originalresult, predictresult)
    
        precision= accuracy_score(originalresult, predictresult)
        recall= recall_score(originalresult, predictresult)
        auc=roc_auc_score(originalresult, possibility)
        print(auc)
        auc_final.append(auc)
        precision_final.append(precision)
        recall_final.append(recall)
        
sess.close() 



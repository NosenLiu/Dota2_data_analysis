#coding:utf-8

"""根据对局数据型训练1个预测模型
   可以对胜率进行预测
   LSTM 两个时间节点，第一个时间节点为天辉英雄的稀疏向量，第二个时间节点为夜魇英雄的稀疏向量
   英雄共117个，最后一个英雄为马尔斯(编号129)，"""

# TODO ====参数区==== TODO
predict_step = 1
team_num = 2   # 多少个月的数据作为训练集
cnn_output_dim = 64 # CNN层输出特征维度
kernel_size = 13
pool_size = 2       # 池化倍数 TODO 注意！！ 需要保证 cnn_output_dim ÷ pool_size 的结果是一个整数
hidden_size = 256  # 神经元数目
epochs = 2000      # 训练轮次
batch_size = 1000  # 批处理大小
model_saved_path = "../model/keras/"   # 模型保存路径
# model_name = '50000_samples_20191029_LSTM'
# model_name = '100000_samples_20191030_CNN+LSTM_kernel_13_batch_1000'
model_name = '100000_samples_20191031_LSTM'
hero_id_max = 129   # 英雄id的最大值
# TODO ====参数区==== TODO

import os
# os.environ['THEANO_FLAGS'] = "device=gpu"  # 改变环境变量添加THEANO_FLAGS,在导入theano的时候可以设置使用gpu
import numpy as np
import math
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Embedding, Masking, Dropout, Conv1D, MaxPooling1D, Reshape
from keras.models import load_model

import time
import datetime
import sys
#from keras.utils.visualize_plots import figures   # TODO：一个手动添加的模块，用来绘制误差值随着迭代次数的曲线，并保存图像
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import matplotlib.dates as mdates       # 让坐标显示月份
import matplotlib as mpl 
mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签  
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号  

# ===================TODO 读取对局数据   TODO========================
with open('../data/matches_list_ranking_all.csv','r',encoding='utf-8') as fo_1:
    line_matches = fo_1.readlines()
    sample_in = []
    sample_out = []
    for i in range(len(line_matches))[1:]:
        split = line_matches[i].split(', ')
        radiant = split[2]
        dire = split[3]
        # print(split[4][:-1])
        if split[4][:-1]=='True':
            win = 1.0
        else:
            win = 0.0
        radiant = list(map(int,radiant.split(',')))
        dire = list(map(int,dire.split(',')))
        radiant_vector = np.zeros(hero_id_max)
        dire_vector = np.zeros(hero_id_max)
        for item in radiant:
            radiant_vector[item-1] = 1
        for item in dire:
            dire_vector[item-1] = 1
        sample_in.append([radiant_vector,dire_vector])
        sample_out.append(win)
# print(sample_in)
# print(sample_out)


def make_samples():
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    validate_x = []
    validate_y = []
    for i in range(len(sample_in)):
        if i%10==8:
            test_x.append(sample_in[i])
            test_y.append(sample_out[i])
        elif i%10==9:
            validate_x.append(sample_in[i])
            validate_y.append(sample_out[i])
        else:
            train_x.append(sample_in[i])
            train_y.append(sample_out[i])
    return train_x,train_y,test_x,test_y,validate_x,validate_y


def training():
    train_x,train_y,test_x,test_y,validate_x,validate_y = make_samples()
    tx = np.array(train_x).reshape(len(train_x),team_num,hero_id_max)
    ty = np.array(train_y).reshape(len(train_y),1)
    test_x = np.array(test_x).reshape(len(test_x),team_num,hero_id_max)
    test_y = np.array(test_y).reshape(len(test_y),1)
    validate_x = np.array(validate_x).reshape(len(validate_x),team_num,hero_id_max)
    validate_y = np.array(validate_y).reshape(len(validate_y),1)
     #TODO ==============样本制作 结束==================
    print('=========== tx.shape:',tx.shape,' ===============')
    print('=========== ty.shape:',ty.shape,' ===============')
    print('=========== test_x.shape:',test_x.shape,' ===============')
    print('=========== test_y.shape:',test_y.shape,' ===============')
    print('=========== validate_x.shape:',validate_x.shape,' ===============')
    print('=========== validate_y.shape:',validate_y.shape,' ===============')

    # TODO CNN + LSTM 模型  ---使用该模型时需注释掉另外两个模型
    # model = Sequential()
    # model.add(Conv1D(cnn_output_dim,kernel_size,padding='same',input_shape=(team_num,hero_id_max)))  #(none,team_num,9) 转换为 (none,team_num,32)
    # model.add(MaxPooling1D(pool_size=pool_size,data_format='channels_first'))  #(none,team_num,32)转换为 (none,team_num,16)
    # model.add(LSTM(hidden_size, input_shape=(team_num,(cnn_output_dim/pool_size)), return_sequences=False))  # 输入(none,team_num,129)  输出向量 (hidden_size,)
    # model.add(Dropout(0.2))
    # model.add(Dense(10))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))              # 全连接到一个元素
    # model.add(Activation('sigmoid'))
    # model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

    # TODO 纯LSTM 模型  ---使用该模型时需注释掉另外两个模型
    model = Sequential()
    model.add(LSTM(hidden_size, input_shape=(team_num,hero_id_max), return_sequences=False))  # 输入(none,team_num,129)  输出向量 (hidden_size,)
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1))              # 全连接到一个元素
    model.add(Activation('sigmoid'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

    # TODO 纯CNN 模型  ---使用该模型时需注释掉另外两个模型
    # model = Sequential()
    # model.add(Conv1D(cnn_output_dim,kernel_size,padding='same',activation='relu',input_shape=(team_num,hero_id_max)))  #(none,team_num,129) 转换为 (none,team_num,32)
    # model.add(MaxPooling1D(pool_size=pool_size,data_format='channels_first'))  #(none,team_num,32)转换为 (none,team_num,16)
    # model.add(Reshape((int(team_num*cnn_output_dim/pool_size),), input_shape=(team_num,int(cnn_output_dim/pool_size))))
    # model.add(Dropout(0.2))
    # model.add(Dense((10),input_shape=(team_num,cnn_output_dim/pool_size)))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))              # 全连接到一个元素
    # model.add(Activation('sigmoid'))
    # model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    


    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='min'),\
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, mode='min',\
             epsilon=0.0001, cooldown=0, min_lr=0)]
    hist = model.fit(tx,ty,batch_size=batch_size,epochs=epochs,shuffle=True,\
        validation_data=(validate_x, validate_y),callbacks=callbacks)
    model.save(model_saved_path+model_name+'.h5')


def testing(tag_line):
    train_x,train_y,test_x,test_y,validate_x,validate_y = make_samples()
    tx = np.array(train_x).reshape(len(train_x),team_num,hero_id_max)
    ty = np.array(train_y).reshape(len(train_y),1)
    test_x = np.array(test_x).reshape(len(test_x),team_num,hero_id_max)
    test_y = np.array(test_y).reshape(len(test_y),1)
    validate_x = np.array(validate_x).reshape(len(validate_x),team_num,hero_id_max)
    validate_y = np.array(validate_y).reshape(len(validate_y),1)

    # tx = tx[40000:]
    # ty = ty[40000:]
    # test_x = test_x[5000:]
    # test_y = test_y[5000:]
    # validate_x = validate_x[5000:]
    # validate_y = validate_y[5000:]

     #TODO ==============样本制作 结束==================
    print('=========== tx.shape:',tx.shape,' ===============')
    print('=========== ty.shape:',ty.shape,' ===============')
    print('=========== test_x.shape:',test_x.shape,' ===============')
    print('=========== test_y.shape:',test_y.shape,' ===============')
    print('=========== validate_x.shape:',validate_x.shape,' ===============')
    print('=========== validate_y.shape:',validate_y.shape,' ===============')
    # TODO 开始加载模型
    keras.backend.clear_session()    # 计算图清空，防止越来越慢
    model = load_model(model_saved_path+model_name+'.h5')
    out0 = model.predict(test_x)
    correct_num = 0
    for i in range(len(out0)):
        if out0[i][0]<0.5:
            temp_result = 0.0
        else:
            temp_result = 1.0
        if temp_result==test_y[i][0]:
            correct_num += 1
    print('测试集准确率：',float(correct_num)/len(test_x))

    out1 = model.predict(tx)
    correct_num = 0
    for i in range(len(out1)):
        if out1[i][0]<0.5:
            temp_result = 0.0
        else:
            temp_result = 1.0
        if temp_result==ty[i][0]:
            correct_num += 1
    print('训练集准确率：',float(correct_num)/len(tx))

    out2 = model.predict(validate_x)
    correct_num = 0
    for i in range(len(out2)):
        if out2[i][0]<0.5:
            temp_result = 0.0
        else:
            temp_result = 1.0
        if temp_result==validate_y[i][0]:
            correct_num += 1
    print('验证集准确率：',float(correct_num)/len(validate_x))

    correct_num = 0
    compare_num = 0
    for i in range(len(out0)):
        if out0[i][0]<(1.0-tag_line) or out0[i][0]>tag_line:
            compare_num += 1
            if out0[i][0]<0.5:
                temp_result = 0.0
            else:
                temp_result = 1.0
            if temp_result==test_y[i][0]:
                correct_num += 1
    print('测试集,预测胜率在'+str(tag_line)+'以上的准确率：',float(correct_num)/compare_num,\
        ' ('+str(correct_num)+'/'+str(compare_num)+')')
    
    for i in range(5):
        tag_line = 0.75+0.05*i
        correct_num = 0
        compare_num = 0
        for i in range(len(out0)):
            if out0[i][0]<(1.0-tag_line) or out0[i][0]>tag_line:
                compare_num += 1
                if out0[i][0]<0.5:
                    temp_result = 0.0
                else:
                    temp_result = 1.0
                if temp_result==test_y[i][0]:
                    correct_num += 1
        if compare_num!=0:
            print('测试集,预测胜率在'+str(tag_line)+'以上的准确率：',float(correct_num)/compare_num,\
                ' ('+str(correct_num)+'/'+str(compare_num)+')')
        else:
            print('测试集,预测胜率在'+str(tag_line)+'以上的准确率：','0.0',\
                ' ('+str(correct_num)+'/'+str(compare_num)+')')



if __name__ == "__main__":
    training()
    tag_line = 0.6
    testing(tag_line)


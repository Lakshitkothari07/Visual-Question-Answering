from google.colab import drive
drive._mount('/content/drive')

from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import  shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

import collections
import operator

ann_file="/content/drive/MyDrive/VQA/archive/annotations_train/abstract_v002_train2015_annotations.json"

with open(ann_file,'r') as f:
  ann=json.load(f)

all_ans=[]
all_ans_qids=[]
all_img_name_vector=[]

for annot in ann['annotations']:
  ans_dic=collections.defaultdict(int)
  for each in annot['answers']:
    diffans=each['answer']
    if diffans in ans_dic:
      if each['answer_confidence']=='yes':
        ans_dic[diffans]+=4
      if each['answer_confidence']=='maybe':
        ans_dic[diffans]+=2
      if each['answer_confidence']=='no':
        ans_dic[diffans]+=1
    else:
      if each['answer_confidence']=='yes':
        ans_dic[diffans]=4
      if each['answer_confidence']=='maybe':
        ans_dic[diffans]=2
      if each['answer_confidence']=='no':
        ans_dic[diffans]=1

  most_fav=max(ans_dic.items(),key=operator.itemgetter(1))[0]
  caption='<start> ' + most_fav + ' <end>'

  img_id=annot['image_id']
  ques_id=annot['question_id']
  full_img_path= '/content/drive/MyDrive/VQA/archive/img_train/abstract_v002_train2015_' + '%012d.png' %(img_id)

  all_img_name_vector.append(full_img_path)
  all_ans.append(caption)
  all_ans_qids.append(ques_id)
  
  
  ques_file="/content/drive/MyDrive/VQA/archive/questions_train/OpenEnded_abstract_v002_train2015_questions.json"

with open(ques_file,'r') as f:
  ques=json.load(f)

ques_id=[]
all_ques=[]
all_img_name_vector_2=[]

for annot in ques['questions']:
  caption='<start> ' + annot['question'] + ' <end>'
  img_id=annot['image_id']
  full_img_path='/content/drive/MyDrive/VQA/archive/img_train/abstract_v002_train2015_' + '%012d.png' %(img_id)

  all_img_name_vector_2.append(full_img_path)
  all_ques.append(caption)
  ques_id.append(annot['question_id'])
  
  print(len(all_img_name_vector),len(all_ans),len(all_ans_qids))
print(all_img_name_vector[10:15])
print(all_ans[10:15])
print(all_ans_qids[10:15])
print(len(all_img_name_vector_2),len(all_ques),len(ques_id))
print(all_img_name_vector_2[10:15])
print(all_ques[10:15])
print(ques_id[10:15])

train_ans,train_ques,img_name_vector=shuffle(all_ans,all_ques,all_img_name_vector,random_state=1)

num_examples=1000
train_ans=train_ans[:num_examples]
train_ques=train_ques[:num_examples]
img_name_vector=img_name_vector[:num_examples]

print(img_name_vector[0],train_ques[0],train_ans[0])
print(len(img_name_vector),len(train_ques),len(train_ans))

def load_img(img_path):
  img=tf.io.read_file(img_path)
  img=tf.image.decode_png(img,channels=3)
  img=tf.image.resize(img,(299,299))
  img=tf.keras.applications.inception_v3.preprocess_input(img)
  return img,img_path

img_model=tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')

new_input=img_model.input
hidden_layer=img_model.layers[-1].output

img_features_extract_model=tf.keras.Model(new_input,hidden_layer)

encode_train=sorted(set(img_name_vector))

img_dataset=tf.data.Dataset.from_tensor_slices(encode_train)
img_dataset=img_dataset.map(load_img,num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img,path in img_dataset:
  batch_features=img_features_extract_model(img)
  batch_features=tf.reshape(batch_features,(batch_features.shape[0],-1,batch_features.shape[3]))

  for bf,p in zip(batch_features,path):
    path_of_feature=p.numpy().decode("utf=8")
    np.save(path_of_feature,bf.numpy())
    
    
def calc_max_length(tensor):
  return max(len(t) for t in tensor)

top_k=10000
tokenizer=tf.keras.preprocessing.text.Tokenizer(num_words=top_k,oov_token="<unk>",filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_ques)
train_ques_seqs=tokenizer.texts_to_sequences(train_ques)

print(tokenizer.word_index)
ques_vocab=tokenizer.word_index

tokenizer.word_index['<pad>']=0
tokenizer.index_word[0]='<pad>'

train_ques_seqs=tokenizer.texts_to_sequences(train_ques)

ques_vector=tf.keras.preprocessing.sequence.pad_sequences(train_ques_seqs,padding='post')

max_length=calc_max_length(train_ques_seqs)
print(max_length)

max_q=max_length

from numpy import array,argmax
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

data=train_ans
values=array(data)
print(values[:10])

label_encoder=LabelEncoder()
integer_encoded=label_encoder.fit_transform(values)

ans_vocab={l:i for i , l in enumerate(label_encoder.classes_)}
print(ans_vocab)

onehot_encoder=OneHotEncoder(sparse=False)
integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded=onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded[0],len(onehot_encoded))

ans_vector=onehot_encoded
len_ans_vocab=len(onehot_encoded[0])
print(ans_vector)

print(len(ques_vector[0]),len(ans_vector[0]))

img_name_train,img_name_val,ques_train,ques_val,ans_train,ans_val=train_test_split(img_name_vector,ques_vector,ans_vector,test_size=0.2,random_state=0)

print(len(img_name_train),len(img_name_val),len(ques_train),len(ques_val),len(ans_train),len(ans_val))

ques_train.shape

BATCH_SIZE=16
BUFFER_SIZE=1
num_steps=len(img_name_train)//BATCH_SIZE
features_shape=2048
attention_features_shape=64

def map_func(img_name,cap,ans):
  img_tensor=np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor,cap,ans

dataset=tf.data.Dataset.from_tensor_slices((img_name_train,ques_train.astype(np.float32),ans_train.astype(np.float32)))

dataset=dataset.map(lambda item1,item2,item3: tf.numpy_function(map_func,[item1,item2,item3],[tf.float32,tf.float32,tf.float32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset=dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset=tf.data.Dataset.from_tensor_slices((img_name_val,ques_val.astype(np.float32),ans_val.astype(np.float32)))

test_dataset=test_dataset.map(lambda item1,item2,item3: tf.numpy_function(map_func,[item1,item2,item3],[tf.float32,tf.float32,tf.float32]),num_parallel_calls=tf.data.experimental.AUTOTUNE)

test_dataset=test_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset=test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

dataset

test_dataset

class AppendImageAsWordModel(tf.keras.Model):
  def __init__(self,embedding_size, rnn_size, output_size):
    super(AppendImageAsWordModel, self).__init__()
    self.flatten = tf.keras.layers.Flatten()
    self.condense = tf.keras.layers.Dense(embedding_size, activation='relu')
    # add embedding layer for questions
    self.embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1,output_dim=embedding_size)

    # create the input
    self.gru = tf.keras.layers.GRU(rnn_size,return_sequences=False,return_state=False)
    self.logits = tf.keras.layers.Dense(output_size,activation='softmax')

  def call(self,x,sents,hidden):
    flattended_output = self.flatten(x)
    condensed_out = self.condense(flattended_output)
    #    print(condensed_out.shape)
    condensed_out = tf.expand_dims(condensed_out, axis=1)
    #    print(condensed_out.shape)
    sents = self.embedding(sents)
    #    print(sents.shape)
    input_s = tf.concat([sents, condensed_out], axis=1)
    #    print(input_s.shape)
    output = self.gru(input_s, initial_state=hidden)
    final_output = self.logits(output)
    #    print(final_output.shape)
    return final_output

  def init_state(self, batch_size, rnn_size):
    return tf.zeros((batch_size, rnn_size))


append_image_word_model = AppendImageAsWordModel(256,256, len_ans_vocab)
crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def calc_loss(labels, logits):
  return crossentropy(labels, logits)

optimizer_append = tf.keras.optimizers.Adam()

#  @tf.function
def train_step(input_imgs, input_sents, labels, initial_state):
  with tf.GradientTape() as tape:
    my_model_output = append_image_word_model(input_imgs, input_sents, initial_state)
    loss = calc_loss(labels, my_model_output)
  variables = append_image_word_model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer_append.apply_gradients(zip(gradients, variables))
  return loss

EPOCHS = 40
for epoch in range(EPOCHS):
  init_states = append_image_word_model.init_state(BATCH_SIZE, 256)
  for (batch, (img_tensor, question, answer)) in enumerate(dataset):
    loss = train_step(img_tensor, question, answer, init_states)
  if epoch%10 == 0:
    print("Epoch #%d, Loss %.4f" % (epoch,loss))
    
class PrependImageAsWordModel(tf.keras.Model):
 def __init__(self, embedding_size, rnn_size, output_size):
  super(PrependImageAsWordModel, self).__init__()
  self.flatten = tf.keras.layers.Flatten()
  self.condense = tf.keras.layers.Dense(embedding_size, activation='relu')

  # add embedding layer for questions
  self.embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) +1, output_dim=embedding_size)

  # create the input
  self.gru = tf.keras.layers.GRU(rnn_size,return_sequences=False,return_state=False)
  self.logits = tf.keras.layers.Dense(output_size, activation='softmax')

 def call(self, x, sents, hidden):
   flattended_output = self.flatten(x)
   condensed_out = self.condense(flattended_output)
#   print(condensed_out.shape)
   condensed_out = tf.expand_dims(condensed_out, axis=1)
#   print(condensed_out.shape)
   sents = self.embedding(sents)
#   print(sents.shape)
   input_s = tf.concat([condensed_out, sents], axis=1)
#   print(input_s.shape)
   output = self.gru(input_s, initial_state=hidden)
   final_output = self.logits(output)
   return final_output 

 def init_state(self, batch_size, rnn_size):
   return tf.zeros((batch_size, rnn_size))


prepend_image_word_model = PrependImageAsWordModel(256, 256, len_ans_vocab)
crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def calc_loss(labels,logits):
  return crossentropy(labels, logits)

optimizer_prepend = tf.keras.optimizers.Adam()

# @tf.function
def train_step(input_imgs, input_sents, labels, initial_state):
  with tf.GradientTape() as tape:
    my_model_output = prepend_image_word_model(input_imgs, input_sents, initial_state)
    loss = calc_loss(labels, my_model_output)
  variables = prepend_image_word_model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer_prepend.apply_gradients(zip(gradients, variables))
  return loss

EPOCHS = 40
for epoch in range(EPOCHS):
  init_states = prepend_image_word_model.init_state(BATCH_SIZE, 256)
  for (batch, (img_tensor, question, answer)) in enumerate(dataset):
    loss = train_step(img_tensor, question, answer, init_states)
  if epoch%10 == 0:
    print("EPOCH #%d, LOSS %.4f" % (epoch, loss))
    
    
class SeperateImageAsWordModel(tf.keras.Model):
    def __init__(self, embedding_size, rnn_size, output_size):
        super(SeperateImageAsWordModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.condense = tf.keras.layers.Dense(embedding_size, activation='relu')
        # add ambedding layer for questions
        self.embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_size)
        # create the input
        self.gru = tf.keras.layers.GRU(rnn_size, return_sequences=False, return_state=False)
        self.logits = tf.keras.layers.Dense(output_size, activation='softmax')

    def call(self, x, sents, hidden):
        flattended_output = self.flatten(x)
        condensed_out = self.condense(flattended_output)
#         print(condensed_out.shape)
        condensed_out = tf.expand_dims(condensed_out, axis=1)
#         print(condensed_out.shape)
        sents = self.embedding(sents)
        sent_lstm_output = self.gru(sents, initial_state=hidden) #   run LSTM on question sents
        sent_lstm_output = tf.expand_dims(sent_lstm_output, axis=1)
#         print(sent_lstm_output.shape)
        output = tf.concat([sent_lstm_output, condensed_out], axis=2) # word and image embeddings side by side
#         print(output.shape)
        final_output = self.logits(output)
        return final_output

    def init_state(self, batch_size, rnn_size):
        return tf.zeros((batch_size, rnn_size))
      
  seperate_image_word_model = SeperateImageAsWordModel(256, 256, len_ans_vocab)
crossentropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def calc_loss(labels, logits):
    return crossentropy(labels, logits)

optimizer_seperate = tf.keras.optimizers.Adam()

# @tf.function
def train_step(input_imgs, input_sents, labels, initial_state):
  with tf.GradientTape() as tape:
    my_model_output = seperate_image_word_model(input_imgs, input_sents, initial_state)
    loss = calc_loss(labels, my_model_output)
  variables = seperate_image_word_model.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer_seperate.apply_gradients(zip(gradients, variables))
  return loss

EPOCHS = 40
for epoch in range(EPOCHS):
  init_states = seperate_image_word_model.init_state(BATCH_SIZE, 256)
  for (batch, (img_tensor, question, answer)) in enumerate(dataset):
    loss = train_step(img_tensor, question, answer, init_states)
  if epoch%10 == 0:
    print("EPOCH #%d, LOSS %.4f" % (epoch, loss))
    
    
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Dropout,Embedding,LSTM,Activation,ZeroPadding1D,Conv1D

class CoattentionModel(tf.keras.Model):
  def __init__(self,ans_vocab,max_q,ques_vocab):
    super(CoattentionModel, self).__init__(name='CoattentionModel')
    self.ans_vocab = ans_vocab
    self.max_q = max_q
    self.ques_vocab = ques_vocab

    self.ip_dense = Dense(256, activation='relu', input_shape=(512,))
    num_words = len(ques_vocab)+2
    self.word_level_feats = Embedding(input_dim = len(ques_vocab)+2,output_dim = 256)
    self.lstm_layer = LSTM(256,return_sequences=True,input_shape=(None,max_q,256))
    self.dropout_layer = Dropout(0.5)
    self.tan_layer = Activation('tanh')
    self.dense_image = Dense(256, activation='relu', input_shape=(256,))
    self.dense_text = Dense(256, activation='relu', input_shape=(256,))
    self.image_attention = Dense(1, activation='softmax', input_shape=(256,))
    self.text_attention = Dense(1, activation='softmax', input_shape=(256,))
    self.dense_word_level = Dense(256, activation='relu', input_shape=(256,))
    self.dense_phrase_level = Dense(256, activation='relu', input_shape=(2*256,))
    self.dense_sent_level = Dense(256, activation='relu', input_shape=(2*256,))
    self.dense_final = Dense(len(ans_vocab), activation='relu', input_shape=(256,))

  def affinity(self,image_feat,text_feat,level,prev_att):
    img = self.dense_image(image_feat)
    text = self.dense_text(text_feat)

    if level==0:
      return self.dropout_layer(self.tan_layer(text))

    elif level==1:
      level = tf.expand_dims(self.dense_text(prev_att),1)
      return self.dropout_layer(self.tan_layer(img + level))

    elif level==2:
      level = tf.expand_dims(self.dense_image(prev_att),1)
      return self.dropout_layer(self.tan_layer(text + level))

  def attention_ques(self,text_feat,text):
    return tf.reduce_sum(self.text_attention(text) * text_feat,1)

  def attention_img(self,image_feat,img):
    return tf.reduce_sum(self.image_attention(img) * image_feat,1)

  def call(self,image_feat,question_encoding):
    # Processing the image
    image_feat = self.ip_dense(image_feat)

    # Text features

    # Text: Word level
    word_feat = self.word_level_feats(question_encoding)

    # Text: Sentence level
    sent_feat = self.lstm_layer(word_feat)

  	#Apply attention to features on both the levels

    # Applying attention on word level features
    word_text_attention = self.attention_ques(word_feat, self.affinity(image_feat,word_feat,0,0))
    word_img_attention = self.attention_img(image_feat, self.affinity(image_feat,word_feat,1,word_text_attention))
    word_text_attention = self.attention_ques(word_feat, self.affinity(image_feat,word_feat,2,word_img_attention))

    word_pred = self.dropout_layer(self.tan_layer(self.dense_word_level(word_img_attention + word_text_attention)))

    # Applying attention on sentence level features
    sent_text_attention = self.attention_ques(sent_feat,self.affinity(image_feat,sent_feat,0,0))
    sent_img_attention = self.attention_img(image_feat,self.affinity(image_feat,sent_feat,1,sent_text_attention))
    sent_text_attention = self.attention_ques(sent_feat,self.affinity(image_feat,sent_feat,2,sent_img_attention))

    sent_pred = self.dropout_layer(self.tan_layer(self.dense_sent_level( tf.concat([sent_img_attention + sent_text_attention, word_pred],-1))))


    return self.dense_final(sent_pred)
  
model=CoattentionModel(ans_vocab,max_q,ques_vocab)

loss_function=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam()
train_loss_metric=tf.keras.metrics.Mean(name='train_loss')
test_loss_metric=tf.keras.metrics.Mean(name='test_loss')

train_accuracy_metric=tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
test_accuracy_metric=tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

def train_step(images, questions, answers ,model):
  with tf.GradientTape() as tape:    
    # Forward pass
    predictions = model(images, questions)
    train_loss = loss_function(y_true=answers, y_pred=predictions)
  
  # Backward pass
  gradients = tape.gradient(train_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # Record results
  train_loss_metric(train_loss)
  train_accuracy_metric(answers, predictions)

def test_step(images,questions, answers,model):
  predictions = model(images,questions)
  test_loss = loss_function(y_true=answers, y_pred=predictions)
  
  # Record results
  test_loss_metric(test_loss)
  test_accuracy_metric(answers, predictions)

  EPOCHS=40
train_loss =[]
test_loss=[]
train_acc=[]
test_acc=[]
for epoch in range(EPOCHS):
			#init_state = model.init_state(16)
			for (batch, (img_tensor, question, answer)) in enumerate(dataset):
				train_step(img_tensor, question, answer ,model)
		  
			for (batch, (img_tensor, question, answer)) in enumerate(test_dataset):
				test_step(img_tensor, question, answer,model)

			template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}, Test loss: {:.4f}, Test accuracy: {:.2f}'
			train_loss.append(train_loss_metric.result())
			test_loss.append(test_loss_metric.result())
			train_acc.append(train_accuracy_metric.result() * 100)
			test_acc.append(test_accuracy_metric.result() * 100)
			print (template.format(epoch +1, 
								 train_loss_metric.result(), 
								 train_accuracy_metric.result() * 100, 
								 test_loss_metric.result(), 
								 test_accuracy_metric.result() * 100))
      
models = [PrependImageAsWordModel, AppendImageAsWordModel,SeperateImageAsWordModel, CoattentionModel]
result = [[],[],[],[]]
a,b,c,d = [],[],[],[]

from numpy.ma.core import reshape
import pandas as pd
# from all_imports import *
# from func_defs import *
# import argparse
# from models import *
# from prep_data import get_data

def train_model(model_idx):
		model_name = models[model_idx]
		if model_idx == 3:
			model = model_name(len(ans_vocab), len(ques_vocab), max_q)
		else:
			model = model_name(len(ans_vocab), len(ques_vocab))
		train_loss =[]
		test_loss=[]
		train_acc=[]
		test_acc=[]
		for epoch in range(EPOCHS):
			#init_state = model.init_state(16)
			for (batch, (m_name, img_tensor, question, answer)) in enumerate(dataset):
				train_step(img_tensor, question, answer ,model)
		  
			for (batch, (name, img_tensor, question, answer)) in enumerate(test_dataset):
				test_step(img_tensor, question, answer,model)

			template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}, Test loss: {:.4f}, Test accuracy: {:.2f}'
			train_loss.append(train_loss_metric.result())
			test_loss.append(test_loss_metric.result())
			train_acc.append(train_accuracy_metric.result() * 100)
			test_acc.append(test_accuracy_metric.result() * 100)
			print (template.format(epoch +1,train_loss_metric.result(), train_accuracy_metric.result() * 100, test_loss_metric.result(), test_accuracy_metric.result() * 100))

		for (batch, (name, img_tensor, question, answer)) in enumerate(test_dataset):
			pred = test_step(img_tensor, question, answer,model)
			a = name.numpy()
			b = tokenizer.sequences_to_texts(question.numpy())
			c = label_encoder.classes_[[int(x) for x in answer.numpy()]]
			d = label_encoder.classes_[tf.argmax(pred, axis=2).numpy()]
       
			result[0] = np.hstack((result[0], a))
			result[1] = np.hstack((result[1], b))
			result[2] = np.hstack((result[2], c))
			result[3] = np.hstack((result[3], d))
			
		res = pd.DataFrame(result)
		res = res.transpose()
		res.to_csv(str(model_idx))
    # print(res)

train_model(3)


    


  












  

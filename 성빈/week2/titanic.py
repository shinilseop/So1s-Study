#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
print(tf.__version__)


# In[3]:


TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)

column_names = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare',
       'class', 'deck', 'embark_town', 'alone']

df = pd.read_csv(TRAIN_DATA_URL)
# pd.set_option('display.max_rows', None)
print(df)
# 데이터 정보 확인
print(df.info())
# 수치형 데이터 확인
print(df.describe())
# 범주형 데이터 확인
df.describe(include = np.object_)
# 결측치 확인
df.isnull().sum()


# In[4]:


# 데이터 요약

print("전체 데이터 수:", df.shape[0] * df.shape[1])
print(f"결측치 수: {df.isnull().sum().sum()}")
print("총 인원 수:", df["age"].count())
print("중복된 데이터:",df.duplicated().sum())


# In[16]:


class_names = ['UnSurvived', 'Survived']

feature_names = column_names[1:]
label_name = column_names[0]

print(feature_names)
print(label_name)

batch_size = 32
train_dataset = tf.data.experimental.make_csv_dataset(
    train_file_path,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

test_dataset = tf.data.experimental.make_csv_dataset(
    train_file_path,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

features, labels = next(iter(train_dataset))

print(f"##features \n {features}")


# In[17]:


# 불필요한 컬럼 삭제 -> embark_town
def remove_columns(features, labels):
    del(features['embark_town'])
    return features, labels

train_dataset = train_dataset.map(remove_columns)
test_dataset = test_dataset.map(remove_columns)
features, labels = next(iter(train_dataset))

print(f"features \n {features}")


# In[18]:


# 문자열 처리 -> sex, deck, alone, class를 문자열로
def convert_to_int(feature, label):
    if feature['sex'] == 'male':
        feature['sex'] = 0.
    else:
        feature['sex'] = 1.
    
    feature['sex'] = tf.cast(feature['sex'], tf.float32)
    
    return feature, label

CAT_COLUMNS = ['sex', 'deck', 'alone', 'class']
NUM_COLUMNS = ['age', 'fare', 'n_siblings_spouses', 'parch']

feature_cols = []

# Create IndicatorColumn for categorical features
for feature in CAT_COLUMNS:
  vocab = df[feature].unique()
  feature_cols.append(tf.feature_column.indicator_column(
      tf.feature_column.categorical_column_with_vocabulary_list(feature, vocab)))

# Create NumericColumn for numerical features
for feature in NUM_COLUMNS:
  feature_cols.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))

print(feature_cols)


# In[15]:


# 모델 생성
model = tf.keras.Sequential()
model.add(tf.keras.layers.DenseFeatures(feature_cols))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(train_dataset, epochs=100)
        


# In[25]:


# 모델 평가하기
model.predict(test_dataset)


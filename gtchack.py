# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:28:57 2018

@author: Ankit Gokhroo
"""
#importing pandas 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the given datasets
 
df=pd.read_excel("Corpus.xlsx")
train=pd.read_excel("MnA_Training.xlsx",error_bad_lines=False)
test=pd.read_excel("MnA_test.xlsx",error_bad_lines=False)

#after visualising datasets
#   we merge datasets on basis of system id

new=train.merge(df,how='left',on='system_id')

#the   formed new dataset is in categorial type
## so we need to encode and split the dataset using sklearn and encoders techniques   

# encoded date

new['date'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')


# encoded source

from sklearn.preprocessing import OneHotEncoder

one_hot=pd.get_dummies(new['source'])

new=new.drop('source',axis=1)
new=new.join(one_hot)

plt.plot(new['system_id'],new['globenewswire.com'])
plt.xlabel('globenewswire.com')
plt.ylabel('system_id')
plt.show()
plt.plot(new['marketwatch.com'],new['system_id'])
plt.xlabel('marketwatch.com')
plt.ylabel('system_id')
plt.show()
plt.plot(new['reuters.com'],new['system_id'])
plt.xlabel('reuters.com')
plt.ylabel('system_id')
plt.show()
plt.plot(new['techcrunch.com'],new['system_id'])
plt.xlabel('techcrunch.com')
plt.ylabel('system_id')
plt.show()

plt.scatter(new['reuters.com'],new['system_id'])
plt.xlabel('reuters.com')
plt.ylabel('system_id')
plt.show()

plt.scatter(new['techcrunch.com'],new['system_id'])
plt.xlabel('techcrunch.com')
plt.ylabel('system_id')
plt.show()


merger=new[new['raw_article'].str.contains('merg')]

acquire=new[new['raw_article'].str.contains('acquir')]

acquire ['boolean']=1
merger ['boolean']=0

bigdata=acquire.append(merger,ignore_index=True)

features=bigdata.iloc[:,[5,7,8,9,10,11]]

label1=bigdata.iloc[:,[1]]
label2=bigdata.iloc[:,[2]]

from sklearn.model_selection import train_test_split
features_train,features_test,label1_train,label1_test=train_test_split(features,label1,test_size=0.2,random_state=0)
features_train,features_test,label2_train,label2_test=train_test_split(features,label2,test_size=0.2,random_state=0)

new1=test.merge(df,how='left',on='system_id')

new1['date'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

from sklearn.preprocessing import OneHotEncoder



one_hot=pd.get_dummies(new1['source'])

new1=new1.drop('source',axis=1)
new1=new1.join(one_hot)


from sklearn.model_selection import train_test_split
features_train,features_test,label1_train,label1_test=train_test_split(features,label1,test_size=0.2,random_state=0)
features_train,features_test,label2_train,label2_test=train_test_split(features,label2,test_size=0.2,random_state=0)

from sklearn import datasets,linear_model
lm=linear_model.LinearRegression()
model=lm.fit(features_train,label1_train)
model=lm.fit(features_train,label2_train)
predictions=lm.predict(features_test)
predictions[0:5]

import re 
import string
def countWords(dataframe,selected_words):
    words_dict={}
    


for i in new.iloc[:,0] :
    for j in test.iloc[:,0] :
        if i==j:
            test['Acq Final']=new['Acq Final']
            test['Tar Final']=new['Tar Final']



plt.scatter(bigdata['boolean'][0:30],bigdata['Acq Final'])








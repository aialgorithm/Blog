#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt

spam_df = pd.read_csv('./data/spam.csv', header=0, encoding="ISO-8859-1")

# 数据展示
_, ax = plt.subplots(1,2,figsize=(10,5))
spam_df['label'].value_counts().plot(ax=ax[0], kind="bar", rot=90, title='label');
spam_df['label'].value_counts().plot(ax=ax[1], kind="pie", rot=90, title='label', ylabel='');
print("Dataset size: ", spam_df.shape)

spam_df.head(5)


# In[1]:


# 导入相关的库
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.data import load
from nltk.stem import SnowballStemmer
from string import punctuation

import re  # 正则匹配
stop_words = set(stopwords.words('english'))
non_words = list(punctuation)


# 词形、词干还原
# from nltk.stem import WordNetLemmatizer
# wnl = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
def stem_tokens(tokens, stemmer):
    stems = []
    for token in tokens:
        stems.append(stemmer.stem(token))
    return stems


### 清除非英文词汇并替换数值x
def clean_non_english_xdig(txt,isstem=True, gettok=True):
    txt = re.sub('[0-9]', 'x', txt) # 去数字替换为x
    txt = txt.lower() # 统一小写
    txt = re.sub('[^a-zA-Z]', ' ', txt) #去除非英文字符并替换为空格
    word_tokens = word_tokenize(txt) # 分词
    if not isstem: #是否做词干还原
        filtered_word = [w for w in word_tokens if not w in stop_words]  # 删除停用词
    else:
        filtered_word = [stemmer.stem(w) for w in word_tokens if not w in stop_words]   # 删除停用词及词干还原
    if gettok:   #返回为字符串或分词列表
        return filtered_word
    else:
        return " ".join(filtered_word)


# In[20]:


# 数据清洗
spam_df['token'] = spam_df.message.apply(lambda x:clean_non_english_xdig(x))

# 标签整数编码
spam_df['label'] = (spam_df.label=='spam').astype(int)

spam_df.head(3)


# In[5]:


# 训练词向量 Fasttext embed模型
from gensim.models import FastText,word2vec


fmodel = FastText(spam_df.token,  size=100,sg=1, window=3, min_count=1, iter=10, min_n=3, max_n=6,word_ngrams=1,workers=12) 
print("输出hello的词向量",fmodel.wv['hello']) # 词向量
# fmodel.save('./data/fasttext100dim')


# In[12]:


fmodel = FastText.load('./data/fasttext100dim')



#对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(sentence,w2v_model,size=100):
    sen_vec=np.zeros((size,))
    count=0
    for word in sentence:
        try:
            sen_vec+=w2v_model[word]#.reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec

# 句向量
sents_vec = []
for sent in spam_df['token']:
    sents_vec.append(build_sentence_vector(sent,fmodel,size=100))
        
print(len(sents_vec))


# In[38]:


### 训练文本分类模型
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

train_x, test_x, train_y, test_y = train_test_split(sents_vec, spam_df.label,test_size=0.2,shuffle=True,random_state=42)
result = []
clf = LGBMClassifier(class_weight='balanced',n_estimators=300, num_leaves=64, reg_alpha= 1,reg_lambda= 1,random_state=42)
#clf = LogisticRegression(class_weight='balanced',random_state=42)


clf.fit(train_x,train_y)

import pickle
# 保存模型
# pickle.dump(clf, open('./saved_models/spam_clf.pkl', 'wb'))

# 加载模型
model = pickle.load(open('./saved_models/spam_clf.pkl', 'rb'))


# In[40]:


from sklearn.metrics import auc,roc_curve,f1_score,precision_score,recall_score
def model_metrics(model, x, y,tp='auc'):
    """ 评估 """
    yhat = model.predict(x)
    yprob = model.predict_proba(x)[:,1]
    fpr,tpr,_ = roc_curve(y, yprob,pos_label=1)
    metrics = {'AUC':auc(fpr, tpr),'KS':max(tpr-fpr),
               'f1':f1_score(y,yhat),'P':precision_score(y,yhat),'R':recall_score(y,yhat)}
    
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")


    return metrics

print('train ',model_metrics(clf,  train_x, train_y,tp='ks'))
print('test ',model_metrics(clf, test_x,test_y,tp='ks'))






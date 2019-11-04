#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import word2vec
import gensim
import logging


# In[2]:


# 新闻语料Word2Vec处理，标准普尔数据打上标签
class DataProcess:
    def __init__(self, 
                 dow_jons_path=r'E:\SIOA\Program\PersonalProfit\DowJonesPredict\SourceData\DowJones.csv',
                 news_path=r'E:\SIOA\Program\PersonalProfit\DowJonesPredict\SourceData\News.csv',
                 save_model_file=r'E:\SIOA\Program\PersonalProfit\DowJonesPredict\SourceData\\',
                 padding_size = 1500,
                 vector_size = 64
                ):
        self.dow_jons_path = dow_jons_path
        self.news_path = news_path
        # 初始化两个数据表格
        self.dow_jons_pd,self.news_pd = self.read_files()
        # 词向量模型
        self.WordVectorModel = None
        # padding size
        self.padding_size = padding_size
        # vector size
        self.vector_size = vector_size
        # 代表每天新闻情况的矩阵
        self.news_matrix = []
        self.save_model_file=save_model_file
        
    # 读取新闻语料和dowjons数据，整合日期
    def read_files(self):
        news_pd = pd.read_csv(self.news_path)
        dow_jons_pd = pd.read_csv(self.dow_jons_path)
        news_pd = news_pd.loc[news_pd.Date.isin(dow_jons_pd.Date)]
        return dow_jons_pd, news_pd
        
     # 输入text,进行缩写词拓展，去除停用词处理
    def news_process(self, text, remove_stopwords=False):
        # 全部小写
        text = text.lower()
        text = re.sub(r'b\"', ' ', text)
        text = re.sub(r'b\'', ' ', text)
        text = re.sub(r'\'', '', text)
        text = re.sub(r'\"', '', text)
        # 替换不规则的词
        text = re.sub(r'&amp;', '', text) 
        text = re.sub(r'0,0', '00', text) 
        text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
        text = re.sub(r'\'', ' ', text)
        text = re.sub(r'\$', ' $ ', text)
        text = re.sub(r'u s ', ' united states ', text)
        text = re.sub(r'u n ', ' united nations ', text)
        text = re.sub(r'u k ', ' united kingdom ', text)
        text = re.sub(r'j k ', ' jk ', text)
        text = re.sub(r' s ', ' ', text)
        text = re.sub(r' yr ', ' year ', text)
        text = re.sub(r' l g b t ', ' lgbt ', text)
        text = re.sub(r'0km ', '0 km ', text)
        
        # 将缩写词进行拓展
        contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}
        new_text = []
        text = text.split()
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
                continue
            new_text.append(word)
        if remove_stopwords:
            stops_set = set(stopwords.words("english"))
            new_text = [word for word in new_text if word not in stops_set]
        return " ".join(new_text)
    
    # 新闻语料数据进行清洗，去除停用词处理
    # 添加DataFrame 中Cleaned_News列
    def clean_news_text(self):
        news_text_list = []
        for text in self.news_pd.News:
            news_text_list.append(self.news_process(text, remove_stopwords=True))
        self.news_pd['Cleaned_News'] = news_text_list
        # 当日所有新闻首尾连接
        dates_list = list(self.news_pd.Date.drop_duplicates())
        news_list = []
        for date in dates_list:
            selected_pd = self.news_pd.loc[self.news_pd.Date == date]
            news_list.append(" ".join(list(selected_pd.Cleaned_News)))
        new_data_pd = pd.DataFrame({'Date':dates_list, 'Cleaned_News':news_list})
        self.news_pd = new_data_pd
        # 将清理好的新闻输出成果文档
        with open(self.save_model_file + r'Cleaned_news.txt','w') as f:
            for senteces in self.news_pd.Cleaned_News:
                f.write(senteces)
                f.write('\n')
        
    # Word2vec训练
    def model_train(self): 
        # model_file_name为训练语料的路径,save_model为保存模型名
        # 模型训练，生成词向量
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        sentences = word2vec.Text8Corpus(self.save_model_file + r'Cleaned_news.txt')  # 加载语料
        self.WordVectorModel = gensim.models.Word2Vec(sentences, size=self.vector_size)  # 训练模型; 
        self.WordVectorModel.save(self.save_model_file + r'WordVector.model')
        self.WordVectorModel.wv.save_word2vec_format(self.save_model_file + "WordVectorModel.bin", binary=True)
        
    # 将每天的新闻转成一个固定大小的矩阵
    def transfer_news_to_matrix(self):
        res_matrix = []
        for sentence in self.news_pd.Cleaned_News[:-1]:
            news_matrix = []
            word_list = sentence.split()
            for i in range(len(word_list)):
                if word_list[i] in self.WordVectorModel:
                    news_matrix.append(self.WordVectorModel[word_list[i]].tolist())
                if len(news_matrix) >= self.padding_size:
                    break
            while len(news_matrix) < self.padding_size:
                news_matrix.append([0] * self.vector_size)
            res_matrix.append(news_matrix)
        return np.array(res_matrix)
    
    # 处理价格数据
    def process_price(self):
        self.dow_jons_pd.drop(['High', 'Low', 'Close', 'Volume', 'Adj Close'],axis=1,inplace=True)
        price_array = self.dow_jons_pd['Open']
        self.dow_jons_pd['price_flag'] = [1 if price_array[i+1]>price_array[i] else 0 for i in range(self.dow_jons_pd.shape[0]-1)] + [0]
        self.dow_jons_pd.drop(index=self.dow_jons_pd.shape[0]-1, axis=0,inplace=True)
        self.dow_jons_pd['today_news'] = self.news_pd.Cleaned_News
    # 主运行程序    
    def run_engine(self):
        self.clean_news_text()
#         if os.path.isfile(self.save_model_file + "WordVectorModel.bin"):
#             self.WordVectorModel = gensim.models.KeyedVectors.load_word2vec_format(self.save_model_file + 'WordVectorModel.bin',binary=True)
#         else:    
#             self.model_train()
#         self.news_matrix = self.transfer_news_to_matrix()
        self.process_price()
        return self.dow_jons_pd


# In[ ]:





#etemadir@ryerson.ca
#2021-06-01 cikm2021

import numpy as np
import tensorflow as tf
from networkx import to_numpy_matrix
import networkx as nx
import datetime
import sys
import os
import pickle
try:
    import ujson as json
except:
    import json
import math
from scipy.linalg import fractional_matrix_power
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
from sklearn.cluster import KMeans
from sklearn.metrics import ndcg_score 
import ml_metrics 


class CoExperts:    
    def  __init__(self,data):        
        self.dataset=data        
    def init_model(self):
        self.loadG()
        self.d=32 #embeding dim
        self.GCNW_1 =CoExperts.weight_variable((self.G.number_of_nodes(), 16)) 
        print("GCNW1=",self.GCNW_1.shape)
        self.GCNW_2 =CoExperts.weight_variable((self.GCNW_1.shape[1], self.d))
        print("GCNW2=",self.GCNW_2.shape)
        
        #MLP layer
        self.regindim=6*self.GCNW_2.shape[1]+11
        self.W1=CoExperts.weight_variable((self.regindim,16))
        #self.W2=CoExperts.weight_variable((self.W1.shape[1],8))
        #self.W3=CoExperts.weight_variable((self.W2.shape[1],16))
        self.W4 = CoExperts.weight_variable2(self.W1.shape[1])
        #self.W4 = CoExperts.weight_variable2(4*self.GCNW_2.shape[1])
        self.b = tf.Variable(random.uniform(0, 1))
        self.inputs=[]
        self.outputs=[]      
        
        #kernels setting
        self.n_bins=11 
        self.embedding_size=300        
        self.lamb = 0.5
        self.mus = CoExperts.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = CoExperts.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        self.embeddings = tf.Variable(tf.random.uniform([self.vocab_size+1, self.embedding_size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
       
        
        
    def weight_variable(shape):        
        tmp = np.sqrt(6.0) / np.sqrt(shape[0]+shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)
    
    def weight_variable2(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape)
        initial = tf.random.uniform([shape,1], minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)
    
    def load_traindata(self):
        data_dir=self.dataset
        INPUT=data_dir+"/Data_LDA_Topics/train.txt"        
        fin=open(INPUT)
        train=fin.readline().strip()
        train_q_contributers={}
        all_contrs=[]
        while train:
            data=train.split(",")
            q_id=data[0]            
            cntr_id=int(data[5]) 
            if q_id not in train_q_contributers:
                train_q_contributers[q_id]=[cntr_id]
            else:    
                train_q_contributers[str(q_id)].append(cntr_id)
            train=fin.readline().strip()
            if cntr_id not in all_contrs:
                all_contrs.append(cntr_id)
        fin.close()
        fin=open(data_dir+"/properties_LDA_topics.txt","r",encoding="utf-8")
        RepoNum=int(fin.readline().split("=")[1])
        ExpertNum=int(fin.readline().split("=")[1])
        StartExpertID=RepoNum
        fin.close()
        
        #load train data
        self.train_data=[]
        self.train_label=[]
        
        INPUT=data_dir+"/Data_LDA_Topics/train.txt"
        fin_train=open(INPUT)        
        train=fin_train.readline().strip()
        
        train_q_negs={}
        while train:
            data=train.split(",")            
            lst=[int(data[0]),int(data[1])]
            topics=[]
            if len(data[2].strip())>0:
                for d in data[2].split(" "):
                    topics.append(int(d))
            lst.append(topics)
            languages=[]
            if len(data[3].strip())>0:
                for d in data[3].split(" "):
                    languages.append(int(d))
            lst.append(languages)
            lst.append(int(data[4]))
            lst1=lst.copy()
            lst1.append(int(data[5]))
            
            self.train_data.append(lst1)
            labels=data[6].split(" ")
            #print(labels)
            if labels[3]=='0':
                label=0
            else:
                label=(float(labels[0])/float(labels[3]))*15
            self.train_label.append(label+4)
            
            #start add negative sample to train data
            q_id=int(data[0])
            flag=True
            while flag:
                ran=random.randint(0,ExpertNum-1)+StartExpertID
                if (ran in all_contrs) and (ran not in train_q_contributers[str(q_id)]) and ( 
                    (str(q_id) not in train_q_negs) or  (ran not in train_q_negs[str(q_id)])) :
                    lst1=lst.copy()
                    lst1.append(int(ran))
                    self.train_data.append(lst1)
                    self.train_label.append(0.0)
                    if str(q_id) not in train_q_negs:
                        train_q_negs[str(q_id)]=[ran]
                    else:
                        train_q_negs[str(q_id)].append(ran)
                    flag=False
            #end add negative sample to train data
            
            train=fin_train.readline().strip()
            
        
        fin_train.close()
        
        self.train_data=np.array(self.train_data)        
        self.train_label=np.array(self.train_label)
        
        #load test as validation 
        INPUT=data_dir+"/Data_LDA_Topics/test.txt"        
        fin=open(INPUT)
        test=fin.readline().strip()
        val_q_contributers={}
        while test:
            data=test.split(",")
            q_id=data[0]            
            cntr_id=int(data[5]) 
            if q_id not in val_q_contributers:
                val_q_contributers[q_id]=[cntr_id]
            else:    
                val_q_contributers[str(q_id)].append(cntr_id)
            test=fin.readline().strip()
            
        fin.close()
        self.val_data=[]
        self.val_label=[]
        INPUT=data_dir+"/Data_LDA_Topics/test.txt"
        fin_val=open(INPUT)        
        val=fin_val.readline().strip()
        val_q_negs={}
        while val:
            data=val.split(",")            
            lst=[int(data[0]),int(data[1])]
            topics=[]
            if len(data[2].strip())>0:
                for d in data[2].split(" "):
                    topics.append(int(d))
            lst.append(topics)
            languages=[]
            if len(data[3].strip())>0:
                for d in data[3].split(" "):
                    languages.append(int(d))
            lst.append(languages)
            lst.append(int(data[4]))
            lst1=lst.copy()
            lst1.append(int(data[5]))
            
            self.val_data.append(lst1)
            labels=data[6].split(" ")
            #print(labels)
            if labels[3]=='0':
                label=0
            else:
                label=(float(labels[0])/float(labels[3]))*15
            self.val_label.append(label+4)
            
            
            #start add negative sample to train data
            q_id=int(data[0])
            flag=True
            while flag:
                ran=random.randint(0,ExpertNum-1)+StartExpertID
                if (ran in all_contrs) and (ran not in val_q_contributers[str(q_id)]) and ( 
                    (str(q_id) not in train_q_negs) or  (ran not in train_q_negs[str(q_id)])):
                    lst1=lst.copy()
                    lst1.append(int(ran))
                    self.val_data.append(lst1)
                    self.val_label.append(0.0)
                    if str(q_id) not in val_q_negs:
                        val_q_negs[str(q_id)]=[ran]
                    else:
                        val_q_negs[str(q_id)].append(ran)
                    flag=False
            #end add negative sample to train data
            val=fin_val.readline().strip()
        fin_val.close()
        
        self.val_data=np.array(self.val_data)        
        self.val_label=np.array(self.val_label)
        
        #load text for train
        vocab=[]
        
        INPUT=data_dir+"/vocabs.txt"
        fin=open( INPUT, "r", encoding="utf-8")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")
            if v[1].strip()!="" and int(v[2].strip())>4:
                vocab.append(v[1])
            line=fin.readline().strip()
        fin.close()    
        self.vocab_size=len(vocab)
        print(self.vocab_size)
        
        txt_cntrs={}
        fin=open(data_dir+"/Data_LDA_Topics/txt_contrs.txt","r",encoding="utf8")
        line=fin.readline().strip()
        txt_contrs_len=[]
        while line:
            data=line.split(";;;",1)
            cid=data[0]
            txt_cntrs[cid]={}
            txts=data[1].split(";;;")
            all_txt=""
            
            for tx in txts:
                tc=tx.split(",",1)
                tccode=""
                for w in tc[1].split(" "):
                    if w in vocab:
                        tccode+=" "+str(vocab.index(w)+1)
                all_txt+=" "+tccode.strip()
                txt_cntrs[cid][tc[0]]=tccode.strip()
            
            txt_contrs_len.append(len(all_txt.split(" ")))
            line=fin.readline().strip()    
        #print(txt_cntrs["466"])
        fin.close()
        
        txt_repos={}
        fin=open(data_dir+"/Data_LDA_Topics/txt_repo.txt","r",encoding="utf8")
        line=fin.readline().strip()
        txt_repos_len=[]
        while line:
            data=line.split(",",1)
            rid=data[0]
            txt_code=""
            for w in data[1].split(" "):
                if w in vocab:
                    txt_code+=" "+str(vocab.index(w)+1)
            txt_code=txt_code.strip()
            txt_repos_len.append(len(txt_code.split(" ")))
            if rid not in txt_repos:
                txt_repos[rid]=txt_code
            else:
                txt_repos[rid].append(txt_code)
            line=fin.readline().strip()    
        #print(txt_repos["0"])
        fin.close() 
        #print(txt_repos)
        self.max_q_len=math.floor(max(txt_repos_len))
        self.max_d_len=math.floor(np.mean(txt_contrs_len))-self.max_q_len
        print(self.max_q_len,self.max_d_len)
        
        self.qatext=[]
        
        for t in self.train_data:
            repo_id=t[0]
            repo_txt=[ int(wid) for wid in txt_repos[str(repo_id)].split(" ")]
            contrs_id=t[5]
            
            c_txt=""
            for rkey in txt_cntrs[str(contrs_id)].keys():
                if rkey!=str(repo_id):
                    c_txt+=" "+txt_cntrs[str(contrs_id)][rkey]
            
            cntrs_txt=[ int(wid) for wid in c_txt.strip().split(" ")] #[:self.max_d_len]]
            #cntrs_txt=[ int(wid) for wid in txt_cntrs[str(contrs_id)].split(" ")[:self.max_d_len]]
            self.qatext.append([repo_txt,cntrs_txt])   
        self.qatext=np.array(self.qatext)
        
        self.val_data_text=[]
        for t in self.val_data:
            repo_id=t[0]
            repo_txt=[ int(wid) for wid in txt_repos[str(repo_id)].split(" ")]
            contrs_id=t[5]
            c_txt=""
            for rkey in txt_cntrs[str(contrs_id)].keys():
                if rkey!=str(repo_id):
                    c_txt+=" "+txt_cntrs[str(contrs_id)][rkey]
            cntrs_txt=[ int(wid) for wid in c_txt.strip().split(" ")]#[:self.max_d_len]]
            #cntrs_txt=[ int(wid) for wid in txt_cntrs[str(contrs_id)].split(" ")[:self.max_d_len]]
            self.val_data_text.append([repo_txt,cntrs_txt])
        
        self.val_data_text=np.array(self.val_data_text)
        #sys.exit(0)

    def loadG(self):
        INPUT=self.dataset+"/Data_LDA_Topics/Gedgs.txt"
        self.G=nx.Graph();        
        self.G=nx.read_weighted_edgelist(INPUT)
        #self.G=nx.read_edgelist(INPUT)
        order = [str(i) for i in range(len(self.G.nodes()))]
        #print(order)
        A_hat =to_numpy_matrix(self.G, nodelist=order,dtype=np.float32)
        
        I = np.eye(self.G.number_of_nodes(),dtype=np.float32)        
        A_hat = A_hat + I        
        print("A_hat=",A_hat.shape)
        
        Dhat05 = np.array(np.sum(A_hat, axis=0),dtype=np.float32)[0]       
        Dhat05 = np.matrix(np.diag(Dhat05),dtype=np.float32)        
        Dhat05=fractional_matrix_power(Dhat05,-0.5)  
        
        self.DAD=np.array(Dhat05*A_hat*Dhat05,dtype=np.float32)
        print("DAD=",self.DAD.shape)
    
    #adopted from knrm paper ref:https://github.com/AdeDZY/K-NRM
    @staticmethod
    def kernal_mus(n_kernels, use_exact):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        if use_exact:
            l_mu = [1]
        else:
            l_mu = [2]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    #adpoted from knrm paper copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
    @staticmethod
    def kernel_sigmas(n_kernels, lamb, use_exact):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.00001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [bin_size * lamb] * (n_kernels - 1)
        return l_sigma
    
    def q_a_rbf(self,inputs_q,inputs_d):    
        # look up embeddings for each term. [nbatch, qlen, emb_dim]

        self.max_q_len=len(inputs_q[0])
        self.max_d_len=len(inputs_d[0])
        
        q_embed = tf.nn.embedding_lookup(self.embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.embeddings, inputs_d, name='demb')
        batch_size=1
        
        # normalize and compute similarity matrix using l2 norm         
        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_embed), 2))
        #print(norm_q)
        norm_q=tf.reshape(norm_q,(len(norm_q),len(norm_q[0]),1))
        #print(norm_q)
        normalized_q_embed = q_embed / norm_q
        #print(normalized_q_embed)
        norm_d = tf.sqrt(tf.reduce_sum(tf.square(d_embed), 2))
        norm_d=tf.reshape(norm_d,(len(norm_d),len(norm_d[0]),1))
        normalized_d_embed = d_embed / norm_d
        #print(normalized_d_embed)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])
        #print(tmp)
        sim =tf.matmul(normalized_q_embed, tmp)
        #print(sim)        
        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [batch_size, self.max_q_len, self.max_d_len, 1])
        #print(rs_sim)
        
        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, self.mus)) / (tf.multiply(tf.square(self.sigmas), 2)))
        #print(tmp)
        
        feats = []  # store the soft-TF features from each field.
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]
        
        #print(kde)
        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        
        #q_weights=np.where(np.array(inputs_q)>0,1,0)
        #q_weights=tf.dtypes.cast(q_weights, tf.float32)
        #q_weights = tf.reshape(q_weights, shape=[batch_size, self.max_q_len, 1])
        
        aggregated_kde = tf.reduce_sum(kde , [1])  # [batch, n_bins]   *q_weights
        #print( aggregated_kde)
        feats.append(aggregated_kde) # [[batch, nbins]]
        feats_tmp = tf.concat( feats,1)  # [batch, n_bins]
        #print ("batch feature shape:", feats_tmp.get_shape())
        
        # Reshape. (maybe not necessary...)
        feats_flat = tf.reshape(feats_tmp, [-1, self.n_bins])
        feats_flat2=tf.reshape(feats_flat, [1,self.n_bins])
        
        return(feats_flat2)               
        
    def GCN_layer1(self):      
        return tf.matmul(self.DAD,self.GCNW_1)
    
    def GCN_layer2(self,i,X):
        a=tf.matmul([self.DAD[i,:]],X) 
        return tf.matmul(a,self.GCNW_2)
    
    def GCN_layers(self,i):
        H_1 = self.GCN_layer1()        
        H_2 = self.GCN_layer2(i,H_1)
        return H_2
              
    def model_test(self):
        """used for test"""
        embed=[]
        #print(self.inputs)
        for k in range(len(self.inputs)): 
            ind=self.inputs[k]
            qtext=[self.qatextinput[k][0]]
            #print(qtext)
            atext=[self.qatextinput[k][1]]
            q_a_rbf=self.q_a_rbf(qtext,atext)
            
            repoembed=tf.constant([np.zeros(self.d)],dtype=tf.float32)
                        
            #print(qembed)
            ownerembed=self.GCN_layers(ind[1])
            LDA_topic_embed=self.GCN_layers(ind[4])
            
            contributer_embed=self.GCN_layers(ind[5])
            
            if len(ind[2])>0:
                i=1             
                lst=[self.GCN_layers(ind[2][0])]
                for indx in  ind[2][1:]:
                    lst.append(self.GCN_layers(indx))
                    i=i+1
                topicsembed=tf.math.reduce_sum(lst, axis=0)/i 
            else:  
                topicsembed=tf.constant([np.zeros(self.d)],dtype=tf.float32)
            
            if len(ind[3])>0:
                i=1             
                lst=[self.GCN_layers(ind[3][0])]
                for indx in  ind[3][1:]:
                    lst.append(self.GCN_layers(indx))
                    i=i+1
                languagesembed=tf.math.reduce_sum(lst, axis=0)/i 
            else:  
                languagesembed=tf.constant([np.zeros(self.d)],dtype=tf.float32)
            
            embed1=tf.concat([q_a_rbf,repoembed,ownerembed,topicsembed,languagesembed,LDA_topic_embed,contributer_embed],1, name='concat')
            #embed1=tf.concat([qembed,askerembed,answererembed,tagsembed],1, name='concat')
            embed.append(embed1)
        embed=tf.reshape(embed,[len(self.inputs),self.regindim])    
        #return  tf.reshape(tf.matmul(embed,self.W4),[len(self.inputs)]) + self.b
        #print(embed)
        #print(len(embed))
        #print(len(embed[0]))
        w1out=tf.nn.tanh(tf.matmul(embed,self.W1))
        #print(w1out.shape)
        #w2out=tf.nn.tanh(tf.matmul(w1out,self.W2))
        #print(w2out.shape)
        #w3out=tf.nn.tanh(tf.matmul(w2out,self.W3))
        #print(w3out.shape)   
        return  tf.reshape(tf.matmul(w1out,self.W4),[len(self.inputs)]) + self.b
    
    
    def model(self):
        embed=[]
        #print(self.inputs)
        for k in range(len(self.inputs)): 
            ind=self.inputs[k]
            qtext=[self.qatextinput[k][0]]
            #print(qtext)
            atext=[self.qatextinput[k][1]]
            q_a_rbf=self.q_a_rbf(qtext,atext)
            
            repoembed=self.GCN_layers(ind[0])
            #print(qembed)
            ownerembed=self.GCN_layers(ind[1])
            LDA_topic_embed=self.GCN_layers(ind[4])
            contributer_embed=self.GCN_layers(ind[5])
            
            if len(ind[2])>0:
                i=1             
                lst=[self.GCN_layers(ind[2][0])]
                for indx in  ind[2][1:]:
                    lst.append(self.GCN_layers(indx))
                    i=i+1
                topicsembed=tf.math.reduce_sum(lst, axis=0)/i 
            else:  
                topicsembed=tf.constant([np.zeros(self.d)],dtype=tf.float32)
            
            if len(ind[3])>0:
                i=1             
                lst=[self.GCN_layers(ind[3][0])]
                for indx in  ind[3][1:]:
                    lst.append(self.GCN_layers(indx))
                    i=i+1
                languagesembed=tf.math.reduce_sum(lst, axis=0)/i 
            else:  
                languagesembed=tf.constant([np.zeros(self.d)],dtype=tf.float32)
            
            embed1=tf.concat([q_a_rbf,repoembed,ownerembed,topicsembed,languagesembed,LDA_topic_embed,contributer_embed],1, name='concat')
            #embed1=tf.concat([qembed,askerembed,answererembed,tagsembed],1, name='concat')
            embed.append(embed1)
        embed=tf.reshape(embed,[len(self.inputs),self.regindim])    
        #return  tf.reshape(tf.matmul(embed,self.W4),[len(self.inputs)]) + self.b
        #print(embed)
        #print(len(embed))
        #print(len(embed[0]))
        w1out=tf.nn.tanh(tf.matmul(embed,self.W1))
        #print(w1out.shape)
        #w2out=tf.nn.tanh(tf.matmul(w1out,self.W2))
        #print(w2out.shape)
        #w3out=tf.nn.tanh(tf.matmul(w2out,self.W3))
        #print(w3out.shape)   
        return  tf.reshape(tf.matmul(w1out,self.W4),[len(self.inputs)]) + self.b
        
    
    def loss(self):
        self.L= tf.reduce_mean(tf.square(self.model() - self.outputs))
        return self.L  
        
    def train(self,model_save_name,l_rate=0.0005,eps=5e-7,decay_step=500,epoch_nums=20): 
        self.load_traindata()
        self.init_model()
        
        print("train data loaded!!")
      
        val_len=math.ceil(0.1*len(self.train_data))
       
        len_train_data=len(self.train_data)
        
        loss_=0
        epochs = range(epoch_nums)
        self.batch_size=1
        global_step = tf.Variable(0, trainable=False)
        self.l_rate=l_rate
        self.epsilon=eps
        self.decay_step=decay_step
        self.optimizer="Adam"
        decayed_lr = tf.compat.v1.train.exponential_decay(self.l_rate,
                                        global_step,self.decay_step,
                                        0.95, staircase=True)
        opt = tf.keras.optimizers.Adam(learning_rate=decayed_lr,epsilon=self.epsilon)#(decayed_lr,epsilon=5e-6)
        logfile=open(self.dataset+"/log.txt","w")
        t_loss=[]
        v_loss=[]
        eps=[]
        
        for epoch in epochs:
            ind_new=[i for i in range(len_train_data)]
            np.random.shuffle(ind_new)
            self.train_data=self.train_data[ind_new,]
            self.train_label=self.train_label[ind_new,]           
            self.qatext=self.qatext[ind_new,] 
            
            start=0
            end=0
            for i in range(math.ceil(len_train_data/self.batch_size)):
                if ((i+1)*self.batch_size)<len_train_data:                    
                    start=i*self.batch_size
                    end=(i+1)*self.batch_size
                else:                    
                    start=i*self.batch_size
                    end=len_train_data
                    
                self.inputs=self.train_data[start:end]
                self.outputs=self.train_label[start:end]
                self.qatextinput=self.qatext[start:end]
                opt.minimize(self.loss, var_list=[self.embeddings,self.GCNW_1,self.GCNW_2,self.W1,self.W4,self.b])
                   
                loss_+=self.L 
                
                global_step.assign_add(1)
                opt._decayed_lr(tf.float32)
                
                #print(self.Loss)
                #sys.exit(0)
                if (i+1)%50==0:                    
                    rep=(epoch*math.ceil(len_train_data/self.batch_size))+((i+1))
                    txt='Epoch %2d: i  %2d  out of  %4d     loss=%2.5f' %(epoch, i*self.batch_size, len_train_data, loss_/(rep))
                    logfile.write(txt+"\n")
                    print(txt)    
            opt._decayed_lr(tf.float32)
            #print(self.W4)
            #validate the results
            
            print("\n************\nValidation started....\n")
            val_loss=0
            
            for ii in range(math.ceil(val_len/self.batch_size)):
                if ((ii+1)*self.batch_size)<val_len:
                    start=ii*self.batch_size
                    end=(ii+1)*self.batch_size
                else:
                    start=ii*self.batch_size
                    end=val_len
                self.inputs=self.val_data[start:end]
                self.outputs=self.val_label[start:end]
                self.qatextinput=self.val_data_text[start:end]
                
                val_loss+=self.loss()
                #print(self.loss())
                #print(val_loss)
                if (ii+1)%50==0:                   
                    txt='Epoch %2d: ii  %2d  out of  %4d     validation loss=%2.5f' %(epoch, ii*self.batch_size, val_len, val_loss/(ii+1))
                    logfile.write(txt+"\n")
                    print(txt)
            txt='Epoch %2d: ii  %2d  out of  %4d     validation loss=%2.5f' %(epoch, ii*self.batch_size, val_len, val_loss/(ii+1))
            logfile.write(txt+"\n")
            print(txt)
            
            if epoch%1==0:
                pkl_filename =self.dataset+ "/Data_LDA_Topics/st_co/pickle_model.pkl_"+model_save_name+str(epoch)
                with open(pkl_filename, 'wb') as file:
                    pickle.dump(self, file)
                print("model was saved")
            t_loss.append(loss_/(rep))
            v_loss.append(val_loss/math.ceil(val_len/self.batch_size))
            eps.append(epoch)
            plt.figure(figsize=(10,7))
            plt.plot(eps,t_loss,'r-o',label = "train loss")
            plt.plot(eps,v_loss,'b-*',label = "validation loss")
            plt.title("train and validation losses")
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc="upper right")
            #plt.savefig(self.dataset+ "/QRouting_noTags_noText/QRresults1/loss.png")
            plt.show()
        print("train model done!!")
        logfile.close() 

    def test_all(data_name,path, modelname,model_num):    
        """ranks all experts"""
        pkl_filename =data_name+path+"/st_co/"+modelname+model_num
        # Load from file
        with open(pkl_filename, 'rb') as file:
            ob = pickle.load(file)
        print("model was loaded!!")
        
        cntrs_ids=[]
        
        file_result=data_name+path+"/st_co/"+modelname+"_all_result_allexperts.txt"
        if not os.path.isfile(file_result):
            f_result=open(file_result,"w")
        else:
            f_result=open(file_result,"a+")
            
        fin=open(data_name+path+"txt_contrs.txt","r",encoding="utf8")
        line=fin.readline().strip()
        while line:
            data=line.split(";;;",1)
            cid=data[0]            
            cntrs_ids.append(int(cid))
            line=fin.readline().strip()    
        #print(txt_cntrs["466"])
        fin.close()
         
        INPUT=data_name+path+"test.txt"        
        fin_test=open(INPUT)
        test=fin_test.readline().strip()
        test_data=[]
        tst_q_ids=[]
        tst_q_contributers={}
        
        while test:
            data=test.split(",")
            
            q_id=int(data[0])            
            cntr_id=int(data[5])
            
            if q_id not in tst_q_ids:
                q_owner_id=int(data[1])
            
                topic_ids=[]            
                if data[2].strip()!='':
                    for tid in data[2].split(" "):
                        topic_ids.append(int(tid))
                l_ids=[]            
                if data[3].strip()!='':
                    for lid in data[3].split(" "):
                        l_ids.append(int(lid))

                lda_topic_id=int(data[4])
                
                tst_q_ids.append(q_id)
                tst_q_contributers[str(q_id)]=[cntr_id]
                test_data.append([q_id,q_owner_id,topic_ids,l_ids,lda_topic_id])
                
            else:
                tst_q_contributers[str(q_id)].append(cntr_id)
            test=fin_test.readline().strip()
        fin_test.close() 
        
        data_dir=data_name+"/"
        fin=open(data_dir +"properties_LDA_topics.txt","r",encoding="utf-8")
        RepoNum=int(fin.readline().split("=")[1])
        ExpertNum=int(fin.readline().split("=")[1])
        StartExpertID=RepoNum
        fin.close()
        
        total_match=0
        k=1
        filename =data_name+path+"/st_co/"+modelname+model_num+"_results1_allexperts.txt"
        
        fout=open(filename,"w")
        fout_test=open(data_name+path+"/st_co/"+modelname+model_num+"_test_results1_allexperts.txt","w")
        
        #load text for train
        vocab=[]
        
        INPUT=data_name+"/vocabs.txt"
        fin=open( INPUT, "r", encoding="utf-8")
        line=fin.readline()
        line=fin.readline().strip()
        while line:
            v = line.split(" ")
            if v[1].strip()!="" and int(v[2].strip())>4:
                vocab.append(v[1])
            line=fin.readline().strip()
        fin.close()    
        
        
        txt_cntrs={}
        fin=open(data_name+path+"txt_contrs.txt","r",encoding="utf8")
        line=fin.readline().strip()
        txt_contrs_len=[]
        while line:
            data=line.split(";;;",1)
            cid=data[0]
            txt_cntrs[cid]={}
            txts=data[1].split(";;;")
            all_txt=""
            
            for tx in txts:
                tc=tx.split(",",1)
                tccode=""
                for w in tc[1].split(" "):
                    if w in vocab:
                        tccode+=" "+str(vocab.index(w)+1)
                all_txt+=" "+tccode.strip()
                txt_cntrs[cid][tc[0].strip()]=tccode.strip()
            
            txt_contrs_len.append(len(all_txt.split(" ")))
            line=fin.readline().strip()    
        #print(txt_cntrs["466"])
        fin.close()
        
        txt_repos={}
        fin=open(data_name+path+"txt_repo.txt","r",encoding="utf8")
        line=fin.readline().strip()
        txt_repos_len=[]
        while line:
            data=line.split(",",1)
            rid=data[0]
            txt_code=""
            for w in data[1].split(" "):
                if w in vocab:
                    txt_code+=" "+str(vocab.index(w)+1)
            txt_code=txt_code.strip()
            txt_repos_len.append(len(txt_code.split(" ")))
            if rid not in txt_repos:
                txt_repos[rid]=txt_code
            else:
                txt_repos[rid].append(txt_code)
            line=fin.readline().strip()    
        #print(txt_repos["0"])
        fin.close() 
        #print(txt_repos)
        max_q_len=ob.max_q_len
        max_d_len=ob.max_d_len
        print(max_q_len,max_d_len)
        
        print("num test qs=",len(test_data))
        for tst_q in test_data:
            #print("\n\ntest q "+str(k)+":")
            inputs=[]
            true_cntrs=tst_q_contributers[str(tst_q[0])]
            repo_id=int(tst_q[0])
            
            inputtext=[]
            
            repo_txt=[ int(wid) for wid in txt_repos[str(repo_id)].split(" ")]            
            
            
            for c_id in cntrs_ids:
                input_item=tst_q.copy()
                input_item.append(c_id)
                inputs.append(input_item)
                c_txt=""
                for rkey in txt_cntrs[str(c_id)].keys():
                    if rkey!=str(repo_id):
                        c_txt+=" "+txt_cntrs[str(c_id)][rkey]
                cntrs_txt=[ int(wid) for wid in c_txt.strip().split(" ")]#[:max_d_len]]
                inputtext.append([repo_txt,cntrs_txt])
            print(tst_q[0])

            #print(len(inputs))
            ob.inputs=inputs
            ob.qatextinput=inputtext
            scores=ob.model_test().numpy()
            ids=cntrs_ids.copy()
            
            sorted_scores,sorted_ids=(list(t) for t in zip(*sorted(zip(scores, ids),reverse=True)) ) 
            top_cntrs=sorted_ids[:len(true_cntrs)]
            match=len( set(top_cntrs).intersection(set(true_cntrs))) /len(true_cntrs)
            #print("match\%:"+str(match))
            total_match+=match
            #print("ave match\%:"+str(total_match/k))
            fout.write("match\%:"+str(match)+"ave match\%:"+str(total_match/k)+"\n")
            fout.flush()
            k+=1
            #write rsults into file
            true_contrs=' '.join([str(pcid) for pcid in true_cntrs])
            all_contrs=' '.join([str(pcid) for pcid in cntrs_ids])
            res=""
            for ind in range(len(sorted_ids)):
                res+=" "+str(sorted_ids[ind])+ " "+ str(sorted_scores[ind])
            fout_test.write(str(tst_q[0])+","+true_contrs.strip()+","+all_contrs.strip()+","+res.strip()+"\n")
            fout_test.flush()
            fout_test.flush()    
            
            
        print("ave match\%:"+str(total_match/k))    
        print("test_model done!!") 
        f_result.write("\nmodel="+modelname+model_num+"\n")
        f_result.write("ave match\%:"+str(total_match/k))
        f_result.close()
        
        fout.close()
        fout_test.close()
         
    def discover_teams_baselineMatchZoo_all_cikm2021(self,path,alg,model,team_size):        
        #load baseline results
        #load baseline results
        test_statistics={}
        fin=open(path+"test.txt","r")
        line=fin.readline().strip()
        while line:
            data=line.split(",")
            repo_id=data[0].strip()
            numadds=int(data[6].split(" ")[0])
            if repo_id not in test_statistics:
                test_statistics[repo_id]={"true":[int(data[5])],"adds":[numadds],"neg":[]}
            else:
                test_statistics[repo_id]["true"].append(int(data[5]))
                test_statistics[repo_id]["adds"].append(numadds)
            line=fin.readline().strip()
        fin.close()
        
        
        cntrs_ids=[]
        fin=open(path+"txt_contrs.txt","r",encoding="utf8")
        line=fin.readline().strip()
        while line:
            data=line.split(";;;",1)
            cid=int(data[0])
            cntrs_ids.append(cid)
            line=fin.readline().strip()    
        #print(txt_cntrs["466"])
        fin.close()
        fin=open(path +"G_properties.txt","r",encoding="utf-8")
        self.N=int(fin.readline().split("=")[1])
        self.NumRepos=int(fin.readline().split("=")[1].split(" ")[0])
        self.NumExperts=int(fin.readline().split("=")[1].split(" ")[0])
        fin.close()
        #load baseline results
        baseline_results=np.loadtxt(path+alg+"/"+model)
        
        #shortest path
        G_avr_sh_p=0
        num_p=0
        sh_path={}
        fin=open(path+"G_shortest_path.txt","r")
        line=fin.readline()
        while line:
            data=line.split(" ")
            i=data[0]+","+data[1]
            sh=int(data[3])
            #print(i,sh)
            sh_path[i]=sh
            G_avr_sh_p+=sh
            num_p+=1
            #sys.exit(0)
            line=fin.readline()
        fin.close()
        print("G avr sh path="+str(G_avr_sh_p/num_p))
        start=0
        total_avr_sh_path=[]
        GM=[]
        PN=[]
        MSC=0
        total_c_sh_path=[]
        total_bc_sh_path=[]
        ndcg=[]
        MAP=[]
        for test_id in test_statistics.keys():

            num_true_cntrs=len(test_statistics[test_id]["true"])
            end=start+len(cntrs_ids)
            c_scores=baseline_results[start:end]
            start=end
            c_ids=cntrs_ids.copy()
            sorted_c_scores,sorted_c_ids=(list(t) for t in zip(*sorted(zip(c_scores, c_ids),reverse=True)) )
                        
            true_cntrs=test_statistics[test_id]["true"]
            hat_cntrs=sorted_c_ids[:team_size]
        
            match=len(set(true_cntrs).intersection(set(hat_cntrs)))/len(true_cntrs)
            GM.append(match)
            if match>0:
                MSC+=1
                
            match=len(set(true_cntrs).intersection(set(hat_cntrs)))/len(hat_cntrs)
            PN.append(match)
            
            true_ranks=test_statistics[test_id]["adds"]
            true_ranks=np.array(true_ranks)
            true_ranks=(true_ranks/true_ranks.sum())*15+4
            
            true_scores=np.zeros(len(sorted_c_ids))
            #print(true_ranks)
            #print(true_cntrs)
            hat_score=[]
            for ii in range(len(true_cntrs)):
                eid=true_cntrs[ii]
                scoree=true_ranks[ii]
                eindex=sorted_c_ids.index(eid)
                true_scores[eindex]=scoree
                hat_score.append(sorted_c_scores[eindex])
            #print(true_scores)
            
            hat_score=np.array(hat_score)
           # print(hat_score)
           # print(true_ranks)
            if min(hat_score)<0:
                hat_score=hat_score-min(hat_score) 
                
#             print(ndcg_score([hat_score], [true_ranks], k=team_size))
            if min(sorted_c_scores)<0:
                sorted_c_scores=sorted_c_scores-min(sorted_c_scores) 
                        
            
            #ndcg.append(ndcg_score([true_scores], [sorted_c_scores],  k=team_size))
            ndcg.append(ndcg_score([true_ranks], [hat_score], k=team_size))
            
            sorted_true_scores,sorted_true_ids=(list(t) for t in zip(*sorted(zip(true_ranks, true_cntrs),reverse=True)) )
            mapatk=ml_metrics.mapk([sorted_true_ids], [hat_cntrs],k=team_size)
            MAP.append(mapatk)
            
            #compute average shortest path
            avg_sh_c,num_path_c=0,0
            hat_cntrs=list(set(hat_cntrs))
            hat_cntrs.sort()
            for i in range(len(hat_cntrs)-1):
                for j in range(i+1,len(hat_cntrs)):
                    ci=hat_cntrs[i]
                    cj=hat_cntrs[j]
                    sh_p=sh_path[str(ci)+","+str(cj)]
                    avg_sh_c+=sh_p
                    num_path_c+=1
                        
            if num_path_c==0:
                total_bc_sh_path.append(0)
            else:
                total_bc_sh_path.append(avg_sh_c/num_path_c)
                    
            t_c_shp=0
            for j in range(len(hat_cntrs)):
                    ri=test_id
                    cj=hat_cntrs[j]
                    if int(cj)>int(ri):
                        ind=str(ri)+","+str(cj)
                    else:
                        ind=str(cj)+","+str(ri)
                    
                    t_c_shp+=sh_path[ind]
                    
            
            total_c_sh_path.append(t_c_shp/len(hat_cntrs))
            
        total_c_sh_path=np.array(total_c_sh_path)
        
        GM=np.array(GM)
        print("mean GM="+str(np.mean(GM)))
        
        PN=np.array(PN)
        print("mean PN="+str(np.mean(PN)))
        
        MSC=MSC/len(test_statistics.keys())
        
        print("MSC=",MSC)
        
        av_ndcg=np.mean(ndcg)
        
        print("ndcg=",av_ndcg)
        av_map=np.mean(np.array(MAP))
        print("map=",av_map)
        
        print("\n\nmean shortest path from test repo to selected es="+str(np.mean(total_c_sh_path)))
        print("var shortest path from test repo to selected es="+str(np.var(total_c_sh_path)))
        print("Standard Deviation shortest path from test repo to selected es="+str(np.std(total_c_sh_path)))
        
        
        print("\n\nmean shortest path between es="+str(np.mean(total_bc_sh_path)))
        print("var shortest path between es="+str(np.var(total_bc_sh_path)))
        print("Standard Deviation shortest path between es="+str(np.std(total_bc_sh_path)))
        
        print("dicovering teams done!")
        return round(np.mean(GM),3),round(np.mean(PN),3),round(
            MSC,3),round(np.mean(total_c_sh_path),3), round(np.mean(total_bc_sh_path),3), round(av_ndcg,3),round(av_map,3)
    
    def discover_teams_TF_cikm2021(self,path,alg): 
        """compute cikm metrics GM,PN, MSC, ASPL_c"""
        #load results
        test_statistics={}
        fin=open(path+"/Data_LDA_Topics/test.txt","r")
        line=fin.readline().strip()
        test_ids=[]
        while line:
            data=line.split(",")
            repo_id=data[0].strip()
            test_ids.append(repo_id)
            if repo_id not in test_statistics:
                test_statistics[repo_id]={"true":[int(data[5])]}
            else:
                test_statistics[repo_id]["true"].append(int(data[5]))
            line=fin.readline().strip()
        fin.close()
        
        test_res={}
        fin=open(path+"/Data_LDA_Topics/DBLPformat/all"+alg+"results.txt","r")
        line=fin.readline()
        while line:
            r=line.strip().split(" ")
            team=[]
            for c in r[1:]:
                if int(c) not in team:
                    team.append(int(c))
            test_id=test_ids[int(r[0])]
            test_res[test_id]=team
            line=fin.readline()
        fin.close()
        
        #shortest path
        G_avr_sh_p=0
        num_p=0
        sh_path={}
        fin=open(path+"/Data_LDA_Topics/G_shortest_path.txt","r")
        line=fin.readline()
        while line:
            data=line.split(" ")
            i=data[0]+","+data[1]
            sh=int(data[3])
            #print(i,sh)
            sh_path[i]=sh
            G_avr_sh_p+=sh
            num_p+=1
            #sys.exit(0)
            line=fin.readline()
        fin.close()
        print("G avr sh path="+str(G_avr_sh_p/num_p))
        
        #load embeddings        
        # Load model from file
#         with open(path+"/st_co/"+model, 'rb') as file:
#             ob = pickle.load(file)
        #print("model was loaded!!")  
        
        fin=open(path +"/Data_LDA_Topics/G_properties.txt","r",encoding="utf-8")
        self.N=int(fin.readline().split("=")[1])
        self.NumRepos=int(fin.readline().split("=")[1].split(" ")[0])
        self.NumExperts=int(fin.readline().split("=")[1].split(" ")[0])
        fin.close() 
        

        
        total_avr_sh_path=[]
        
        GM=[]
        PN=[]
        MSC=0
        total_c_sh_path=[]
        total_bc_sh_path=[]
        av_team_size=0
        for test_id in test_res.keys():
            
            avg_sh=0
            num_path=0
                        
            true_cntrs=test_statistics[test_id]["true"]  #true team  
            
            hat_cntrs=test_res[test_id]  #discoverd team with size team_size
            
            av_team_size+=len(hat_cntrs)
            
            #compute average shortest path
            avg_sh_c,num_path_c=0,0
            
            hat_cntrs=list(set(hat_cntrs))
            match=len(set(hat_cntrs).intersection(set(true_cntrs)))/len(true_cntrs)
            GM.append(match)
            
            if match >0:
                MSC+=1
            
            match=len(set(hat_cntrs).intersection(set(true_cntrs)))/len(hat_cntrs)
            PN.append(match)
            
            hat_cntrs.sort()
            for i in range(len(hat_cntrs)-1):
                for j in range(i+1,len(hat_cntrs)):
                    ci=hat_cntrs[i]
                    cj=hat_cntrs[j]
                    sh_p=sh_path[str(ci)+","+str(cj)]
                    avg_sh_c+=sh_p
                    num_path_c+=1
            
            if num_path_c==0:
                total_bc_sh_path.append(0)
            else:
                total_bc_sh_path.append(avg_sh_c/num_path_c)
                    
            t_c_shp=0
            for j in range(len(hat_cntrs)):
                    ri=test_id
                    cj=hat_cntrs[j]
                    if int(cj)>int(ri):
                        ind=str(ri)+","+str(cj)
                    else:
                        ind=str(cj)+","+str(ri)
                    
                    t_c_shp+=sh_path[ind]
            
            total_c_sh_path.append(t_c_shp/len(hat_cntrs))
        
        total_c_sh_path=np.array(total_c_sh_path)
        
        GM=np.array(GM)
        print("mean GM="+str(np.mean(GM)))
        
        PN=np.array(PN)
        print("mean PN="+str(np.mean(PN)))
        
        MSC=MSC/len(test_res.keys())
        
        print("MSC=",MSC)
        print("avr team size=", av_team_size/len(test_res.keys() ))
        
        print("\n\nmean shortest path from test repo to selected es="+str(np.mean(total_c_sh_path)))
        print("var shortest path from test repo to selected es="+str(np.var(total_c_sh_path)))
        print("Standard Deviation shortest path from test repo to selected es="+str(np.std(total_c_sh_path)))
        
        
        print("\n\nmean shortest path between es="+str(np.mean(total_bc_sh_path)))
        print("var shortest path between es="+str(np.var(total_bc_sh_path)))
        print("Standard Deviation shortest path between es="+str(np.std(total_bc_sh_path)))
        
        print("dicovering teams done!")
        return round(np.mean(GM),5),round(np.mean(PN),5),round(
            MSC,5),round(np.mean(total_c_sh_path),3), round(np.mean(total_bc_sh_path),3)
    
    def discover_teams_propose_vs_TF_cikm2021(self,path,contrs_ranking_results,model,alg): 
        """compute cikm metrics GM,PN, MSC, ASPL_c"""
        #load test results
        test_conts_ranks={}
        fin=open(path+"/st_co/"+contrs_ranking_results,"r")
        line=fin.readline()
        test_ids=[]
        while line:
            data=line.split(",")
            repo_id=data[0]
            true_contrs=[int(ids) for ids in data[1].split(" ")]
            neg_contrs=[int(ids) for ids in data[2].split(" ")]
            ids_ranks=data[3].split(" ")
            ids=[int(id_) for id_ in ids_ranks[0::2]]
            ranks=[float(r) for r in ids_ranks[1::2]]
            
            test_conts_ranks[repo_id]={"true":true_contrs,"neg":neg_contrs,"ids":ids,"sim":ranks}
            test_ids.append(repo_id)
            line=fin.readline()
        fin.close()
        
        test_res={}
        fin=open(path+"/DBLPformat/all"+alg+"results.txt","r")
        line=fin.readline()
        while line:
            r=line.strip().split(" ")
            team=[]
            for c in r[1:]:
                if int(c) not in team:
                    team.append(int(c))
            test_id=test_ids[int(r[0])]
            test_res[test_id]=team
            line=fin.readline()
        fin.close()
        
        #shortest path
        G_avr_sh_p=0
        num_p=0
        sh_path={}
        fin=open(path+"G_shortest_path.txt","r")
        line=fin.readline()
        while line:
            data=line.split(" ")
            i=data[0]+","+data[1]
            sh=int(data[3])
            #print(i,sh)
            sh_path[i]=sh
            G_avr_sh_p+=sh
            num_p+=1
            #sys.exit(0)
            line=fin.readline()
        fin.close()
        print("G avr sh path="+str(G_avr_sh_p/num_p))
        
        #load embeddings        
        # Load model from file
#         with open(path+"/st_co/"+model, 'rb') as file:
#             ob = pickle.load(file)
        #print("model was loaded!!")  
        
        fin=open(path +"G_properties.txt","r",encoding="utf-8")
        self.N=int(fin.readline().split("=")[1])
        self.NumRepos=int(fin.readline().split("=")[1].split(" ")[0])
        self.NumExperts=int(fin.readline().split("=")[1].split(" ")[0])
        fin.close() 
        
       
        total_avr_sh_path=[]
        
        GM=[]
        PN=[]
        MSC=0
        total_c_sh_path=[]
        total_bc_sh_path=[]
        
        for test_id in test_res.keys():
            avg_sh=0
            num_path=0
            team_size=len(test_res[test_id])
            
            test=test_conts_ranks[test_id]

            res=str(test_id)
            
            true_cntrs=test["true"]  #true team            
            hat_cntrs=test["ids"][:team_size]  #discoverd team with size team_size
            
            
            
            #compute average shortest path
            avg_sh_c,num_path_c=0,0
            
            hat_cntrs=list(set(hat_cntrs))
            match=len(set(hat_cntrs).intersection(set(true_cntrs)))/len(true_cntrs)
            GM.append(match)
            
            if match >0:
                MSC+=1
            
            match=len(set(hat_cntrs).intersection(set(true_cntrs)))/len(hat_cntrs)
            PN.append(match)
            
            hat_cntrs.sort()
            for i in range(len(hat_cntrs)-1):
                for j in range(i+1,len(hat_cntrs)):
                    ci=hat_cntrs[i]
                    cj=hat_cntrs[j]
                    sh_p=sh_path[str(ci)+","+str(cj)]
                    avg_sh_c+=sh_p
                    num_path_c+=1
            
            if num_path_c==0:
                total_bc_sh_path.append(0)
            else:
                total_bc_sh_path.append(avg_sh_c/num_path_c)
                    
            t_c_shp=0
            for j in range(len(hat_cntrs)):
                    ri=test_id
                    cj=hat_cntrs[j]
                    if int(cj)>int(ri):
                        ind=str(ri)+","+str(cj)
                    else:
                        ind=str(cj)+","+str(ri)
                    
                    t_c_shp+=sh_path[ind]
            
            total_c_sh_path.append(t_c_shp/len(hat_cntrs))
            
        
        
       
        total_c_sh_path=np.array(total_c_sh_path)
        
        GM=np.array(GM)
        print("mean GM="+str(np.mean(GM)))
        
        PN=np.array(PN)
        print("mean PN="+str(np.mean(PN)))
        
        MSC=MSC/len(test_res.keys())
        
        print("MSC=",MSC)
        
        
        print("\n\nmean shortest path from test repo to selected es="+str(np.mean(total_c_sh_path)))
        print("var shortest path from test repo to selected es="+str(np.var(total_c_sh_path)))
        print("Standard Deviation shortest path from test repo to selected es="+str(np.std(total_c_sh_path)))
        
        
        print("\n\nmean shortest path between es="+str(np.mean(total_bc_sh_path)))
        print("var shortest path between es="+str(np.var(total_bc_sh_path)))
        print("Standard Deviation shortest path between es="+str(np.std(total_bc_sh_path)))
        
        print("dicovering teams done!")
        return round(np.mean(GM),3),round(np.mean(PN),3),round(
            MSC,3),round(np.mean(total_c_sh_path),3), round(np.mean(total_bc_sh_path),3)
    
    def discover_teams_cikm2021(self,path,contrs_ranking_results,model,team_size): 
        """compute cikm metrics GM,PN, MSC, ASPL_c,ASPL_bc ndcg map"""
        #load baseline results
        test_statistics={}
        fin=open(path+"test.txt","r")
        line=fin.readline().strip()
        while line:
            data=line.split(",")
            repo_id=data[0].strip()
            numadds=int(data[6].split(" ")[0])
            if repo_id not in test_statistics:
                test_statistics[repo_id]={"true":[int(data[5])],"adds":[numadds],"neg":[]}
            else:
                test_statistics[repo_id]["true"].append(int(data[5]))
                test_statistics[repo_id]["adds"].append(numadds)
            line=fin.readline().strip()
        fin.close()
        
        #load test results
        test_conts_ranks={}
        fin=open(path+"/st_co/"+contrs_ranking_results,"r")
        line=fin.readline()
        while line:
            data=line.split(",")
            repo_id=data[0]
            true_contrs=[int(ids) for ids in data[1].split(" ")]
            neg_contrs=[int(ids) for ids in data[2].split(" ")]
            ids_ranks=data[3].split(" ")
            ids=[int(id_) for id_ in ids_ranks[0::2]]
            ranks=[float(r) for r in ids_ranks[1::2]]
            
            test_conts_ranks[repo_id]={"true":true_contrs,"neg":neg_contrs,"ids":ids,"sim":ranks}
            
            line=fin.readline()
        fin.close()
        
        
        
        #shortest path
        G_avr_sh_p=0
        num_p=0
        sh_path={}
        fin=open(path+"G_shortest_path.txt","r")
        line=fin.readline()
        while line:
            data=line.split(" ")
            i=data[0]+","+data[1]
            sh=int(data[3])
            #print(i,sh)
            sh_path[i]=sh
            G_avr_sh_p+=sh
            num_p+=1
            #sys.exit(0)
            line=fin.readline()
        fin.close()
        print("G avr sh path="+str(G_avr_sh_p/num_p))
 
        
        fin=open(path +"G_properties.txt","r",encoding="utf-8")
        self.N=int(fin.readline().split("=")[1])
        self.NumRepos=int(fin.readline().split("=")[1].split(" ")[0])
        self.NumExperts=int(fin.readline().split("=")[1].split(" ")[0])
        fin.close() 
        
        total_avr_sh_path=[]
        fouts=open(path+"st_co/"+contrs_ranking_results[:-4]+"_teams_teamsize"+str(team_size)+".txt","w")
        GM=[]
        PN=[]
        MSC=0
        total_c_sh_path=[]
        total_bc_sh_path=[]
        ndcg=[]
        MAP=[]
        for test_id in test_conts_ranks.keys():
            avg_sh=0
            num_path=0
            test=test_conts_ranks[test_id]

            res=str(test_id)
            
            true_cntrs=test["true"]  #true team            
            hat_cntrs=test["ids"][:team_size]  #discoverd team with size team_size
            
            res+=","+" ".join([str(c) for c in hat_cntrs]).strip()
            
            fouts.write(res+"\n")
            fouts.flush()
            
            #compute average shortest path
            avg_sh_c,num_path_c=0,0
            
            hat_cntrs=list(set(hat_cntrs))
            match=len(set(hat_cntrs).intersection(set(true_cntrs)))/len(true_cntrs)
            GM.append(match)
            
            if match >0:
                MSC+=1
            
            match=len(set(hat_cntrs).intersection(set(true_cntrs)))/len(hat_cntrs)
            PN.append(match)
            
            true_ranks=test_statistics[test_id]["adds"]
            true_ranks=np.array(true_ranks)
            true_ranks=(true_ranks/true_ranks.sum())*15+4
            
            sorted_c_scores=np.array(test["sim"])
            true_scores=np.zeros(len(test["ids"]))
            hat_score=[]
            for ii in range(len(true_cntrs)):
                eid=true_cntrs[ii]
                scoree=true_ranks[ii]
                eindex=test["ids"].index(eid)
                true_scores[eindex]=scoree
                hat_score.append(sorted_c_scores[eindex])
            #print(true_scores)
            
            hat_score=np.array(hat_score)
            if min(hat_score)<0:
                hat_score=hat_score-min(hat_score) 
                
#             print(ndcg_score([hat_score], [true_ranks], k=team_size))
            if min(sorted_c_scores)<0:
                sorted_c_scores=sorted_c_scores-min(sorted_c_scores) 
                        
            
            #ndcg.append(ndcg_score([true_scores], [sorted_c_scores],  k=team_size))
            ndcg.append(ndcg_score([true_ranks], [hat_score], k=team_size))
            
            sorted_true_scores,sorted_true_ids=(list(t) for t in zip(*sorted(zip(true_ranks, true_cntrs),reverse=True)) )
            mapatk=ml_metrics.mapk([sorted_true_ids], [hat_cntrs],k=team_size)
            MAP.append(mapatk)
            hat_cntrs.sort()
            for i in range(len(hat_cntrs)-1):
                for j in range(i+1,len(hat_cntrs)):
                    ci=hat_cntrs[i]
                    cj=hat_cntrs[j]
                    sh_p=sh_path[str(ci)+","+str(cj)]
                    avg_sh_c+=sh_p
                    num_path_c+=1
            
            if num_path_c==0:
                total_bc_sh_path.append(0)
            else:
                total_bc_sh_path.append(avg_sh_c/num_path_c)
                    
            t_c_shp=0
            for j in range(len(hat_cntrs)):
                    ri=test_id
                    cj=hat_cntrs[j]
                    if int(cj)>int(ri):
                        ind=str(ri)+","+str(cj)
                    else:
                        ind=str(cj)+","+str(ri)
                    
                    t_c_shp+=sh_path[ind]
            
            total_c_sh_path.append(t_c_shp/len(hat_cntrs))
            
        
        fouts.close()
       
        total_c_sh_path=np.array(total_c_sh_path)
        
        GM=np.array(GM)
        print("mean GM="+str(np.mean(GM)))
        
        PN=np.array(PN)
        print("mean PN="+str(np.mean(PN)))
        
        MSC=MSC/len(test_conts_ranks.keys())
        
        print("MSC=",MSC)
        
        av_ndcg=np.mean(ndcg)
        
        print("ndcg=",av_ndcg)
        av_map=np.mean(np.array(MAP))
        print("map=",av_map)
        
        print("\n\nmean shortest path from test repo to selected es="+str(np.mean(total_c_sh_path)))
        print("var shortest path from test repo to selected es="+str(np.var(total_c_sh_path)))
        print("Standard Deviation shortest path from test repo to selected es="+str(np.std(total_c_sh_path)))
        
        
        print("\n\nmean shortest path between es="+str(np.mean(total_bc_sh_path)))
        print("var shortest path between es="+str(np.var(total_bc_sh_path)))
        print("Standard Deviation shortest path between es="+str(np.std(total_bc_sh_path)))
        
        print("dicovering teams done!")
        return round(np.mean(GM),3),round(np.mean(PN),3),round(
            MSC,3),round(np.mean(total_c_sh_path),3), round(np.mean(total_bc_sh_path),3),round(av_ndcg,3),round(av_map,3)

dataset=["CNCF","Java","ML","NE"] 
data=dataset[3]

trian=True
print("Start")
if trian==True:
    ob=CoExperts(data)
    ob.train("a",0.0005,5e-6,500,15) 
else:  
    CoExperts.test_all(data,"/Data_LDA_Topics/","pickle_model.pkl_i",str(1))    
print("Done!")
        
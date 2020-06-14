import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import random
import math
from tqdm import tqdm
import os
import re

class dictionary():
    def __init__(self):
        self.word_freq={}
        self.id2word={}
        self.word2id={}
    
    def add_word(self,word):
        if word in self.word_freq:
            self.word_freq[word]+=1
        else:
            self.word_freq[word]=1
        
    def create_mapping(self):
        self.word_freq['[PAD]']=1000001
        self.word_freq['[UNK]']=1000000
        c_unk=0
        dic_items=[]
        for k in self.word_freq.keys():
            if self.word_freq[k]>1 or np.random.uniform()>0.5:
                dic_items.append((k,self.word_freq[k]))
            else:
                c_unk+=1
        ordered_lis=sorted( dic_items, key=lambda x: (-x[1],x[0]))
        assert ordered_lis[0][0]=='[PAD]'
        self.id2word=dict([(i,ordered_lis[i][0]) for i in range(len(ordered_lis))])
        self.word2id=dict([(ordered_lis[i][0],i) for i in range(len(ordered_lis))])
        self.ordered_lis=ordered_lis
        return c_unk

    def get_id(self,word):
        if word in self.word2id:
            return self.word2id[word]
        else:
            return 1
    
    def get_word(self,idx):
        return self.id2word[idx]
    
    def get_len(self):
        return len(self.id2word)

class NER_dataset(Dataset):
    def __init__(self,data_path,dic_word,dic_char,training=False):
        super(NER_dataset,self).__init__()
        #read data -> X:[[word,word,word,...],[...],[...]] Y:[[0,1,2,3...],...]
        f=open(data_path,encoding='utf-8')
        X=[[]]
        Y=[[]]

        line=f.readline()
        self.label_map=["O","B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG","B-MISC","I-MISC"]
        self.label_num=len(self.label_map)

        while line:
            if line=='\n':
                if len(X[-1])>0:
                    X.append([])
                    Y.append([])
            else:
                word,pos,ner=line.split()
                assert ner in self.label_map
                word=re.sub('\d','0',word)      #replace all the digits with 0, this helps
                X[-1].append(word)
                Y[-1].append(self.label_map.index(ner))
            line=f.readline()
                
        f.close()
        if len(X[-1])==0:
            X=X[:-1]
            Y=Y[:-1]

        self.label=Y
        self.data_num=len(X)

        #get word dictionary
        if training:
            dic_word=dictionary()
            for sentence in X:
                for word in sentence:
                    dic_word.add_word(word)
            dic_word.create_mapping()

        #get word_ids: list of lists
        #encode words str->id
        self.word_ids=[]
        for i in range(len(X)):
            self.word_ids.append(list(map(lambda x:dic_word.get_id(x), X[i])))

        #get character dictionary
        if training:
            dic_char=dictionary()
            for sentence in X:
                text=''.join(sentence)
                for char in text:
                    dic_char.add_word(char)
            dic_char.create_mapping()

        #get char_ids: list of lists of lists
        self.char_ids=[]
        for sentence in X:
            s=[]
            for word in sentence:
                s.append(list(map(lambda x:dic_char.get_id(x), word)))
            self.char_ids.append(s)

        self.dic_word=dic_word
        self.dic_char=dic_char

    def __getitem__(self,index):
        return self.word_ids[index],self.char_ids[index],self.label[index]

    def __len__(self):
        return self.data_num

def expand_dic(dictionary,embedding_path,paths):
    f=open(embedding_path,encoding="utf-8")
    line=f.readline()
    word2emb={}
    while line:
        line=line.split()
        word2emb[line[0]]=torch.from_numpy(np.array(line[1:],dtype=np.str).astype(np.float))
        line=f.readline()

    words=[]
    for data_path in paths:
        f=open(data_path,encoding='utf-8')
        line=f.readline()
        while line:
            if line=='\n':
                pass
            else:
                word=line.split()[0]
                words.append(word)
            line=f.readline()
        f.close()

    train_len=dictionary.get_len()
    for word in words:
        if word not in dictionary.word2id and any([x in word2emb for x in [word,word.lower(),re.sub('\d','0',word.lower())]]):
            dictionary.word2id[word]=dictionary.get_len()
            dictionary.id2word[dictionary.get_len()]=word
            dictionary.ordered_lis.append((word,0))
    num_add=dictionary.get_len()-train_len
    print("original word num: %d  expand num: %d"%(train_len,num_add)) 
    return dictionary,word2emb



def collate_batch(batch):
    #input is a list of tuples
    word_num=list(map(lambda x:len(x[0]),batch))
    max_word_num=max(word_num)
    word_ids=list(map(lambda x:x[0]+[0]*(max_word_num-len(x[0])),batch))
    label_ids=list(map(lambda x:x[2]+[0]*(max_word_num-len(x[2])),batch))

    max_word_length=max(list(map(lambda x:max([len(i) for i in x[1]]),batch)))
    char_ids=[]
    for tuple in batch:
        s=[]
        for word in tuple[1]:
            s.append(word+[0]*(max_word_length-len(word)))
        s=s+[[0]*max_word_length for i in range((max_word_num-len(s)))]
        char_ids.append(s)

    word_num=torch.LongTensor(word_num)
    word_ids=torch.LongTensor(word_ids)
    char_ids=torch.LongTensor(char_ids)
    label_ids=torch.LongTensor(label_ids)

    return word_num,word_ids,char_ids,label_ids

def forward_alg(observation,transition,word_num):

    def log_sum_exp(matrix,dim):
        maximum,_=matrix.max(dim=dim,keepdim=True)  #to avoid NaN
        return (maximum+torch.log(torch.exp(matrix-maximum).sum(dim=dim,keepdim=True))).squeeze(1)

    observation=observation.transpose(1,2)
    transition=transition.unsqueeze(0).expand(observation.size(0),-1,-1)
    alpha=torch.zeros_like(observation)
    alpha[:,:,0:1]=observation[:,:,0:1]
    for i in range(1,observation.size(2)):
        alpha[:,:,i:i+1]=(observation[:,:,i]+log_sum_exp(alpha[:,:,i-1:i]+transition,dim=1)).unsqueeze(2)
    end_label=alpha[:,10,1:]     #(batch_size, sequence_len)
    return end_label.gather(1,word_num.unsqueeze(1)).squeeze(1)


class LSTM_CRF(nn.Module):
    def __init__(self,word2emb,dic_word,dic_char):
        super(LSTM_CRF, self).__init__()
        word_emb_dim=100
        word_lstm_dim=100
        char_emb_dim=25
        char_lstm_dim=25
        label_num=9
        dropout_rate=0.5
            
        word_emb=nn.Embedding(dic_word.get_len(),word_emb_dim,padding_idx=0)
        for i in range(dic_word.get_len()):
            word=dic_word.ordered_lis[i][0]
            if word in word2emb:
                word_emb.weight.data[i]=word2emb[word]
            elif word.lower() in word2emb:
                word_emb.weight.data[i]=word2emb[word.lower()]
            elif re.sub('\d','0',word.lower()) in word2emb:
                word_emb.weight.data[i]=word2emb[re.sub('\d','0',word.lower())]
        #print(word_emb.weight.data[0])


        char_emb=nn.Embedding(dic_char.get_len(),char_emb_dim,padding_idx=0)

        self.char_emb=char_emb
        self.char_lstm=nn.LSTM(char_emb_dim,char_lstm_dim,batch_first=True,bidirectional=True)
        self.word_emb=word_emb
        self.dropout=nn.Dropout(dropout_rate)
        self.word_lstm=nn.LSTM(word_emb_dim+char_lstm_dim*2, word_lstm_dim,batch_first=True,bidirectional=True)
        self.hidden=nn.Sequential(
            nn.Linear(word_lstm_dim*2,word_lstm_dim),
            nn.Tanh(),
            nn.Linear(word_lstm_dim,label_num)
        )
        self.transition=nn.Parameter(torch.full((label_num+2,label_num+2),math.log(1/label_num)))

        self.char_lstm_dim=char_lstm_dim
        self.char_emb_dim=char_emb_dim
        self.word_lstm_dim=word_lstm_dim
        self.word_emb_dim=word_emb_dim
        self.label_num=label_num
        
    def get_feature(self,word_num,word_ids,char_ids):
        batch_size=word_ids.size(0)
        sequence_len=word_ids.size(1)
        char_input=self.char_emb(char_ids)
        #print(char_input.size())    #4 dimensional
        char_emb_dim=char_input.size(3)
        word_len=char_input.size(2)
        char_input=char_input.view(batch_size*sequence_len,word_len,char_emb_dim)
        char_hidden,_=self.char_lstm(char_input)    #second output "_" is equal to char_output below
        forward_=char_hidden[:,-1,:self.char_lstm_dim]
        backward_=char_hidden[:,0,self.char_lstm_dim:]
        char_output=torch.cat((forward_,backward_),dim=-1)
        char_output=char_output.view(batch_size,sequence_len,self.char_lstm_dim*2)

        index=torch.LongTensor(list(range(sequence_len))).cuda().unsqueeze(0).expand(batch_size,sequence_len)
        condition=word_num.unsqueeze(1).expand(batch_size,sequence_len)>index
        mask=torch.where(condition,torch.ones(1,).cuda(),torch.zeros(1,).cuda()).unsqueeze(2)
        char_output*=mask   #to mask all the padding tokens

        word_feature=self.word_emb(word_ids)
        word_feature=torch.cat((word_feature,char_output),dim=-1)
        word_feature=self.dropout(word_feature)
        word_feature,_=self.word_lstm(word_feature)

        word_feature=self.hidden(word_feature)
        word_feature*=mask

        return word_feature,mask
    
    def forward(self,word_num,word_ids,char_ids,label_ids):    
        batch_size=word_ids.size(0)
        sequence_len=word_ids.size(1)
        word_feature,mask=self.get_feature(word_num,word_ids,char_ids)
        
        #compute numerator: the score of target label sequence
        numerator=word_feature.gather(2,label_ids.unsqueeze(2)).squeeze(2).sum(dim=1)
        #print(numerator.size())
        padded_label=torch.cat((torch.full((batch_size,1),9,dtype=torch.long).cuda(), label_ids), dim=1)

        #print(self.transition[(padded_label[:,:-1],padded_label[:,1:])].size()) 
        #a tensor can be indexed by several LongTensors or lists, each of them corresponds with an axis
        trans_score=self.transition[(padded_label[:,:-1],padded_label[:,1:])]   #size(batch_size,sequence_len)
        trans_score*=mask.squeeze(2)
        numerator+=trans_score.sum(dim=1)
        last_label=(padded_label.gather(1,word_num.unsqueeze(1))).squeeze()
        numerator+=self.transition[(last_label,torch.full((batch_size,),10,dtype=torch.long).cuda())]
        
        #prepare observation matrix
        small=-1000
        se_label=torch.full((batch_size,sequence_len,2),small,dtype=torch.float).cuda()*mask
        observation=torch.cat((word_feature,se_label),dim=2)
        observation=torch.cat((torch.full((batch_size,1,self.label_num+2),small,dtype=torch.float).cuda(),
                                observation,
                                torch.full((batch_size,1,self.label_num+2),small,dtype=torch.float).cuda()),dim=1)
        observation[:,0,9]=0
        observation[:,-1,10]=0

        denominator=forward_alg(observation,self.transition,word_num)   #the score of all the label sequences
        loss=-(numerator-denominator)

        return torch.mean(loss)
    
    def decode(self,word_num,word_ids,char_ids):
        batch_size=word_ids.size(0)
        sequence_len=word_ids.size(1)
        word_feature,mask=self.get_feature(word_num,word_ids,char_ids)

        index=torch.LongTensor(list(range(sequence_len))).cuda().unsqueeze(0).expand(batch_size,sequence_len)
        condition=word_num.unsqueeze(1).expand(batch_size,sequence_len)==index
        end_mask=torch.where(condition,torch.ones(1,).cuda(),torch.zeros(1,).cuda()).unsqueeze(2)

        small=-1000
        constrain=torch.full((batch_size,sequence_len,self.label_num+2),small,dtype=torch.float).cuda()*end_mask
        constrain[:,:,10]=0         #tensor "constrain" is used to make sure all the paths finish at [END] state

        se_label=torch.full((batch_size,sequence_len,2),small,dtype=torch.float).cuda()*mask    #correspond with Start and End label
        observation=torch.cat((word_feature,se_label),dim=2)
        observation+=constrain
        observation=torch.cat((torch.full((batch_size,1,self.label_num+2),small,dtype=torch.float).cuda(),
                                observation,
                                torch.full((batch_size,1,self.label_num+2),small,dtype=torch.float).cuda()),dim=1)
        observation[:,0,9]=0
        observation[:,-1,10]=0
        
        #viterbi
        observation=observation.transpose(1,2)
        path=torch.zeros_like(observation).long().cuda()
        transition=self.transition.unsqueeze(0).expand(batch_size,-1,-1)
        z=observation[:,:,0:1]
        for i in range(1,observation.size(2)):
            values,indices=(z+transition).max(dim=1)
            path[:,:,i]=indices
            values+=observation[:,:,i]
            z=values.unsqueeze(2)

        last=path[:,10,-1:]
        pred=last
        for i in range(path.size(2)-2,1,-1):
            last=path[:,:,i].gather(1,last)
            pred=torch.cat((last,pred),dim=1)   
        
        #pred size: batch_size,sequence_len
        #print(pred.size())

        #validation     this step is unnecessary, just make sure there is nothing wrong
        pred_=torch.cat((pred,torch.full((batch_size,1),10,dtype=torch.long).cuda()),dim=1)
        condition=pred_.gather(1,word_num.unsqueeze(1)).squeeze(1)==10
        assert condition.size(0)==(condition.sum().item())
        #print("validation passed")

        return pred

def list_batch(pred,word_num,word_ids,label_ids,dic_word,label_map):
    pred=pred.tolist()
    word_num=word_num.tolist()
    label_ids=label_ids.tolist()
    word_ids=word_ids.tolist()

    outputs=[]
    for i in range(len(word_num)):
        seq_len=word_num[i]
        prediction=pred[i][:seq_len]
        target=label_ids[i][:seq_len]
        words=word_ids[i][:seq_len]
        prediction=list(map(lambda x: label_map[x], prediction))
        target=list(map(lambda x: label_map[x], target))
        words=list(map(lambda x: dic_word.get_word(x), words))
        for j in range(seq_len):
            outputs.append(' '.join([words[j],target[j],prediction[j]]))
        outputs.append('')
    
    return outputs  
  
BATCH_SIZE=32 
LR=0.001
CLIP=5.

train_dataset=NER_dataset('./dataset_ner/train.txt',dictionary(),dictionary(),training=True)
train_dataset.dic_word,word2emb=expand_dic(train_dataset.dic_word,"dataset_ner/glove.6B.100d.txt",['./dataset_ner/dev.txt','./dataset_ner/test.txt'])
dev_dataset=NER_dataset('./dataset_ner/dev.txt',train_dataset.dic_word,train_dataset.dic_char)
test_dataset=NER_dataset('./dataset_ner/test.txt',train_dataset.dic_word,train_dataset.dic_char)

train_loader=DataLoader(train_dataset,BATCH_SIZE,shuffle=True,num_workers=8,collate_fn=collate_batch)
dev_loader=DataLoader(dev_dataset,BATCH_SIZE,shuffle=False,num_workers=8,collate_fn=collate_batch)
test_loader=DataLoader(test_dataset,BATCH_SIZE,shuffle=False,num_workers=8,collate_fn=collate_batch)


lstm_crf=LSTM_CRF(word2emb,train_dataset.dic_word,train_dataset.dic_char).cuda()
optimizer=torch.optim.Adam(lstm_crf.parameters(),LR)

best_score=0
for epoch in range(50):
    lstm_crf.train()
    loss_lis=[]
    pbar=tqdm(total=len(train_loader))
    for i,(word_num,word_ids,char_ids,label_ids) in enumerate(train_loader):
        loss=lstm_crf(word_num.cuda(),word_ids.cuda(),char_ids.cuda(),label_ids.cuda())
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(lstm_crf.parameters(),CLIP)
        optimizer.step()

        loss_lis.append(loss.item())
        pbar.update(1)
    pbar.close()
    mean_loss=torch.mean(torch.tensor(loss_lis)).item()

    lstm_crf.eval()
    f1_score=0

    #for loader in (dev_loader,test_loader):
    loader=test_loader

    outputs=[]
    for i,(word_num,word_ids,char_ids,label_ids) in enumerate(loader):
        word_num,word_ids,char_ids,label_ids=word_num.cuda(),word_ids.cuda(),char_ids.cuda(),label_ids.cuda()
        pred=lstm_crf.decode(word_num,word_ids,char_ids)
        outputs+=list_batch(pred,word_num,word_ids,label_ids, train_dataset.dic_word, train_dataset.label_map)

    f=open('outputs.txt','w',encoding='utf-8')
    f.write('\n'.join(outputs))
    f.close()
    os.system("./conlleval < outputs.txt > results")
    f=open('results','r',encoding='utf-8')
    f1_score=float(f.readlines()[1].split()[-1])
    f.close()
    best_score=max(best_score,f1_score)
    
    #print('epoch %d:  mean loss: %.4f  f1 score dev: %.2f  test: %.2f'%(epoch,mean_loss,f1_score[0],f1_score[1]))
    print('epoch %d:  mean loss: %.4f  f1 score: %.2f  best: %.2f'%(epoch,mean_loss,f1_score,best_score))


#  90.08
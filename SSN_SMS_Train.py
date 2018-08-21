#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:33:46 2018

@author: ssn-9
"""

    
def file_read(arg1):
    import glob   
    import nltk
    import itertools
    path = arg1+'/*.txt'   
    files=glob.glob(path)   
    fw=open('Temp/bow.txt','w')
    bow = []
    for file in files: 
        #print (file)
        f=open(file,'r',encoding='utf-8')
        content=f.read().splitlines()
    #print content
    #bow = []
        for sentence in content:
            sentence = sentence.replace('.',' ')
            sentence = sentence.replace(',',' ')
            sentence = sentence.replace('?',' ')
            sentence = sentence.replace(';',' ')
            sentence = sentence.replace(':',' ')
            sentence = sentence.replace('=',' ')
            sentence = sentence.replace('*',' ')
            words = nltk.word_tokenize(str(sentence))
            if (words!='?'):
                bow.append(words)
             
       
    
    #print (bow)
    bow_full=list(itertools.chain.from_iterable(bow))
    #bow_full.remove('?')
    val='?'
    while val in bow_full:
            bow_full.remove(val)
    #print (bow_full)
   
    #print (len(bow_full))
    bow = list(set(bow_full))
    newbow = []
    for bows in bow:
        newbow.append(bows.lower())
    #print(newbow)
    #print (bow)
    newbow=set(newbow)
    #print (len(newbow))
    fea=' '.join(newbow)     #31290 features   #23956
    fw.write(fea)
    fw.close()
    f.close()
    #return newbow
    

def file_read_all(arg1):
    import nltk
    import itertools
    
    fw=open('Temp/feature.txt','w')
    #fw.write('Text'+'\t'+'FileNo'+'\n')
    for i in range(1,351):
        infileno=format(i, '03d')
        infilename=arg1+'/author_id_'+infileno+'.txt'
        f=open(infilename,'r')
        content=f.read().splitlines()
        #print content
                
        bow = []
        for sentence in content:
            sentence = sentence.replace('.',' ')
            sentence = sentence.replace(',',' ')
            sentence = sentence.replace('?',' ')
            sentence = sentence.replace(';',' ')
            sentence = sentence.replace(':',' ')
            sentence = sentence.replace('=',' ')
            sentence = sentence.replace('*',' ')
            words = nltk.word_tokenize(str(sentence))
            
            if (words!='?'):
                bow.append(words)
    
        bow_full=list(itertools.chain.from_iterable(bow))
        #bow_full.remove('?')
        bow = list(set(bow_full))
        newbow = []
        for bows in bow:
            newbow.append(bows.lower())
        #print(newbow)
        #print (bow)
        #print (len(newbow))
        fea=' '.join(newbow)
        #fea.replace(",", " ")
        fw.write(fea)
        #fw.write('\t'+str(infileno))
        fw.write('\n')
        f.close()
    fw.close()
    
    
  
def chisquare(arg1):
    import nltk
    i=1
    a=b=c=d=0
    ea=eb=ec=ed=0
    n=0
    
    import pandas as pd
    filename=arg1+'/Truth.csv'
    gender = pd.read_csv(filename, usecols=['Gender'])
    #age = pd.read_csv('Truth.csv', usecols=['Age Group'])
    
    gen=list(gender['Gender'])
    #ag=list(age['Age Group'])
    
    fw=open('Temp/bowgen.txt','w')
    
    fr=open('Temp/feature.txt','r')
    content=fr.read().splitlines()
    
    f=open('Temp/bow.txt','r')
    bow=f.read().splitlines()
    bow = bow[0].replace('.', ' ')
    #print (bow)
    
    bow_gen=[]
    words = nltk.word_tokenize(str(bow))
    #n = len(words)
    for x in words:
        a=b=c=d=0
        ea=eb=ec=ed=0
        for y in range(0,350):
            
            if (x in content[y]) and (gen[y]=='male'):
                a=a+1
            elif (x in content[y]) and (gen[y]=='female'):
                b=b+1
            elif (x not in content[y]) and (gen[y]=='male'):
                c=c+1
            elif (x not in content[y]) and (gen[y]=='female'):
                d=d+1
            
        n=a+b+c+d
        if(a==0 or b==0 or c==0 or d==0):
            chi=0;
        else:
            ea=((a+b)*(a+c))/n;
            eb=((a+b)*(b+d))/n;
            ec=((a+c)*(c+d))/n;
            ed=((b+d)*(c+d))/n;
            
            if(ea==0 or eb==0 or ec==0 or ed==0):
                chi=0
            else:
                chi = (((a-ea)*(a-ea))/ea) + (((b-eb)*(b-eb))/eb) + (((c-ec)*(c-ec))/ec) + (((d-ed)*(d-ed))/ed)
        
        
        
        i=i+1
        #print (i, x, chi)
        #if  (chi>=3.841):    # 2869 features
        if  (chi>=2.706):     # 4233 features   #2256  #1343
            #print (i, x, chi)
            bow_gen.append(x)
            
    #print (bow_gen)
    #print (len(bow_gen))
    #print (n)
    fea=' '.join(bow_gen)
    fw.write(fea)
    fw.close()
    
def fea_vec():                   # Feature vector with Chi Square
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    #from sklearn.feature_extraction.text import TfidfVectorizer
    fr=open('Temp/bowgen.txt','r')
    content=fr.read().splitlines()
    words = nltk.word_tokenize(str(content[0]))
    words=set(words)
    #print (len(words))
    
    
    fr=open('Temp/feature.txt','r')
    content=fr.read().splitlines()
    word_vec = content
    cv = CountVectorizer(vocabulary=words)
    #gen_vec=cv.fit_transform(word_vec).toarray()
    gen_vec=cv.transform(word_vec).toarray()
    #print (cv)
    #print (gen_vec.shape)
       
    np.savetxt('Temp/feavec.csv',gen_vec, delimiter=',')
    

    

def gen_model(arg1, arg2):
    
    from sklearn.naive_bayes import MultinomialNB 
    from sklearn.neural_network import MLPClassifier
    
    import pandas as pd
    import numpy as np
    
    print ("Building Model for Gender\n")
    
    filename = arg1+'/Truth.csv'
    gender = pd.read_csv(filename, usecols=['Gender'])
        
    gen_target=list(gender['Gender'])
    
    #print (gen_target)
    
    gen_train = np.genfromtxt('Temp/feavec.csv', delimiter=',')
    
    
    MNBclf = MultinomialNB().fit(gen_train, gen_target)
    MLPclf = MLPClassifier().fit(gen_train, gen_target)
    
    import pickle
    
    modelname1=arg2+"/gender_model_run2.pkl"
    modelname2=arg2+"/gender_model_run1.pkl"
    # save the classifier
    with open(modelname1, 'wb') as gm1:
        pickle.dump(MNBclf, gm1)    
    
    with open(modelname2, 'wb') as gm2:
        pickle.dump(MLPclf, gm2)    

    gm1.close()
    gm2.close()
    
    #return MNBclf,MLPclf

  

def chisquare_age(arg1):
    import nltk
    i=1
    a=b=c=d=e=f=0
    ea=eb=ec=ed=ee=ef=0
    n=0
    
    import pandas as pd
    #gender = pd.read_csv('Truth.csv', usecols=['Gender'])
    filename = arg1+ '/Truth.csv'
    age = pd.read_csv(filename, usecols=['Age Group'])
    
    #gen=list(gender['Gender'])
    ag=list(age['Age Group'])
    
    fw=open('Temp/bowage.txt','w')
    
    fr=open('Temp/feature.txt','r')
    content=fr.read().splitlines()
    
    f=open('Temp/bow.txt','r')
    bow=f.read().splitlines()
    bow = bow[0].replace('.', ' ')
    #print (bow)
    
    bow_age=[]
    words = nltk.word_tokenize(str(bow))
    #n = len(words)
    for x in words:
        a=b=c=d=e=f=0
        ea=eb=ec=ed=ee=ef=0
        for y in range(0,350):
            
            if (x in content[y]) and (ag[y]== '15-19'):
                a=a+1
            elif (x in content[y]) and (ag[y]=='20-24'):
                b=b+1
            elif (x in content[y]) and (ag[y]=='25-xx'):
                e=e+1
            elif (x not in content[y]) and (ag[y]=='15-19'):
                c=c+1
            elif (x not in content[y]) and (ag[y]=='20-24'):
                d=d+1
            elif (x not in content[y]) and (ag[y]=='25-xx'):
                f=f+1
            
        n=a+b+c+d+e+f
        if(a==0 or b==0 or c==0 or d==0 or e==0 or f==0):
            chi=0;
        else:
           
            ea=((a+b+e)*(a+c))/n;
            eb=((a+b+e)*(b+d))/n;
            ec=((a+c)*(c+d+f))/n;
            ed=((b+d)*(c+d+f))/n;
            ee=((a+b+e)*(e+f))/n;
            ef=((e+f)*(c+d+f))/n;
            
            if(ea==0 or eb==0 or ec==0 or ed==0 or ee==0 or ef==0):
                chi=0
            else:
                chi = (((a-ea)*(a-ea))/ea) + (((b-eb)*(b-eb))/eb) + (((c-ec)*(c-ec))/ec) + (((d-ed)*(d-ed))/ed) + (((e-ee)*(e-ee))/ee) + (((f-ef)*(f-ef))/ef)
        
        
        
        i=i+1
        #print (i, x, chi)
        #if  (chi>=3.841):    # 2869 features
        if  (chi>=4.605):     # 1091
            #print (i, x, chi)
            bow_age.append(x)
            
    #print (bow_age)
    #print (len(bow_age))
    #print (n)
    fea=' '.join(bow_age)
    fw.write(fea)
    fw.close()
    
def fea_vec_age():                   # Feature vector with Chi Square for AGE
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    #from sklearn.feature_extraction.text import TfidfVectorizer
    fr=open('Temp/bowage.txt','r')
    content=fr.read().splitlines()
    words = nltk.word_tokenize(str(content[0]))
    words=set(words)
    #print (len(words))
        
    fr=open('Temp/feature.txt','r')
    content=fr.read().splitlines()
    word_vec = content
    cv = CountVectorizer(vocabulary=words)
    #gen_vec=cv.fit_transform(word_vec).toarray()
    gen_vec=cv.transform(word_vec).toarray()
    #print (cv)
    #print (gen_vec.shape)
       
    np.savetxt('Temp/feavecage.csv',gen_vec, delimiter=',')
    


    

def age_model(arg1, arg2):
    
    from sklearn.naive_bayes import MultinomialNB 
    from sklearn.neural_network import MLPClassifier
    
    import pandas as pd
    import numpy as np
    
    print ("Building Model for Age\n")
    
    filename = arg1+"/Truth.csv"
    
    age = pd.read_csv(filename, usecols=['Age Group'])
    #age = pd.read_csv('Training/Truth.csv', usecols=['Age Group'])
        
    age_target=list(age['Age Group'])
    
    #print (gen_target)
    
    age_train = np.genfromtxt('Temp/feavecage.csv', delimiter=',')
    
    
    MNBclf = MultinomialNB().fit(age_train, age_target)
    MLPclf = MLPClassifier().fit(age_train, age_target)
    
    import pickle
    
    modelname1=arg2+"/age_model_run2.pkl"
    modelname2=arg2+"/age_model_run1.pkl"
    # save the classifier
    with open(modelname1, 'wb') as gm1:
        pickle.dump(MNBclf, gm1)    
    
    with open(modelname2, 'wb') as gm2:
        pickle.dump(MLPclf, gm2)    

    gm1.close()
    gm2.close()
    
    return MNBclf,MLPclf


      

import sys
program_name = sys.argv[0]
arg1 = sys.argv[1]
arg2 = sys.argv[2]

print (program_name)

file_read(arg1)
file_read_all(arg1)
chisquare(arg1)
fea_vec()


gen_model(arg1, arg2)
#gen_pred()

chisquare_age(arg1)
fea_vec_age()
#fea_vec_test_age()
age_model(arg1, arg2)
#age_pred()
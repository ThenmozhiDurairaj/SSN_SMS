#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:33:46 2018

@author: ssn-9
"""

def file_read_test(arg1):
    import nltk
    import itertools
    #fw=open('feature.csv','w')
    #fw.write('Text'+'\t'+'FileNo'+'\n')
    fw=open('Temp/feature_test.txt','w')
    #fw.write('Text'+'\t'+'FileNo'+'\n')
    for i in range(1,151):
        #print("file ", i)
        testfileno=format(i, '03d')
        infilename=arg1+'/author_id_'+testfileno+'.txt'
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


def fea_vec_test():                   # Feature vector with Chi Square
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    #from sklearn.feature_extraction.text import TfidfVectorizer
    fr=open('Temp/bowgen.txt','r')
    content=fr.read().splitlines()
    words = nltk.word_tokenize(str(content[0]))
    words=set(words)
    #print (len(words))
   
    
    fr=open('Temp/feature_test.txt','r')
    content=fr.read().splitlines()
    word_vec = content
    cv = CountVectorizer(vocabulary=words)
    #gen_vec=cv.fit_transform(word_vec).toarray()
    gen_vec=cv.transform(word_vec).toarray()
    #print (cv)
    #print (gen_vec.shape)
       
    np.savetxt('Temp/feavec_test.csv',gen_vec, delimiter=',')
    

def fea_vec_test_age():                   # Feature vector for age with Chi Square
    import nltk
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    #from sklearn.feature_extraction.text import TfidfVectorizer
    fr=open('Temp/bowage.txt','r')
    content=fr.read().splitlines()
    words = nltk.word_tokenize(str(content[0]))
    words=set(words)
    #print (len(words))
    
    
    fr=open('Temp/feature_test.txt','r')
    content=fr.read().splitlines()
    word_vec = content
    cv = CountVectorizer(vocabulary=words)
    #gen_vec=cv.fit_transform(word_vec).toarray()
    gen_vec=cv.transform(word_vec).toarray()
    #print (cv)
    #print (gen_vec.shape)
       
    np.savetxt('Temp/feavecage_test.csv',gen_vec, delimiter=',')
    
  

def gen_pred(arg1, arg2, arg3):
    
    import numpy as np
    import csv
    #from itertools import izip
    
    resultfile1=arg3+"/Run2_Gender_Result.csv"
    resultfile2=arg3+"/Run1_Gender_Result.csv"
    fw1=open(resultfile1,'w')
    fw2=open(resultfile2,'w')
    writer1 = csv.DictWriter(fw1, fieldnames = ["Test_Author_Profile_Id", "Gender"])
    writer1.writeheader()
    
    writer2 = csv.DictWriter(fw2, fieldnames = ["Test_Author_Profile_Id", "Gender"])
    writer2.writeheader()
    
    testname=[]
    for i in range(1,151):
        
        testid=format(i, '03d')
        tname='Test-Document-'+testid
        testname.append(tname)
    
    #model_NB, model_MLP = gen_model()
    
    import pickle
    
    
    # load it again
    with open('Model/gender_model_run2.pkl', 'rb') as gm1:
        model_NB = pickle.load(gm1)
    
    with open('Model/gender_model_run1.pkl', 'rb') as gm2:
        model_MLP = pickle.load(gm2)
    
    gen_test = np.genfromtxt('Temp/feavec_test.csv', delimiter=',')
    NBout = model_NB.predict(gen_test)
    #print (NBout)
    
    MLPout = model_MLP.predict(gen_test)
    
    for i in range(0,150):
        print (testname[i], NBout[i], MLPout[i], '\n')
        
    
    
    for i,j in zip(testname,NBout):
        fw1.write(str(i)+","+str(j)+"\n")
        
    for i,j in zip(testname,MLPout):
        fw2.write(str(i)+","+str(j)+"\n")
        
    fw1.close()
    fw2.close()

  



def age_pred(arg1, arg2, arg3):
    
    import numpy as np
    import csv
    #from itertools import izip
    
    resultfile1=arg3+"/Run2_Age_Result.csv"
    resultfile2=arg3+"/Run1_Age_Result.csv"
    fw1=open(resultfile1,'w')
    fw2=open(resultfile2,'w')
    
    writer1 = csv.DictWriter(fw1, fieldnames = ["Test_Author_Profile_Id", "Age"])
    writer1.writeheader()
    
    writer2 = csv.DictWriter(fw2, fieldnames = ["Test_Author_Profile_Id", "Age"])
    writer2.writeheader()
    
    testname=[]
    for i in range(1,151):
        
        testid=format(i, '03d')
        tname='Test-Document-'+testid
        testname.append(tname)
    
    #model_NB, model_MLP = age_model()
    
    import pickle
    
    modelname1=arg2+"/age_model_run2.pkl"
    modelname2=arg2+"/age_model_run1.pkl"
    
    # load it again
    with open(modelname1, 'rb') as gm1:
        model_NB = pickle.load(gm1)
    
    with open(modelname2, 'rb') as gm2:
        model_MLP = pickle.load(gm2)
    
    gen_test = np.genfromtxt('Temp/feavecage_test.csv', delimiter=',')
    NBout = model_NB.predict(gen_test)
    print (NBout)
    
    MLPout = model_MLP.predict(gen_test)
    
    for i in range(0,150):
        print (testname[i], NBout[i], MLPout[i], '\n')
        
    
    
    for i,j in zip(testname,NBout):
        fw1.write(str(i)+","+str(j)+"\n")
        
    for i,j in zip(testname,MLPout):
        fw2.write(str(i)+","+str(j)+"\n")
        
    fw1.close()
    fw2.close()
      

import sys
program_name = sys.argv[0]
arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]

print (program_name)

#file_read()
#file_read_all()
#chisquare()
#fea_vec()
file_read_test(arg1)
fea_vec_test()
#gen_model()
gen_pred(arg1, arg2, arg3)

#chisquare_age()
#fea_vec_age()
fea_vec_test_age()
#age_model()
age_pred(arg1, arg2, arg3)
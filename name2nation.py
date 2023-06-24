#read data
import pandas as pd
df_train=pd.read_csv('./data/train.txt',sep='\t',header=None)
df_train.shape
df_train.head()



#clean data
df_train.columns=['name','nation']
df_y=df_train['nation'].value_counts().rename_axis('nation').reset_index(name='counts')
# df_y.columns=['nation','count']
df_y.head()
tmp_df=df_y.loc[df_y.counts<50,:]
varList=tmp_df['nation'].tolist()

df_train.dropna(axis=0, how='any', thresh=None, subset=['nation'], inplace=True)
df_train.shape

for var in varList:
    df_train=df_train.loc[df_train['nation']!=var,:]
df_train['nation'].value_counts()

df_train['name']=df_train['name'].apply(lambda x:x.lower())
df_train['name']=df_train['name'].apply(lambda x:x.replace(' ',''))
df_train['name']=df_train['name'].apply(lambda x:x.replace('▁','  '))
df_train['name']=df_train['name'].apply(lambda x:x.replace('.',''))
df_train.head(10)

numOfClasses=len(pd.unique(df_train['nation']))
print('numOfClasses:'+str(numOfClasses))

# load packages
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn import decomposition, ensemble
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas, xgboost, numpy, string
import pandas as pd

#split data
from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(df_train['name'], df_train['nation'],test_size=0.3,random_state=42, stratify=df_train['nation'])

train_labels = train_y
valid_labels = valid_y
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
encoder.fit(df_train['nation'])
train_y = encoder.transform(train_y)
valid_y = encoder.transform(valid_y)
print(train_x.shape)
print(valid_x.shape)

#encode classes
class_df=pd.DataFrame(encoder.classes_)
class_df=class_df.reset_index()
class_df.columns=['class','label']
# class_df.to_csv('./class2nation.csv',index=True,header=True)
class_df['class']=class_df['class'].apply(lambda x:str(x))

dict1 = dict(zip(class_df['class'],class_df['label']))
test=str(dict1)
with open('class2label.txt','w') as f:
    f.write(test)



#create TfidfVectorizer
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=3000,use_idf=True)
# tfidf_vect_ngram_chars =HashingVectorizer(n_features=5000)
tfidf_vect_ngram_chars.fit(df_train['name'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

idf_index=tfidf_vect_ngram_chars.get_feature_names()
df_idf=pd.DataFrame(tfidf_vect_ngram_chars.idf_,index=idf_index)
df_idf=df_idf.round(8)



#save idf data
df_idf=df_idf.reset_index()
df_idf.columns=['pattern','idf']
# df_idf.to_csv('./forOnline/idf_1022.csv',header=True,index=False,encoding='utf-8')
df_idf

dict_tmp={}
for index, row in df_idf.iterrows():
    dict_tmp[df_idf.loc[index]['pattern']]=df_idf.loc[index]['idf']
test=str(dict_tmp)
with open('idf_dict_8.txt','w') as f:
    f.write(test)

bToken=tfidf_vect_ngram_chars.build_tokenizer()
bAnalyzer=tfidf_vect_ngram_chars.build_analyzer()
print(bToken('nokaj  valbone'))
print(bAnalyzer('nokaj  valbone'))



#save tfidf dictionary
import pandas as pd
df_dict=pd.DataFrame.from_dict(tfidf_vect_ngram_chars.vocabulary_, orient='index',columns=['value'])
# df_dict.to_csv('./forOnline/tfidf_dict_f3000_iter3000_1010.csv',header=True,index=True)

#prepare for online with java scripts
test=str(tfidf_vect_ngram_chars.vocabulary_)
with open('./forOnline/vocabulary.txt','w') as f:
    f.write(test)



#transfer model to pmml format
from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
clf=linear_model.LogisticRegression(solver="lbfgs",multi_class="multinomial",max_iter=3000)
classifier = PMMLPipeline([("classifier", clf)])
# training
classifier.fit(xtrain_tfidf_ngram_chars, train_y) #, sample_weight=classes_weights
print('fit done')


# predict the labels on validation dataset
predictions = classifier.predict(xvalid_tfidf_ngram_chars)
print('predict done')
accuracy=metrics.accuracy_score(predictions, valid_y)
print("lr, CharLevel Vectors: ", accuracy)

save_model_pmml_file='./model/lr_20221010_3000.pmml'
# save
sklearn2pmml(classifier, save_model_pmml_file, with_repr=True)
print('save2pmml done')

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(valid_y, predictions))

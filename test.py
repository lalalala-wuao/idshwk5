
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# train_data=pd.read_csv('./train.txt',header=None,names=['domin','label'])
# for index_train in train_data.index :
def str_num_count(str) :
    count=0
    for i in str :
        if i.isdigit() :
            count+=1
    return count

def label_num(str) :
    if str=='dga' :
        return 1
    if str == 'notdga' :
        return 0

def num_label(num) :
    if num==1 :
        return 'dga'
    if num==0 :
        return 'notdga'
    
def create_csv() :
    train_data=pd.read_csv('./train.txt',header=None,names=['domin','label'])
    # print(train_data.loc[1].values)
    for index_train in train_data.index :
        # print(index_train)
        train_data_list=train_data.loc[index_train].values
        domin_name=train_data_list[0]
        domin_label=label_num(train_data_list[1])
        domin_length=len(domin_name)
        domin_count_num=str_num_count(domin_name)
        domin_entropy_letter=(domin_length-domin_count_num)/domin_length
        
        
        dataframe_domin=pd.DataFrame(
            [[domin_name,domin_length,domin_count_num,domin_entropy_letter,domin_label]]
            # ,columns=['name','length','num_count','entropy_char','label']
            )
        dataframe_domin.to_csv('./train.csv',mode='a',header=False,index=None)

def train() : 
    df=pd.read_csv('./train.csv'
    ,engine='c'
    ,header=None
    ,names=['name','length','num_count','entropy_char','label'])
    X=df.values[:,1:4]
    y=df.values[:,-1]
    # print(X)
    y=y.astype('int')
    # print(y)
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X,y)
    return clf

def test(clf) :
    test_data_df=pd.read_csv('./test.txt')
    # result_data=pd.read_csv('./result.txt')
    for index_test in test_data_df.index :
        domin_name=test_data_df.loc[index_test].values[0]
        domin_length=len(domin_name)
        domin_count_num=str_num_count(domin_name)
        domin_entropy_letter=(domin_length-domin_count_num)/domin_length
        res_pre=num_label(clf.predict([[domin_length,domin_count_num,domin_entropy_letter]])[0])
        with open("result.txt","a") as file :
            file.write(domin_name+','+res_pre+'\n')
        
if __name__=='__main__' :
    create_csv()
    clf=train()
    test(clf)

    
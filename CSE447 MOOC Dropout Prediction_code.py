##read and write dataset
import pandas as pd

##read the datasets
enrollment = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/enrollment_list.csv")
enrollment.head()

useridn = enrollment.groupby("user_id").count()
useridn['user_id']=useridn.index
useridn.columns=['course_num','reng','user_id']
course_num = useridn.drop('reng',1)
feature_data=enrollment.merge(course_num,how='left',on='user_id')


activity = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/activity_log.csv")
activity.head()

problem_freq = activity[activity['event']=='problem'].groupby('enrollment_id').count()
problem_freq['enrollment_id']=problem_freq.index
problem_freq.columns=['problem_freq','reng','enrollment_id']
problem_freq = problem_freq.drop('reng',1)
feature_data=feature_data.merge(problem_freq,how='left',on='enrollment_id')

access_freq = activity[activity['event']=='access'].groupby('enrollment_id').count()
access_freq['enrollment_id']=access_freq.index
access_freq.columns=['access_freq','reng','enrollment_id']
access_freq = access_freq.drop('reng',1)
feature_data=feature_data.merge(access_freq,how='left',on='enrollment_id')

navigate_freq = activity[activity['event']=='navigate'].groupby('enrollment_id').count()
navigate_freq['enrollment_id']=navigate_freq.index
navigate_freq.columns=['navigate_freq','reng','enrollment_id']
navigate_freq = navigate_freq.drop('reng',1)
feature_data=feature_data.merge(navigate_freq,how='left',on='enrollment_id')

discussion_freq = activity[activity['event']=='discussion'].groupby('enrollment_id').count()
discussion_freq['enrollment_id']=discussion_freq.index
discussion_freq.columns=['discussion_freq','reng','enrollment_id']
discussion_freq = discussion_freq.drop('reng',1)
feature_data=feature_data.merge(discussion_freq,how='left',on='enrollment_id')

wiki_freq = activity[activity['event']=='wiki'].groupby('enrollment_id').count()
wiki_freq['enrollment_id']=wiki_freq.index
wiki_freq.columns=['wiki_freq','reng','enrollment_id']
wiki_freq = wiki_freq.drop('reng',1)
feature_data=feature_data.merge(wiki_freq,how='left',on='enrollment_id')

page_close_freq = activity[activity['event']=='page_close'].groupby('enrollment_id').count()
page_close_freq['enrollment_id']=page_close_freq.index
page_close_freq.columns=['page_close_freq','reng','enrollment_id']
page_close_freq = page_close_freq.drop('reng',1)
feature_data=feature_data.merge(page_close_freq,how='left',on='enrollment_id')

video_freq = activity[activity['event']=='video'].groupby('enrollment_id').count()
video_freq['enrollment_id']=video_freq.index
video_freq.columns=['video_freq','reng','enrollment_id']
video_freq = video_freq.drop('reng',1)
feature_data=feature_data.merge(video_freq,how='left',on='enrollment_id')

duration = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/duration.csv")
duration.head()
feature_data=feature_data.merge(duration,how='left',on='enrollment_id')

train = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/train_label.csv")
train.head()
feature_data=feature_data.merge(train,how='left',on='enrollment_id')

courseidn = enrollment.groupby("course_id").count()
courseidn['course_id']=courseidn.index
courseidn.columns=['course_capacity','reng','course_id']
course_capacity=courseidn.drop('reng',1)
feature_data=feature_data.merge(course_capacity,how='left',on='course_id')

course_drop = feature_data[feature_data['dropout_prob']==1].groupby('course_id').count()
course_drop['course_id']=course_drop.index
course_drop.columns=['course_drop','reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12','course_id']
course_drop=course_drop.drop(columns=['reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12'])
feature_data=feature_data.merge(course_drop,how='left',on='course_id')

user_drop = feature_data[feature_data['dropout_prob']==1].groupby('user_id').count()
user_drop['user_id']=user_drop.index
user_drop.columns=['user_drop','reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12','reng13','user_id']
user_drop=user_drop.drop(columns=['reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12','reng13'])
feature_data=feature_data.merge(user_drop,how='left',on='user_id')

feature_data=feature_data.fillna(value=0)

data_use=feature_data.head(72325)
course_capuse = data_use.groupby("course_id").count()
course_capuse['course_id']=course_capuse.index
course_capuse.columns=['course_capuse','reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12','reng13','reng14','course_id']
course_capuse=course_capuse.drop(columns=['reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12','reng13','reng14'])

user_capuse = data_use.groupby("user_id").count()
user_capuse['user_id']=user_capuse.index
user_capuse.columns=['user_capuse','reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12','reng13','reng14','user_id']
user_capuse=user_capuse.drop(columns=['reng1','reng2','reng3','reng4','reng5','reng6','reng7','reng8','reng9','reng10','reng11','reng12','reng13','reng14'])

feature_data['problem_dens']=1000*feature_data['problem_freq']/feature_data['duration']
feature_data['access_dens']=1000*feature_data['access_freq']/feature_data['duration']
feature_data['navigate_dens']=1000*feature_data['navigate_freq']/feature_data['duration']
feature_data['discussion_dens']=1000*feature_data['discussion_freq']/feature_data['duration']
feature_data['wiki_dens']=1000*feature_data['wiki_freq']/feature_data['duration']
feature_data['page_close_dens']=1000*feature_data['page_close_freq']/feature_data['duration']
feature_data['video_dens']=1000*feature_data['video_freq']/feature_data['duration']

course_droprate=pd.DataFrame(course_drop['course_drop']/course_capuse['course_capuse'])
course_droprate['course_id']=course_droprate.index
course_droprate.columns=['course_droprate','course_id']
feature_data=feature_data.merge(course_droprate,how='left',on='course_id')

user_droprate=pd.DataFrame(user_drop['user_drop']/user_capuse['user_capuse'])
user_droprate['user_id']=user_droprate.index
user_droprate.columns=['user_droprate','user_id']
user_droprate=user_droprate.fillna(value=0)
feature_data=feature_data.merge(user_droprate,how='left',on='user_id')

feature_data["course_id"] = feature_data["course_id"].astype("category")
feature_data["course_id"].cat.set_categories(list(courseidn.index),inplace=True)
feature_data["user_id"] = feature_data["user_id"].astype("category")
feature_data["user_id"].cat.set_categories(list(useridn.index),inplace=True)

feature_data=feature_data.drop(columns=['course_drop','user_drop','dropout_prob',])
feature_data=feature_data=feature_data.merge(train,how='left',on='enrollment_id')

data_use=feature_data.head(72325)
data_predict=feature_data.tail(48217)
feature_data.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/feature_data.csv", sep=',')
data_use.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_use.csv", sep=',')
data_predict.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_predict.csv", sep=',')



#######################without ratio


########Build Model
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

data_use=pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_use.csv")
index=rd.sample(range(1,72325), 43395)
# print(index)
X_use=np.array(data_use.iloc[:,[3,4,5,6,7,8,10,11,13,14,18,20]])
Y_use=np.array(data_use.iloc[:,-1])
data_xy=data_use.iloc[index,:]
X=X_use[index,:]
Y=Y_use[index]
X_test=np.delete(X_use, index, axis=0)
Y_test=np.delete(Y_use, index, axis=0)

##Group CV
groups = data_xy.iloc[:,-2]
gkf = GroupKFold(n_splits=7)
for train_index, test_index in gkf.split(X, Y, groups):
    X_train = X[train_index]
    X_validate = X[test_index]
    Y_train = Y[train_index]
    Y_validate = Y[test_index]
print(X_train.shape)
print(Y_train.shape)


##Establish Model
model_logi = LogisticRegression().fit(X=X_train, y=Y_train, sample_weight=None)
model_ccv = CalibratedClassifierCV().fit(X_train, Y_train)
model_gbc = GradientBoostingClassifier().fit(X_train, Y_train)
model_rfc = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, Y_train)
model_abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME.R",n_estimators=100).fit(X_train, Y_train)

##Calculate logloss on test set
Y_validate_pre = model_gbc.predict_proba(X_validate)[:, 1]
Y_test_pre = model_gbc.predict_proba(X_test)[:, 1]
# Y_validate_pre[np.where(Y_validate_pre < 0.15)]=0.01
# Y_validate_pre[np.where((Y_validate_pre > 0.2)&(Y_validate_pre < 0.4))]=0.4
# Y_validate_pre[np.where(Y_validate_pre > 0.9)]=0.99

loos1=metrics.log_loss(Y_validate,Y_validate_pre)
print(loos1)
loos2=metrics.log_loss(Y_test,Y_test_pre)
print(loos2)


# #Get predictions
# data_predict = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_predict.csv")
# X_pre=data_predict.iloc[:,[2,4,7,8,9,13,14,15,16,17,18,19]]
# Y_pre=model_rfc.predict_proba(X_pre)[:,1]
#
# # Y_pre[np.where(Y_pre < 0.08)]=0.01
# # Y_pre[np.where((Y_pre > 0.2)&(Y_pre < 0.5))]=0.5
# # Y_pre[np.where(Y_pre > 0.7)]=0.999
#
# submission_rfc= pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/sample_submission.csv")
# submission_rfc['dropout_prob']=Y_pre
# print(submission_rfc)
# submission_rfc.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/submission_rfc.csv", sep=',')





###############with ratio

data_use=pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_use.csv")
index=rd.sample(range(1,72325), 43395)
# print(index)
X_use=np.array(data_use.iloc[:,[3,4,5,6,7,8,10,11,13,16,18,19,20]])
Y_use=np.array(data_use.iloc[:,-1])
data_xy=data_use.iloc[index,:]
X=X_use[index,:]
Y=Y_use[index]
X_test=np.delete(X_use, index, axis=0)
Y_test=np.delete(Y_use, index, axis=0)

##Group CV
groups = data_xy.iloc[:,-2]
gkf = GroupKFold(n_splits=7)
for train_index, test_index in gkf.split(X, Y, groups):
    X_train = X[train_index]
    X_validate = X[test_index]
    Y_train = Y[train_index]
    Y_validate = Y[test_index]
print(X_train.shape)
print(Y_train.shape)


##Establish Model
model_gbc = GradientBoostingClassifier().fit(X_train, Y_train)
model_logi = LogisticRegression().fit(X=X_train, y=Y_train, sample_weight=None)
model_rfc = RandomForestClassifier(max_depth=10, random_state=0).fit(X_train, Y_train)

model_voting = VotingClassifier(estimators=[('gbc', model_gbc), ('rfc', model_rfc)], voting='soft', weights=[7,1]).fit(X_train, Y_train)


##Calculate logloss on test set
Y_validate_pre1 = model_gbc.predict_proba(X_validate)[:, 1]
Y_test_pre1 = model_gbc.predict_proba(X_test)[:, 1]

Y_validate_pre2 = model_logi.predict_proba(X_validate)[:, 1]
Y_test_pre2 = model_logi.predict_proba(X_test)[:, 1]

Y_validate_pre3 = model_rfc.predict_proba(X_validate)[:, 1]
Y_test_pre3 = model_rfc.predict_proba(X_test)[:, 1]

Y_validate_pre4 = model_voting.predict_proba(X_validate)[:, 1]
Y_test_pre4 = model_voting.predict_proba(X_test)[:, 1]


##hard pull
# a=np.maximum(Y_validate_pre1,Y_validate_pre2,Y_validate_pre3)
Y_validate_pre4[np.where((Y_validate_pre1 < 0.08)&(Y_validate_pre2<0.08)&(Y_validate_pre3<0.08))]=0.001
# Y_validate_pre1[np.where((Y_validate_pre1 < 0.8) & (Y_validate_pre1 > 0.1) & (Y_validate_pre2 < 0.8) & (Y_validate_pre2 > 0.1) & (Y_validate_pre3 < 0.8) & (Y_validate_pre3 > 0.1))]=a
Y_validate_pre4[np.where((Y_validate_pre1 > 0.95)&(Y_validate_pre2>0.95)&(Y_validate_pre3>0.95))]=0.999


index_mid=np.where(((Y_test_pre1 < 0.5)&(Y_test_pre1 > 0.1))|((Y_test_pre2<0.5)&(Y_test_pre2 >0.1))|((Y_test_pre3<0.5)&(Y_test_pre3 >0.1)))
# print(len(Y_test_pre1[index_mid]))
# aaa=np.array([Y_test_pre1[index_mid],Y_test_pre2[index_mid],Y_test_pre3[index_mid]]).max(axis=0)
# Y_test_pre1[index_mid]=aaa


loos1=metrics.log_loss(Y_validate,Y_validate_pre4)
print(loos1)
loos2=metrics.log_loss(Y_test,Y_test_pre4)
print(loos2)






#Get predictions
data_predict = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_predict.csv")
X_pre=data_predict.iloc[:,[3,4,5,6,7,8,10,11,13,16,18,19,20]]
Y_pre1=model_gbc.predict_proba(X_pre)[:,1]
Y_pre2=model_logi.predict_proba(X_pre)[:,1]
Y_pre3=model_rfc.predict_proba(X_pre)[:,1]
Y_pre4=model_voting.predict_proba(X_pre)[:,1]

##hard
Y_pre4[np.where((Y_pre1 < 0.1) & (Y_pre2 < 0.1) & (Y_pre3 < 0.1))]=0.001
# Y_pre1[np.where((Y_pre1 < 0.8) & (Y_pre1 > 0.1) & (Y_pre2 < 0.8) & (Y_pre2 > 0.1) & (Y_pre3 < 0.8) & (Y_pre3 > 0.1))]=max(Y_pre1,Y_pre2,Y_pre3)
Y_pre4[np.where((Y_pre1 > 0.9) & (Y_pre2 > 0.9) & (Y_pre3 > 0.9))]=0.999

submission_voting= pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/sample_submission.csv")
submission_voting['dropout_prob']=Y_pre4
print(submission_voting)
submission_voting.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/submission_voting2.csv", sep=',')



###############knn
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier

data_use = pd.read_csv("W:\MOOC_DROPOUT\data_use.csv")
X=np.array(data_use.iloc[:,4:11])
Y=np.array(data_use.iloc[:,-1])


##Group CV
groups = data_use.iloc[:,-2]
for num_split in range(17,28):
    gkf = GroupKFold(n_splits=num_split)
    for train_index, test_index in gkf.split(X, Y, groups):
        X_train = X[train_index]
        X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
    # print(X_train.shape)
    # print(Y_train.shape)
    for num_nb in range(93, 94):
        # num_nb = 150
        ##Establish Model
        model_knn = KNeighborsClassifier(n_neighbors=num_nb)
        model_knn.fit(X_train, Y_train)
        ##Calculate logloss on test set

        Y_test_pre = model_knn.predict_proba(X_test)[:, 1]

        # Y_test_pre[np.where(Y_test_pre < 0.15)]=0.01
        # Y_test_pre[np.where((Y_test_pre > 0.2)&(Y_test_pre < 0.4))]=0.4
        # Y_test_pre[np.where(Y_test_pre > 0.9)]=0.99

        loos = metrics.log_loss(Y_test, Y_test_pre)
        print(num_split,num_nb, loos)

        # #Get predictions
        # data_predict = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_predict.csv")
        # X_pre=data_predict.iloc[:,[4,7,8,9,13,14,15,16,17,18,19]]
        # Y_pre=model_rfc.predict_proba(X_pre)[:,1]
        #
        # # Y_pre[np.where(Y_pre < 0.08)]=0.01
        # # Y_pre[np.where((Y_pre > 0.2)&(Y_pre < 0.5))]=0.5
        # # Y_pre[np.where(Y_pre > 0.7)]=0.999
        #
        # submission_rfc= pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/sample_submission.csv")
        # submission_rfc['dropout_prob']=Y_pre
        # print(submission_rfc)
        # submission_rfc.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/submission_rfc.csv", sep=',')
    # for num_split in range(2,10):







#access = activity[activity['event']=='access'].groupby('enrollment_id').count()
#EVENT=pd.DataFrame(columns=['enrollment_id','problem_freq','access','navigate','discussion','wiki','page_close','video'])
#EVENT['enrollment_id']=enrollment["enrollment_id"]

# drop_rate=pd.DataFrame(0,index=np.arange(39),columns=['course_id','drop_rate'])
# i=0
# for courseid in set(data_use['course_id']):
#     data_use1=data_use[data_use['course_id']==courseid]['dropout_prob']
#     #print(data_use1)
#     drop_rate['course_id'].iloc[i]=courseid
#     #print(courseid)
#     drop_rate['drop_rate'].iloc[i]=sum(data_use1)/len(data_use1)
#     #print(len(data_use1))
#     i=i+1
# #print (drop_rate)
# feature_data = pd.read_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/feature_data.csv")
# feature_data=feature_data.merge(drop_rate,how="left",on="course_id")
# columnsTitles=["drop_rate","dropout_prob"]
# feature_data=feature_data.reindex(columns=columnsTitles)
# data_use=data_use.merge(drop_rate,how="left",on="course_id")
# data_use=data_use.reindex(columns=columnsTitles)
# feature_data.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/feature_data.csv")
# data_use.to_csv("/Users/Emma/Desktop/Lehigh/CSE447/Project/data_use.csv")


#import csv
#with open('/Users/Emma/Desktop/Lehigh/CSE447/Project/train_label.csv') as train_label:
#     train = csv.reader(train_label, delimiter=',')
#     print (train)

#    enrollids = []
#    drops = []
#    for row in train:
#        enrollid = row[0]
#        drop = row[1]

#        enrollids.append(enrollid)
#        drops.append(drop)


#with open('/Users/Emma/Desktop/Lehigh/CSE447/Project/enrollment_list.csv') as enrollment_list:
#    enrollment = csv.reader(enrollment_list, delimiter=',')
#    print (enrollment)
#    for row in enrollment:
#        userid = row[1]
#        course = row[2]


#with open('/Users/Emma/Desktop/Lehigh/CSE447/Project/activity_log.csv') as activity_log:
#    activity = csv.reader(activity_log, delimiter=',')
#    print (activity)

#    problem = []
#    video = []
#    access = []
#    wiki = []
#    discussion = []
#    navigation = []
#    page_close = []

#    act = ['problem', 'video', 'access', 'wiki', 'discussion', 'navigation', 'page_close']
#    EVENT = {a: for element in act }
#    print
#    months
#    for row in activity:
#        enrollid = row[0]
#        event = row[2]
#        for i in range(len(enrollid)):








##    for row in train:
##        print(row)
##        print(row[o],row[1])



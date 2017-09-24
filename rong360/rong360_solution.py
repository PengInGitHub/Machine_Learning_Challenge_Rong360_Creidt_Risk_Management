
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

"""
Created on Sun Jul 30 16:27:29 2017

@author: pengchengliu
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.serif'] = ['SimHei']
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
p = sns.color_palette()
sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})


####################################################
#                                                  #
#            1.loading and merging data            #
#                                                  #
####################################################  

##########################
#   subset: loan time    #
##########################

#read data and change name
train_loan_time = pd.read_csv("loan_time_train.txt",header=None,names=['userid','loan_time'])
#the time stamp is masked
train_loan_time['loan_time']=train_loan_time['loan_time']//86400
       
##########################
#   subset: user info    #
##########################    
train_user_info = pd.read_csv("user_info_train.txt",header=None,
                    names=['userid','usersex','userjob','useredu',
                           'usermarriage', 'userhukou'])

###################################
#   subset: credit card detail    #
###################################  

train_bill_detail=pd.read_csv("bill_detail_train.txt",header=None,
                    names=['userid','time','bank_id','pre_amount_of_bill','pre_repayment','credit_amount',
                           'amount_of_bill_left','least_repayment','consume_amount','amount_of_bill','adjust_amount',
                          'circ_interest','avail_amount','prepare_amount','repayment_state'])
#the time stamp is masked
#divied by seconds of a day (86400)
train_bill_detail['time']=train_bill_detail['time']//86400
#merge loan time table and credit card detail table        
train_bill_detail = pd.merge(train_bill_detail, train_loan_time,how='inner', on = "userid")

###################################
#      subset: browse history     #
################################### 
userbrowse_behavior = pd.read_csv("browse_history_train.txt",header=None,
                    names=['userid','browse_time','browse_behavior','browse_behavior_number'])
userbrowse_behavior['browse_time']=userbrowse_behavior['browse_time']//86400
      
###################################
#       subset: bank detail       #
###################################       
      
bank_detail=pd.read_csv("bank_detail_train.txt",header=None,names=['userid','bank_detailtime','extype','examount',
                           'sal_inc_mark'])  
bank_detail['bank_detailtime']=bank_detail['bank_detailtime']//86400
      
###################################
#       subset: target variable   #
################################### 
      
train = pd.read_csv("overdue_train.txt",header=None,
                    names=['userid','target'])
#merge overdue table and user info table
train = pd.merge(train,train_user_info,how='inner',on = "userid")
#further merge with loan time table
train = pd.merge(train,train_loan_time,how='inner',on = "userid")  
#55,596 * 8

#unbalance rate: 13%
d=train
len(d[(d['target']==1)])/(len(d[(d['target']==1)])+len(d[(d['target']==0)]))



####################################################
#                                                  #
#                       2.EDA                      #
#                                                  #
#################################################### 

#######################################
#        subset: user info table      #
####################################### 
#relationship of overdue and features in user info table 

job_distribute=train.groupby('userjob',as_index=False)['target'].agg({'overdue' : 'sum','sum' : 'count'})
job_distribute['job_overdue_ratio']=job_distribute['overdue']/job_distribute['sum']
job_distribute

sex_distribute=train.groupby('usersex',as_index=False)['target'].agg({'overdue' : 'sum','sum' : 'count'})
sex_distribute['sex_overdue_ratio']=sex_distribute['overdue']/sex_distribute['sum']
sex_distribute      


usermarriage=train.groupby('usermarriage',as_index=False)['target'].agg({'overdue' : 'sum','sum' : 'count'})
usermarriage['marriage_overdue_ratio']=usermarriage['overdue']/usermarriage['sum']
usermarriage

edu_distribute=train.groupby('useredu',as_index=False)['target'].agg({'overdue' : 'sum','sum' : 'count'})
edu_distribute['edu_overdue_ratio']=edu_distribute['overdue']/edu_distribute['sum']
edu_distribute

userhukou=train.groupby('userhukou',as_index=False)['target'].agg({'overdue' : 'sum','sum' : 'count'})
userhukou['hukou_overdue_ratio']=userhukou['overdue']/userhukou['sum']
userhukou

#plot the relationship of overdue and features in user info table

fig = plt.figure(figsize=(20, 20))

ax1 = fig.add_subplot(3, 2, 1)
ax1=sns.barplot(job_distribute.index, job_distribute.job_overdue_ratio, alpha=0.8, color=p[0], label='rate')
ax1.legend()
ax1.set_title(u'overdue rate by job type') 
ax1.set_xlabel(u'job type')
ax1.set_ylabel(u'overdue rate')

ax2 = fig.add_subplot(3, 2, 2)
ax2=sns.barplot(sex_distribute.index, sex_distribute.sex_overdue_ratio, alpha=0.8, color=p[1], label='rate')
ax2.legend()
ax2.set_title(u'overdue rate by sex type') 
ax2.set_xlabel(u'sex type')
ax2.set_ylabel(u'overdue rate')

ax3 = fig.add_subplot(3, 2, 3)
ax3=sns.barplot(edu_distribute.index, edu_distribute.edu_overdue_ratio, alpha=0.8, color=p[2], label='rate')
ax3.legend()
ax3.set_title(u'overdue rate by education type') 
ax3.set_xlabel(u'education type')
ax3.set_ylabel(u'overdue rate')

ax4 = fig.add_subplot(3, 2, 4)
ax4=sns.barplot(usermarriage.index, usermarriage.marriage_overdue_ratio, alpha=0.8, color=p[3], label='rate')
ax4.legend()
ax4.set_title(u'overdue rate by marriage status') 
ax4.set_xlabel(u'marriage status')
ax4.set_ylabel(u'overdue rate')

ax5 = fig.add_subplot(3, 2, 5)
ax5=sns.barplot(userhukou.index, userhukou.hukou_overdue_ratio, alpha=0.8, color=p[4], label='rate')
ax5.legend()
ax5.set_title(u'overdue rate by hukou type') 
ax5.set_xlabel(u'hukou type')
ax5.set_ylabel(u'overdue_user_ratio')

#################################################
#        subset: credit card details table      #
#################################################
 
################
#bill_detail.txt analysis
#the difference of time in bill details and the loan_time (the time user gets the loan) is critical
#we could do analysis the behavioral, finalcial and other conditional difference before and after getting the loan
#among those have bill details later than he/she gets the loan, we could see their conditions change when the get the loan 
#calculate the count of records of each user after he/she gets the loan
#do statistics according to the days difference: by one day, two days or even more


t1=train_bill_detail[(train_bill_detail['time']>train_bill_detail['loan_time'])].groupby("userid",as_index=False)
t2=train_bill_detail[(train_bill_detail['time']>train_bill_detail['loan_time']+1)].groupby("userid",as_index=False)
t3=train_bill_detail[(train_bill_detail['time']>train_bill_detail['loan_time']+2)].groupby("userid",as_index=False)

#count of such records and corresponding uid for three levels
x=t1['time'].apply(lambda x:np.unique(x).size)
x1=t1['time'].agg({'t1' : 'count'})
x1['x1']=x

x=t2['time'].apply(lambda x:np.unique(x).size)
x2=t2['time'].agg({'t2' : 'count'})
x2['x2']=x

x=t3['time'].apply(lambda x:np.unique(x).size)
x3=t3['time'].agg({'t3' : 'count'})
x3['x3']=x
  
#put the info into the training table  
train=pd.merge(train,x1,how='left',on = "userid")
train=pd.merge(train,x2,how='left',on = "userid")
train=pd.merge(train,x3,how='left',on = "userid")
#missing value here are replaced by 0, which means user has no such records
train=train.fillna(0)

#########
#  plot #
#########
d=train
l0_x1=d[(d['target']==0)].groupby("x1",as_index=False)
l1_x1=d[(d['target']==1)].groupby("x1",as_index=False)

#see the distribution of x1 btw overdued cases and non-overdue cases
l0_x1=l0_x1['x1'].agg({'l0_x1' : 'count'})#label is 0
l1_x1=l1_x1['x1'].agg({'l1_x1' : 'count'})#label is 1

l1_x1.plot(kind="scatter", x="x1", y="l1_x1")
l0_x1.plot(kind="scatter", x="x1", y="l0_x1")
d.plot(kind="scatter", x="x1", y="target")

sns.jointplot(x="x1", y="target", data=d, size=3)
lala=sns.FacetGrid(d, hue="target", size=5).map(plt.scatter, "x1", "target").add_legend()
plt.subplots_adjust(top=0.9)
lala.fig.suptitle('counts of bill detail after getting loan by overdue condition')
lala.set_axis_labels("counts of bank details after gettig loan","overdue condition")


#################################################
#            subset: bank details table         #
#################################################

#bank_detail.txt analysis
#calculate the difference btw the montly or overall income and expense per user
train_bank_detail = pd.merge(bank_detail,train_loan_time,how='left',on = "userid")
#define expense and group by user
user_exp=train_bank_detail[(train_bank_detail['extype']==1)].groupby("userid",as_index=False)
#define income and group by user
user_inc=train_bank_detail[(train_bank_detail['extype']==0)].groupby("userid",as_index=False)
#income from salary
user_sal_inc=train_bank_detail[(train_bank_detail['sal_inc_mark']==1)].groupby("userid",as_index=False)
#aggregrate the amount per person
user_exp=user_exp['examount'].agg({'user_exp' : 'sum'})
#aggregrate the amount per person
user_inc=user_inc['examount'].agg({'user_inc' : 'sum'})
#aggregrate the amount per person
user_sal_inc=user_sal_inc['examount'].agg({'user_sal_inc' : 'sum'})
#merge these three columns with training table by user id
stat = train
stat=pd.merge(stat,user_exp,how='left',on = "userid")
stat=pd.merge(stat,user_inc,how='left',on = "userid")
stat=pd.merge(stat,user_sal_inc,how='left',on = "userid")   
stat=stat.fillna(0)
#calculate the difference of expense and income, name this difference as over_spending_index 
stat['over_spending_index']=stat['user_exp']-stat['user_inc']
#the effect of this feature seems not so big   
sns.FacetGrid(stat, hue="target", size=5).map(plt.scatter, "over_spending_index", "target").add_legend()


#####################
#merge features into statistics table:
stat=pd.merge(stat, sex_distribute,how='inner', on = 'usersex')
stat=pd.merge(stat, job_distribute,how='inner', on = 'userjob')
stat=pd.merge(stat, usermarriage,how='inner', on = 'usermarriage')
stat=pd.merge(stat, edu_distribute,how='inner', on = 'useredu')
stat=pd.merge(stat, userhukou,how='inner', on = 'userhukou')

#??? info in stat

####################################################
#                                                  #
#              3.feature construction              #
#                                                  #
####################################################  
#credut card bill detail table
#################################################
#                time is known                  #
#################################################
#time is known(larger than zero)
d=train_bill_detail[(train_bill_detail['time']>0)]


#additional feature：55596 rows × 466 columns
#loan time is the benchmark
feature=train_loan_time
#################################################
#          stat info before loan time           #
#################################################
#add features of stat info: sum count max min mean std var etc.
#compare time and loan time, choose records that have earlier time than loan time
gb=d[(d['time']<=d['loan_time'])].loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
                                  'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount','repayment_state']].groupby(["userid"],as_index=False)


#sum value by user 
#be_loan means before loan time
#sum of bills before loan time
be_loan_bill_sum=gb.sum()
#rename columns

be_loan_bill_sum.columns = ['userid', 'be_loan_pre_amount_of_bill_sum', 'be_loan_pre_repayment_sum','be_loan_credit_amount_sum','be_loan_amount_of_bill_left_sum',
                     'be_loan_least_repayment_sum','be_loan_consume_amount_sum','be_loan_amount_of_bill_sum','be_loan_adjust_amount_sum','be_loan_circ_interest_sum',
                     'be_loan_avail_amount_sum','be_loan_prepare_amount_sum','be_loan_repayment_state_sum']

#merge sum table with loan time table
feature=pd.merge(feature, be_loan_bill_sum,how='left', on = "userid")

#start from here


#calculate difference btw previous_payback_before_loan_sum and previous_balance_before_loan_sum
feature['be_loan_pre_repayment_sum_and_be_loan_pre_amount_of_bill_sum_diff']=feature['be_loan_pre_repayment_sum']-feature['be_loan_pre_amount_of_bill_sum']
#calculate difference btw credit_limits_before_loan_sum and current_balance_before_loan_sum
feature['be_loan_credit_amount_sum_and_be_loan_amount_of_bill_left_sum_diff']=feature['be_loan_credit_amount_sum']-feature['be_loan_amount_of_bill_left_sum']
#calculate difference btw available_balance_before_loan_sum and cash_advance_quota_before_loan_sum
feature['be_loan_avail_amount_sum_and_be_loan_prepare_amount_sum_diff']=feature['be_loan_avail_amount_sum']-feature['be_loan_prepare_amount_sum']
#calculate difference btw current_lowest_payment_before_loan_sum and interest_before_loan_sum
feature['be_loan_least_repayment_sum_and_be_loan_circ_interest_sum_sum']=feature['be_loan_least_repayment_sum']+feature['be_loan_circ_interest_sum']

#count value by user 
be_loan_bill_count=gb.count()
be_loan_bill_count.columns = ['userid', 'be_loan_pre_amount_of_bill_count', 'be_loan_pre_repayment_count','be_loan_credit_amount_count','be_loan_amount_of_bill_left_count',
                     'be_loan_least_repayment_count','be_loan_consume_amount_count','be_loan_amount_of_bill_count','be_loan_adjust_amount_count','be_loan_circ_interestcount',
                     'be_loan_avail_amount_count','be_loan_prepare_amount_count','be_loan_repayment_state_count']
feature=pd.merge(feature, be_loan_bill_count,how='left', on = "userid")

#max value by user 
be_loan_bill_max=gb.max()
be_loan_bill_max.columns = ['userid', 'be_loan_pre_amount_of_bill_max', 'be_loan_pre_repayment_max','be_loan_credit_amount_max','be_loan_amount_of_bill_left_max',
                     'be_loan_least_repayment_max','be_loan_consume_amount_max','be_loan_amount_of_bill_max','be_loan_adjust_amount_max','be_loan_circ_interest_max',
                     'be_loan_avail_amount_max','be_loan_prepare_amount_max','be_loan_repayment_state_max']
feature=pd.merge(feature, be_loan_bill_max,how='left', on = "userid")
#diff of max values
feature['be_loan_pre_repayment_max_and_be_loan_pre_amount_of_bill_max_diff']=feature['be_loan_pre_repayment_max']-feature['be_loan_pre_amount_of_bill_max']
feature['be_loan_credit_amount_max_and_be_loan_amount_of_bill_left_max_diff']=feature['be_loan_credit_amount_max']-feature['be_loan_amount_of_bill_left_max']
feature['be_loan_avail_amount_max_and_be_loan_prepare_amount_max_diff']=feature['be_loan_avail_amount_max']-feature['be_loan_prepare_amount_max']
feature['be_loan_least_repayment_max_and_be_loan_circ_interest_max_sum']=feature['be_loan_least_repayment_max']+feature['be_loan_circ_interest_max']

#min value by user 
be_loan_bill_min=gb.min()
be_loan_bill_min.columns = ['userid', 'be_loan_pre_amount_of_bill_min', 'be_loan_pre_interest_min','be_loan_credit_amount_min','be_loan_amount_of_bill_left_min',
                     'be_loan_least_interest_min','be_loan_consume_amount_min','be_loan_amount_of_bill_min','be_loan_adjust_amount_min','be_loan_circ_interest_min',
                     'be_loan_avail_amount_min','be_loan_prepare_amount_min','be_loan_repayment_state_min']

feature=pd.merge(feature, be_loan_bill_min,how='left', on = "userid")
#diff of min values
feature['be_loan_pre_interest_min_and_be_loan_pre_amount_of_bill_min_diff']=feature['be_loan_pre_interest_min']-feature['be_loan_pre_amount_of_bill_min']
feature['be_loan_credit_amount_min_and_be_loan_amount_of_bill_left_min_diff']=feature['be_loan_credit_amount_min']-feature['be_loan_amount_of_bill_left_min']
feature['be_loan_avail_amount_min_and_be_loan_prepare_amount_min_diff']=feature['be_loan_avail_amount_min']-feature['be_loan_prepare_amount_min']
feature['be_loan_least_interest_min_and_be_loan_circ_interest_min_sum']=feature['be_loan_least_interest_min']+feature['be_loan_circ_interest_min']

#mean value by user 
be_loan_billmean=gb.mean()
be_loan_billmean.columns = ['userid', 'be_loan_pre_amount_of_billmean', 'be_loan_pre_interest_mean','be_loan_credit_amount_mean','be_loan_amount_of_bill_leftmean',
                     'be_loan_least_interest_mean','be_loan_consume_amount_mean','be_loan_amount_of_billmean','be_loan_adjust_amount_mean','be_loan_circ_interestmean',
                     'be_loan_avail_amount_mean','be_loan_prepare_amount_mean','be_loan_repayment_state_mean']

feature=pd.merge(feature, be_loan_billmean,how='left', on = "userid")
#diff of mean values
feature['be_loan_pre_interest_mean_and_be_loan_pre_amount_of_billmean_diff']=feature['be_loan_pre_interest_mean']-feature['be_loan_pre_amount_of_billmean']
feature['be_loan_credit_amount_mean_and_be_loan_amount_of_bill_leftmean_diff']=feature['be_loan_credit_amount_mean']-feature['be_loan_amount_of_bill_leftmean']
feature['be_loan_avail_amount_mean_and_be_loan_prepare_amount_mean_diff']=feature['be_loan_avail_amount_mean']-feature['be_loan_prepare_amount_mean']
feature['be_loan_least_interest_mean_and_be_loan_circ_interestmean_sum']=feature['be_loan_least_interest_mean']+feature['be_loan_circ_interestmean']

#median value by user 
be_loan_bill_median=gb.median()
be_loan_bill_median.columns = ['userid', 'be_loan_pre_amount_of_bill_median', 'be_loan_pre_repayment_median','be_loan_credit_amount_median','be_loan_amount_of_bill_left_median',
                  'be_loan_least_repayment_median','be_loan_consume_amount_median','be_loan_amount_of_bill_median','be_loan_adjust_amount_median',
                  'be_loan_circ_interest_median','be_loan_avail_amount_median','be_loan_prepare_amount_median','be_loan_repayment_state_median']

feature=pd.merge(feature, be_loan_bill_median,how='left', on = "userid")
#diff of median values
feature['be_loan_pre_repayment_median_and_be_loan_pre_amount_of_bill_median_diff']=feature['be_loan_pre_repayment_median']-feature['be_loan_pre_amount_of_bill_median']
feature['be_loan_credit_amount_median_and_be_loan_amount_of_bill_left_median_diff']=feature['be_loan_credit_amount_median']-feature['be_loan_amount_of_bill_left_median']
feature['be_loan_avail_amount_median_and_be_loan_prepare_amount_median_diff']=feature['be_loan_avail_amount_median']-feature['be_loan_prepare_amount_median']
feature['be_loan_least_repayment_median_and_be_loan_circ_interest_median_sum']=feature['be_loan_least_repayment_median']+feature['be_loan_circ_interest_median']

#standarddeviation value by user 
be_loan_billstd=gb.std()
be_loan_billstd.columns = ['userid', 'be_loan_pre_amount_of_billstd', 'be_loan_pre_repayment_std','be_loan_credit_amount_std','be_loan_amount_of_bill_left_std',
                     'be_loan_least_repayment_std','be_loan_consume_amount_std','be_loan_amount_of_billstd','be_loan_adjust_amount_std','be_loan_circ_intereststd',
                     'be_loan_avail_amount_std','be_loan_prepare_amount_std','be_loan_repayment_statestd']

feature=pd.merge(feature, be_loan_billstd,how='left', on = "userid")

#diff of std values
feature['be_loan_pre_repayment_std_and_be_loan_pre_amount_of_billstd_diff']=feature['be_loan_pre_repayment_std']-feature['be_loan_pre_amount_of_billstd']
feature['be_loan_credit_amount_std_and_be_loan_amount_of_bill_left_std_diff']=feature['be_loan_credit_amount_std']-feature['be_loan_amount_of_bill_left_std']
feature['be_loan_avail_amount_std_and_be_loan_prepare_amount_std_diff']=feature['be_loan_avail_amount_std']-feature['be_loan_prepare_amount_std']
feature['be_loan_least_repayment_std_and_be_loan_circ_intereststd_sum']=feature['be_loan_least_repayment_std']+feature['be_loan_circ_intereststd']

#variance value by user 
be_loan_bill_var=gb.var()
be_loan_bill_var.columns = ['userid', 'be_loan_pre_amount_of_bill_var', 'be_loan_pre_repayment_var','be_loan_credit_amount_var','be_loan_amount_of_bill_left_var',
                     'be_loan_least_repayment_var','be_loan_consume_amount_var','be_loan_amount_of_bill_var','be_loan_adjust_amount_var','be_loan_circ_interestvar',
                     'be_loan_avail_amount_var','be_loan_prepare_amount_var','be_loan_repayment_statevar']

feature=pd.merge(feature, be_loan_bill_var,how='left', on = "userid")
#diff of var values
feature['be_loan_pre_repayment_var_and_be_loan_pre_amount_of_bill_var_diff']=feature['be_loan_pre_repayment_var']-feature['be_loan_pre_amount_of_bill_var']
feature['be_loan_credit_amount_var_and_be_loan_amount_of_bill_left_var_diff']=feature['be_loan_credit_amount_var']-feature['be_loan_amount_of_bill_left_var']
feature['be_loan_avail_amount_var_and_be_loan_prepare_amount_var_diff']=feature['be_loan_avail_amount_var']-feature['be_loan_prepare_amount_var']
feature['be_loan_least_repayment_var_and_be_loan_circ_interestvar_sum']=feature['be_loan_least_repayment_var']+feature['be_loan_circ_interestvar']

feature1=feature.fillna(-1)
#count of missing values per column
print((feature1==-1).sum(axis=0))
#count of missing values per row 
print((feature1==-1).sum(axis=1))
feature
#55,596 * 126

#################################################
#                drop-duplicate                 #
#################################################
feature_copy=feature
#drop-duplicate by user id, bank and time
#max
data=d[(d['time']<=d['loan_time'])].loc[:,['userid','time','bank_id','pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left'
                                  ,'least_repayment','consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount'
                                  ,'prepare_amount']].groupby(["userid","time","bank_id"],as_index=False).max()

gb=data.loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
               'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount']].groupby(["userid"],as_index=False)

drop_d_be_loan_bill_sum=gb.sum()
drop_d_be_loan_bill_sum.columns = ['userid', 'drop_d_be_loan_pre_amount_of_bill_sum', 'drop_d_be_loan_pre_repayment_sum','drop_d_be_loan_credit_amount_sum'
                       ,'drop_d_be_loan_amount_of_bill_left_sum','drop_d_be_loan_least_repayment_sum','drop_d_be_loan_consume_amount_sum'
                       ,'drop_d_be_loan_amount_of_bill_sum','drop_d_be_loan_adjust_amount_sum','drop_d_be_loan_circ_interest_sum','drop_d_be_loan_avail_amount_sum'
                       ,'drop_d_be_loan_prepare_amount_sum']
feature=pd.merge(feature, drop_d_be_loan_bill_sum,how='left', on = "userid")

feature['drop_d_be_loan_pre_repayment_sum_and_be_loan_pre_amount_of_bill_sum_diff']=feature['drop_d_be_loan_pre_repayment_sum']-feature['drop_d_be_loan_pre_amount_of_bill_sum']
feature['drop_d_be_loan_credit_amount_sum_and_be_loan_amount_of_bill_left_sum_diff']=feature['drop_d_be_loan_credit_amount_sum']-feature['drop_d_be_loan_amount_of_bill_left_sum']
feature['drop_d_be_loan_avail_amount_sum_and_be_loan_prepare_amount_sum_diff']=feature['drop_d_be_loan_avail_amount_sum']-feature['drop_d_be_loan_prepare_amount_sum']
feature['drop_d_be_loan_least_repayment_sum_and_be_loan_circ_interest_sum_sum']=feature['drop_d_be_loan_least_repayment_sum']+feature['drop_d_be_loan_circ_interest_sum']

drop_d_be_loan_bill_count=gb.count()
drop_d_be_loan_bill_count.columns = ['userid', 'drop_d_be_loan_pre_amount_of_bill_count', 'drop_d_be_loan_pre_repayment_count','drop_d_be_loan_credit_amount_count'
                         ,'drop_d_be_loan_amount_of_bill_left_count','drop_d_be_loan_least_repayment_count','drop_d_be_loan_consume_amount_count'
                         ,'drop_d_be_loan_amount_of_bill_count','drop_d_be_loan_adjust_amount_count','drop_d_be_loan_circ_interestcount'
                         ,'drop_d_be_loan_avail_amount_count','drop_d_be_loan_prepare_amount_count']
feature=pd.merge(feature, drop_d_be_loan_bill_count,how='left', on = "userid")

drop_d_be_loan_bill_max=gb.max()
drop_d_be_loan_bill_max.columns = ['userid', 'drop_d_be_loan_pre_amount_of_bill_max', 'drop_d_be_loan_pre_repayment_max','drop_d_be_loan_credit_amount_max'
                       ,'drop_d_be_loan_amount_of_bill_left_max','drop_d_be_loan_least_repayment_max','drop_d_be_loan_consume_amount_max'
                       ,'drop_d_be_loan_amount_of_bill_max','drop_d_be_loan_adjust_amount_max','drop_d_be_loan_circ_interest_max'
                       ,'drop_d_be_loan_avail_amount_max','drop_d_be_loan_prepare_amount_max']
feature=pd.merge(feature, drop_d_be_loan_bill_max,how='left', on = "userid")

feature['drop_d_be_loan_pre_repayment_max_and_be_loan_pre_amount_of_bill_max_diff']=feature['drop_d_be_loan_pre_repayment_max']-feature['drop_d_be_loan_pre_amount_of_bill_max']
feature['drop_d_be_loan_credit_amount_max_and_be_loan_amount_of_bill_left_max_diff']=feature['drop_d_be_loan_credit_amount_max']-feature['drop_d_be_loan_amount_of_bill_left_max']
feature['drop_d_be_loan_avail_amount_max_and_be_loan_prepare_amount_max_diff']=feature['drop_d_be_loan_avail_amount_max']-feature['drop_d_be_loan_prepare_amount_max']
feature['drop_d_be_loan_least_repayment_max_and_be_loan_circ_interest_max_sum']=feature['drop_d_be_loan_least_repayment_max']+feature['drop_d_be_loan_circ_interest_max']

drop_d_be_loan_bill_min=gb.min()
drop_d_be_loan_bill_min.columns = ['userid', 'drop_d_be_loan_pre_amount_of_bill_min', 'drop_d_be_loan_pre_interest_min','drop_d_be_loan_credit_amount_min'
                       ,'drop_d_be_loan_amount_of_bill_left_min','drop_d_be_loan_least_interest_min','drop_d_be_loan_consume_amount_min'
                       ,'drop_d_be_loan_amount_of_bill_min','drop_d_be_loan_adjust_amount_min','drop_d_be_loan_circ_interest_min'
                       ,'drop_d_be_loan_avail_amount_min','drop_d_be_loan_prepare_amount_min']
feature=pd.merge(feature, drop_d_be_loan_bill_min,how='left', on = "userid")
feature['drop_d_be_loan_pre_interest_min_and_be_loan_pre_amount_of_bill_min_diff']=feature['drop_d_be_loan_pre_interest_min']-feature['drop_d_be_loan_pre_amount_of_bill_min']
feature['drop_d_be_loan_credit_amount_min_and_be_loan_amount_of_bill_left_min_diff']=feature['drop_d_be_loan_credit_amount_min']-feature['drop_d_be_loan_amount_of_bill_left_min']
feature['drop_d_be_loan_avail_amount_min_and_be_loan_prepare_amount_min_diff']=feature['drop_d_be_loan_avail_amount_min']-feature['drop_d_be_loan_prepare_amount_min']
feature['drop_d_be_loan_least_interest_min_and_be_loan_circ_interest_min_sum']=feature['drop_d_be_loan_least_interest_min']+feature['drop_d_be_loan_circ_interest_min']

drop_d_be_loan_billmean=gb.mean()
drop_d_be_loan_billmean.columns = ['userid', 'drop_d_be_loan_pre_amount_of_billmean', 'drop_d_be_loan_pre_interest_mean','drop_d_be_loan_credit_amount_mean'
                        ,'drop_d_be_loan_amount_of_bill_leftmean','drop_d_be_loan_least_interest_mean','drop_d_be_loan_consume_amount_mean'
                        ,'drop_d_be_loan_amount_of_billmean','drop_d_be_loan_adjust_amount_mean','drop_d_be_loan_circ_interestmean'
                        ,'drop_d_be_loan_avail_amount_mean','drop_d_be_loan_prepare_amount_mean']
feature=pd.merge(feature, drop_d_be_loan_billmean,how='left', on = "userid")
feature['drop_d_be_loan_pre_interest_mean_and_be_loan_pre_amount_of_billmean_diff']=feature['drop_d_be_loan_pre_interest_mean']-feature['drop_d_be_loan_pre_amount_of_billmean']
feature['drop_d_be_loan_credit_amount_mean_and_be_loan_amount_of_bill_leftmean_diff']=feature['drop_d_be_loan_credit_amount_mean']-feature['drop_d_be_loan_amount_of_bill_leftmean']
feature['drop_d_be_loan_avail_amount_mean_and_be_loan_prepare_amount_mean_diff']=feature['drop_d_be_loan_avail_amount_mean']-feature['drop_d_be_loan_prepare_amount_mean']
feature['drop_d_be_loan_least_interest_mean_and_be_loan_circ_interestmean_sum']=feature['drop_d_be_loan_least_interest_mean']+feature['drop_d_be_loan_circ_interestmean']


drop_d_be_loan_bill_median=gb.median()
drop_d_be_loan_bill_median.columns = ['userid', 'drop_d_be_loan_pre_amount_of_bill_median', 'drop_d_be_loan_pre_repayment_median'
                          ,'drop_d_be_loan_credit_amount_median','drop_d_be_loan_amount_of_bill_left_median','drop_d_be_loan_least_repayment_median'
                          ,'drop_d_be_loan_consume_amount_median','drop_d_be_loan_amount_of_bill_median','drop_d_be_loan_adjust_amount_median'
                          ,'drop_d_be_loan_circ_interest_median','drop_d_be_loan_avail_amount_median','drop_d_be_loan_prepare_amount_median']
feature=pd.merge(feature, drop_d_be_loan_bill_median,how='left', on = "userid")
feature['drop_d_be_loan_pre_repayment_median_and_be_loan_pre_amount_of_bill_median_diff']=feature['drop_d_be_loan_pre_repayment_median']-feature['drop_d_be_loan_pre_amount_of_bill_median']
feature['drop_d_be_loan_credit_amount_median_and_be_loan_amount_of_bill_left_median_diff']=feature['drop_d_be_loan_credit_amount_median']-feature['drop_d_be_loan_amount_of_bill_left_median']
feature['drop_d_be_loan_avail_amount_median_and_be_loan_prepare_amount_median_diff']=feature['drop_d_be_loan_avail_amount_median']-feature['drop_d_be_loan_prepare_amount_median']
feature['drop_d_be_loan_least_repayment_median_and_be_loan_circ_interest_median_sum']=feature['drop_d_be_loan_least_repayment_median']+feature['drop_d_be_loan_circ_interest_median']


drop_d_be_loan_billstd=gb.std()
drop_d_be_loan_billstd.columns = ['userid', 'drop_d_be_loan_pre_amount_of_billstd', 'drop_d_be_loan_pre_repayment_std','drop_d_be_loan_credit_amount_std'
                       ,'drop_d_be_loan_amount_of_bill_left_std','drop_d_be_loan_least_repayment_std','drop_d_be_loan_consume_amount_std'
                       ,'drop_d_be_loan_amount_of_billstd','drop_d_be_loan_adjust_amount_std','drop_d_be_loan_circ_intereststd'
                       ,'drop_d_be_loan_avail_amount_std','drop_d_be_loan_prepare_amount_std']
feature=pd.merge(feature, drop_d_be_loan_billstd,how='left', on = "userid")
feature['drop_d_be_loan_pre_repayment_std_and_be_loan_pre_amount_of_billstd_diff']=feature['drop_d_be_loan_pre_repayment_std']-feature['drop_d_be_loan_pre_amount_of_billstd']
feature['drop_d_be_loan_credit_amount_std_and_be_loan_amount_of_bill_left_std_diff']=feature['drop_d_be_loan_credit_amount_std']-feature['drop_d_be_loan_amount_of_bill_left_std']
feature['drop_d_be_loan_avail_amount_std_and_be_loan_prepare_amount_std_diff']=feature['drop_d_be_loan_avail_amount_std']-feature['drop_d_be_loan_prepare_amount_std']
feature['drop_d_be_loan_least_repayment_std_and_be_loan_circ_intereststd_sum']=feature['drop_d_be_loan_least_repayment_std']+feature['drop_d_be_loan_circ_intereststd']

drop_d_be_loan_bill_var=gb.var()
drop_d_be_loan_bill_var.columns = ['userid', 'drop_d_be_loan_pre_amount_of_bill_var', 'drop_d_be_loan_pre_repayment_var','drop_d_be_loan_credit_amount_var'
                       ,'drop_d_be_loan_amount_of_bill_left_var','drop_d_be_loan_least_repayment_var','drop_d_be_loan_consume_amount_var'
                       ,'drop_d_be_loan_amount_of_bill_var','drop_d_be_loan_adjust_amount_var','drop_d_be_loan_circ_interestvar','drop_d_be_loan_avail_amount_var','drop_d_be_loan_prepare_amount_var']
feature=pd.merge(feature, drop_d_be_loan_bill_var,how='left', on = "userid")
feature['drop_d_be_loan_pre_repayment_var_and_be_loan_pre_amount_of_bill_var_diff']=feature['drop_d_be_loan_pre_repayment_var']-feature['drop_d_be_loan_pre_amount_of_bill_var']
feature['drop_d_be_loan_credit_amount_var_and_be_loan_amount_of_bill_left_var_diff']=feature['drop_d_be_loan_credit_amount_var']-feature['drop_d_be_loan_amount_of_bill_left_var']
feature['drop_d_be_loan_avail_amount_var_and_be_loan_prepare_amount_var_diff']=feature['drop_d_be_loan_avail_amount_var']-feature['drop_d_be_loan_prepare_amount_var']
feature['drop_d_be_loan_least_repayment_var_and_be_loan_circ_interestvar_sum']=feature['drop_d_be_loan_least_repayment_var']+feature['drop_d_be_loan_circ_interestvar']

#feature 242 columns now
#################################################
#           stat info after loan time           #
#################################################


feature_copy=feature
#add features of stat info: sum count max min mean std var etc.
#compare time and loan time, choose records that have later time than loan time

gb=d[(d['time']>d['loan_time'])].loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
                                  'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount']].groupby(["userid"],as_index=False)

after_loan_bill_sum=gb.sum()
after_loan_bill_sum.columns = ['userid', 'after_loan_pre_amount_of_bill_sum', 'after_loan_pre_repayment_sum','after_loan_credit_amount_sum','after_loan_amount_of_bill_left_sum',
                     'after_loan_least_repayment_sum','after_loan_consume_amount_sum','after_loan_amount_of_bill_sum','after_loan_adjust_amount_sum','after_loan_circ_interest_sum',
                     'after_loan_avail_amount_sum','after_loan_prepare_amount_sum']

feature=pd.merge(feature, after_loan_bill_sum,how='left', on = "userid")
feature['after_loan_pre_repayment_sum_and_after_loan_pre_amount_of_bill_sum_diff']=feature['after_loan_pre_repayment_sum']-feature['after_loan_pre_amount_of_bill_sum']
feature['after_loan_credit_amount_sum_and_after_loan_amount_of_bill_left_sum_diff']=feature['after_loan_credit_amount_sum']-feature['after_loan_amount_of_bill_left_sum']
feature['after_loan_avail_amount_sum_and_after_loan_prepare_amount_sum_diff']=feature['after_loan_avail_amount_sum']-feature['after_loan_prepare_amount_sum']
feature['after_loan_least_repayment_sum_and_after_loan_circ_interest_sum_sum']=feature['after_loan_least_repayment_sum']+feature['after_loan_circ_interest_sum']

after_loan_bill_count=gb.count()
after_loan_bill_count.columns = ['userid', 'after_loan_pre_amount_of_bill_count', 'after_loan_pre_repayment_count','after_loan_credit_amount_count','after_loan_amount_of_bill_left_count',
                     'after_loan_least_repayment_count','after_loan_consume_amount_count','after_loan_amount_of_bill_count','after_loan_adjust_amount_count','after_loan_circ_interestcount',
                     'after_loan_avail_amount_count','after_loan_prepare_amount_count']
feature=pd.merge(feature, after_loan_bill_count,how='left', on = "userid")

after_loan_bill_max=gb.max()
after_loan_bill_max.columns = ['userid', 'after_loan_pre_amount_of_bill_max', 'after_loan_pre_repayment_max','after_loan_credit_amount_max','after_loan_amount_of_bill_left_max',
                     'after_loan_least_repayment_max','after_loan_consume_amount_max','after_loan_amount_of_bill_max','after_loan_adjust_amount_max','after_loan_circ_interest_max',
                     'after_loan_avail_amount_max','after_loan_prepare_amount_max']
feature=pd.merge(feature, after_loan_bill_max,how='left', on = "userid")
feature['after_loan_pre_repayment_max_and_after_loan_pre_amount_of_bill_max_diff']=feature['after_loan_pre_repayment_max']-feature['after_loan_pre_amount_of_bill_max']
feature['after_loan_credit_amount_max_and_after_loan_amount_of_bill_left_max_diff']=feature['after_loan_credit_amount_max']-feature['after_loan_amount_of_bill_left_max']
feature['after_loan_avail_amount_max_and_after_loan_prepare_amount_max_diff']=feature['after_loan_avail_amount_max']-feature['after_loan_prepare_amount_max']
feature['after_loan_least_repayment_max_and_after_loan_circ_interest_max_sum']=feature['after_loan_least_repayment_max']+feature['after_loan_circ_interest_max']

after_loan_bill_min=gb.min()
after_loan_bill_min.columns = ['userid', 'after_loan_pre_amount_of_bill_min', 'after_loan_pre_interest_min','after_loan_credit_amount_min','after_loan_amount_of_bill_left_min',
                     'after_loan_least_interest_min','after_loan_consume_amount_min','after_loan_amount_of_bill_min','after_loan_adjust_amount_min','after_loan_circ_interest_min',
                     'after_loan_avail_amount_min','after_loan_prepare_amount_min']
feature=pd.merge(feature, after_loan_bill_min,how='left', on = "userid")
feature['after_loan_pre_interest_min_and_after_loan_pre_amount_of_bill_min_diff']=feature['after_loan_pre_interest_min']-feature['after_loan_pre_amount_of_bill_min']
feature['after_loan_credit_amount_min_and_after_loan_amount_of_bill_left_min_diff']=feature['after_loan_credit_amount_min']-feature['after_loan_amount_of_bill_left_min']
feature['after_loan_avail_amount_min_and_after_loan_prepare_amount_min_diff']=feature['after_loan_avail_amount_min']-feature['after_loan_prepare_amount_min']
feature['after_loan_least_interest_min_and_after_loan_circ_interest_min_sum']=feature['after_loan_least_interest_min']+feature['after_loan_circ_interest_min']

after_loan_billmean=gb.mean()
after_loan_billmean.columns = ['userid', 'after_loan_pre_amount_of_billmean', 'after_loan_pre_interest_mean','after_loan_credit_amount_mean','after_loan_amount_of_bill_leftmean',
                     'after_loan_least_interest_mean','after_loan_consume_amount_mean','after_loan_amount_of_billmean','after_loan_adjust_amount_mean','after_loan_circ_interestmean',
                     'after_loan_avail_amount_mean','after_loan_prepare_amount_mean']
feature=pd.merge(feature, after_loan_billmean,how='left', on = "userid")
feature['after_loan_pre_interest_mean_and_after_loan_pre_amount_of_billmean_diff']=feature['after_loan_pre_interest_mean']-feature['after_loan_pre_amount_of_billmean']
feature['after_loan_credit_amount_mean_and_after_loan_amount_of_bill_leftmean_diff']=feature['after_loan_credit_amount_mean']-feature['after_loan_amount_of_bill_leftmean']
feature['after_loan_avail_amount_mean_and_after_loan_prepare_amount_mean_diff']=feature['after_loan_avail_amount_mean']-feature['after_loan_prepare_amount_mean']
feature['after_loan_least_interest_mean_and_after_loan_circ_interestmean_sum']=feature['after_loan_least_interest_mean']+feature['after_loan_circ_interestmean']

after_loan_bill_median=gb.median()
after_loan_bill_median.columns = ['userid', 'after_loan_pre_amount_of_bill_median', 'after_loan_pre_repayment_median','after_loan_credit_amount_median','after_loan_amount_of_bill_left_median',
                  'after_loan_least_repayment_median','after_loan_consume_amount_median','after_loan_amount_of_bill_median','after_loan_adjust_amount_median',
                  'after_loan_circ_interest_median','after_loan_avail_amount_median','after_loan_prepare_amount_median']
feature=pd.merge(feature, after_loan_bill_median,how='left', on = "userid")
feature['after_loan_pre_repayment_median_and_after_loan_pre_amount_of_bill_median_diff']=feature['after_loan_pre_repayment_median']-feature['after_loan_pre_amount_of_bill_median']
feature['after_loan_credit_amount_median_and_after_loan_amount_of_bill_left_median_diff']=feature['after_loan_credit_amount_median']-feature['after_loan_amount_of_bill_left_median']
feature['after_loan_avail_amount_median_and_after_loan_prepare_amount_median_diff']=feature['after_loan_avail_amount_median']-feature['after_loan_prepare_amount_median']
feature['after_loan_least_repayment_median_and_after_loan_circ_interest_median_sum']=feature['after_loan_least_repayment_median']+feature['after_loan_circ_interest_median']

after_loan_billstd=gb.std()
after_loan_billstd.columns = ['userid', 'after_loan_pre_amount_of_billstd', 'after_loan_pre_repayment_std','after_loan_credit_amount_std','after_loan_amount_of_bill_left_std',
                     'after_loan_least_repayment_std','after_loan_consume_amount_std','after_loan_amount_of_billstd','after_loan_adjust_amount_std','after_loan_circ_intereststd',
                     'after_loan_avail_amount_std','after_loan_prepare_amount_std']
feature=pd.merge(feature, after_loan_billstd,how='left', on = "userid")
feature['after_loan_pre_repayment_std_and_after_loan_pre_amount_of_billstd_diff']=feature['after_loan_pre_repayment_std']-feature['after_loan_pre_amount_of_billstd']
feature['after_loan_credit_amount_std_and_after_loan_amount_of_bill_left_std_diff']=feature['after_loan_credit_amount_std']-feature['after_loan_amount_of_bill_left_std']
feature['after_loan_avail_amount_std_and_after_loan_prepare_amount_std_diff']=feature['after_loan_avail_amount_std']-feature['after_loan_prepare_amount_std']
feature['after_loan_least_repayment_std_and_after_loan_circ_intereststd_sum']=feature['after_loan_least_repayment_std']+feature['after_loan_circ_intereststd']

after_loan_bill_var=gb.var()
after_loan_bill_var.columns = ['userid', 'after_loan_pre_amount_of_bill_var', 'after_loan_pre_repayment_var','after_loan_credit_amount_var','after_loan_amount_of_bill_left_var',
                     'after_loan_least_repayment_var','after_loan_consume_amount_var','after_loan_amount_of_bill_var','after_loan_adjust_amount_var','after_loan_circ_interestvar',
                     'after_loan_avail_amount_var','after_loan_prepare_amount_var']
feature=pd.merge(feature, after_loan_bill_var,how='left', on = "userid")
feature['after_loan_pre_repayment_var_and_after_loan_pre_amount_of_bill_var_diff']=feature['after_loan_pre_repayment_var']-feature['after_loan_pre_amount_of_bill_var']
feature['after_loan_credit_amount_var_and_after_loan_amount_of_bill_left_var_diff']=feature['after_loan_credit_amount_var']-feature['after_loan_amount_of_bill_left_var']
feature['after_loan_avail_amount_var_and_after_loan_prepare_amount_var_diff']=feature['after_loan_avail_amount_var']-feature['after_loan_prepare_amount_var']
feature['after_loan_least_repayment_var_and_after_loan_circ_interestvar_sum']=feature['after_loan_least_repayment_var']+feature['after_loan_circ_interestvar']    
#################################################
#            drop-duplicate                     #
#################################################

feature_copy=feature
#358 columns now 
data=d[(d['time']>d['loan_time'])].loc[:,['userid','time','bank_id','pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left'
                                  ,'least_repayment','consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount'
                                  ,'prepare_amount']].groupby(["userid","time","bank_id"],as_index=False).max()

gb=data.loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
               'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount']].groupby(["userid"],as_index=False)

drop_d_after_loan_bill_sum=gb.sum()
drop_d_after_loan_bill_sum.columns = ['userid', 'drop_d_after_loan_pre_amount_of_bill_sum', 'drop_d_after_loan_pre_repayment_sum','drop_d_after_loan_credit_amount_sum'
                       ,'drop_d_after_loan_amount_of_bill_left_sum','drop_d_after_loan_least_repayment_sum','drop_d_after_loan_consume_amount_sum'
                       ,'drop_d_after_loan_amount_of_bill_sum','drop_d_after_loan_adjust_amount_sum','drop_d_after_loan_circ_interest_sum','drop_d_after_loan_avail_amount_sum'
                       ,'drop_d_after_loan_prepare_amount_sum']
feature=pd.merge(feature, drop_d_after_loan_bill_sum,how='left', on = "userid")
feature['drop_d_after_loan_pre_repayment_sum_and_after_loan_pre_amount_of_bill_sum_diff']=feature['drop_d_after_loan_pre_repayment_sum']-feature['drop_d_after_loan_pre_amount_of_bill_sum']
feature['drop_d_after_loan_credit_amount_sum_and_after_loan_amount_of_bill_left_sum_diff']=feature['drop_d_after_loan_credit_amount_sum']-feature['drop_d_after_loan_amount_of_bill_left_sum']
feature['drop_d_after_loan_avail_amount_sum_and_after_loan_prepare_amount_sum_diff']=feature['drop_d_after_loan_avail_amount_sum']-feature['drop_d_after_loan_prepare_amount_sum']
feature['drop_d_after_loan_least_repayment_sum_and_after_loan_circ_interest_sum_sum']=feature['drop_d_after_loan_least_repayment_sum']+feature['drop_d_after_loan_circ_interest_sum']

drop_d_after_loan_bill_count=gb.count()
drop_d_after_loan_bill_count.columns = ['userid', 'drop_d_after_loan_pre_amount_of_bill_count', 'drop_d_after_loan_pre_repayment_count','drop_d_after_loan_credit_amount_count'
                         ,'drop_d_after_loan_amount_of_bill_left_count','drop_d_after_loan_least_repayment_count','drop_d_after_loan_consume_amount_count'
                         ,'drop_d_after_loan_amount_of_bill_count','drop_d_after_loan_adjust_amount_count','drop_d_after_loan_circ_interestcount'
                         ,'drop_d_after_loan_avail_amount_count','drop_d_after_loan_prepare_amount_count']
feature=pd.merge(feature, drop_d_after_loan_bill_count,how='left', on = "userid")

drop_d_after_loan_bill_max=gb.max()
drop_d_after_loan_bill_max.columns = ['userid', 'drop_d_after_loan_pre_amount_of_bill_max', 'drop_d_after_loan_pre_repayment_max','drop_d_after_loan_credit_amount_max'
                       ,'drop_d_after_loan_amount_of_bill_left_max','drop_d_after_loan_least_repayment_max','drop_d_after_loan_consume_amount_max'
                       ,'drop_d_after_loan_amount_of_bill_max','drop_d_after_loan_adjust_amount_max','drop_d_after_loan_circ_interest_max'
                       ,'drop_d_after_loan_avail_amount_max','drop_d_after_loan_prepare_amount_max']
feature=pd.merge(feature, drop_d_after_loan_bill_max,how='left', on = "userid")
feature['drop_d_after_loan_pre_repayment_max_and_after_loan_pre_amount_of_bill_max_diff']=feature['drop_d_after_loan_pre_repayment_max']-feature['drop_d_after_loan_pre_amount_of_bill_max']
feature['drop_d_after_loan_credit_amount_max_and_after_loan_amount_of_bill_left_max_diff']=feature['drop_d_after_loan_credit_amount_max']-feature['drop_d_after_loan_amount_of_bill_left_max']
feature['drop_d_after_loan_avail_amount_max_and_after_loan_prepare_amount_max_diff']=feature['drop_d_after_loan_avail_amount_max']-feature['drop_d_after_loan_prepare_amount_max']
feature['drop_d_after_loan_least_repayment_max_and_after_loan_circ_interest_max_sum']=feature['drop_d_after_loan_least_repayment_max']+feature['drop_d_after_loan_circ_interest_max']

drop_d_after_loan_bill_min=gb.min()
drop_d_after_loan_bill_min.columns = ['userid', 'drop_d_after_loan_pre_amount_of_bill_min', 'drop_d_after_loan_pre_interest_min','drop_d_after_loan_credit_amount_min'
                       ,'drop_d_after_loan_amount_of_bill_left_min','drop_d_after_loan_least_interest_min','drop_d_after_loan_consume_amount_min'
                       ,'drop_d_after_loan_amount_of_bill_min','drop_d_after_loan_adjust_amount_min','drop_d_after_loan_circ_interest_min'
                       ,'drop_d_after_loan_avail_amount_min','drop_d_after_loan_prepare_amount_min']
feature=pd.merge(feature, drop_d_after_loan_bill_min,how='left', on = "userid")
feature['drop_d_after_loan_pre_interest_min_and_after_loan_pre_amount_of_bill_min_diff']=feature['drop_d_after_loan_pre_interest_min']-feature['drop_d_after_loan_pre_amount_of_bill_min']
feature['drop_d_after_loan_credit_amount_min_and_after_loan_amount_of_bill_left_min_diff']=feature['drop_d_after_loan_credit_amount_min']-feature['drop_d_after_loan_amount_of_bill_left_min']
feature['drop_d_after_loan_avail_amount_min_and_after_loan_prepare_amount_min_diff']=feature['drop_d_after_loan_avail_amount_min']-feature['drop_d_after_loan_prepare_amount_min']
feature['drop_d_after_loan_least_interest_min_and_after_loan_circ_interest_min_sum']=feature['drop_d_after_loan_least_interest_min']+feature['drop_d_after_loan_circ_interest_min']

drop_d_after_loan_billmean=gb.mean()
drop_d_after_loan_billmean.columns = ['userid', 'drop_d_after_loan_pre_amount_of_billmean', 'drop_d_after_loan_pre_interest_mean','drop_d_after_loan_credit_amount_mean'
                        ,'drop_d_after_loan_amount_of_bill_leftmean','drop_d_after_loan_least_interest_mean','drop_d_after_loan_consume_amount_mean'
                        ,'drop_d_after_loan_amount_of_billmean','drop_d_after_loan_adjust_amount_mean','drop_d_after_loan_circ_interestmean'
                        ,'drop_d_after_loan_avail_amount_mean','drop_d_after_loan_prepare_amount_mean']
feature=pd.merge(feature, drop_d_after_loan_billmean,how='left', on = "userid")
feature['drop_d_after_loan_pre_interest_mean_and_after_loan_pre_amount_of_billmean_diff']=feature['drop_d_after_loan_pre_interest_mean']-feature['drop_d_after_loan_pre_amount_of_billmean']
feature['drop_d_after_loan_credit_amount_mean_and_after_loan_amount_of_bill_leftmean_diff']=feature['drop_d_after_loan_credit_amount_mean']-feature['drop_d_after_loan_amount_of_bill_leftmean']
feature['drop_d_after_loan_avail_amount_mean_and_after_loan_prepare_amount_mean_diff']=feature['drop_d_after_loan_avail_amount_mean']-feature['drop_d_after_loan_prepare_amount_mean']
feature['drop_d_after_loan_least_interest_mean_and_after_loan_circ_interestmean_sum']=feature['drop_d_after_loan_least_interest_mean']+feature['drop_d_after_loan_circ_interestmean']


drop_d_after_loan_bill_median=gb.median()
drop_d_after_loan_bill_median.columns = ['userid', 'drop_d_after_loan_pre_amount_of_bill_median', 'drop_d_after_loan_pre_repayment_median'
                          ,'drop_d_after_loan_credit_amount_median','drop_d_after_loan_amount_of_bill_left_median','drop_d_after_loan_least_repayment_median'
                          ,'drop_d_after_loan_consume_amount_median','drop_d_after_loan_amount_of_bill_median','drop_d_after_loan_adjust_amount_median'
                          ,'drop_d_after_loan_circ_interest_median','drop_d_after_loan_avail_amount_median','drop_d_after_loan_prepare_amount_median']
feature=pd.merge(feature, drop_d_after_loan_bill_median,how='left', on = "userid")
feature['drop_d_after_loan_pre_repayment_median_and_after_loan_pre_amount_of_bill_median_diff']=feature['drop_d_after_loan_pre_repayment_median']-feature['drop_d_after_loan_pre_amount_of_bill_median']
feature['drop_d_after_loan_credit_amount_median_and_after_loan_amount_of_bill_left_median_diff']=feature['drop_d_after_loan_credit_amount_median']-feature['drop_d_after_loan_amount_of_bill_left_median']
feature['drop_d_after_loan_avail_amount_median_and_after_loan_prepare_amount_median_diff']=feature['drop_d_after_loan_avail_amount_median']-feature['drop_d_after_loan_prepare_amount_median']
feature['drop_d_after_loan_least_repayment_median_and_after_loan_circ_interest_median_sum']=feature['drop_d_after_loan_least_repayment_median']+feature['drop_d_after_loan_circ_interest_median']


drop_d_after_loan_billstd=gb.std()
drop_d_after_loan_billstd.columns = ['userid', 'drop_d_after_loan_pre_amount_of_billstd', 'drop_d_after_loan_pre_repayment_std','drop_d_after_loan_credit_amount_std'
                       ,'drop_d_after_loan_amount_of_bill_left_std','drop_d_after_loan_least_repayment_std','drop_d_after_loan_consume_amount_std'
                       ,'drop_d_after_loan_amount_of_billstd','drop_d_after_loan_adjust_amount_std','drop_d_after_loan_circ_intereststd'
                       ,'drop_d_after_loan_avail_amount_std','drop_d_after_loan_prepare_amount_std']
feature=pd.merge(feature, drop_d_after_loan_billstd,how='left', on = "userid")
feature['drop_d_after_loan_pre_repayment_std_and_after_loan_pre_amount_of_billstd_diff']=feature['drop_d_after_loan_pre_repayment_std']-feature['drop_d_after_loan_pre_amount_of_billstd']
feature['drop_d_after_loan_credit_amount_std_and_after_loan_amount_of_bill_left_std_diff']=feature['drop_d_after_loan_credit_amount_std']-feature['drop_d_after_loan_amount_of_bill_left_std']
feature['drop_d_after_loan_avail_amount_std_and_after_loan_prepare_amount_std_diff']=feature['drop_d_after_loan_avail_amount_std']-feature['drop_d_after_loan_prepare_amount_std']
feature['drop_d_after_loan_least_repayment_std_and_after_loan_circ_intereststd_sum']=feature['drop_d_after_loan_least_repayment_std']+feature['drop_d_after_loan_circ_intereststd']

drop_d_after_loan_bill_var=gb.var()
drop_d_after_loan_bill_var.columns = ['userid', 'drop_d_after_loan_pre_amount_of_bill_var', 'drop_d_after_loan_pre_repayment_var','drop_d_after_loan_credit_amount_var'
                       ,'drop_d_after_loan_amount_of_bill_left_var','drop_d_after_loan_least_repayment_var','drop_d_after_loan_consume_amount_var'
                       ,'drop_d_after_loan_amount_of_bill_var','drop_d_after_loan_adjust_amount_var','drop_d_after_loan_circ_interestvar','drop_d_after_loan_avail_amount_var','drop_d_after_loan_prepare_amount_var']
feature=pd.merge(feature, drop_d_after_loan_bill_var,how='left', on = "userid")
feature['drop_d_after_loan_pre_repayment_var_and_after_loan_pre_amount_of_bill_var_diff']=feature['drop_d_after_loan_pre_repayment_var']-feature['drop_d_after_loan_pre_amount_of_bill_var']
feature['drop_d_after_loan_credit_amount_var_and_after_loan_amount_of_bill_left_var_diff']=feature['drop_d_after_loan_credit_amount_var']-feature['drop_d_after_loan_amount_of_bill_left_var']
feature['drop_d_after_loan_avail_amount_var_and_after_loan_prepare_amount_var_diff']=feature['drop_d_after_loan_avail_amount_var']-feature['drop_d_after_loan_prepare_amount_var']
feature['drop_d_after_loan_least_repayment_var_and_after_loan_circ_interestvar_sum']=feature['drop_d_after_loan_least_repayment_var']+feature['drop_d_after_loan_circ_interestvar']

feature.to_csv("feature_time_known.csv",index=None)
#55596 rows × 474 columns

#credit card bill detail table
#################################################
#               time is unknown                 #
#################################################
#time is unknown
train_bill_detail['time'].describe()

d=train_bill_detail[(train_bill_detail['time']==0)]
#do the above processing agian

feature=train_loan_time[['userid']]
gb=d.loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
                                  'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount','repayment_state']].groupby(["userid"],as_index=False)

time_unknown_bill_sum=gb.sum()
time_unknown_bill_sum.columns = ['userid', 'time_unknown_pre_amount_of_bill_sum', 'time_unknown_pre_repayment_sum','time_unknown_credit_amount_sum','time_unknown_amount_of_bill_left_sum',
                     'time_unknown_least_repayment_sum','time_unknown_consume_amount_sum','time_unknown_amount_of_bill_sum','time_unknown_adjust_amount_sum','time_unknown_circ_interest_sum',
                     'time_unknown_avail_amount_sum','time_unknown_prepare_amount_sum','time_unknown_repayment_state_sum']
feature=pd.merge(feature, time_unknown_bill_sum,how='left', on = "userid")
feature['time_unknown_pre_repayment_sum_and_time_unknown_pre_amount_of_bill_sum_diff']=feature['time_unknown_pre_repayment_sum']-feature['time_unknown_pre_amount_of_bill_sum']
feature['time_unknown_credit_amount_sum_and_time_unknown_amount_of_bill_left_sum_diff']=feature['time_unknown_credit_amount_sum']-feature['time_unknown_amount_of_bill_left_sum']
feature['time_unknown_avail_amount_sum_and_time_unknown_prepare_amount_sum_diff']=feature['time_unknown_avail_amount_sum']-feature['time_unknown_prepare_amount_sum']
feature['time_unknown_least_repayment_sum_and_time_unknown_circ_interest_sum_sum']=feature['time_unknown_least_repayment_sum']+feature['time_unknown_circ_interest_sum']

time_unknown_bill_count=gb.count()
time_unknown_bill_count.columns = ['userid', 'time_unknown_pre_amount_of_bill_count', 'time_unknown_pre_repayment_count','time_unknown_credit_amount_count','time_unknown_amount_of_bill_left_count',
                     'time_unknown_least_repayment_count','time_unknown_consume_amount_count','time_unknown_amount_of_bill_count','time_unknown_adjust_amount_count','time_unknown_circ_interestcount',
                     'time_unknown_avail_amount_count','time_unknown_prepare_amount_count','time_unknown_repayment_state_count']
feature=pd.merge(feature, time_unknown_bill_count,how='left', on = "userid")

time_unknown_bill_max=gb.max()
time_unknown_bill_max.columns = ['userid', 'time_unknown_pre_amount_of_bill_max', 'time_unknown_pre_repayment_max','time_unknown_credit_amount_max','time_unknown_amount_of_bill_left_max',
                     'time_unknown_least_repayment_max','time_unknown_consume_amount_max','time_unknown_amount_of_bill_max','time_unknown_adjust_amount_max','time_unknown_circ_interest_max',
                     'time_unknown_avail_amount_max','time_unknown_prepare_amount_max','time_unknown_repayment_state_max']

feature=pd.merge(feature, time_unknown_bill_max,how='left', on = "userid")
feature['time_unknown_pre_repayment_max_and_time_unknown_pre_amount_of_bill_max_diff']=feature['time_unknown_pre_repayment_max']-feature['time_unknown_pre_amount_of_bill_max']
feature['time_unknown_credit_amount_max_and_time_unknown_amount_of_bill_left_max_diff']=feature['time_unknown_credit_amount_max']-feature['time_unknown_amount_of_bill_left_max']
feature['time_unknown_avail_amount_max_and_time_unknown_prepare_amount_max_diff']=feature['time_unknown_avail_amount_max']-feature['time_unknown_prepare_amount_max']
feature['time_unknown_least_repayment_max_and_time_unknown_circ_interest_max_sum']=feature['time_unknown_least_repayment_max']+feature['time_unknown_circ_interest_max']

time_unknown_bill_min=gb.min()
time_unknown_bill_min.columns = ['userid', 'time_unknown_pre_amount_of_bill_min', 'time_unknown_pre_interest_min','time_unknown_credit_amount_min','time_unknown_amount_of_bill_left_min',
                     'time_unknown_least_interest_min','time_unknown_consume_amount_min','time_unknown_amount_of_bill_min','time_unknown_adjust_amount_min','time_unknown_circ_interest_min',
                     'time_unknown_avail_amount_min','time_unknown_prepare_amount_min','time_unknown_repayment_state_min']
feature=pd.merge(feature, time_unknown_bill_min,how='left', on = "userid")
feature['time_unknown_pre_interest_min_and_time_unknown_pre_amount_of_bill_min_diff']=feature['time_unknown_pre_interest_min']-feature['time_unknown_pre_amount_of_bill_min']
feature['time_unknown_credit_amount_min_and_time_unknown_amount_of_bill_left_min_diff']=feature['time_unknown_credit_amount_min']-feature['time_unknown_amount_of_bill_left_min']
feature['time_unknown_avail_amount_min_and_time_unknown_prepare_amount_min_diff']=feature['time_unknown_avail_amount_min']-feature['time_unknown_prepare_amount_min']
feature['time_unknown_least_interest_min_and_time_unknown_circ_interest_min_sum']=feature['time_unknown_least_interest_min']+feature['time_unknown_circ_interest_min']

time_unknown_billmean=gb.mean()
time_unknown_billmean.columns = ['userid', 'time_unknown_pre_amount_of_billmean', 'time_unknown_pre_interest_mean','time_unknown_credit_amount_mean','time_unknown_amount_of_bill_leftmean',
                     'time_unknown_least_interest_mean','time_unknown_consume_amount_mean','time_unknown_amount_of_billmean','time_unknown_adjust_amount_mean','time_unknown_circ_interestmean',
                     'time_unknown_avail_amount_mean','time_unknown_prepare_amount_mean','time_unknown_repayment_state_mean']
feature=pd.merge(feature, time_unknown_billmean,how='left', on = "userid")
feature['time_unknown_pre_interest_mean_and_time_unknown_pre_amount_of_billmean_diff']=feature['time_unknown_pre_interest_mean']-feature['time_unknown_pre_amount_of_billmean']
feature['time_unknown_credit_amount_mean_and_time_unknown_amount_of_bill_leftmean_diff']=feature['time_unknown_credit_amount_mean']-feature['time_unknown_amount_of_bill_leftmean']
feature['time_unknown_avail_amount_mean_and_time_unknown_prepare_amount_mean_diff']=feature['time_unknown_avail_amount_mean']-feature['time_unknown_prepare_amount_mean']
feature['time_unknown_least_interest_mean_and_time_unknown_circ_interestmean_sum']=feature['time_unknown_least_interest_mean']+feature['time_unknown_circ_interestmean']

time_unknown_bill_median=gb.median()
time_unknown_bill_median.columns = ['userid', 'time_unknown_pre_amount_of_bill_median', 'time_unknown_pre_repayment_median','time_unknown_credit_amount_median','time_unknown_amount_of_bill_left_median',
                  'time_unknown_least_repayment_median','time_unknown_consume_amount_median','time_unknown_amount_of_bill_median','time_unknown_adjust_amount_median',
                  'time_unknown_circ_interest_median','time_unknown_avail_amount_median','time_unknown_prepare_amount_median','time_unknown_repayment_state_median']
feature=pd.merge(feature, time_unknown_bill_median,how='left', on = "userid")
feature['time_unknown_pre_repayment_median_and_time_unknown_pre_amount_of_bill_median_diff']=feature['time_unknown_pre_repayment_median']-feature['time_unknown_pre_amount_of_bill_median']
feature['time_unknown_credit_amount_median_and_time_unknown_amount_of_bill_left_median_diff']=feature['time_unknown_credit_amount_median']-feature['time_unknown_amount_of_bill_left_median']
feature['time_unknown_avail_amount_median_and_time_unknown_prepare_amount_median_diff']=feature['time_unknown_avail_amount_median']-feature['time_unknown_prepare_amount_median']
feature['time_unknown_least_repayment_median_and_time_unknown_circ_interest_median_sum']=feature['time_unknown_least_repayment_median']+feature['time_unknown_circ_interest_median']

time_unknown_billstd=gb.std()
time_unknown_billstd.columns = ['userid', 'time_unknown_pre_amount_of_billstd', 'time_unknown_pre_repayment_std','time_unknown_credit_amount_std','time_unknown_amount_of_bill_left_std',
                     'time_unknown_least_repayment_std','time_unknown_consume_amount_std','time_unknown_amount_of_billstd','time_unknown_adjust_amount_std','time_unknown_circ_intereststd',
                     'time_unknown_avail_amount_std','time_unknown_prepare_amount_std','time_unknown_repayment_statestd']
feature=pd.merge(feature, time_unknown_billstd,how='left', on = "userid")
feature['time_unknown_pre_repayment_std_and_time_unknown_pre_amount_of_billstd_diff']=feature['time_unknown_pre_repayment_std']-feature['time_unknown_pre_amount_of_billstd']
feature['time_unknown_credit_amount_std_and_time_unknown_amount_of_bill_left_std_diff']=feature['time_unknown_credit_amount_std']-feature['time_unknown_amount_of_bill_left_std']
feature['time_unknown_avail_amount_std_and_time_unknown_prepare_amount_std_diff']=feature['time_unknown_avail_amount_std']-feature['time_unknown_prepare_amount_std']
feature['time_unknown_least_repayment_std_and_time_unknown_circ_intereststd_sum']=feature['time_unknown_least_repayment_std']+feature['time_unknown_circ_intereststd']

time_unknown_bill_var=gb.var()
time_unknown_bill_var.columns = ['userid', 'time_unknown_pre_amount_of_bill_var', 'time_unknown_pre_repayment_var','time_unknown_credit_amount_var','time_unknown_amount_of_bill_left_var',
                     'time_unknown_least_repayment_var','time_unknown_consume_amount_var','time_unknown_amount_of_bill_var','time_unknown_adjust_amount_var','time_unknown_circ_interestvar',
                     'time_unknown_avail_amount_var','time_unknown_prepare_amount_var','time_unknown_repayment_statevar']
feature=pd.merge(feature, time_unknown_bill_var,how='left', on = "userid")
feature['time_unknown_pre_repayment_var_and_time_unknown_pre_amount_of_bill_var_diff']=feature['time_unknown_pre_repayment_var']-feature['time_unknown_pre_amount_of_bill_var']
feature['time_unknown_credit_amount_var_and_time_unknown_amount_of_bill_left_var_diff']=feature['time_unknown_credit_amount_var']-feature['time_unknown_amount_of_bill_left_var']
feature['time_unknown_avail_amount_var_and_time_unknown_prepare_amount_var_diff']=feature['time_unknown_avail_amount_var']-feature['time_unknown_prepare_amount_var']
feature['time_unknown_least_repayment_var_and_time_unknown_circ_interestvar_sum']=feature['time_unknown_least_repayment_var']+feature['time_unknown_circ_interestvar']
feature.shape
#55596 rows × 125 columns

#################################################
#                drop-duplicate                 #
#################################################


feature_copy=feature
data=d.loc[:,['userid','time','bank_id','pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left'
              ,'least_repayment','consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount'
              ,'prepare_amount']].groupby(["userid","time","bank_id"],as_index=False).max()

gb=data.loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
               'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount']].groupby(["userid"],as_index=False)

drop_d_time_unknown_bill_sum=gb.sum()
drop_d_time_unknown_bill_sum.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_bill_sum', 'drop_d_time_unknown_pre_repayment_sum','drop_d_time_unknown_credit_amount_sum'
                       ,'drop_d_time_unknown_amount_of_bill_left_sum','drop_d_time_unknown_least_repayment_sum','drop_d_time_unknown_consume_amount_sum'
                       ,'drop_d_time_unknown_amount_of_bill_sum','drop_d_time_unknown_adjust_amount_sum','drop_d_time_unknown_circ_interest_sum','drop_d_time_unknown_avail_amount_sum'
                       ,'drop_d_time_unknown_prepare_amount_sum']
feature=pd.merge(feature, drop_d_time_unknown_bill_sum,how='left', on = "userid")
feature['drop_d_time_unknown_pre_repayment_sum_and_time_unknown_pre_amount_of_bill_sum_diff']=feature['drop_d_time_unknown_pre_repayment_sum']-feature['drop_d_time_unknown_pre_amount_of_bill_sum']
feature['drop_d_time_unknown_credit_amount_sum_and_time_unknown_amount_of_bill_left_sum_diff']=feature['drop_d_time_unknown_credit_amount_sum']-feature['drop_d_time_unknown_amount_of_bill_left_sum']
feature['drop_d_time_unknown_avail_amount_sum_and_time_unknown_prepare_amount_sum_diff']=feature['drop_d_time_unknown_avail_amount_sum']-feature['drop_d_time_unknown_prepare_amount_sum']
feature['drop_d_time_unknown_least_repayment_sum_and_time_unknown_circ_interest_sum_sum']=feature['drop_d_time_unknown_least_repayment_sum']+feature['drop_d_time_unknown_circ_interest_sum']

drop_d_time_unknown_bill_count=gb.count()
drop_d_time_unknown_bill_count.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_bill_count', 'drop_d_time_unknown_pre_repayment_count','drop_d_time_unknown_credit_amount_count'
                         ,'drop_d_time_unknown_amount_of_bill_left_count','drop_d_time_unknown_least_repayment_count','drop_d_time_unknown_consume_amount_count'
                         ,'drop_d_time_unknown_amount_of_bill_count','drop_d_time_unknown_adjust_amount_count','drop_d_time_unknown_circ_interestcount'
                         ,'drop_d_time_unknown_avail_amount_count','drop_d_time_unknown_prepare_amount_count']
feature=pd.merge(feature, drop_d_time_unknown_bill_count,how='left', on = "userid")

drop_d_time_unknown_bill_max=gb.max()
drop_d_time_unknown_bill_max.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_bill_max', 'drop_d_time_unknown_pre_repayment_max','drop_d_time_unknown_credit_amount_max'
                       ,'drop_d_time_unknown_amount_of_bill_left_max','drop_d_time_unknown_least_repayment_max','drop_d_time_unknown_consume_amount_max'
                       ,'drop_d_time_unknown_amount_of_bill_max','drop_d_time_unknown_adjust_amount_max','drop_d_time_unknown_circ_interest_max'
                       ,'drop_d_time_unknown_avail_amount_max','drop_d_time_unknown_prepare_amount_max']
feature=pd.merge(feature, drop_d_time_unknown_bill_max,how='left', on = "userid")
feature['drop_d_time_unknown_pre_repayment_max_and_time_unknown_pre_amount_of_bill_max_diff']=feature['drop_d_time_unknown_pre_repayment_max']-feature['drop_d_time_unknown_pre_amount_of_bill_max']
feature['drop_d_time_unknown_credit_amount_max_and_time_unknown_amount_of_bill_left_max_diff']=feature['drop_d_time_unknown_credit_amount_max']-feature['drop_d_time_unknown_amount_of_bill_left_max']
feature['drop_d_time_unknown_avail_amount_max_and_time_unknown_prepare_amount_max_diff']=feature['drop_d_time_unknown_avail_amount_max']-feature['drop_d_time_unknown_prepare_amount_max']
feature['drop_d_time_unknown_least_repayment_max_and_time_unknown_circ_interest_max_sum']=feature['drop_d_time_unknown_least_repayment_max']+feature['drop_d_time_unknown_circ_interest_max']

drop_d_time_unknown_bill_min=gb.min()
drop_d_time_unknown_bill_min.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_bill_min', 'drop_d_time_unknown_pre_interest_min','drop_d_time_unknown_credit_amount_min'
                       ,'drop_d_time_unknown_amount_of_bill_left_min','drop_d_time_unknown_least_interest_min','drop_d_time_unknown_consume_amount_min'
                       ,'drop_d_time_unknown_amount_of_bill_min','drop_d_time_unknown_adjust_amount_min','drop_d_time_unknown_circ_interest_min'
                       ,'drop_d_time_unknown_avail_amount_min','drop_d_time_unknown_prepare_amount_min']
feature=pd.merge(feature, drop_d_time_unknown_bill_min,how='left', on = "userid")
feature['drop_d_time_unknown_pre_interest_min_and_time_unknown_pre_amount_of_bill_min_diff']=feature['drop_d_time_unknown_pre_interest_min']-feature['drop_d_time_unknown_pre_amount_of_bill_min']
feature['drop_d_time_unknown_credit_amount_min_and_time_unknown_amount_of_bill_left_min_diff']=feature['drop_d_time_unknown_credit_amount_min']-feature['drop_d_time_unknown_amount_of_bill_left_min']
feature['drop_d_time_unknown_avail_amount_min_and_time_unknown_prepare_amount_min_diff']=feature['drop_d_time_unknown_avail_amount_min']-feature['drop_d_time_unknown_prepare_amount_min']
feature['drop_d_time_unknown_least_interest_min_and_time_unknown_circ_interest_min_sum']=feature['drop_d_time_unknown_least_interest_min']+feature['drop_d_time_unknown_circ_interest_min']

drop_d_time_unknown_billmean=gb.mean()
drop_d_time_unknown_billmean.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_billmean', 'drop_d_time_unknown_pre_interest_mean','drop_d_time_unknown_credit_amount_mean'
                        ,'drop_d_time_unknown_amount_of_bill_leftmean','drop_d_time_unknown_least_interest_mean','drop_d_time_unknown_consume_amount_mean'
                        ,'drop_d_time_unknown_amount_of_billmean','drop_d_time_unknown_adjust_amount_mean','drop_d_time_unknown_circ_interestmean'
                        ,'drop_d_time_unknown_avail_amount_mean','drop_d_time_unknown_prepare_amount_mean']
feature=pd.merge(feature, drop_d_time_unknown_billmean,how='left', on = "userid")
feature['drop_d_time_unknown_pre_interest_mean_and_time_unknown_pre_amount_of_billmean_diff']=feature['drop_d_time_unknown_pre_interest_mean']-feature['drop_d_time_unknown_pre_amount_of_billmean']
feature['drop_d_time_unknown_credit_amount_mean_and_time_unknown_amount_of_bill_leftmean_diff']=feature['drop_d_time_unknown_credit_amount_mean']-feature['drop_d_time_unknown_amount_of_bill_leftmean']
feature['drop_d_time_unknown_avail_amount_mean_and_time_unknown_prepare_amount_mean_diff']=feature['drop_d_time_unknown_avail_amount_mean']-feature['drop_d_time_unknown_prepare_amount_mean']
feature['drop_d_time_unknown_least_interest_mean_and_time_unknown_circ_interestmean_sum']=feature['drop_d_time_unknown_least_interest_mean']+feature['drop_d_time_unknown_circ_interestmean']


drop_d_time_unknown_bill_median=gb.median()
drop_d_time_unknown_bill_median.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_bill_median', 'drop_d_time_unknown_pre_repayment_median'
                          ,'drop_d_time_unknown_credit_amount_median','drop_d_time_unknown_amount_of_bill_left_median','drop_d_time_unknown_least_repayment_median'
                          ,'drop_d_time_unknown_consume_amount_median','drop_d_time_unknown_amount_of_bill_median','drop_d_time_unknown_adjust_amount_median'
                          ,'drop_d_time_unknown_circ_interest_median','drop_d_time_unknown_avail_amount_median','drop_d_time_unknown_prepare_amount_median']
feature=pd.merge(feature, drop_d_time_unknown_bill_median,how='left', on = "userid")
feature['drop_d_time_unknown_pre_repayment_median_and_time_unknown_pre_amount_of_bill_median_diff']=feature['drop_d_time_unknown_pre_repayment_median']-feature['drop_d_time_unknown_pre_amount_of_bill_median']
feature['drop_d_time_unknown_credit_amount_median_and_time_unknown_amount_of_bill_left_median_diff']=feature['drop_d_time_unknown_credit_amount_median']-feature['drop_d_time_unknown_amount_of_bill_left_median']
feature['drop_d_time_unknown_avail_amount_median_and_time_unknown_prepare_amount_median_diff']=feature['drop_d_time_unknown_avail_amount_median']-feature['drop_d_time_unknown_prepare_amount_median']
feature['drop_d_time_unknown_least_repayment_median_and_time_unknown_circ_interest_median_sum']=feature['drop_d_time_unknown_least_repayment_median']+feature['drop_d_time_unknown_circ_interest_median']


drop_d_time_unknown_billstd=gb.std()
drop_d_time_unknown_billstd.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_billstd', 'drop_d_time_unknown_pre_repayment_std','drop_d_time_unknown_credit_amount_std'
                       ,'drop_d_time_unknown_amount_of_bill_left_std','drop_d_time_unknown_least_repayment_std','drop_d_time_unknown_consume_amount_std'
                       ,'drop_d_time_unknown_amount_of_billstd','drop_d_time_unknown_adjust_amount_std','drop_d_time_unknown_circ_intereststd'
                       ,'drop_d_time_unknown_avail_amount_std','drop_d_time_unknown_prepare_amount_std']
feature=pd.merge(feature, drop_d_time_unknown_billstd,how='left', on = "userid")
feature['drop_d_time_unknown_pre_repayment_std_and_time_unknown_pre_amount_of_billstd_diff']=feature['drop_d_time_unknown_pre_repayment_std']-feature['drop_d_time_unknown_pre_amount_of_billstd']
feature['drop_d_time_unknown_credit_amount_std_and_time_unknown_amount_of_bill_left_std_diff']=feature['drop_d_time_unknown_credit_amount_std']-feature['drop_d_time_unknown_amount_of_bill_left_std']
feature['drop_d_time_unknown_avail_amount_std_and_time_unknown_prepare_amount_std_diff']=feature['drop_d_time_unknown_avail_amount_std']-feature['drop_d_time_unknown_prepare_amount_std']
feature['drop_d_time_unknown_least_repayment_std_and_time_unknown_circ_intereststd_sum']=feature['drop_d_time_unknown_least_repayment_std']+feature['drop_d_time_unknown_circ_intereststd']

drop_d_time_unknown_bill_var=gb.var()
drop_d_time_unknown_bill_var.columns = ['userid', 'drop_d_time_unknown_pre_amount_of_bill_var', 'drop_d_time_unknown_pre_repayment_var','drop_d_time_unknown_credit_amount_var'
                       ,'drop_d_time_unknown_amount_of_bill_left_var','drop_d_time_unknown_least_repayment_var','drop_d_time_unknown_consume_amount_var'
                       ,'drop_d_time_unknown_amount_of_bill_var','drop_d_time_unknown_adjust_amount_var','drop_d_time_unknown_circ_interestvar','drop_d_time_unknown_avail_amount_var','drop_d_time_unknown_prepare_amount_var']
feature=pd.merge(feature, drop_d_time_unknown_bill_var,how='left', on = "userid")
feature['drop_d_time_unknown_pre_repayment_var_and_time_unknown_pre_amount_of_bill_var_diff']=feature['drop_d_time_unknown_pre_repayment_var']-feature['drop_d_time_unknown_pre_amount_of_bill_var']
feature['drop_d_time_unknown_credit_amount_var_and_time_unknown_amount_of_bill_left_var_diff']=feature['drop_d_time_unknown_credit_amount_var']-feature['drop_d_time_unknown_amount_of_bill_left_var']
feature['drop_d_time_unknown_avail_amount_var_and_time_unknown_prepare_amount_var_diff']=feature['drop_d_time_unknown_avail_amount_var']-feature['drop_d_time_unknown_prepare_amount_var']
feature['drop_d_time_unknown_least_repayment_var_and_time_unknown_circ_interestvar_sum']=feature['drop_d_time_unknown_least_repayment_var']+feature['drop_d_time_unknown_circ_interestvar']

feature.shape#55596 rows × 241 columns


feature.to_csv("feature_time_unknown.csv",index=None)


######################################################
#             no matter time is known or not         #
######################################################


#start from here
d=train_bill_detail
feature=train_loan_time[['userid']]
###additional feature，increase overall__stat_infosum count max min mean std var
gb=d.loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
                                  'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount','repayment_state']].groupby(["userid"],as_index=False)

overall_bill_sum=gb.sum()
overall_bill_sum.columns = ['userid', 'overall_pre_amount_of_bill_sum', 'overall_pre_repayment_sum','overall_credit_amount_sum','overall_amount_of_bill_left_sum',
                     'overall_least_repayment_sum','overall_consume_amount_sum','overall_amount_of_bill_sum','overall_adjust_amount_sum','overall_circ_interest_sum',
                     'overall_avail_amount_sum','overall_prepare_amount_sum','overall_repayment_state_sum']

feature=pd.merge(feature, overall_bill_sum,how='left', on = "userid")
feature['overall_pre_repayment_sum_and_overall_pre_amount_of_bill_sum_diff']=feature['overall_pre_repayment_sum']-feature['overall_pre_amount_of_bill_sum']
feature['overall_credit_amount_sum_and_overall_amount_of_bill_left_sum_diff']=feature['overall_credit_amount_sum']-feature['overall_amount_of_bill_left_sum']
feature['overall_avail_amount_sum_and_overall_prepare_amount_sum_diff']=feature['overall_avail_amount_sum']-feature['overall_prepare_amount_sum']
feature['overall_least_repayment_sum_and_overall_circ_interest_sum_sum']=feature['overall_least_repayment_sum']+feature['overall_circ_interest_sum']

overall_bill_count=gb.count()
overall_bill_count.columns = ['userid', 'overall_pre_amount_of_bill_count', 'overall_pre_repayment_count','overall_credit_amount_count','overall_amount_of_bill_left_count',
                     'overall_least_repayment_count','overall_consume_amount_count','overall_amount_of_bill_count','overall_adjust_amount_count','overall_circ_interestcount',
                     'overall_avail_amount_count','overall_prepare_amount_count','overall_repayment_state_count']
feature=pd.merge(feature, overall_bill_count,how='left', on = "userid")

overall_bill_max=gb.max()
overall_bill_max.columns = ['userid', 'overall_pre_amount_of_bill_max', 'overall_pre_repayment_max','overall_credit_amount_max','overall_amount_of_bill_left_max',
                     'overall_least_repayment_max','overall_consume_amount_max','overall_amount_of_bill_max','overall_adjust_amount_max','overall_circ_interest_max',
                     'overall_avail_amount_max','overall_prepare_amount_max','overall_repayment_state_max']
feature=pd.merge(feature, overall_bill_max,how='left', on = "userid")
feature['overall_pre_repayment_max_and_overall_pre_amount_of_bill_max_diff']=feature['overall_pre_repayment_max']-feature['overall_pre_amount_of_bill_max']
feature['overall_credit_amount_max_and_overall_amount_of_bill_left_max_diff']=feature['overall_credit_amount_max']-feature['overall_amount_of_bill_left_max']
feature['overall_avail_amount_max_and_overall_prepare_amount_max_diff']=feature['overall_avail_amount_max']-feature['overall_prepare_amount_max']
feature['overall_least_repayment_max_and_overall_circ_interest_max_sum']=feature['overall_least_repayment_max']+feature['overall_circ_interest_max']

overall_bill_min=gb.min()
overall_bill_min.columns = ['userid', 'overall_pre_amount_of_bill_min', 'overall_pre_interest_min','overall_credit_amount_min','overall_amount_of_bill_left_min',
                     'overall_least_interest_min','overall_consume_amount_min','overall_amount_of_bill_min','overall_adjust_amount_min','overall_circ_interest_min',
                     'overall_avail_amount_min','overall_prepare_amount_min','overall_repayment_state_min']
feature=pd.merge(feature, overall_bill_min,how='left', on = "userid")
feature['overall_pre_interest_min_and_overall_pre_amount_of_bill_min_diff']=feature['overall_pre_interest_min']-feature['overall_pre_amount_of_bill_min']
feature['overall_credit_amount_min_and_overall_amount_of_bill_left_min_diff']=feature['overall_credit_amount_min']-feature['overall_amount_of_bill_left_min']
feature['overall_avail_amount_min_and_overall_prepare_amount_min_diff']=feature['overall_avail_amount_min']-feature['overall_prepare_amount_min']
feature['overall_least_interest_min_and_overall_circ_interest_min_sum']=feature['overall_least_interest_min']+feature['overall_circ_interest_min']

overall_billmean=gb.mean()
overall_billmean.columns = ['userid', 'overall_pre_amount_of_billmean', 'overall_pre_interest_mean','overall_credit_amount_mean','overall_amount_of_bill_leftmean',
                     'overall_least_interest_mean','overall_consume_amount_mean','overall_amount_of_billmean','overall_adjust_amount_mean','overall_circ_interestmean',
                     'overall_avail_amount_mean','overall_prepare_amount_mean','overall_repayment_state_mean']
feature=pd.merge(feature, overall_billmean,how='left', on = "userid")
feature['overall_pre_interest_mean_and_overall_pre_amount_of_billmean_diff']=feature['overall_pre_interest_mean']-feature['overall_pre_amount_of_billmean']
feature['overall_credit_amount_mean_and_overall_amount_of_bill_leftmean_diff']=feature['overall_credit_amount_mean']-feature['overall_amount_of_bill_leftmean']
feature['overall_avail_amount_mean_and_overall_prepare_amount_mean_diff']=feature['overall_avail_amount_mean']-feature['overall_prepare_amount_mean']
feature['overall_least_interest_mean_and_overall_circ_interestmean_sum']=feature['overall_least_interest_mean']+feature['overall_circ_interestmean']

overall_bill_median=gb.median()
overall_bill_median.columns = ['userid', 'overall_pre_amount_of_bill_median', 'overall_pre_repayment_median','overall_credit_amount_median','overall_amount_of_bill_left_median',
                  'overall_least_repayment_median','overall_consume_amount_median','overall_amount_of_bill_median','overall_adjust_amount_median',
                  'overall_circ_interest_median','overall_avail_amount_median','overall_prepare_amount_median','overall_repayment_state_median']
feature=pd.merge(feature, overall_bill_median,how='left', on = "userid")
feature['overall_pre_repayment_median_and_overall_pre_amount_of_bill_median_diff']=feature['overall_pre_repayment_median']-feature['overall_pre_amount_of_bill_median']
feature['overall_credit_amount_median_and_overall_amount_of_bill_left_median_diff']=feature['overall_credit_amount_median']-feature['overall_amount_of_bill_left_median']
feature['overall_avail_amount_median_and_overall_prepare_amount_median_diff']=feature['overall_avail_amount_median']-feature['overall_prepare_amount_median']
feature['overall_least_repayment_median_and_overall_circ_interest_median_sum']=feature['overall_least_repayment_median']+feature['overall_circ_interest_median']

overall_billstd=gb.std()
overall_billstd.columns = ['userid', 'overall_pre_amount_of_billstd', 'overall_pre_repayment_std','overall_credit_amount_std','overall_amount_of_bill_left_std',
                     'overall_least_repayment_std','overall_consume_amount_std','overall_amount_of_billstd','overall_adjust_amount_std','overall_circ_intereststd',
                     'overall_avail_amount_std','overall_prepare_amount_std','overall_repayment_statestd']
feature=pd.merge(feature, overall_billstd,how='left', on = "userid")
feature['overall_pre_repayment_std_and_overall_pre_amount_of_billstd_diff']=feature['overall_pre_repayment_std']-feature['overall_pre_amount_of_billstd']
feature['overall_credit_amount_std_and_overall_amount_of_bill_left_std_diff']=feature['overall_credit_amount_std']-feature['overall_amount_of_bill_left_std']
feature['overall_avail_amount_std_and_overall_prepare_amount_std_diff']=feature['overall_avail_amount_std']-feature['overall_prepare_amount_std']
feature['overall_least_repayment_std_and_overall_circ_intereststd_sum']=feature['overall_least_repayment_std']+feature['overall_circ_intereststd']

overall_bill_var=gb.var()
overall_bill_var.columns = ['userid', 'overall_pre_amount_of_bill_var', 'overall_pre_repayment_var','overall_credit_amount_var','overall_amount_of_bill_left_var',
                     'overall_least_repayment_var','overall_consume_amount_var','overall_amount_of_bill_var','overall_adjust_amount_var','overall_circ_interestvar',
                     'overall_avail_amount_var','overall_prepare_amount_var','overall_repayment_statevar']
feature=pd.merge(feature, overall_bill_var,how='left', on = "userid")
feature['overall_pre_repayment_var_and_overall_pre_amount_of_bill_var_diff']=feature['overall_pre_repayment_var']-feature['overall_pre_amount_of_bill_var']
feature['overall_credit_amount_var_and_overall_amount_of_bill_left_var_diff']=feature['overall_credit_amount_var']-feature['overall_amount_of_bill_left_var']
feature['overall_avail_amount_var_and_overall_prepare_amount_var_diff']=feature['overall_avail_amount_var']-feature['overall_prepare_amount_var']
feature['overall_least_repayment_var_and_overall_circ_interestvar_sum']=feature['overall_least_repayment_var']+feature['overall_circ_interestvar']
feature.shape#55596 rows × 125 columns



#################################################
#                drop-duplicate                 #
#################################################

feature_copy=feature
data=d.loc[:,['userid','time','bank_id','pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left'
              ,'least_repayment','consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount'
              ,'prepare_amount']].groupby(["userid","time","bank_id"],as_index=False).max()

gb=data.loc[:,['userid', 'pre_amount_of_bill', 'pre_repayment','credit_amount','amount_of_bill_left','least_repayment',
               'consume_amount','amount_of_bill','adjust_amount','circ_interest','avail_amount','prepare_amount']].groupby(["userid"],as_index=False)

drop_d_overall_bill_sum=gb.sum()
drop_d_overall_bill_sum.columns = ['userid', 'drop_d_overall_pre_amount_of_bill_sum', 'drop_d_overall_pre_repayment_sum','drop_d_overall_credit_amount_sum'
                       ,'drop_d_overall_amount_of_bill_left_sum','drop_d_overall_least_repayment_sum','drop_d_overall_consume_amount_sum'
                       ,'drop_d_overall_amount_of_bill_sum','drop_d_overall_adjust_amount_sum','drop_d_overall_circ_interest_sum','drop_d_overall_avail_amount_sum'
                       ,'drop_d_overall_prepare_amount_sum']
feature=pd.merge(feature, drop_d_overall_bill_sum,how='left', on = "userid")
feature['drop_d_overall_pre_repayment_sum_and_overall_pre_amount_of_bill_sum_diff']=feature['drop_d_overall_pre_repayment_sum']-feature['drop_d_overall_pre_amount_of_bill_sum']
feature['drop_d_overall_credit_amount_sum_and_overall_amount_of_bill_left_sum_diff']=feature['drop_d_overall_credit_amount_sum']-feature['drop_d_overall_amount_of_bill_left_sum']
feature['drop_d_overall_avail_amount_sum_and_overall_prepare_amount_sum_diff']=feature['drop_d_overall_avail_amount_sum']-feature['drop_d_overall_prepare_amount_sum']
feature['drop_d_overall_least_repayment_sum_and_overall_circ_interest_sum_sum']=feature['drop_d_overall_least_repayment_sum']+feature['drop_d_overall_circ_interest_sum']

drop_d_overall_bill_count=gb.count()
drop_d_overall_bill_count.columns = ['userid', 'drop_d_overall_pre_amount_of_bill_count', 'drop_d_overall_pre_repayment_count','drop_d_overall_credit_amount_count'
                         ,'drop_d_overall_amount_of_bill_left_count','drop_d_overall_least_repayment_count','drop_d_overall_consume_amount_count'
                         ,'drop_d_overall_amount_of_bill_count','drop_d_overall_adjust_amount_count','drop_d_overall_circ_interestcount'
                         ,'drop_d_overall_avail_amount_count','drop_d_overall_prepare_amount_count']
feature=pd.merge(feature, drop_d_overall_bill_count,how='left', on = "userid")

drop_d_overall_bill_max=gb.max()
drop_d_overall_bill_max.columns = ['userid', 'drop_d_overall_pre_amount_of_bill_max', 'drop_d_overall_pre_repayment_max','drop_d_overall_credit_amount_max'
                       ,'drop_d_overall_amount_of_bill_left_max','drop_d_overall_least_repayment_max','drop_d_overall_consume_amount_max'
                       ,'drop_d_overall_amount_of_bill_max','drop_d_overall_adjust_amount_max','drop_d_overall_circ_interest_max'
                       ,'drop_d_overall_avail_amount_max','drop_d_overall_prepare_amount_max']
feature=pd.merge(feature, drop_d_overall_bill_max,how='left', on = "userid")
feature['drop_d_overall_pre_repayment_max_and_overall_pre_amount_of_bill_max_diff']=feature['drop_d_overall_pre_repayment_max']-feature['drop_d_overall_pre_amount_of_bill_max']
feature['drop_d_overall_credit_amount_max_and_overall_amount_of_bill_left_max_diff']=feature['drop_d_overall_credit_amount_max']-feature['drop_d_overall_amount_of_bill_left_max']
feature['drop_d_overall_avail_amount_max_and_overall_prepare_amount_max_diff']=feature['drop_d_overall_avail_amount_max']-feature['drop_d_overall_prepare_amount_max']
feature['drop_d_overall_least_repayment_max_and_overall_circ_interest_max_sum']=feature['drop_d_overall_least_repayment_max']+feature['drop_d_overall_circ_interest_max']

drop_d_overall_bill_min=gb.min()
drop_d_overall_bill_min.columns = ['userid', 'drop_d_overall_pre_amount_of_bill_min', 'drop_d_overall_pre_interest_min','drop_d_overall_credit_amount_min'
                       ,'drop_d_overall_amount_of_bill_left_min','drop_d_overall_least_interest_min','drop_d_overall_consume_amount_min'
                       ,'drop_d_overall_amount_of_bill_min','drop_d_overall_adjust_amount_min','drop_d_overall_circ_interest_min'
                       ,'drop_d_overall_avail_amount_min','drop_d_overall_prepare_amount_min']
feature=pd.merge(feature, drop_d_overall_bill_min,how='left', on = "userid")
feature['drop_d_overall_pre_interest_min_and_overall_pre_amount_of_bill_min_diff']=feature['drop_d_overall_pre_interest_min']-feature['drop_d_overall_pre_amount_of_bill_min']
feature['drop_d_overall_credit_amount_min_and_overall_amount_of_bill_left_min_diff']=feature['drop_d_overall_credit_amount_min']-feature['drop_d_overall_amount_of_bill_left_min']
feature['drop_d_overall_avail_amount_min_and_overall_prepare_amount_min_diff']=feature['drop_d_overall_avail_amount_min']-feature['drop_d_overall_prepare_amount_min']
feature['drop_d_overall_least_interest_min_and_overall_circ_interest_min_sum']=feature['drop_d_overall_least_interest_min']+feature['drop_d_overall_circ_interest_min']

drop_d_overall_billmean=gb.mean()
drop_d_overall_billmean.columns = ['userid', 'drop_d_overall_pre_amount_of_billmean', 'drop_d_overall_pre_interest_mean','drop_d_overall_credit_amount_mean'
                        ,'drop_d_overall_amount_of_bill_leftmean','drop_d_overall_least_interest_mean','drop_d_overall_consume_amount_mean'
                        ,'drop_d_overall_amount_of_billmean','drop_d_overall_adjust_amount_mean','drop_d_overall_circ_interestmean'
                        ,'drop_d_overall_avail_amount_mean','drop_d_overall_prepare_amount_mean']
feature=pd.merge(feature, drop_d_overall_billmean,how='left', on = "userid")
feature['drop_d_overall_pre_interest_mean_and_overall_pre_amount_of_billmean_diff']=feature['drop_d_overall_pre_interest_mean']-feature['drop_d_overall_pre_amount_of_billmean']
feature['drop_d_overall_credit_amount_mean_and_overall_amount_of_bill_leftmean_diff']=feature['drop_d_overall_credit_amount_mean']-feature['drop_d_overall_amount_of_bill_leftmean']
feature['drop_d_overall_avail_amount_mean_and_overall_prepare_amount_mean_diff']=feature['drop_d_overall_avail_amount_mean']-feature['drop_d_overall_prepare_amount_mean']
feature['drop_d_overall_least_interest_mean_and_overall_circ_interestmean_sum']=feature['drop_d_overall_least_interest_mean']+feature['drop_d_overall_circ_interestmean']


drop_d_overall_bill_median=gb.median()
drop_d_overall_bill_median.columns = ['userid', 'drop_d_overall_pre_amount_of_bill_median', 'drop_d_overall_pre_repayment_median'
                          ,'drop_d_overall_credit_amount_median','drop_d_overall_amount_of_bill_left_median','drop_d_overall_least_repayment_median'
                          ,'drop_d_overall_consume_amount_median','drop_d_overall_amount_of_bill_median','drop_d_overall_adjust_amount_median'
                          ,'drop_d_overall_circ_interest_median','drop_d_overall_avail_amount_median','drop_d_overall_prepare_amount_median']
feature=pd.merge(feature, drop_d_overall_bill_median,how='left', on = "userid")
feature['drop_d_overall_pre_repayment_median_and_overall_pre_amount_of_bill_median_diff']=feature['drop_d_overall_pre_repayment_median']-feature['drop_d_overall_pre_amount_of_bill_median']
feature['drop_d_overall_credit_amount_median_and_overall_amount_of_bill_left_median_diff']=feature['drop_d_overall_credit_amount_median']-feature['drop_d_overall_amount_of_bill_left_median']
feature['drop_d_overall_avail_amount_median_and_overall_prepare_amount_median_diff']=feature['drop_d_overall_avail_amount_median']-feature['drop_d_overall_prepare_amount_median']
feature['drop_d_overall_least_repayment_median_and_overall_circ_interest_median_sum']=feature['drop_d_overall_least_repayment_median']+feature['drop_d_overall_circ_interest_median']


drop_d_overall_billstd=gb.std()
drop_d_overall_billstd.columns = ['userid', 'drop_d_overall_pre_amount_of_billstd', 'drop_d_overall_pre_repayment_std','drop_d_overall_credit_amount_std'
                       ,'drop_d_overall_amount_of_bill_left_std','drop_d_overall_least_repayment_std','drop_d_overall_consume_amount_std'
                       ,'drop_d_overall_amount_of_billstd','drop_d_overall_adjust_amount_std','drop_d_overall_circ_intereststd'
                       ,'drop_d_overall_avail_amount_std','drop_d_overall_prepare_amount_std']
feature=pd.merge(feature, drop_d_overall_billstd,how='left', on = "userid")
feature['drop_d_overall_pre_repayment_std_and_overall_pre_amount_of_billstd_diff']=feature['drop_d_overall_pre_repayment_std']-feature['drop_d_overall_pre_amount_of_billstd']
feature['drop_d_overall_credit_amount_std_and_overall_amount_of_bill_left_std_diff']=feature['drop_d_overall_credit_amount_std']-feature['drop_d_overall_amount_of_bill_left_std']
feature['drop_d_overall_avail_amount_std_and_overall_prepare_amount_std_diff']=feature['drop_d_overall_avail_amount_std']-feature['drop_d_overall_prepare_amount_std']
feature['drop_d_overall_least_repayment_std_and_overall_circ_intereststd_sum']=feature['drop_d_overall_least_repayment_std']+feature['drop_d_overall_circ_intereststd']

drop_d_overall_bill_var=gb.var()
drop_d_overall_bill_var.columns = ['userid', 'drop_d_overall_pre_amount_of_bill_var', 'drop_d_overall_pre_repayment_var','drop_d_overall_credit_amount_var'
                       ,'drop_d_overall_amount_of_bill_left_var','drop_d_overall_least_repayment_var','drop_d_overall_consume_amount_var'
                       ,'drop_d_overall_amount_of_bill_var','drop_d_overall_adjust_amount_var','drop_d_overall_circ_interestvar','drop_d_overall_avail_amount_var','drop_d_overall_prepare_amount_var']
feature=pd.merge(feature, drop_d_overall_bill_var,how='left', on = "userid")
feature['drop_d_overall_pre_repayment_var_and_overall_pre_amount_of_bill_var_diff']=feature['drop_d_overall_pre_repayment_var']-feature['drop_d_overall_pre_amount_of_bill_var']
feature['drop_d_overall_credit_amount_var_and_overall_amount_of_bill_left_var_diff']=feature['drop_d_overall_credit_amount_var']-feature['drop_d_overall_amount_of_bill_left_var']
feature['drop_d_overall_avail_amount_var_and_overall_prepare_amount_var_diff']=feature['drop_d_overall_avail_amount_var']-feature['drop_d_overall_prepare_amount_var']
feature['drop_d_overall_least_repayment_var_and_overall_circ_interestvar_sum']=feature['drop_d_overall_least_repayment_var']+feature['drop_d_overall_circ_interestvar']

feature.shape#55596 rows × 241 columns



feature.to_csv("feature_time_known_and_unknown.csv",index=None)

####################################################
#                                                  #
#              4.feature engineering               #
#                                                  #
####################################################  
#################################################
#                credit bill table              #
#################################################

d=train_bill_detail
feature=train_loan_time
#----------------------------------------stattistics before loan time------------------------------------------#
#calculate the number of observations where underlying features are neg or equal to 0

gb=d[(d['time']<=d['loan_time'])].groupby(["userid"],as_index=False)['pre_amount_of_bill']
x1=gb.apply(lambda x:x.where(x<0).count())
x2=gb.apply(lambda x:x.where(x==0.000000).count())
x=gb.agg({'be_loan_amount_of_bill_stat' : 'sum'})
x['be_loan_amount_of_bill_is_neg_count']=x1
x['be_loan_amount_of_bill_is_0']=x2

feature=pd.merge(feature, x,how='left', on = "userid")

gb=d[(d['time']<=d['loan_time'])].groupby(["userid"],as_index=False)['pre_repayment']
x1=gb.apply(lambda x:x.where(x<0).count())
x2=gb.apply(lambda x:x.where(x==0.000000).count())
x=gb.agg({'be_loan_repayment_stat' : 'sum'})
x['be_loan_repayment_is_neg_count']=x1
x['be_loan_repayment_is_0']=x2

feature=pd.merge(feature, x,how='left', on = "userid")
feature['be_loan_billrepayment_diff']=feature['be_loan_amount_of_bill_stat']-feature['be_loan_repayment_stat']

#delete observations have neg values in pre_amount_of_bill and pre_repayment
d1=d[(d['pre_amount_of_bill']<=0)].index.tolist()
d=d.drop(d1,axis=0)
d2=d[(d['pre_repayment']<=0)].index.tolist()
d=d.drop(d2,axis=0)
d.shape
#1625621rows


gb=d[(d['time']<=d['loan_time'])].groupby(["userid","time","bank_id"],as_index=False)
x1=gb['pre_amount_of_bill'].agg({'be_loan_userbank_last_monthamount_of_bill_sum' : 'sum','be_loan_userbank_last_monthamount_of_bill_max' : 'max'})
x2=gb['pre_repayment'].agg({'be_loan_userbank_last_month_repayment_sum' : 'sum','be_loan_userbank_repayment_max' : 'max'})
x3=gb['consume_amount'].agg({'userbe_loan_consume_amount_max' : 'max'})
x4=gb['circ_interest'].agg({'userbe_loan_circ_interest_max' : 'max'})

gb1=x1.groupby(["userid"],as_index=False)
gb2=x2.groupby(["userid"],as_index=False)
gb3=x3.groupby(["userid"],as_index=False)
gb4=x4.groupby(["userid"],as_index=False)

x11=gb1['be_loan_userbank_last_monthamount_of_bill_sum'].agg({'be_loan_useramount_of_bill_sum(drop_d)' : 'sum','be_loan_userbill_count(drop_d)' : 'count'})
x12=gb1['be_loan_userbank_last_monthamount_of_bill_max'].agg({'be_loan_useramount_of_bill_max_sum(drop_d)' : 'sum'})

x21=gb2['be_loan_userbank_last_month_repayment_sum'].agg({'be_loan_userbillrepayment_sum(drop_d)' : 'sum'})
x22=gb2['be_loan_userbank_repayment_max'].agg({'be_loan_userbillrepayment_max_sum(drop_d)' : 'sum'})

x31=gb3['userbe_loan_consume_amount_max'].agg({'userbe_loan_consume_amount(drop_d)' : 'sum'})
x41=gb4['userbe_loan_circ_interest_max'].agg({'userbe_loan_circ_interest(drop_d)' : 'sum'})

feature=pd.merge(feature, x11,how='left', on = "userid")
feature=pd.merge(feature, x12,how='left', on = "userid")
feature=pd.merge(feature, x21,how='left', on = "userid")
feature=pd.merge(feature, x22,how='left', on = "userid")
feature=pd.merge(feature, x31,how='left', on = "userid")
feature=pd.merge(feature, x41,how='left', on = "userid")

x=pd.merge(x1, x2,how='inner')
gb3=x[(x['be_loan_userbank_last_monthamount_of_bill_max']>x['be_loan_userbank_repayment_max'])].groupby(["userid"],as_index=False)
gb4=x[(x['be_loan_userbank_last_monthamount_of_bill_max']==x['be_loan_userbank_repayment_max'])].groupby(["userid"],as_index=False)
gb5=x[(x['be_loan_userbank_last_monthamount_of_bill_max']<x['be_loan_userbank_repayment_max'])].groupby(["userid"],as_index=False)

x31=gb3['userid'].agg({'be_loan_bill_larger_than_repayment_count(drop_d)' : 'count'})
x32=gb4['userid'].agg({'be_loan_bill_equals_repayment_count(drop_d)' : 'count'})
x33=gb5['userid'].agg({'be_loan_bill_smaller_than_repayment_count(drop_d)' : 'count'})

feature=pd.merge(feature, x31,how='left', on = "userid")
feature=pd.merge(feature, x32,how='left', on = "userid")
feature=pd.merge(feature, x33,how='left', on = "userid")

feature['be_loan_bill_sumrepayment_diff(drop_d)']=feature['be_loan_useramount_of_bill_sum(drop_d)']-feature['be_loan_userbillrepayment_sum(drop_d)']
feature['be_loan_bill_maxrepayment_diff(drop_d)']=feature['be_loan_useramount_of_bill_max_sum(drop_d)']-feature['be_loan_userbillrepayment_max_sum(drop_d)']

gb=d[(d['time']<=d['loan_time'])].groupby(["userid"],as_index=False)
x1=gb['consume_amount'].agg({'userbe_loan_consume_amount' : 'sum'})
x2=gb['circ_interest'].agg({'userbe_loan_circ_interest' : 'sum'})
x3=gb['credit_amount'].agg({'userbe_loan_credit_amount_max' : 'max'})

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")
feature=pd.merge(feature, x3,how='left', on = "userid")

#----------------------------------------stattistics after loan time------------------------------------------#
d=train_bill_detail
#_statafter_loan_userpre_amount_of_bill_sum_and_useramount_of_bill_is_neg_count_stat
gb=d[(d['time']>d['loan_time'])].groupby(["userid"],as_index=False)['pre_amount_of_bill']
x1=gb.apply(lambda x:x.where(x<0).count())
x2=gb.apply(lambda x:x.where(x==0.000000).count())
x=gb.agg({'after_loan_amount_of_bill_stat' : 'sum'})
x['after_loan_amount_of_bill_is_neg_count']=x1
x['after_loan_amount_of_bill_is_0']=x2

feature=pd.merge(feature, x,how='left', on = "userid")

gb=d[(d['time']>d['loan_time'])].groupby(["userid"],as_index=False)['pre_repayment']
x1=gb.apply(lambda x:x.where(x<0).count())
x2=gb.apply(lambda x:x.where(x==0.000000).count())
x=gb.agg({'after_loan_repayment_stat' : 'sum'})
x['after_loan_repayment_is_neg_count']=x1
x['after_loan_repayment_is_0']=x2

feature=pd.merge(feature, x,how='left', on = "userid")

feature['after_loan_billrepayment_diff']=feature['after_loan_amount_of_bill_stat']-feature['after_loan_repayment_stat']

#delete 0 and neg
d1=d[(d['pre_amount_of_bill']<=0)].index.tolist()
d=d.drop(d1,axis=0)
d2=d[(d['pre_repayment']<=0)].index.tolist()
d=d.drop(d2,axis=0)

#1625621 rows left

gb=d[(d['time']>d['loan_time'])].groupby(["userid","time","bank_id"],as_index=False)
x1=gb['pre_amount_of_bill'].agg({'after_loan_userbank_last_monthamount_of_bill_sum' : 'sum','after_loan_userbank_last_monthamount_of_bill_max' : 'max'})
x2=gb['pre_repayment'].agg({'after_loan_userbank_last_month_repayment_sum' : 'sum','after_loan_user_bank_repayment_max' : 'max'})
x3=gb['consume_amount'].agg({'userafter_loan_consume_amount_max' : 'max'})
x4=gb['circ_interest'].agg({'userafter_loan_circ_interest_max' : 'max'})

gb1=x1.groupby(["userid"],as_index=False)
gb2=x2.groupby(["userid"],as_index=False)
gb3=x3.groupby(["userid"],as_index=False)
gb4=x4.groupby(["userid"],as_index=False)

x11=gb1['after_loan_userbank_last_monthamount_of_bill_sum'].agg({'after_loan_useramount_of_bill_sum(drop_d)' : 'sum','after_loan_userbill_count(drop_d)' : 'count'})
x12=gb1['after_loan_userbank_last_monthamount_of_bill_max'].agg({'after_loan_useramount_of_bill_max_sum(drop_d)' : 'sum'})

x21=gb2['after_loan_userbank_last_month_repayment_sum'].agg({'after_loan_userbillrepayment_sum(drop_d)' : 'sum'})
x22=gb2['after_loan_user_bank_repayment_max'].agg({'after_loan_userbillrepayment_max_sum(drop_d)' : 'sum'})

x31=gb3['userafter_loan_consume_amount_max'].agg({'userafter_loan_consume_amount(drop_d)' : 'sum'})
x41=gb4['userafter_loan_circ_interest_max'].agg({'userafter_loan_circ_interest(drop_d)' : 'sum'})

feature=pd.merge(feature, x11,how='left', on = "userid")
feature=pd.merge(feature, x12,how='left', on = "userid")
feature=pd.merge(feature, x21,how='left', on = "userid")
feature=pd.merge(feature, x22,how='left', on = "userid")
feature=pd.merge(feature, x31,how='left', on = "userid")
feature=pd.merge(feature, x41,how='left', on = "userid")

x=pd.merge(x1, x2,how='inner')
gb3=x[(x['after_loan_userbank_last_monthamount_of_bill_max']>x['after_loan_user_bank_repayment_max'])].groupby(["userid"],as_index=False)
gb4=x[(x['after_loan_userbank_last_monthamount_of_bill_max']==x['after_loan_user_bank_repayment_max'])].groupby(["userid"],as_index=False)
gb5=x[(x['after_loan_userbank_last_monthamount_of_bill_max']<x['after_loan_user_bank_repayment_max'])].groupby(["userid"],as_index=False)

x31=gb3['userid'].agg({'after_loan_bill_larger_than_repayment_count(drop_d)' : 'count'})
x32=gb4['userid'].agg({'after_loan_bill_equals_repayment_count(drop_d)' : 'count'})
x33=gb5['userid'].agg({'after_loan_bill_smaller_than_repayment_count(drop_d)' : 'count'})

feature=pd.merge(feature, x31,how='left', on = "userid")
feature=pd.merge(feature, x32,how='left', on = "userid")
feature=pd.merge(feature, x33,how='left', on = "userid")

feature['after_loan_bill_sumrepayment_diff(drop_d)']=feature['after_loan_useramount_of_bill_sum(drop_d)']-feature['after_loan_userbillrepayment_sum(drop_d)']
feature['after_loan_bill_maxrepayment_diff(drop_d)']=feature['after_loan_useramount_of_bill_max_sum(drop_d)']-feature['after_loan_userbillrepayment_max_sum(drop_d)']

#_stat_be_loan_userconsume_amount，circ_interest_sum
gb=d[(d['time']>d['loan_time'])].groupby(["userid"],as_index=False)
x1=gb['consume_amount'].agg({'userafter_loan_consume_amount' : 'sum'})
x2=gb['circ_interest'].agg({'userafter_loan_circ_interest' : 'sum'})
x3=gb['credit_amount'].agg({'userafter_loan_credit_amount_max' : 'max'})

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")
feature=pd.merge(feature, x3,how='left', on = "userid")

#start from here
#----------------------------------------stattistics for overall loan time------------------------------------------#
d=train_bill_detail
#maxout means amount of bill left is larger than credit amount
gb=d[(d['credit_amount']<d['amount_of_bill_left'])].groupby(["userid"],as_index=False)
x1=gb['time'].apply(lambda x:np.unique(x).size)
x=gb['time'].agg({'maxout_count' : 'count'})
x['maxout_count(drop_d)']=x1
feature=pd.merge(feature, x,how='left', on = "userid")

#count of user holding card
gb=d.groupby(["userid"],as_index=False)
x=gb['bank_id'].apply(lambda x:np.unique(x).size)
x1=gb['bank_id'].agg({'user_bank_bill_count' : 'count'})
x1['user_holdcard_count']=x
feature=pd.merge(feature,x1,how='left', on = "userid")


d=train_bill_detail

t1=d[(d['time']>d['loan_time'])].groupby("userid",as_index=False)
t2=d[(d['time']>d['loan_time']+1)].groupby("userid",as_index=False)
t3=d[(d['time']>d['loan_time']+2)].groupby("userid",as_index=False)

x=t1['time'].apply(lambda x:np.unique(x).size)
x1=t1['time'].agg({'add_feature1' : 'count'})
x1['x1']=x

x=t2['time'].apply(lambda x:np.unique(x).size)
x2=t2['time'].agg({'add_feature2' : 'count'})
x2['x2']=x

x=t3['time'].apply(lambda x:np.unique(x).size)
x3=t3['time'].agg({'add_feature3' : 'count'})
x3['x3']=x

t=feature[['userid']]
t=pd.merge(t,x1,how='left',on = "userid")
t=pd.merge(t,x2,how='left',on = "userid")
t=pd.merge(t,x3,how='left',on = "userid")
t=t[['userid','x1','x2','x3','add_feature1','add_feature2','add_feature3']]

feature=pd.merge(feature, t,how='left', on = "userid")

feature['add_featurex']=(feature['x1']+1)*(feature['x2']+1)*(feature['x3']+1)


feature.shape
feature.to_csv("userbill_feature_train.csv",index=None)


#################################################
#                bank details table             #
#################################################

# **bank_detail_feature_extract:**  
# 55596 rows × 26 columns

feature=train_loan_time
d=train_bank_detail
#----------------------------------------statistics before loan time------------------------------------------#
t=d[(d['bank_detailtime']<=d['loan_time'])]#5684742
gb1=t[(t['extype']==0)].groupby(["userid"],as_index=False)#_inc_stat
gb2=t[(t['extype']==1)].groupby(["userid"],as_index=False)#_exp_stat
gb3=t[(t['sal_inc_mark']==1)].groupby(["userid"],as_index=False)#sal_inc_stat
x1=gb1['examount'].agg({'be_loan_user_inc_count' : 'count','be_loan_user_inc_sum':'sum'})
x2=gb2['examount'].agg({'be_loan_user_exp_count' : 'count','be_loan_user_exp_sum':'sum'})
x3=gb3['examount'].agg({'be_loan_user_sal_inc_count' : 'count','be_loan_user_sal_inc_sum':'sum'})

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")
feature=pd.merge(feature, x3,how='left', on = "userid")

feature['be_loan_user_inc_exp_count_diff']=feature['be_loan_user_inc_count']-feature['be_loan_user_exp_count']
feature['be_loan_user_inc_exp_sum_diff']=feature['be_loan_user_inc_sum']-feature['be_loan_user_exp_sum']
feature['be_loan_user_non_sal_inc_count']=feature['be_loan_user_inc_count']-feature['be_loan_user_sal_inc_count']
feature['be_loan_user_non_sal_inc_sum']=feature['be_loan_user_inc_sum']-feature['be_loan_user_sal_inc_sum']
feature['be_loan_sal_inc_count_times_diff']=feature['be_loan_user_sal_inc_count']*feature['be_loan_user_inc_exp_count_diff']
feature['be_loan_sal_inc_sum_times_diff']=feature['be_loan_user_sal_inc_sum']*feature['be_loan_user_inc_exp_sum_diff']

#----------------------------------------statistics after loan time------------------------------------------#
t=d[(d['bank_detailtime']>d['loan_time'])]#5684742
gb1=t[(t['extype']==0)].groupby(["userid"],as_index=False)#_inc_stat
gb2=t[(t['extype']==1)].groupby(["userid"],as_index=False)#_exp_stat
gb3=t[(t['sal_inc_mark']==1)].groupby(["userid"],as_index=False)#sal_inc_stat
x1=gb1['examount'].agg({'after_loan_user_inc_count' : 'count','after_loan_user_inc_sum':'sum'})
x2=gb2['examount'].agg({'after_loan_user_exp_count' : 'count','after_loan_user_exp_sum':'sum'})
x3=gb3['examount'].agg({'after_loan_user_sal_inc_count' : 'count','after_loan_user_sal_inc_sum':'sum'})

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")
feature=pd.merge(feature, x3,how='left', on = "userid")

feature['after_loan_user_inc_exp_count_diff']=feature['after_loan_user_inc_count']-feature['after_loan_user_exp_count']
feature['after_loan_user_inc_exp_sum_diff']=feature['after_loan_user_inc_sum']-feature['after_loan_user_exp_sum']
feature['after_loan_user_non_sal_inc_count']=feature['after_loan_user_inc_count']-feature['after_loan_user_sal_inc_count']
feature['after_loan_user_non_sal_inc_sum']=feature['after_loan_user_inc_sum']-feature['after_loan_user_sal_inc_sum']
feature['after_loan_sal_inc_count_times_diff']=feature['after_loan_user_sal_inc_count']*feature['after_loan_user_inc_exp_count_diff']
feature['after_loan_sal_inc_sum_times_diff']=feature['after_loan_user_sal_inc_sum']*feature['after_loan_user_inc_exp_sum_diff']
feature.shape
feature.to_csv("userbank_detail_train.csv",index=None)

#################################################
#                user browse table              #
#################################################

userbrowse_behavior.shape

userbrowse_behavior[(userbrowse_behavior['browse_time']==0)].shape#there is no unknown time in this subset

feature=train_loan_time
d= pd.merge(userbrowse_behavior, train_loan_time,how='left', on = "userid")

#----------------------------------------statistics before loan time------------------------------------------#

gb=d[(d['browse_time']<=d['loan_time'])].groupby(["userid"],as_index=False)
x1=gb['browse_behavior'].agg({'be_loan_browse_behavior_sum' : 'sum','be_loan_browse_behavior_max' : 'max','be_loan_browse_behavior_mean' : 'mean'
                    ,'be_loan_browse_behavior_min' : 'min','be_loan_browse_behavior_std' : 'std','be_loan_browse_behavior_var' : 'var'})
    
xx=gb['browse_behavior_number'].apply(lambda x:np.unique(x).size)
x2=gb['browse_behavior_number'].agg({'be_loan_browse_behavior_number_count' : 'count'})
x2['be_loan_browse_behavior_number_count（drop_d）']=xx

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")

d=pd.get_dummies(d,columns=d[['browse_behavior_number']])#22919547 rows × 14 columns
gb=d[(d['browse_time']<=d['loan_time'])].groupby(["userid"],as_index=False)
x1=gb['browse_behavior_number_1'].agg({'be_loan_browse_behavior_number_1' : 'sum'})
x2=gb['browse_behavior_number_2'].agg({'be_loan_browse_behavior_number_2' : 'sum'})
x3=gb['browse_behavior_number_3'].agg({'be_loan_browse_behavior_number_3' : 'sum'})
x4=gb['browse_behavior_number_4'].agg({'be_loan_browse_behavior_number_4' : 'sum'})
x5=gb['browse_behavior_number_5'].agg({'be_loan_browse_behavior_number_5' : 'sum'})
x6=gb['browse_behavior_number_6'].agg({'be_loan_browse_behavior_number_6' : 'sum'})
x7=gb['browse_behavior_number_7'].agg({'be_loan_browse_behavior_number_7' : 'sum'})
x8=gb['browse_behavior_number_8'].agg({'be_loan_browse_behavior_number_8' : 'sum'})
x9=gb['browse_behavior_number_9'].agg({'be_loan_browse_behavior_number_9' : 'sum'})
x10=gb['browse_behavior_number_10'].agg({'be_loan_browse_behavior_number_10' : 'sum'})
x11=gb['browse_behavior_number_11'].agg({'be_loan_browse_behavior_number_11' : 'sum'})

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")
feature=pd.merge(feature, x3,how='left', on = "userid")
feature=pd.merge(feature, x4,how='left', on = "userid")
feature=pd.merge(feature, x5,how='left', on = "userid")
feature=pd.merge(feature, x6,how='left', on = "userid")
feature=pd.merge(feature, x7,how='left', on = "userid")
feature=pd.merge(feature, x8,how='left', on = "userid")
feature=pd.merge(feature, x9,how='left', on = "userid")
feature=pd.merge(feature, x10,how='left', on = "userid")
feature=pd.merge(feature, x11,how='left', on = "userid")

x1=gb['browse_behavior'].agg({'after_loan_browse_behavior_sum' : 'sum','after_loan_browse_behavior_max' : 'max','after_loan_browse_behavior_mean' : 'mean'
                     ,'after_loan_browse_behavior_min' : 'min','after_loan_browse_behavior_std' : 'std','after_loan_browse_behavior_var' : 'var'})
    
xx=gb['browse_behavior_number'].apply(lambda x:np.unique(x).size)
x2=gb['browse_behavior_number'].agg({'after_loan_browse_behavior_number_count' : 'count'})
x2['after_loan_browse_behavior_number_count（drop_d）']=xx

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")

d=pd.get_dummies(d,columns=d[['browse_behavior_number']])#22919547 rows × 14 columns
gb=d[(d['browse_time']<=d['loan_time'])].groupby(["userid"],as_index=False)
x1=gb['browse_behavior_number_1'].agg({'after_loan_browse_behavior_number_1' : 'sum'})
x2=gb['browse_behavior_number_2'].agg({'after_loan_browse_behavior_number_2' : 'sum'})
x3=gb['browse_behavior_number_3'].agg({'after_loan_browse_behavior_number_3' : 'sum'})
x4=gb['browse_behavior_number_4'].agg({'after_loan_browse_behavior_number_4' : 'sum'})
x5=gb['browse_behavior_number_5'].agg({'after_loan_browse_behavior_number_5' : 'sum'})
x6=gb['browse_behavior_number_6'].agg({'after_loan_browse_behavior_number_6' : 'sum'})
x7=gb['browse_behavior_number_7'].agg({'after_loan_browse_behavior_number_7' : 'sum'})
x8=gb['browse_behavior_number_8'].agg({'after_loan_browse_behavior_number_8' : 'sum'})
x9=gb['browse_behavior_number_9'].agg({'after_loan_browse_behavior_number_9' : 'sum'})
x10=gb['browse_behavior_number_10'].agg({'after_loan_browse_behavior_number_10' : 'sum'})
x11=gb['browse_behavior_number_11'].agg({'after_loan_browse_behavior_number_11' : 'sum'})

feature=pd.merge(feature, x1,how='left', on = "userid")
feature=pd.merge(feature, x2,how='left', on = "userid")
feature=pd.merge(feature, x3,how='left', on = "userid")
feature=pd.merge(feature, x4,how='left', on = "userid")
feature=pd.merge(feature, x5,how='left', on = "userid")
feature=pd.merge(feature, x6,how='left', on = "userid")
feature=pd.merge(feature, x7,how='left', on = "userid")
feature=pd.merge(feature, x8,how='left', on = "userid")
feature=pd.merge(feature, x9,how='left', on = "userid")
feature=pd.merge(feature, x10,how='left', on = "userid")
feature=pd.merge(feature, x11,how='left', on = "userid")

feature.shape
feature.to_csv("userbrowse_behavior_train.csv",index=None)



############################################################
#         dataset without further feature engineering      #
############################################################

userbrowse_behavior_train = pd.read_csv('userbrowse_behavior_train.csv')
userbank_detail_train = pd.read_csv('userbank_detail_train.csv')

#merge processed browse and bank details into train
train = pd.merge(userbrowse_behavior_train,userbank_detail_train,on='userid')
train.shape#55,596*40
userbill_feature_train = pd.read_csv('userbill_feature_train.csv')

#merge processed bill details into train
train = pd.merge(train,userbill_feature_train,on='userid')
train.shape#55,596*115

bill_stat_time_known_and_unknown = pd.read_csv('feature_time_known_and_unknown.csv')
bill_stat_time_unknown = pd.read_csv('feature_time_unknown.csv')
bill_stat_time_known = pd.read_csv('feature_time_known.csv')

#merge three bill details statistics info into train
train = pd.merge(train,bill_stat_time_known_and_unknown,on='userid')
train = pd.merge(train,bill_stat_time_unknown,on='userid')
train = pd.merge(train,bill_stat_time_known,on='userid')
train.shape#55,596*1068

overdue = pd.read_csv("overdue_train.txt",header=None,names=['userid','target'])

#merge overdue info into train
train = pd.merge(train,overdue,on='userid')
train.shape#55,596*1069


train_loan_time = pd.read_csv("loan_time_train.txt",header=None,names=['userid','loan_time'])
train_loan_time['loan_time']=train_loan_time['loan_time']//86400
train_loan_time.shape#55,596*1069
          
               
#merge train loan time info into train
train = pd.merge(train,train_loan_time,on='userid')               
train.shape#55,596*1069
    
train_user_info = pd.read_csv("user_info_train.txt",header=None,
                    names=['userid','usersex','userjob','useredu',
                           'usermarriage', 'userhukou'])
    
#merge train loan time info into train
train = pd.merge(train,train_user_info,on='userid')     
train.shape#55,596*1075
import sys
import numpy as np
sys.setrecursionlimit(10000)
train_drop_d = train.T.drop_duplicates().T
train_drop_d.shape#55,596*905

  
train_drop_d.to_csv("rong360_all_feature_withid.csv",index=None)
train_noid = train_drop_d.drop(['userid'],axis=1)
train_noid.shape#55,596*1074
train_noid.to_csv("rong360_all_feature_noid.csv",index=None)
train=train_drop_d
import pandas as pd
rong360=pd.read_csv("/Users/pengchengliu/Documents/Master Thesis/data/rong360/data/rong360_all_feature_withid.csv")



############################################################
#                keep doing feature engineering            #
############################################################

###############################################
#     1.missing value feature construction    #
###############################################


train_xy =pd.read_csv("rong360_all_feature_withid.csv")
train_xy.rename(columns={'target':'y'}, inplace=True)
train_xy.rename(columns={'userid':'uid'}, inplace=True)
#statistics of missing value per row

train = train_xy
train['n_null'] = np.sum(train.isnull(),axis=1)
train['n_null'].describe()

#binning of the missing values per row
train['discret_null'] = train.n_null
train.discret_null[train.discret_null<=323] = 1
train.discret_null[(train.discret_null>323)&(train.discret_null<=356)] = 2
train.discret_null[(train.discret_null>356)&(train.discret_null<=517)] = 3
train.discret_null[(train.discret_null>517)] = 4
train['discret_null'].describe()

train[['uid','n_null','discret_null']].to_csv('train_x_null.csv',index=None)

np.sum(train['y'].isnull(),axis=1)

################################
#   2.num variable rankings    #
################################

#column types
train.dtypes#all float 

#define numeric variables:
#all columns have more than 100 distinct values are seen as numeric varianles
unique_value_stat = pd.Series.sort_values(train.apply(pd.Series.nunique))
numeric_feature = list((unique_value_stat[unique_value_stat>100].index).values)
#clean up the list: remove uid 
if 'uid' in numeric_feature: numeric_feature.remove('uid')
type(numeric_feature)

train_numeric = train[['uid']+numeric_feature]
#55,596*781
train_rank = pd.DataFrame(train_numeric.uid,columns=['uid'])

for feature in numeric_feature:
    train_rank['r'+feature] = train_numeric[feature].rank(method='max')
train_rank.to_csv('train_x_rank.csv',index=None)
train_rank.shape
#40000*157


#####################################
#   3.discretization of rankings    #
#####################################

train_x = train_rank.drop(['uid'],axis=1)

#discretization of ranking features
#each 10% belongs to 1 level
train_x[train_x<6000] = 1
train_x[(train_x>=6000)&(train_x<12000)] = 2
train_x[(train_x>=12000)&(train_x<18000)] = 3
train_x[(train_x>=18000)&(train_x<24000)] = 4
train_x[(train_x>=24000)&(train_x<30000)] = 5
train_x[(train_x>=30000)&(train_x<36000)] = 6
train_x[(train_x>=36000)&(train_x<42000)] = 7
train_x[(train_x>=42000)&(train_x<48000)] = 8
train_x[(train_x>=48000)&(train_x<54000)] = 9
train_x[train_x>=54000] = 10

#rename      
#nameing rules for ranking discretization features, add "d" in front of orginal features
#for instance "x1" would have discretization feature of "dx1"
rename_dict = {s:'d'+s[1:] for s in train_x.columns.tolist()}
train_x = train_x.rename(columns=rename_dict)
train_x['uid'] = train_rank.uid
      
train_x.to_csv('train_x_discretization.csv',index=None)      

#############################################
#   4.frequency of ranking discretization   #
#############################################


train_x['n1'] = (train_x==1).sum(axis=1)
train_x['n2'] = (train_x==2).sum(axis=1)
train_x['n3'] = (train_x==3).sum(axis=1)
train_x['n4'] = (train_x==4).sum(axis=1)
train_x['n5'] = (train_x==5).sum(axis=1)
train_x['n6'] = (train_x==6).sum(axis=1)
train_x['n7'] = (train_x==7).sum(axis=1)
train_x['n8'] = (train_x==8).sum(axis=1)
train_x['n9'] = (train_x==9).sum(axis=1)
train_x['n10'] = (train_x==10).sum(axis=1)
train_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('train_x_nd.csv',index=None)


##############################################
#   5.feature importance of rank features    #
##############################################
#generate a variety of xgboost models to have rank feature importance

import pandas as pd
import xgboost as xgb
import sys
import random
import _pickle as cPickle
import os
from sklearn.model_selection import train_test_split

#craete a dicrectory if it doesn't exist
#https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
#if not makes this more robust
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

##################################################
#                 split dataset                  #
##################################################
#load data
train_x = pd.read_csv("train_x_rank.csv")
train_y = train[['uid']+['y']]
train_xy = pd.merge(train_x,train_y,on='uid')

train_xy, test_xy = train_test_split(train_xy, test_size = 0.2)
#train_xy 32,000*158
#test_xy 8,000*158
train_y = train_xy[['uid']+['y']]
test_y = test_xy[['uid']+['y']]
#label or xgb
y=train_xy.y


###########
#  train  #
###########
#leave features only
train_x= train_xy.drop(["uid",'y'],axis=1)
#convert to percentage 
X = train_x/55596
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

###########
#   test  #
###########
#do the same to test table    
test = test_xy
test_uid = test.uid
test = test.drop(["uid",'y'],axis=1)
test_x = test/55596
dtest = xgb.DMatrix(test_x)

##########################
#   feature importance   #
##########################
#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1350,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    

#train 100 xgb
    for i in list(range(100)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

#run from here
##################################
#   average feature importance   #
##################################
#calculate average feature score for ranking features

#get rank feature importance info from the xgboost models
import pandas as pd 
import os

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('rank_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)

##############################################
#     6.feature importance of raw features   #
##############################################

from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb
import sys,random
import _pickle as cPickle
import os

if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds") 
##load data
train_xy =pd.read_csv("rong360_all_feature_withid.csv")
train_xy.rename(columns={'target':'y'}, inplace=True)
train_xy.rename(columns={'userid':'uid'}, inplace=True)

train_xy, test_xy = train_test_split(train_xy, test_size = 0.2)
#train_xy 32,000*158
#test_xy 8,000*158
train_y = train_xy[['uid']+['y']]
test_y = test_xy[['uid']+['y']]
#flabel or xgb
y=train_xy.y


###########
#  train  #
###########
#leave features only
train_x= train_xy.drop(["uid",'y'],axis=1)
#convert to percentage 
X = train_x
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

###########
#   test  #
###########
#do the same to test table    
test = test_xy
test_uid = test.uid
test = test.drop(["uid",'y'],axis=1)
test_x = test
dtest = xgb.DMatrix(test_x)

##########################
#   feature importance   #
##########################
#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=500,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
        

if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    


#train 10 xgb
    for i in list(range(10)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

##################################
#   average feature importance   #
##################################
#calculate average feature score for ranking features

#get rank feature importance info from the xgboost models

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('raw_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)


#start from here
##################################################
#     7.feature importance of discret features   #
##################################################


#craete a dicrectory if it doesn't exist
#https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
#if not makes this more robust
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

##################################################
#                 split dataset                  #
##################################################
#load data

train_xy = pd.read_csv("rong360_all_feature_withid.csv")
train_xy.rename(columns={'target':'y'}, inplace=True)
train_xy.rename(columns={'userid':'uid'}, inplace=True)


train_x = pd.read_csv("train_x_discretization.csv")
train_y = train_xy[['uid']+['y']]
train_xy = pd.merge(train_x,train_y,on='uid')

train_xy, test_xy = train_test_split(train_xy, test_size = 0.2)
#train_xy 32,000*158
#test_xy 8,000*158
train_y = train_xy[['uid']+['y']]
test_y = test_xy[['uid']+['y']]
#flabel or xgb
y=train_xy.y


###########
#  train  #
###########
#leave features only
train_x= train_xy.drop(["uid",'y'],axis=1)
#convert to percentage 
X = train_x
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

###########
#   test  #
###########
#do the same to test table    
test = test_xy
test_uid = test.uid
test = test.drop(["uid",'y'],axis=1)
test_x = test
dtest = xgb.DMatrix(test_x)

##########################
#   feature importance   #
##########################
#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1000,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    

#train 100 xgb
    for i in list(range(10)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

#run from here
##################################
#   average feature importance   #
##################################
#calculate average feature score for ranking features

#get rank feature importance info from the xgboost models
import pandas as pd 
import os

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('discreet_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)


#####################################################
#      normalization of numeric variables           #
#####################################################

train=train.fillna(-1)


plt.figure(1)
train_numeric.UserInfo_213.apply(lambda x: x/10).hist()
plt.figure(2)
luk = train_numeric.UserInfo_213.apply(lambda x: np.log(x+1).round()).hist()

def numpy_minmax(X):
    xmin =  X.min(axis=0)
    return (X - xmin) / (X.max(axis=0) - xmin)

plt.figure(1)
train_numeric.UserInfo_213.apply(lambda x: x/10).hist()
plt.figure(2)
numpy_minmax(train_numeric.UserInfo_213).hist()
from sklearn import preprocessing

standardized_X = preprocessing.scale(train_numeric.UserInfo_213)
train_numeric.UserInfo_5.apply(lambda x: x/10).hist()

luk = train_numeric.UserInfo_213.apply(lambda x: np.log(x+1).round()).hist()



##################################################
#                 8.xgb bagging                  #
##################################################

###########################
#      prepare data       #
###########################

import pandas as pd
import xgboost as xgb
import sys
import random
import _pickle as cPickle
import os


#count features of ranking discretion
train_nd = pd.read_csv('train_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]

#discret features of count of missing values
train_dnull = pd.read_csv('train_x_null.csv')[['uid','discret_null']]

#considering the size of the features above (only 11) it is not necessary to do feature selection on them
#so they are merged and left alone in the feature selection process
eleven_feature = ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','discret_null']
train_eleven = pd.merge(train_nd,train_dnull,on='uid')



#discreet features
discreet_feature_score = pd.read_csv('discreet_feature_score.csv')
#600 of 735
fs = list(discreet_feature_score.feature[0:600])
discreet_train = pd.read_csv("train_x_discretization.csv")[['uid']+fs]

#ranking features
rank_feature_score = pd.read_csv('rank_feature_score.csv')
rank_feature_score.shape
#600 of 780
fs = list(rank_feature_score.feature[0:600])
rank_train = pd.read_csv("train_x_rank.csv")[['uid']+fs]

rank_train = rank_train[fs] / float(len(rank_train))
rank_train['uid'] = pd.read_csv("train_x_rank.csv").uid

#raw feature
raw_feature_score = pd.read_csv('raw_feature_score.csv')
raw_feature_score.shape
#700 of 838
fs = list(raw_feature_score.feature[0:700])

train_xy =pd.read_csv("rong360_all_feature_withid.csv")
train_xy.rename(columns={'target':'y'}, inplace=True)
train_xy.rename(columns={'userid':'uid'}, inplace=True)
#statistics of missing value per row

raw_train = train_xy[['uid']+fs+['y']]


#merge raw, ranking, discret and other 11 features
train = pd.merge(raw_train,rank_train,on='uid')
train = pd.merge(train,discreet_train,on='uid')
#
train = pd.merge(train,train_eleven,on='uid')
#unify all missing records to -1
train=train.fillna(-1)

train.y

train.to_csv("final.csv",index=None)



######################
#    xgb bagging     #
###################### 
#create randomness in the number of raw,ranking and discreet features 
#create randomness in the meta parameters of  

#by setting the number of feature from a random number from 300 to 500
#feature_num is such a variable

    ####################
    #       xgb        #
    ####################
def pipeline(iteration,random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambda_,subsample,colsample_bytree,min_child_weight):
    #define number of features as a variable feature_num
    raw_feature_selected = list(raw_feature_score.feature[0:feature_num])
    rank_feature_selected = list(rank_feature_score.feature[0:rank_feature_num])
    discreet_feature_selected = list(discreet_feature_score.feature[0:discret_feature_num])

    #construct training dataset from the randomly selected top features from raw, ranking, discret plus untouched 11
    train_xy = train[eleven_feature+raw_feature_selected+rank_feature_selected+discreet_feature_selected+['y']]

    test_x = test[eleven_feature+raw_feature_selected+rank_feature_selected+discreet_feature_selected]

    y = train_xy.y
    X = train_xy.drop(['y'],axis=1)
    

    dtest = xgb.DMatrix(test_x)
    dtrain = xgb.DMatrix(X, label=y)
    
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambda_,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.08,
    	'seed':random_seed,
    	'nthread':8
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1500,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(test_uid,columns=["uid"])
    test_result["score"] = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    
    random_seed = list(range(1000,2000,10))
    feature_num = list(range(200,400,2))
    rank_feature_num = list(range(60,110,2))
    discret_feature_num = list(range(50,80,1))
    gamma = [i/1000.0 for i in list(range(0,300,3))]
    max_depth = [6,7,8]
    lambda_ = list(range(500,700,2))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(250,550,3))]
    random.shuffle(rank_feature_num)
    random.shuffle(random_seed)
    random.shuffle(feature_num)
    random.shuffle(discret_feature_num)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambda_)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambda_,subsample,colsample_bytree,min_child_weight),f)
    
    
    for i in list(range(36)):
        print ("iter:",i)
        pipeline(i,random_seed[i],feature_num[i],rank_feature_num[i],discret_feature_num[i],gamma[i],max_depth[i%3],lambda_[i],subsample[i],colsample_bytree[i],min_child_weight[i])

    ##################################
    #  take average of xgb models    #
    ##################################

from sklearn.metrics import roc_auc_score

files = os.listdir('./preds')
pred = pd.read_csv('./preds/'+files[0])
uid = pred.uid
score = pred.score
for f in files[1:]:
    pred = pd.read_csv('./preds/'+f)
    score += pred.score

score /= len(files)

pred = pd.DataFrame(uid,columns=['uid'])
pred['score'] = score
pred.to_csv('avg_preds.csv',index=None,encoding='utf-8')

####cal auc

val_set = test_y
val_pred = pred
auc = roc_auc_score(val_set.y, val_pred.score.values)
print(auc)  

   



































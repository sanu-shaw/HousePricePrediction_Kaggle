# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 23:45:51 2020

@author: Sanu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dataset=pd.read_csv("train.csv")
dataset=pd.read_csv("test.csv")
dataset=pd.read_csv("TrainingPreparation.csv")
dataset=pd.read_csv("TestPreparation.csv")

X=pd.read_csv("test.csv")



dataset.drop(['Id'], axis = 1, inplace=True) 



dataset.info()
X=pd.read_csv("test.csv")
X.MSZoning.value_counts()
X.info()

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,79].values.reshape(-1,1)

###############Dealing with Outliers

dataset.groupby('YrSold')['SalePrice'].mean().plot().bar()
dataset.groupby('YearRemodAdd')['SalePrice'].mean()

dataset_backup=dataset.copy()

(dataset['SalePrice'].unique()==0).any()

dataset['SalePrice']=np.log10(dataset['SalePrice'])
sns.kdeplot(dataset['SalePrice'])



sns.kdeplot(dataset['LotArea'])

(dataset['LotArea'].unique()==0).any()

dataset['LotArea']=np.log10(dataset['LotArea'])



sns.kdeplot(np.log10(dataset['GrLivArea']))

(dataset['GrLivArea'].unique()==0).any()

dataset['GrLivArea']=np.log10(dataset['GrLivArea'])


dataset.drop(['YearBuilt'], axis=1, inplace=True)


#######################################

lower_qrt,upper_qrt=np.percentile(dataset.SalePrice,[25,75])
interquatile_range=upper_qrt-lower_qrt

lower_bound_value=lower_qrt-(1.5*interquatile_range)
upper_bound_value=upper_qrt+(1.5*interquatile_range)

dataset.drop(dataset[dataset['SalePrice']>upper_bound_value].index, inplace=True)
dataset.drop(dataset[dataset['SalePrice']<lower_bound_value].index, inplace=True)
dataset.reset_index(drop=True, inplace=True)





lower_qrt,upper_qrt=np.percentile(dataset.LotArea,[25,75])
interquatile_range=upper_qrt-lower_qrt

lower_bound_value=lower_qrt-(1.5*interquatile_range)
upper_bound_value=upper_qrt+(1.5*interquatile_range)


dataset.drop(dataset[dataset['LotArea']>upper_bound_value].index, inplace=True)
dataset.drop(dataset[dataset['LotArea']<lower_bound_value].index, inplace=True)
dataset.reset_index(drop=True, inplace=True)





lower_qrt,upper_qrt=np.percentile(dataset.GrLivArea,[25,75])
interquatile_range=upper_qrt-lower_qrt

lower_bound_value=lower_qrt-(1.5*interquatile_range)
upper_bound_value=upper_qrt+(1.5*interquatile_range)

dataset.drop(dataset[dataset['GrLivArea']>upper_bound_value].index, inplace=True)
dataset.drop(dataset[dataset['GrLivArea']<lower_bound_value].index, inplace=True)
dataset.reset_index(drop=True, inplace=True)

#outliers=[]
#mean=np.mean(dataset['LotArea'])
#sd=np.std(dataset['LotArea'])
#threshold=3
#for i in dataset['LotArea']:
#    z=(i-mean)/sd
#    if(np.abs(z)>threshold):
#        outliers.append(i)



#######################################


from sklearn.preprocessing import LabelEncoder,OneHotEncoder


#OneHOt for MSSubClass
dataset['MSSubClass'].value_counts()
dataset['MSSubClass'].isna().sum()
dataset=dataset.fillna(value={'MSSubClass':20})
for i in range(0,len(dataset)):
    if (dataset['MSSubClass'][i]==20 or dataset['MSSubClass'][i]==60 or dataset['MSSubClass'][i]==50 or dataset['MSSubClass'][i]==120 or dataset['MSSubClass'][i]==30 or dataset['MSSubClass'][i]==160 or dataset['MSSubClass'][i]==70 or dataset['MSSubClass'][i]==80 or dataset['MSSubClass'][i]==90 or dataset['MSSubClass'][i]==190 or dataset['MSSubClass'][i]==85 or dataset['MSSubClass'][i]==75 or dataset['MSSubClass'][i]==45 or dataset['MSSubClass'][i]==180 or dataset['MSSubClass'][i]==40):
        continue
    else:
        dataset['MSSubClass'][i]=20

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.MSSubClass.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['z1','z2','z3','z4','z5','z6','z7','z8','z9','z10','z11','z12','z13','z14']
dataset=dataset.join(ohecd)
dataset.drop(['MSSubClass'], axis=1, inplace=True)


## Working with GarageArea
dataset.fillna({'GarageArea':0}, inplace=True)

##Working on MSZoning label+onhot
dataset.MSZoning.isnull().any()
dataset.MSZoning.value_counts()

dataset=dataset.fillna(value={'MSZoning':'RL'})


for i in range(0,len(dataset)):
    if (dataset['MSZoning'][i].lower()=='rl' or dataset['MSZoning'][i].lower()=='rm' or dataset['MSZoning'][i].lower()=='fv' or dataset['MSZoning'][i].lower()=='rh' or dataset['MSZoning'][i].lower()=='c (all)'):
        continue
    else:
        dataset['MSZoning'][i]='rl'

le1=LabelEncoder()
dataset.MSZoning=le1.fit_transform(dataset.MSZoning)

ohecd=pd.DataFrame(oh1.fit_transform(dataset.MSZoning.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['a1','a2','a3','a4']
dataset=dataset.join(ohecd)
dataset.drop(['MSZoning'], axis = 1, inplace=True) 


##Working on LotFrontage  filling nan with 0
dataset.LotFrontage.isnull().sum()
dataset=dataset.fillna(value={'LotFrontage':0})


##Working with Street  only label encoded as it has only 2 options
dataset['Street'].value_counts()
le1=LabelEncoder()
dataset.Street=le1.fit_transform(dataset.Street)


##Working with Alley
dataset.Alley.value_counts()
dataset['Alley']=dataset['Alley'].fillna(value='none')
le1=LabelEncoder()
dataset.Alley=le1.fit_transform(dataset.Alley)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.Alley.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['b1','b2']
dataset=dataset.join(ohecd)
dataset.drop(['Alley'], axis = 1, inplace=True) 


##Working with LotShape
dataset.LotShape.value_counts()
dataset.LotShape.isna().sum()

for i in range(0,len(dataset)):
    if(pd.isnull(dataset['LotShape'][i])):
        dataset['LotShape'][i]='Reg'
        
le1=LabelEncoder()
dataset.LotShape=le1.fit_transform(dataset.LotShape)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.LotShape.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['c1','c2','c3']
dataset=dataset.join(ohecd)
dataset.drop(['LotShape'], axis = 1, inplace=True) 



##Working with TotalBsmtSF
dataset.fillna({'TotalBsmtSF':0}, inplace=True)
dataset.isna().sum()

##Working with LandContour
dataset.LandContour.value_counts()
dataset.LandContour.isna().sum()

for i in range(0,len(dataset)):
    if(pd.isnull(dataset['LandContour'][i])):
        dataset['LandContour'][i]='Lvl'

le1=LabelEncoder()
dataset.LandContour=le1.fit_transform(dataset.LandContour)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.LandContour.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['d1','d2','d3']
dataset=dataset.join(ohecd)
dataset.drop(['LandContour'], axis = 1, inplace=True) 



##Working with Utilities
#X.Utilities.value_counts()
dataset.Utilities.isnull().sum()


for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['Utilities'][i])):
        dataset['Utilities'][i]=0
    elif (dataset['Utilities'][i].lower()=='allpub'):
        dataset['Utilities'][i]=4
    elif (dataset['Utilities'][i].lower()=='nosewr'):
        dataset['Utilities'][i]=3
    elif (dataset['Utilities'][i].lower()=='nosewa'):
        dataset['Utilities'][i]=2
    else:
        dataset['Utilities'][i]=1



#Working with LotConfig
dataset.LotConfig.value_counts()

le1=LabelEncoder()
dataset.LotConfig=le1.fit_transform(dataset.LotConfig)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.LotConfig.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['e1','e2','e3','e4']
dataset=dataset.join(ohecd)
dataset.drop(['LotConfig'], axis = 1, inplace=True) 



##Working with LandSlope
dataset.LandSlope.value_counts()

le1=LabelEncoder()
dataset.LandSlope=le1.fit_transform(dataset.LandSlope)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.LandSlope.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['f1','f2']
dataset=dataset.join(ohecd)
dataset.drop(['LandSlope'], axis = 1, inplace=True) 



##Working with Neighborhood

dataset['Neighborhood'].isnull().values.any()
dataset.Neighborhood.value_counts()

le1=LabelEncoder()
dataset.Neighborhood=le1.fit_transform(dataset.Neighborhood)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.Neighborhood.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['g1','g2','g3','g4','g5','g6','g7','g8','g9','g10','g11','g12','g13','g14','g15','g16','g17','g18','g19','g20','g21','g22','g23','g24']
dataset=dataset.join(ohecd)
dataset.drop(['Neighborhood'], axis = 1, inplace=True) 



##Working with Condition1 & Condition2

dataset.Condition2.isna().sum()
dataset.Condition2.value_counts()


le1=LabelEncoder()
dataset.Condition1=le1.fit_transform(dataset.Condition1)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.Condition1.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['r1','r2','r3','r4','r5','r6','r7','r8']
dataset=dataset.join(ohecd)
dataset.drop(['Condition1'], axis = 1, inplace=True)

dataset.drop(['Condition2'], axis = 1, inplace=True)



##Working with BldgType
dataset.BldgType.value_counts()
dataset.BldgType.isna().sum()

le1=LabelEncoder()
dataset.BldgType=le1.fit_transform(dataset.BldgType)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.BldgType.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['i1','i2','i3','i4']
dataset=dataset.join(ohecd)
dataset.drop(['BldgType'], axis = 1, inplace=True)



#working with HouseStyle
dataset.HouseStyle.value_counts()

for i in range(0,len(dataset)):
    if(dataset['HouseStyle'][i]=='2.5Fin'):
        dataset['HouseStyle'][i]='1Story'



le1=LabelEncoder()
dataset.HouseStyle=le1.fit_transform(dataset.HouseStyle)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.HouseStyle.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['j1','j2','j3','j4','j5','j6']
dataset=dataset.join(ohecd)
dataset.drop(['HouseStyle'], axis = 1, inplace=True)




##Working on RoofStyle
dataset.RoofStyle.value_counts()
dataset.RoofStyle.isnull().any()


le1=LabelEncoder()
dataset.RoofStyle=le1.fit_transform(dataset.RoofStyle)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.RoofStyle.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['k1','k2','k3','k4','k5']
dataset=dataset.join(ohecd)
dataset.drop(['RoofStyle'], axis = 1, inplace=True)




##Workin with RoofMatl
#dataset.RoofMatl.value_counts()
#dataset.RoofMatl.isnull().any()
#X.RoofMatl.value_counts()
#X.RoofMatl.isnull().any()
#
#le1=LabelEncoder()
#dataset.RoofMatl=le1.fit_transform(dataset.RoofMatl)
#
#oh1=OneHotEncoder() 
#ohecd=pd.DataFrame(oh1.fit_transform(dataset.RoofMatl.values.reshape(-1,1)).toarray())
#ohecd.drop([0], axis=1, inplace=True)
#ohecd.columns=['l1','l2','l3','l4','l5','l6','l7']
#dataset=dataset.join(ohecd)
dataset.drop(['RoofMatl'], axis = 1, inplace=True)


##Working with Exterior1st
dataset.Exterior1st.value_counts()
dataset.Exterior1st.isnull().any()



dataset.fillna({'Exterior1st':'VinylSd'}, inplace=True)
tempList=[]
for data in dataset.Exterior1st:
    if(data=='VinylSd' or data=='HdBoard' or data=='MetalSd' or data=='Wd Sdng' or data=='Plywood' or data=='CemntBd' or data=='BrkFace' or data=='WdShing' or data=='Stucco' or data=='AsbShng'):
        tempList.append(data)
    else:
        tempList.append("others")
    
dataset=dataset.join(pd.DataFrame(tempList, columns=['Exterior1st_mod']))

le1=LabelEncoder()
dataset.Exterior1st_mod=le1.fit_transform(dataset.Exterior1st_mod)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.Exterior1st_mod.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10']
dataset=dataset.join(ohecd)
dataset.drop(['Exterior1st_mod'], axis = 1, inplace=True)
dataset.drop(['Exterior1st'], axis = 1, inplace=True)


##Working with Exterior2nd
dataset.Exterior2nd.value_counts()
dataset.Exterior2nd.isnull().any()



dataset['Exterior2nd']=dataset['Exterior2nd'].fillna(value='others')

tempList=[]
for data in dataset.Exterior2nd:
    if(data=='VinylSd' or data=='HdBoard' or data=='MetalSd' or data=='Wd Sdng' or data=='Plywood' or data=='CmentBd' or data=='BrkFace' or data=='WdShing' or data=='Stucco' or data=='AsbShng'):
        tempList.append(data)
    else:
        tempList.append("others")
    
dataset=dataset.join(pd.DataFrame(tempList, columns=['Exterior2nd_mod']))
dataset.Exterior2nd_mod.value_counts()

le1=LabelEncoder()
dataset.Exterior2nd_mod=le1.fit_transform(dataset.Exterior2nd_mod)


oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.Exterior2nd_mod.values.reshape(-1,1)).toarray())


ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']
dataset=dataset.join(ohecd)
dataset.drop(['Exterior2nd_mod'], axis = 1, inplace=True)
dataset.drop(['Exterior2nd'], axis = 1, inplace=True)


##Working with MasVnrType
dataset.MasVnrType.value_counts()
dataset.MasVnrType.isnull()




dataset['MasVnrType']=(dataset['MasVnrType'].fillna(dataset['MasVnrType'].mode()[0]))


le1=LabelEncoder()
dataset.MasVnrType=le1.fit_transform(dataset.MasVnrType)
oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.MasVnrType.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['o1','o2','o3']
dataset=dataset.join(ohecd)
dataset.drop(['MasVnrType'], axis = 1, inplace=True)



##Working with MasVnrArea
dataset.MasVnrArea=dataset.MasVnrArea.fillna(0)



##Working with ExterQual
len(dataset)

dataset.ExterQual.value_counts()

for i in range(0,len(dataset)):
    if (dataset['ExterQual'][i].lower()=='ex'):
        dataset['ExterQual'][i]=4
    elif (dataset['ExterQual'][i].lower()=='gd'):
        dataset['ExterQual'][i]=3
    elif (dataset['ExterQual'][i].lower()=='ta'):
        dataset['ExterQual'][i]=2
    elif (dataset['ExterQual'][i].lower()=='fa'):
        dataset['ExterQual'][i]=1
    else:
        dataset['ExterQual'][i]=0



##Working with ExterCond
dataset.ExterCond.isnull().any()
dataset.ExterCond.value_counts()

for i in range(0,len(dataset)):
    if (dataset['ExterCond'][i].lower()=='ex'):
        dataset['ExterCond'][i]=4
    elif (dataset['ExterCond'][i].lower()=='gd'):
        dataset['ExterCond'][i]=3
    elif (dataset['ExterCond'][i].lower()=='ta'):
        dataset['ExterCond'][i]=2
    elif (dataset['ExterCond'][i].lower()=='fa'):
        dataset['ExterCond'][i]=1
    else:
        dataset['ExterCond'][i]=0



##Working with Foundation
dataset.Foundation.value_counts()
dataset.Foundation.isnull().any()



le1=LabelEncoder()
dataset.Foundation=le1.fit_transform(dataset.Foundation)
oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.Foundation.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['p1','p2','p3','p4','p5']
dataset=dataset.join(ohecd)
dataset.drop(['Foundation'], axis = 1, inplace=True)



##Working with BsmtQual
dataset.BsmtQual.value_counts()
dataset.BsmtQual.isnull().any()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['BsmtQual'][i])):
        dataset['BsmtQual'][i]=0
    elif (dataset['BsmtQual'][i].lower()=='ex'):
        dataset['BsmtQual'][i]=5
    elif (dataset['BsmtQual'][i].lower()=='gd'):
        dataset['BsmtQual'][i]=4
    elif (dataset['BsmtQual'][i].lower()=='ta'):
        dataset['BsmtQual'][i]=3
    elif (dataset['BsmtQual'][i].lower()=='fa'):
        dataset['BsmtQual'][i]=2
    else:
        dataset['BsmtQual'][i]=1




##Working with BsmtCond
dataset.BsmtCond.value_counts()
dataset.BsmtCond.isnull().any()


for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['BsmtCond'][i]) or dataset['BsmtCond'][i].lower()=='po'):
        dataset['BsmtCond'][i]=0
    elif (dataset['BsmtCond'][i].lower()=='ex'):
        dataset['BsmtCond'][i]=4
    elif (dataset['BsmtCond'][i].lower()=='gd'):
        dataset['BsmtCond'][i]=3
    elif (dataset['BsmtCond'][i].lower()=='ta'):
        dataset['BsmtCond'][i]=2
    else:
        dataset['BsmtCond'][i]=1



##Working with BsmtExposure
dataset.BsmtExposure.value_counts()
dataset.BsmtExposure.isnull().any()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['BsmtExposure'][i]) or dataset['BsmtExposure'][i].lower()=='no'):
        dataset['BsmtExposure'][i]=0
    elif (dataset['BsmtExposure'][i].lower()=='gd'):
        dataset['BsmtExposure'][i]=3
    elif (dataset['BsmtExposure'][i].lower()=='av'):
        dataset['BsmtExposure'][i]=2
    else:
        dataset['BsmtExposure'][i]=1




##Working with BsmtFinType1
dataset.BsmtFinType1.value_counts()
dataset.BsmtFinType1.isnull().any()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['BsmtFinType1'][i])):
        dataset['BsmtFinType1'][i]=0
    elif (dataset['BsmtFinType1'][i].lower()=='glq'):
        dataset['BsmtFinType1'][i]=6
    elif (dataset['BsmtFinType1'][i].lower()=='alq'):
        dataset['BsmtFinType1'][i]=5
    elif (dataset['BsmtFinType1'][i].lower()=='blq'):
        dataset['BsmtFinType1'][i]=4
    elif (dataset['BsmtFinType1'][i].lower()=='rec'):
        dataset['BsmtFinType1'][i]=3
    elif (dataset['BsmtFinType1'][i].lower()=='lwq'):
        dataset['BsmtFinType1'][i]=2
    else:
        dataset['BsmtFinType1'][i]=1
        


#Working with BsmtFinType2
dataset.BsmtFinType2.value_counts()
dataset.BsmtFinType2.isnull().sum()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['BsmtFinType2'][i])):
        dataset['BsmtFinType2'][i]=0
    elif (dataset['BsmtFinType2'][i].lower()=='unf'):
        dataset['BsmtFinType2'][i]=1
    else:
        dataset['BsmtFinType2'][i]=2



##Wroking with BsmtFinSF1 BsmtFinSF2 BsmtUnfSF    
dataset.drop(['BsmtFinSF1'], axis = 1, inplace=True)
dataset.drop(['BsmtFinSF2'], axis = 1, inplace=True)
dataset.drop(['BsmtUnfSF'], axis = 1, inplace=True)



##Working with Heating
dataset.Heating.value_counts()
dataset.Heating.isnull().any()

for i in range(0,len(dataset)):
    #print(i)
    if (dataset['Heating'][i].lower()=='gasa'):
        dataset['Heating'][i]=1
    else:
        dataset['Heating'][i]=0


##HeatingQC
dataset.HeatingQC.value_counts()
dataset.HeatingQC.isnull().sum()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['HeatingQC'][i])):
        dataset['HeatingQC'][i]=0
    elif (dataset['HeatingQC'][i].lower()=='ex'):
        dataset['HeatingQC'][i]=3
    elif (dataset['HeatingQC'][i].lower()=='gd'):
        dataset['HeatingQC'][i]=2
    elif (dataset['HeatingQC'][i].lower()=='ta'):
        dataset['HeatingQC'][i]=1
    else:
        dataset['HeatingQC'][i]=0        


##Working with CentralAir
dataset.CentralAir.value_counts()
dataset.CentralAir.isnull().any()


le1=LabelEncoder()
dataset['CentralAir']=le1.fit_transform(dataset['CentralAir'])


##Working with Electrical
dataset.Electrical.value_counts()
dataset.Electrical.isnull().sum()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['Electrical'][i])):
        dataset['Electrical'][i]=0
    elif (dataset['Electrical'][i].lower()=='sbrkr'):
        dataset['Electrical'][i]=1
    else:
        dataset['Electrical'][i]=0  


##Working with 1stFlrSF & 2ndFlrSF
dataset['1stFlrSF'].dtypes
dataset['2ndFlrSF'].dtypes
dataset['totalSF']=dataset['1stFlrSF']+dataset['2ndFlrSF']

dataset.drop(['1stFlrSF'], axis = 1, inplace=True)
dataset.drop(['2ndFlrSF'], axis = 1, inplace=True)



##Working with LowQualFinSF  GrLivArea
dataset.LowQualFinSF.isnull().sum()
dataset['LowQualFinSF'].dtypes

dataset.GrLivArea.isnull().sum()
dataset['GrLivArea'].dtypes

##Working with BsmtFullBath & BsmtHalfBath
dataset.drop(['BsmtFullBath'], axis = 1, inplace=True)
dataset.drop(['BsmtHalfBath'], axis = 1, inplace=True)


##Working with KitchenQual
dataset.KitchenQual.isnull().sum()
dataset.KitchenQual.value_counts()



dataset.fillna({'KitchenQual':'TA'}, inplace=True)

for i in range(0,len(dataset)):
    #print(i)
    if(dataset['KitchenQual'][i].lower()=='ex'):
        dataset['KitchenQual'][i]=4
    elif (dataset['KitchenQual'][i].lower()=='gd'):
        dataset['KitchenQual'][i]=3
    elif (dataset['KitchenQual'][i].lower()=='ta'):
        dataset['KitchenQual'][i]=2
    elif (dataset['KitchenQual'][i].lower()=='fa'):
        dataset['KitchenQual'][i]=1
    else:
        dataset['KitchenQual'][i]=0  
        


##Working with Functional
dataset.Functional.isnull().any()
dataset.Functional.value_counts()


for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['Functional'][i])):
        dataset['Functional'][i]=0
    elif (dataset['Functional'][i].lower()=='typ'):
        dataset['Functional'][i]=8
    elif (dataset['Functional'][i].lower()=='min1'):
        dataset['Functional'][i]=7
    elif (dataset['Functional'][i].lower()=='min2'):
        dataset['Functional'][i]=6
    elif (dataset['Functional'][i].lower()=='mod'):
        dataset['Functional'][i]=5
    elif (dataset['Functional'][i].lower()=='maj1'):
        dataset['Functional'][i]=4
    elif (dataset['Functional'][i].lower()=='maj2'):
        dataset['Functional'][i]=3
    elif (dataset['Functional'][i].lower()=='sev'):
        dataset['Functional'][i]=2        
    else:
        dataset['Functional'][i]=1


##Working with FireplaceQu
dataset.FireplaceQu.value_counts()
dataset.FireplaceQu.isnull().sum()


for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['FireplaceQu'][i])):
        dataset['FireplaceQu'][i]=0
    elif (dataset['FireplaceQu'][i].lower()=='ex'):
        dataset['FireplaceQu'][i]=5
    elif (dataset['FireplaceQu'][i].lower()=='gd'):
        dataset['FireplaceQu'][i]=4
    elif (dataset['FireplaceQu'][i].lower()=='ta'):
        dataset['FireplaceQu'][i]=3
    elif (dataset['FireplaceQu'][i].lower()=='fa'):
        dataset['FireplaceQu'][i]=2       
    else:
        dataset['FireplaceQu'][i]=1



##Working with GarageFinish
dataset.GarageFinish.value_counts()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['GarageFinish'][i])):
        dataset['GarageFinish'][i]=0
    elif (dataset['GarageFinish'][i].lower()=='fin'):
        dataset['GarageFinish'][i]=3
    elif (dataset['GarageFinish'][i].lower()=='rfn'):
        dataset['GarageFinish'][i]=2
    else:
        dataset['GarageFinish'][i]=1
        


##Working with GarageCars & GarageYrBlt & GarageType
dataset.drop(['GarageCars'], axis = 1, inplace=True)  
dataset.drop(['GarageYrBlt'], axis = 1, inplace=True)
dataset.drop(['GarageType'], axis = 1, inplace=True)      


###Working with GarageQual
dataset.GarageQual.value_counts()

for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['GarageQual'][i])):
        dataset['GarageQual'][i]=0
    elif (dataset['GarageQual'][i].lower()=='ex'):
        dataset['GarageQual'][i]=5
    elif (dataset['GarageQual'][i].lower()=='gd'):
        dataset['GarageQual'][i]=4
    elif (dataset['GarageQual'][i].lower()=='ta'):
        dataset['GarageQual'][i]=3
    elif (dataset['GarageQual'][i].lower()=='fa'):
        dataset['GarageQual'][i]=2       
    else:
        dataset['GarageQual'][i]=1



##Working with GarageCond
dataset.drop(['GarageCond'], axis = 1, inplace=True)  


##Working with PavedDrive
for i in range(0,len(dataset)):
    if(pd.isnull(dataset['PavedDrive'][i])):
        dataset['PavedDrive'][i]=0
    elif (dataset['PavedDrive'][i].lower()=='y'):
        dataset['PavedDrive'][i]=3
    elif (dataset['PavedDrive'][i].lower()=='p'):
        dataset['PavedDrive'][i]=2
    else:
        dataset['PavedDrive'][i]=1
        
        
##Working with        
dataset['OpenPorchSF'].dtype
dataset['EnclosedPorch'].dtype
dataset['3SsnPorch'].dtype
dataset['ScreenPorch'].dtype


dataset['TotalPorchSF']=dataset['OpenPorchSF']+dataset['EnclosedPorch']+dataset['3SsnPorch']+dataset['ScreenPorch']
dataset.drop(['OpenPorchSF'], axis = 1, inplace=True) 
dataset.drop(['EnclosedPorch'], axis = 1, inplace=True) 
dataset.drop(['3SsnPorch'], axis = 1, inplace=True) 
dataset.drop(['ScreenPorch'], axis = 1, inplace=True) 


##Working with PoolQC
dataset.PoolQC.value_counts()
for i in range(0,len(dataset)):
    #print(i)
    if(pd.isnull(dataset['PoolQC'][i])):
        dataset['PoolQC'][i]=0
    elif (dataset['PoolQC'][i].lower()=='ex'):
        dataset['PoolQC'][i]=5
    elif (dataset['PoolQC'][i].lower()=='gd'):
        dataset['PoolQC'][i]=4
    elif (dataset['PoolQC'][i].lower()=='ta'):
        dataset['PoolQC'][i]=3
    elif (dataset['PoolQC'][i].lower()=='fa'):
        dataset['PoolQC'][i]=2       
    else:
        dataset['PoolQC'][i]=1


##Working with Fence
dataset.Fence.value_counts()
dataset.Fence.isnull().any()


dataset['Fence']=dataset['Fence'].fillna(value='none')
le1=LabelEncoder()
dataset.Fence=le1.fit_transform(dataset.Fence)

oh1=OneHotEncoder() 
ohecd=pd.DataFrame(oh1.fit_transform(dataset.Fence.values.reshape(-1,1)).toarray())
ohecd.drop([0], axis=1, inplace=True)
ohecd.columns=['q1','q2','q3','q4']
dataset=dataset.join(ohecd)
dataset.drop(['Fence'], axis = 1, inplace=True)


##Working with MiscFeature & SaleType &  YrSold & MoSold & SaleCondition
dataset.drop(['MiscFeature'], axis = 1, inplace=True)
dataset.drop(['SaleType'], axis = 1, inplace=True)
dataset.drop(['YrSold'], axis = 1, inplace=True)
dataset.drop(['MoSold'], axis = 1, inplace=True)
dataset.drop(['SaleCondition'], axis = 1, inplace=True)


#dataset.to_csv("TrainingPreparation.csv") 
#dataset.to_csv("TestPreparation.csv") 


X=dataset.drop(['SalePrice'], axis = 1).values
y=dataset.SalePrice.values.reshape(-1,1)


################################

from sklearn.feature_selection import SelectKBest,f_regression
kBest=SelectKBest(score_func=f_regression, k='all')
model=kBest.fit(X_train_mm,y_train_mm)
len(model.scores_)
df_score=pd.DataFrame(model.scores_, columns=['score'])
df_col=pd.DataFrame((dataset.drop(['SalePrice'], axis = 1)).columns, columns=['features'])
df_final_KB=pd.concat([df_col,df_score],axis=1)

X_test_KB=df_final_KB.nlargest(120,'score')['features']
X_test_KB=X_test_KB.reset_index(drop=True)
KB_list=(X_test_KB).tolist()
KB_list_1=dataset[KB_list]
#X_test_ETR1_1.drop(['SalePrice'], axis = 1, inplace=True)
X=KB_list_1.values



from sklearn.ensemble import ExtraTreesRegressor
model_ETR=ExtraTreesRegressor()
model_ETR.fit(X,y)
len(model_ETR.feature_importances_)
df_score_ETR=pd.DataFrame(model_ETR.feature_importances_, columns=['Features Importance'])
df_col_ETR=pd.DataFrame((dataset.drop(['SalePrice'], axis = 1)).columns, columns=['features'])
df_final_ETR1=pd.concat([df_col_ETR,df_score_ETR],axis=1)

X_test_ETR1=df_final_ETR1.nlargest(120,'Features Importance')['features']
X_test_ETR1=X_test_ETR1.reset_index(drop=True)
ETR_list=(X_test_ETR1).tolist()
X_test_ETR1_1=dataset[ETR_list]
#X_test_ETR1_1.drop(['SalePrice'], axis = 1, inplace=True)
X=X_test_ETR1_1.values



###############################

X_testing=(dataset[KB_list]).values

#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(X, y, random_state=0, test_size=0.20) 



from sklearn.preprocessing import MinMaxScaler
mm_X=MinMaxScaler()
X_train_mm=mm_X.fit_transform(X)


X_train_mm_test=mm_X.transform(X_testing)

mm_y=MinMaxScaler()
y_train_mm=mm_y.fit_transform(y)




from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train_sc,y_train_sc)
y_pred_sc=regressor.predict(X_test_sc)



#########################
from sklearn.model_selection import cross_val_score
estimator=cross_val_score(estimator=regressor, X=X,y=y, scoring='neg_mean_squared_error', cv=10)
#estimator_sc=cross_val_score(estimator=regressor, X=X_train_sc,y=y_train_sc, scoring='neg_mean_squared_error', cv=10)
np.sqrt(np.abs(estimator.mean()))

from sklearn.metrics import SCORERS
SCORERS.keys()
########################

###Used this#############################
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train_mm,y_train_mm)
y_pred_mm=regressor.predict(X_test_mm)

y_pred_mm_test=regressor.predict(X_train_mm_test)
X['SalePrice']=mm_y.inverse_transform(y_pred_mm_test.reshape(-1,1))

import openpyxl
X.to_csv("Submission2.csv") 


from sklearn.metrics import mean_squared_error
rmse=np.sqrt(mean_squared_error(y_test_mm,y_pred_mm))


regressor=RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train_rs,y_train_rs)
y_pred_rs=regressor.predict(X_test_rs)

from sklearn.metrics import r2_score
r2=r2_score(y_test_mm,y_pred_mm)

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test_sc,y_pred_sc)
mse_mm=mean_squared_error(y_test_mm,y_pred_mm)
mse_rs=mean_squared_error(y_test_rs,y_pred_rs)



from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import mean_squared_error

model=Sequential()
#model.add(Dense(activation='relu',input_dim=170, output_dim=500, init='uniform'))
model.add(Dense(activation='relu',input_dim=120, output_dim=500, init='uniform'))
model.add(Dense(activation='relu',output_dim=500))
model.add(Dense(activation='relu',output_dim=500))
model.add(Dense(activation='relu',output_dim=500))
model.add(Dense(activation='relu',output_dim=500))
model.add(Dense(activation='linear',output_dim=1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
model.fit(X_train_mm,y_train_mm,batch_size=50, epochs=100)
y_pred_nural=model.predict(X_test_Shrink)
model.summary()

mse_nural=mean_squared_error(y_test_mm,y_pred_nural)


plt.hist(y_train_mm)
plt.show()

import sklearn.metrics
sklearn.metrics.SCORERS.keys()

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

parameters={'alpha':[1e-15,1e-10,1e-5,1e-2,1,10,15,20,50,100]}
lasso=Lasso()
gs=GridSearchCV(estimator=lasso,param_grid=parameters,cv=5, scoring='neg_mean_squared_error')

gs.fit(X_train_mm,y_train_mm)
print(gs.best_params_)
print(gs.best_score_)

lasso1=Lasso(alpha=1e-05)
lasso1.fit(X_train_mm,y_train_mm)
len(lasso1.coef_)

###############TEST NEST
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
selector = SelectFromModel(estimator=Lasso(alpha=1e-05,random_state=0)).fit(X_train_mm,y_train_mm)
features_SFM=list(np.take((dataset.drop(['SalePrice'], axis = 1)).columns, np.where(selector.get_support() == True)[0]))

selector.estimator_.coef_
selector.threshold_
selector.get_support().sum()
selector.estimator

X_train_Shrink=selector.transform(X_train_mm)
X_test_Shrink=selector.transform(X_test_mm)

regressor=RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X_train_Shrink,y_train_mm)
y_pred_ls=regressor.predict(X_test_Shrink)
mse_ls=mean_squared_error(y_test_mm,y_pred_ls)

#################################################

from xgboost import XGBRegressor
test_regressor=XGBRegressor(objective='reg:linear', eval_metric='rmse')
regressor=XGBRegressor(objective='reg:linear',n_estimators= 1000, max_depth=2, learning_rate=0.1, eval_metric='rmse')
regressor.fit(X,y)
y_pred=[]
y_pred=regressor.predict(X)


y_pred_actual=10**y_pred


sub4=pd.DataFrame(y_pred_actual, columns=['SalePrice'])
X=X.join(sub4)
X.to_csv("Submission5.csv") 

import pickle
with open('regressorXBG.pickle', 'wb') as fh:
   pickle.dump(regressor, fh)



parametersXGB={
        'learning_rate':[0.1,0.15,0.2,0.25],
        'max_depth':[1,2,3],
        'n_estimators':[900,1000,1100]
        }
from sklearn.model_selection import RandomizedSearchCV
rand_GS=RandomizedSearchCV(estimator=test_regressor, param_distributions=parametersXGB, cv=10, scoring='neg_mean_squared_error')
rand_GS=rand_GS.fit(X,y)
best_parama_XGB=rand_GS.best_params_
rand_GS.best_score_

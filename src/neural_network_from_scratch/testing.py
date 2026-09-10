import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as norm
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from full_neural_net import Neural_net
Taxi_data= pd.read_csv(r"C:\Users\rakhi\Downloads\chicago_taxi_train (1).csv")

#data understanding
Taxi_data.shape


# sanity check
Taxi_data.isna().sum().T
Taxi_data.duplicated().sum()#0 duplicates

#EDA
#descriptive stats
Taxi_data.describe()

#data visualization
#histogram
#for i in Taxi_data.select_dtypes(include="number").columns:

    #sns.histplot(data=Taxi_data,x=i)
    #plt.show()

#boxplots
#for i in Taxi_data.select_dtypes(include="number").columns:
    #list[i]
    #sns.boxplot(data=Taxi_data,x=i)
    #plt.show()

#scatterplot
#for i in Taxi_data.select_dtypes(include="number").columns:
    #list[i]
    #sns.scatterplot(data=Taxi_data,x=i,y='FARE')#TRIP MILES TRIP SECONDS , TRIP TOTAL
    #plt.show()


#CORRELATION
#corr=Taxi_data.corr(numeric_only=True)
#sns.heatmap(corr,annot=True)
#plt.show()

#information gain
Taxi_data['speed'] = Taxi_data['TRIP_MILES'] / Taxi_data['TRIP_SECONDS'].replace(0, np.nan)

for i in Taxi_data.select_dtypes(include='number').columns:
    if Taxi_data[i].isnull().any() :
        Taxi_data.fillna({i:Taxi_data[i].mean()},inplace=True)

features= Taxi_data.select_dtypes(include="number").drop(columns=('FARE')).copy()
info_gain=mutual_info_regression(X=Taxi_data[['speed']],y=Taxi_data['FARE'])
pd.Series(data=info_gain,index=Taxi_data[['speed']].columns)

#dropping unnecesarry features
Taxi_data = Taxi_data[['TRIP_SECONDS','TRIP_MILES','FARE','TIPS','TRIP_START_HOUR','speed','TIP_RATE','TOLLS','TRIP_TOTAL']]
#splitting data
# change the split to keep all three features
Temp_x, X_test, Temp_y, Y_test = train_test_split(
    Taxi_data.select_dtypes(include='number').drop(columns=['FARE']),
    Taxi_data["FARE"], test_size=0.2, shuffle=True, random_state=42)
X_train, X_Val, Y_train, Y_val = train_test_split(
    Temp_x, Temp_y, test_size=0.25, shuffle=True, random_state=42)

#Normalizing the dataset
normal= norm()
normalizer = normal.fit(X=X_train)
X_train = pd.DataFrame(normalizer.fit_transform(X_train),columns=X_train.columns)
X_Val =  pd.DataFrame(normalizer.transform(X_Val),columns=X_train.columns)
X_test = pd.DataFrame(normalizer.transform(X_test),columns=X_train.columns)


X, y = make_multilabel_classification(
    n_samples=100000, n_features=8, n_classes=3, random_state=42
)


Y_train =Y_train.values.reshape(-1,1)           
            
import matplotlib.pyplot as plt

sgd_net = Neural_net(loss='mse')
sgd_net.add_layer(neurons=16, features=8, activation='relu')
sgd_net.add_layer(neurons=8, features=16, activation='relu')
sgd_net.add_layer(neurons=3, features=8, activation='softmax')
sgd_loss = sgd_net.train(X=X, labels=y,batch_size=20, learning_rate=0.00001, epoch=1, optimizer='sgd')

momentum_net = Neural_net(loss='mse')
momentum_net.add_layer(neurons=16, features=8, activation='relu')
momentum_net.add_layer(neurons=8, features=16, activation='relu')
momentum_net.add_layer(neurons=3, features=8, activation='softmax')
momentum_loss = momentum_net.train(X=X, labels=y,batch_size=20, learning_rate=0.00001, epoch=1, optimizer='momentum')

rmsprop_net = Neural_net(loss='mse')
rmsprop_net.add_layer(neurons=16, features=8, activation='relu')
rmsprop_net.add_layer(neurons=8, features=16, activation='relu')
rmsprop_net.add_layer(neurons=3, features=8, activation='softmax')
rmsprop_loss = rmsprop_net.train(X=X, labels=y,batch_size=20, learning_rate=0.00001, epoch=1, optimizer='rmsprop')

adam_net = Neural_net(loss='mse')
adam_net.add_layer(neurons=16, features=8, activation='relu')
adam_net.add_layer(neurons=8, features=16, activation='relu')
adam_net.add_layer(neurons=3, features=8, activation='softmax')
adam_loss = adam_net.train(X=X, labels=y,batch_size=20, learning_rate=0.00001, epoch=1, optimizer='adam')
print(f'adam_loss:{adam_loss}\n  rmsprop_loss: {rmsprop_loss} \n momentum : {momentum_loss} \n sgd: {sgd_loss}')
plt.figure(figsize=(10, 6))
plt.plot(sgd_loss, label='SGD')
plt.plot(momentum_loss, label='Momentum')
plt.plot(rmsprop_loss, label='RMSprop')
plt.plot(adam_loss, label='Adam')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('SGD vs Momentum vs RMSprop vs Adam')
plt.legend()
plt.show()





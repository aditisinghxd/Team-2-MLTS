# MLTS Project Proposal 
## Wind Power and New York Taxi Traffic!
### Wind power In Wind Turbines, Scada Systems measure and save data’s like wind speed, wind direction, generated power etc. for 10 minutes intervals. Our data [Eri18] was taken from a wind turbine’s scada system that is working and generating power in Turkey. Realising trends and seasonality of the data will ensure better maintenance of the system in terms of the daily requirement, storage of excess produced energy and handling fluctuations in production and demand. NYC Taxi Traffic is a similar dataset, whose study is vital in managing traffic and understanding demand patterns throughout the day. The dataset consists of aggregating the total number of taxi passengers into 30-minute buckets.
Anomaly detection: In an information driven world the detection of anomalies can help
detect fraud or intrusions, monitor health status or find errors in the data to remove or handle
outliers. [CC19] An Autoencoder is a powerful tool for unsupervised learning that is trained to
copy its input to its output, while creating a compressed encoding. [GBC16] Autoencoder are able to be trained to detect anomalies by checking the reconstruction probability of the input. [AC15]
Auto Regressive Integrated Moving Averages: An autoregressive integrated moving
average, or ARIMA, is a statistical analysis model that uses time-series data to either better understand the data set or to predict future trends. A statistical model is autoregressive if it predicts
future values based on past values [MSA18].
Random forest models are an ensemble of many decision trees where the decision trees are
known as weak learners. It can be applied to classification and regression problems. Also, it is a
widely used model for regression analysis. The regression procedure using random forest can be
accomplished in the following steps:
1. Data splitting: The process goes through the splitting of features and each row is responsible
for the creation of decision trees.
2. Decision making: Every tree makes its individual decision based on the data.
3. Decision aggregation: In this step average value predictions from trees become the final
result. This average of decisions from trees makes random forest regression stronger than
any other algorithm.
LSTM The LSTM is a class of RNN model, that is able to learn long time dependencies[GBC16].
After fitting the model to a given time series, it can be used to predict one or several time steps
[MEE20].
For anomaly detection, we need to build a model that is able to detect anomalies in the data,
especially in the New York Taxi Traffic dataset we want to be able to detect the "five anomalies
[that] occur during the NYC marathon, Thanksgiving, Christmas, New Years day, and a snow
storm"[jul22]
In order to investigate the trend and seasonality of the production of wind power- green energy, based on production variables recorded in the dataset, Auto-Regressive Integerated Moving Averages
model will be used. The major steps include time series components detection via decomposition
and then removal of the same. Further making the data stationary in order to implement ARIMA model and chose the best model using ACF (Auto-correlation Function) and PACF (Partial
Auto-correlation Function) graphs. This model will then allow us to predict the future values.
In order to use a Deep Learning method, we want to use an LSTM. The major steps are to
choose a proper architecture, learning rate, optimizer and hyperparameters, then fit the model to
the data and finally compare it to a baseline, in our case the previous mentioned ARIMA model.



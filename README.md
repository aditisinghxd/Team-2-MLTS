# MLTS Project Proposal 
## Wind Power and New York Taxi Traffic!
### Wind power 
In Wind Turbines, Scada Systems measure and save data’s like wind speed, wind direction, generated power etc. for 10 minutes intervals. Our data was taken from a wind turbine’s scada system that is working and generating power in Turkey. Realising trends and seasonality of the data will ensure better maintenance of the system in terms of the daily requirement, storage of excess produced energy and handling fluctuations in production and demand. 
### NYC Taxi Traffic 
NYC Taxi Traffic is a similar dataset, whose study is vital in managing traffic and understanding demand patterns throughout the day. The dataset consists of aggregating the total number of taxi passengers into 30-minute buckets.
### Anomaly detection: 
In an information driven world the detection of anomalies can help detect fraud or intrusions, monitor health status or find errors in the data to remove or handle
outliers. An Autoencoder is a powerful tool for unsupervised learning that is trained to copy its input to its output, while creating a compressed encoding. Autoencoder are able to be trained to detect anomalies by checking the reconstruction probability of the input.
### Auto Regressive Integrated Moving Averages: 
An autoregressive integrated moving average, or ARIMA, is a statistical analysis model that uses time-series data to either better understand the data set or to predict future trends. A statistical model is autoregressive if it predicts future values based on past values.
### Random forest:
Random forest models are an ensemble of many decision trees where the decision trees are known as weak learners. It can be applied to classification and regression problems. Also, it is a widely used model for regression analysis. The regression procedure using random forest can be accomplished in the following steps:
1. Data splitting: The process goes through the splitting of features and each row is responsible
for the creation of decision trees.
2. Decision making: Every tree makes its individual decision based on the data.
3. Decision aggregation: In this step average value predictions from trees become the final
result. This average of decisions from trees makes random forest regression stronger than
any other algorithm.
### LSTM:
The LSTM is a class of RNN model, that is able to learn long time dependencies. After fitting the model to a given time series, it can be used to predict one or several time steps. 

For anomaly detection, we need to build a model that is able to detect anomalies in the data, especially in the New York Taxi Traffic dataset we want to be able to detect the "five anomalies [that] occur during the NYC marathon, Thanksgiving, Christmas, New Years day, and a snow storm".

In order to investigate the trend and seasonality of the production of wind power- green energy, based on production variables recorded in the dataset, Auto-Regressive Integerated Moving Averages model will be used. The major steps include time series components detection via decomposition and then removal of the same. Further making the data stationary in order to implement ARIMA model and chose the best model using ACF (Auto-correlation Function) and PACF (Partial Auto-correlation Function) graphs. This model will then allow us to predict the future values. In order to use a Deep Learning method, we want to use an LSTM. The major steps are to choose a proper architecture, learning rate, optimizer and hyperparameters, then fit the model to the data and finally compare it to a baseline, in our case the previous mentioned ARIMA model.

The proposed work consists of the following parts:
• Exploratory data analysis of dataset 
• Building an Autoencoder model for Anomaly detection
• Building an ARIMA model
• Building an LSTM model for predicting future values and evaluating its performance
• Building a Regression Model based on Random Forest

The expected outputs of this projects are:
• An Autoencoder model that is able to detect anomalies in simple time series data
• An ARIMA model that studies the present data and predicts the future values
• An LSTM model, that is able to predict future values up to a satisfying accuracy.
• A regressor Model that is able to predict future values based on an average prediction across decision trees.

### Literature:
1. Jinwon An and Sungzoon Cho. Variational autoencoder based anomaly detection using reconstruction probability. 2015.
2. Raghavendra Chalapathy and Sanjay Chawla. Deep learning for anomaly detection: A survey. CoRR, abs/1901.03407, 2019.
3. Berk Erisen. Wind turbine scada dataset, 2018. Retrieved [25.11.2022] from https://www.kaggle.com/datasets/berkerisen/wind-turbine-scada-dataset.
4. Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning. MIT Press, 2016. Chapter 14, http://www.deeplearningbook.org.
5. julienjta. Nyc taxi traffic, 2022. Retrieved [23.11.2022] from https://www.kaggle.com/datasets/julienjta/nyc-taxi-traffic.
6. Karim Moharm, Mohamed Eltahan, and Ehab Elsaadany. Wind speed forecast using lstm and bi-lstm algorithms over gabal el-zayt wind farm. In 2020 International Conference on Smart Grids and Energy Systems (SGES), pages 922–927, 2020
7. Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. Statistical and machine learning forecasting methods: Concerns and ways forward. PLOS ONE, 13(3):1–26, 03 2018.
8. Statistical and Machine Learning forecasting methods: Concerns and ways forward Spyros Makridakis,Evangelos Spiliotis ,Vassilios Assimakopoulos.


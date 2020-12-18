from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from keras import backend as K

# load data
data = pd.read_csv("normalized_data.csv", error_bad_lines=False)
data = shuffle(data)
print(data['loan_status'].value_counts())

# split data and labels
data_X = data[['term', 'loan_amnt', 'annual_inc', 'installment', 'dti', 'verification_status']].to_numpy()
data_y = data[['loan_status']].to_numpy()
data_y = np.reshape(data_y, (data_y.shape[0],))
print(data_X.shape)
print(data_y.shape)

# split train test data
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=1234)

batch_size = 128


# custom precision metric
def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted
        # raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio

    precision.__name__ = 'precision_{}'.format(threshold)
    return precision


# custom recall metric
def recall_threshold(threshold=0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted
        # raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio

    recall.__name__ = 'recall_{}'.format(threshold)
    return recall


# custom precision metric
def recommendation_rate_threshold(threshold=0.5):
    def recommendation_rate(y_true, y_pred):
        """Recommendation rate metric.
        Computes the the percentage of recommended loans with the threshold.
        """
        threshold_value = threshold
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        predicted_positives = K.sum(y_pred)
        recommendation_ratio = predicted_positives / batch_size
        return recommendation_ratio * 100

    recommendation_rate.__name__ = 'recommendation_threshold_{}'.format(threshold)
    return recommendation_rate


# create model
model = Sequential()
model.add(Dense(1000, input_dim=6, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=["accuracy",
                       tf.keras.metrics.AUC(),
                       precision_threshold(0.5),
                       precision_threshold(0.6),
                       precision_threshold(0.7),
                       precision_threshold(0.8),
                       precision_threshold(0.9),
                       recommendation_rate_threshold(0.5),
                       recommendation_rate_threshold(0.6),
                       recommendation_rate_threshold(0.7),
                       recommendation_rate_threshold(0.8),
                       recommendation_rate_threshold(0.9),
                       recall_threshold(0.5),
                       recall_threshold(0.6),
                       recall_threshold(0.7),
                       recall_threshold(0.8),
                       recall_threshold(0.9),
                       ])

# fit model
model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_data=(X_test, y_test))
# model.save('model_1')

# get metrics
metrics = model.evaluate(X_test, y_test)
count = 0
for name in model.metrics_names:
    print(name, metrics[count])
    count += 1

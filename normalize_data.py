import pandas as pd
import matplotlib.pyplot as plt

processed_data = pd.read_csv("processed_data.csv", error_bad_lines=False)


def drop(data):
    print("Size before dti drop:", data.shape)
    print(data['dti'].max())
    data = data[data['dti'] < 50]
    print("Size after dti drop:", data.shape)

    print("Size before annual income drop:", data.shape)
    print(data['annual_inc'].max())
    data = data[data['annual_inc'] < 200000]
    print("Size after annual income drop:", data.shape)
    return data


def normalizeData(data):
    data[data.columns] = data[data.columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return data


def viewHistograms(data):
    plt.hist(data['annual_inc'], 10)
    plt.show()


def saveData(data):
    data.to_csv(r'normalized_data.csv', index=False)


dropped_data = drop(processed_data)
normalized_data = normalizeData(dropped_data)
viewHistograms(normalized_data)
saveData(normalized_data)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.datasets import ad_hoc_data
import pandas as pd
import seaborn as sns

datasets = {"digit": load_digits(n_class=2),
            "wine": load_wine(),
            "iris": load_iris(),
            "breast_cancer": load_breast_cancer()}


def caricaDataset(nome):
    if nome == "ad hoc":
        train_x, train_y, test_x, test_y = ad_hoc_data(training_size=20, test_size=5, n=2, gap=0.3,
                                                       plot_data=True, one_hot=False,
                                                       include_sample_total=False)
        return 2, train_x, train_y, test_x, test_y
    else:
        data = datasets[nome]
        X, Y = data.data, data.target
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=22)
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA

    n_dim = min(2, train_x.shape[1])
    pca = PCA(n_components=n_dim).fit(train_x)
    sample_train = pca.transform(train_x)
    sample_test = pca.transform(test_x)

    train = pd.DataFrame(sample_train)
    label = pd.DataFrame(train_y)

    train["labels"] = label
    sns.FacetGrid(train, hue="labels").map(plt.scatter, 0, 1).add_legend()
    plt.show()
    n_dim = train_x.shape[1]

    # Normalise
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    # Scale
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    # Select
    train_size = 250
    sample_train = sample_train[:train_size]
    label_train = train_y[:train_size]

    test_size = 50
    sample_test = sample_test[:test_size]
    label_test = test_y[:test_size]

    return n_dim, sample_train, label_train, sample_test, label_test

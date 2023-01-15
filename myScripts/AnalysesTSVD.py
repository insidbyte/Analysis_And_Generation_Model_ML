import seaborn as sns
import matplotlib
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, IncrementalPCA


stop_words = stopwords.words('english')


class Analyses:
    def __init__(self, df):
        np.random.seed(24)
        train_data = df
        train_li = []
        for i in range(len(train_data)):
            if train_data['sentiment'][i] == 'positive':
                train_li.append(1)
            else:
                train_li.append(0)
        train_data['Binary'] = train_li

        train_data = train_data.sample(frac=1)
        print(train_data.head())
        data_vect = train_data['review'].values
        target_vect = train_data['Binary'].values
        train_data_cv, cv = self.tfidf(data_vect)
        self.dimen_reduc_plot(train_data_cv, target_vect)

    def tfidf(self, data):
        tfidfv = TfidfVectorizer(dtype=np.float32, ngram_range=(1, 2), stop_words="english",#max_df=0.75,
                                  sublinear_tf=True, use_idf=True)
        fit_data_tfidf = tfidfv.fit_transform(data)
        print(fit_data_tfidf.shape)
        return fit_data_tfidf, tfidfv

    def dimen_reduc_plot(self, test_data, test_label):
        tsvd = TruncatedSVD(n_components=100, algorithm="randomized",  random_state=24, n_oversamples=10, n_iter=15)

        incremental = IncrementalPCA(n_components=30, whiten=True, batch_size=150)
        print("tsvd fit transform...")
        tsvd_result = tsvd.fit_transform(test_data)
        print("tsvd transform...")
        X = tsvd.transform(test_data)
        print("tsvd finish")
        #with np.printoptions(threshold=np.inf):
            #print(X)
        plt.figure(figsize=(10, 8))
        colors = ['orange', 'red']

        sns.scatterplot(x=tsvd_result[:, 0], y=tsvd_result[:, 1], hue=test_label)

        plt.show()
        plt.figure(figsize=(10, 10))
        plt.scatter(tsvd_result[:, 0], tsvd_result[:, 1], c=test_label,
                    cmap=matplotlib.colors.ListedColormap(colors))
        color_red = mpatches.Patch(color='red', label='Negative Review')
        color_orange = mpatches.Patch(color='orange', label='Positive Review')
        plt.legend(handles=[color_orange, color_red])
        plt.title("TSVD")
        plt.show()


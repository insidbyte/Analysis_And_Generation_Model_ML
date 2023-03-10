import os
from joblib import dump, load
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
import re

class ModelsGenerator:

    def __init__(self, option):
        np.random.seed(24)
        li = os.listdir("../")
        ll = []
        for l in li:
            l = re.findall('(\S+\S+\.joblib)$',l)
            if str(l) != '[]':
                ll.append(l)
        for l in ll:
            for lll in l:
                print(lll)
        vo = input("selezionare vocabolario (.joblib incluso) da utilizzare:\n")
        self.vocabulary = load(f"../{vo}")
        self.rus = RandomUnderSampler(random_state=24)
        nome = self.select_df()
        self.df = pd.read_csv(f"../Dataset_processed/{nome}")
        self.df = self.df.sample(frac=1)
        self.df = self.df.astype('U')
        count_p = self.df[self.df['sentiment'] == 'positive']
        count_n = self.df[self.df['sentiment'] == 'negative']
        if len(count_n) != len(count_p):
            print(f'Dataset sbilanciato : \nPositive: {len(count_p)}\nNegative: {len(count_n)}')
            print('Bilanciamento in corso...')
            self.df, self.df['sentiment'] = self.rus.fit_resample(self.df, self.df['sentiment'])
            count_p = self.df[self.df['sentiment'] == 'positive']
            count_n = self.df[self.df['sentiment'] == 'negative']
            print(f'Dataset bilanciato : \nPositive: {len(count_p)}\nNegative: {len(count_n)} !')
        else:
            print('Dataset bilanciato !')
        if option == 1:
            self.array_df = self.df_to_array()
            self.vectorizer = TfidfVectorizer(ngram_range=(1, 4), min_df=200, max_df=40000,#45000
                                              vocabulary=self.vocabulary, use_idf=True, smooth_idf=True,
                                              sublinear_tf=True, analyzer='word')


            # self.decompositor = IncrementalPCA(n_components=20, whiten=True, batch_size=20)
            # self.decompositor = TruncatedSVD(n_components=1000, algorithm="randomized", n_iter=15, n_oversamples=10,
                                            # random_state=24)
            self.data_trans = self.tfidf()
            # self.tsvd_result = self.reduce()
            self.y = self.df['sentiment']
            self.x = self.df['review']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data_trans, self.y,
                                                                                    test_size=0.2,
                                                                                    random_state=24)
            """
            self.model = svm.LinearSVC(C=0.2, class_weight='balanced', dual=True, fit_intercept=True,
                                       intercept_scaling=0.1, loss='squared_hinge', max_iter=75000, penalty='l2',
                                       tol=0.001)
            self.model = CalibratedClassifierCV(self.model)
            """
            self.model = LogisticRegression(penalty='l2', dual=True, fit_intercept=True, C=13.9, max_iter=7500,
                                            class_weight="balanced", solver='liblinear',
                                            intercept_scaling=8, tol=0.00001)

        elif option == 2:
            self.array_df = self.df_to_array()
            self.y = self.df['sentiment']

    def select_df(self):
        list = os.listdir("../Dataset_processed")
        print("Inserisci nome del dataset che vuoi utilizzare per la generazione del modello ML (.csv incluso)\n")
        for l in list:
            print(l)
        name = str(input())
        return  name

    def df_to_array(self):
        array = []
        a = 0
        for r in self.df.iloc:
            text = self.df.iloc[a]['review']
            array.append(text)
            a = a + 1
        return array

    def tfidf(self):
        print(f'[0]-Vettorizzazione con tfidf in corso, fit...')
        self.vectorizer.fit(self.array_df)
        print(f'[1]-Vettorizzazione con tfidf in corso, fit_transform...')
        data_trans = self.vectorizer.fit_transform(self.array_df)
        print(f'[2]-Vettorizzazione con tfidf in corso, transform...')
        self.vectorizer.transform(self.array_df)
        print(f'[3]-Vettorizzazione con tfidf completata !')
        ve = input("Si vuole salvare il vettorizzatore in venvServer? \nY/N ?\n").lower()
        if ve == 'y':
            nome = input("Inserire nome del file joblib da salvare in venvServer (.joblib escluso)\n")
            print(f'[4]-Caricamento file joblib in venvServer in corso...')
            dump(value=self.vectorizer,
                 filename=f"../../serverAdmin/Model_ML_New/{nome}.joblib")
            print(f'[5]-Caricamento file joblib completato !')
        return data_trans

    def reduce(self):
        print(f'[6]-Riduzione con Truncated SVD in corso, fit...')
        self.decompositor.fit(self.data_trans)
        print(f'[7]-Riduzione con Truncated SVD in corso, fit completato !')
        print(f'[8]-Riduzione con Truncated SVD in corso, fit_transform...')
        tsvd_result = self.decompositor.fit_transform(self.data_trans)
        print(f'[9]-Riduzione con Truncated SVD in corso, transform...')
        self.decompositor.transform(self.data_trans)
        print(f'[10]-Riduzione con Truncated completata')
        ve = input("Si vuole salvare il vettorizzatore in venvServer? \nY/N ?\n").lower()
        if ve == 'y':
            nome = input("Inserire nome del file joblib da salvare in venvServer (.joblib escluso)\n")
            print(f'[11]-Caricamento file joblib in venvServer in corso...')
            dump(value=self.decompositor,
                 filename=f"../../serverAdmin/Model_ML_New/{nome}.joblib")
            print(f'[12]-Caricamento file joblib in venvServer completato...')
        return tsvd_result

    def generate(self):

        print('[13]-Modello in addestramento...')
        fit = self.model.fit(self.X_train, self.y_train)
        print('[14]-Modello addestrato\nRisultati:')
        p_train = self.model.predict(self.X_train)
        p_test = self.model.predict(self.X_test)

        acc_train = accuracy_score(self.y_train, p_train)
        scc_test = accuracy_score(self.y_test, p_test)

        cm_lr = confusion_matrix(self.y_test, p_test)

        tn, fp, fn, tp = confusion_matrix(self.y_test, p_test).ravel()

        tnt, fpt, fnt, tpt = confusion_matrix(self.y_train, p_train).ravel()

        print(f'\nValori test: \nTN: {tn}\nFP: {fp}\nFN: {fn}\nTP: {tp}')

        print(f'\nValori train: \nTN: {tnt}\nFP: {fpt}\nFN: {fnt}\nTP: {tpt}')

        tpr_lr = round(tp / (tp + fn), 4)
        tnr_lr = round(tn / (tn + fp), 4)
        print(f'\nValori test:\nTRUE POSITIVE RATES: {tpr_lr}\nTRUE NEGATIVE RATES: {tnr_lr}')

        tpr_lrt = round(tpt / (tpt + fnt), 4)
        tnr_lrt = round(tnt / (tnt + fpt), 4)
        print(f'\nValori train: \nTRUE POSITIVE RATES: {tpr_lrt}\nTRUE NEGATIVE RATES: {tnr_lrt}')

        accuracy_train = accuracy_score(self.y_train, p_train)
        accuracy_test = accuracy_score(self.y_test, p_test)
        print(f'\nTRAIN ACCURACY SCORE: {accuracy_train}\nTEST ACCURACY SCORE {accuracy_test}')

        f1score_train = f1_score(self.y_train, p_train, pos_label='positive')
        f1score_test = f1_score(self.y_test, p_test, pos_label='positive')
        print(f'\nTRAIN F1 SCORE: {f1score_train}\nTEST F1 SCORE {f1score_test}')

        precisionscore_train = precision_score(self.y_train, p_train, pos_label='positive')
        precisionscore_test = precision_score(self.y_test, p_test, pos_label='positive')
        print(f'\nTRAIN PRECISION SCORE: {precisionscore_train}\nTEST PRECISION SCORE:  {precisionscore_test}')

        recallscore_train = recall_score(self.y_train, p_train, pos_label='positive')
        recallscore_test = recall_score(self.y_test, p_test, pos_label='positive')
        print(f'\nTRAIN RECALL SCORE:  {recallscore_train}\nTEST RECALL SCORE:  {recallscore_test}')

        cl = input("Si vuole salvare il classificatore in venvServer? \nY/N?\n").lower()
        if cl == 'y':
            nome = input("Inserire nome del file joblib da creare in venvServer\n")
            print(f'[15]-Caricamento file joblib in venvServer in corso...')
            dump(value=self.model,
                 filename=f"../../serverAdmin/Model_ML_New/{nome}.joblib")
            print(f'[16]-Caricamento file joblib in venvServer completato...')

    def tuning(self):
        c = 0
        print(f'[{c}]-Settaggio pipeline in corso...')
        pipeline = Pipeline([
            ('vect', TfidfVectorizer(vocabulary=self.vocabulary)),#TfidfVectorizer(stop_words='english', vocabulary=self.vocabulary)),
            # ('rd', IncrementalPCA()),
            #('clf', svm.SVC(kernel='linear'))
            #('clf', svm.LinearSVC()),
            ('clf', LogisticRegression(solver='liblinear'))
        ])
        c = c + 1
        print(f'[{c}]-Settaggio pipeline completato')
        c = c + 1
        print(f'[{c}]-Settaggio parametri per GridSearchCV in corso...')
        """
        best for logistic
            'clf__penalty': ['l2'],
            'clf__dual': [True],
            'clf__fit_intercept': [True],
            'clf__class_weight': ['balanced'],
            'clf__C': [13.9],#np.arange(start=13.0, stop=15.1, step=0.1, dtype=float),
            
            best_score: 0.8959227421876367
            best_params: {'clf__C': 13.9, 'clf__class_weight': 'balanced', 'clf__dual': True, 
            'clf__fit_intercept': True, 'clf__penalty': 'l2', 'vect__analyzer': 'word', 'vect__max_df': 45000, 
            'vect__min_df': 290, 'vect__ngram_range': (1, 5), 'vect__smooth_idf': True, 'vect__sublinear_tf': True, 
            'vect__use_idf': True, 'clf__warm_start': True}
        896712
        8970
        """
        parameters = {
            # 'rd__n_components': [10, 20],
            # 'rd__whiten': [True],
            # 'rd__batch_size': [20, 30],
            # 'rd__algorithm': ['randomized'],
            # 'rd__random_state': [24],
            # 'rd__power_iteration_normalizer': ['auto', 'LU', 'none'],
            # 'rd__n_oversamples': [10],
            # 'rd__n_iter': [15],
            'clf__max_iter': [7500],
            'clf__penalty': ['l2'],
            #'clf__loss': ['squared_hinge'],
            'clf__dual': [True],
            'clf__fit_intercept': [True],
            'clf__intercept_scaling': [8],
            'clf__class_weight': ['balanced'],
            'clf__tol':  [0.00001],#np.arange(start=0.00001,stop=0.00011,step=0.00001,dtype=float),#[0.0001],
            'clf__C': [13.9],#np.arange(start=0.1, stop=1.1, step=0.1, dtype=float),
            'clf__warm_start': [True],
            #'clf__l1_ratio': [1],
            #'clf__verbose': [1],
            #'clf__random_state':[0, 24, 32, 42],
            'vect__min_df': np.arange(start=1,stop=100,step=10,dtype=int),
            'vect__max_df': [40000],#np.arange(start=40000, stop=46000, step=1000, dtype=int),
            'vect__analyzer': ['word'],
            'vect__use_idf': [True],
            #'vect__token_pattern': [r"(?u)\b\w\w+\b"],
            #'vect__stop_words': [None],
            'vect__smooth_idf': [True],
            'vect__sublinear_tf': [True],
            #'vect__binary': [False],
            'vect__ngram_range': [(1, 4)],
            #'vect__norm': ['l2']
        }

        c = c + 1
        print(f'[{c}]-Settaggio parametri per GridSearchCV completato')
        grid = GridSearchCV(pipeline, parameters, verbose=3, cv=3, n_jobs=-1)
        c = c + 1
        print(f'[{c}]-GridSearchCV lanciato...')
        grid.fit(self.array_df, self.y)
        c = c + 1
        print(f'[{c}]-GridSearchCV terminato')
        return grid.best_score_, grid.best_params_



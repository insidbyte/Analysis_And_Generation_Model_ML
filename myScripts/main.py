from ModelGenerator import ModelsGenerator
from AnalysesTSVD import Analyses
import time
import pandas as pd
import sys

if __name__ == '__main__':

    print('Selezionare un opzione:\n1)-Generare modello\n2)-Fare tuning iperparamentri con GridSearchCV\n'
          '3)-Vedere il plot con tfidf e Truncated SVD')
    option = input()
    if option != '1' and option != '2' and option != '3':
        sys.exit("Inserire un opzione corretta SYSTEM EXIT !")
    else:
        if option == '1' or option == '2':
            option = int(option)
            gen = ModelsGenerator(option=option)
        if option == '3':
            df = pd.read_csv("../Dataset_processed/proc_correct_lemma_nostop_100_union.csv")
            analyses = Analyses(df.astype('U'))
            sys.exit(100)
    if option == 1:
        time_i = time.time()
        gen.generate()
        time_f = time.time()
        print(f'Tempo di esecuzione: {time_f - time_i} sec')
    if option == 2:
        time_i = time.time()
        score, par = gen.tuning()
        print(f'best_score: {score}\nbest_params: {par}')
        time_f = time.time()
        print(f'Tempo di esecuzione: {time_f - time_i} sec')
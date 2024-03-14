import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC
import sklearn.metrics as met
from AISearchEngine.Libs.Dataframe_results_files import Df_Result_lib as Df



def main():
    file = r'../Datasets/Sale_Report.csv'
    # read in the data using pandas
    df = pd.read_csv(file)
    # replace na values by 0 and convert to strin number values
    df = Df.prepare_df_data(df)

    # X data is everything without Category
    X = df.drop(columns=['Category'])
    # separate target values
    y = df['Category'].values
    # # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
    # vectorize column
    column_X = 'Design No.'
    count_vect = HashingVectorizer()
    # use knn neighbours to predict the data
    knn = KNeighborsClassifier(n_neighbors=5)
    # create a loop through all columns in x
    # for idx, col_train in enumerate(X_train.columns):
    X_train_vec = count_vect.fit_transform(X_train[column_X])
    # X_train_vec = lsa.fit_transform(X_train_vec)
    X_train_vec = Normalizer(copy=False).fit_transform(X_train_vec)
    # # get tfidf method to convert vectors
    knn.fit(X_train_vec, y_train)

    # get to predict categories by nearest neighbours
    test_doc = ['AN', 'NA', '201', 'AN20X']
    new_test_vec = count_vect.transform(test_doc)
    predicted = knn.predict(new_test_vec)
    # classification report
    # print('Classification report for k-neighbours\n')
    # print(met.classification_report(test_doc, predicted))
    # save results fo file
    file_path = r'../Search_results/Knn_neighboor.txt'
    Df.write_result_2_file(file_path, test_doc, predicted, df)
    # the same for SGDClassifier
    sdg = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,
                        max_iter=5, tol=None)

    sdg.fit(X_train_vec, y_train)
    # get to predict categories by nearest neighbours
    # new_test_tfidf = tfidf_transformer.transform(new_test_vec)
    predicted = sdg.predict(new_test_vec)
    # print('Classification report for SGDC\n')
    # print(met.classification_report(test_doc, predicted))
    # save results fo file
    file_path = r'../Search_results/SGDC.txt'
    Df.write_result_2_file(file_path, test_doc, predicted, df)
    # the same for naive bayes categorical nb
    rf = LinearSVC()
    rf.fit(X_train_vec, y_train)
    predicted = rf.predict(new_test_vec)
    file_path = r'../Search_results/LinearSVC.txt'
    Df.write_result_2_file(file_path, test_doc, predicted, df)
    


main()

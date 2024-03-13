import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

file = r'../Datasets/Sale_Report.csv'
# read in the data using pandas
df = pd.read_csv(file)
# replace na values by 0
df.fillna(0)
rows_num, col_num = df.shape
# convert number values to string
for col in df.columns:
    for i in range(rows_num):
        # fill not printable values ND value
        if not str(df.at[i, col]).isprintable():
            df.at[i, col] = "ND"
        # if type of value is not string convert it to string
        elif not isinstance(df.at[i, col], str):
            df.at[i, col] = str(df.at[i, col])

# X data is everything without SKU
# create a dataframe with all training data except the target column
X = df.drop(columns=['SKU Code'])

# # separate target values
y = df['SKU Code'].values
# # split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# # transform data by tfidf method
column_X = 'Color'
count_vect = CountVectorizer()
# vect_y = CountVectorizer(analyzer='word', tokenizer=my_tokenizer, min_df=1)
X_train_vec = count_vect.fit_transform(X_train[column_X])
X_test_vec = count_vect.transform(X_test[column_X])
print(X_train_vec.shape, X_test_vec.shape)
# # get tfidf method to convert vectors
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_vec)
X_test_tfidf = tfidf_transformer.transform(X_test_vec)
# print(X_train_tfidf.shape)
# print(Y_train_tfidf.shape)
# # Create KNN classifier
# use knn neighbours to predict the data
knn = KNeighborsClassifier(n_neighbors=45)
knn.fit(X_train_tfidf, y_train)
# get to predict categories by naive bayes algoritm

docs_new = X_test[column_X]
new_test = [' ', 'creaM', 'WINE', 'Multi']
new_test_vec = count_vect.transform(new_test)
new_test_tfidf = tfidf_transformer.transform(new_test_vec)
predicted = knn.predict(X_test_tfidf)

i = 0
for category, sku in zip(new_test, predicted):
    print('predicted by nearest neighbour %r => %s' % (category, sku))
    print(' real from dataframe')
    print(df[df['SKU Code'] == sku])
    i += 1
    if i > 10:
        break

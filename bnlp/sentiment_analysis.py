import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score

class SentimentAnalysis:
  def __init__(self):
    self.svm = SVC(kernel="linear", degree=5, probability=True)
    self.logistic_regression = LogisticRegression()
    self.random_forest = RandomForestClassifier()
    self.knn = KNeighborsClassifier(n_neighbors=12)
    self.adaboost = AdaBoostClassifier(random_state=0, n_estimators=200)
    self.tfidf_vectorizer = CountVectorizer()
  
  def sentiment(self, model, vectors, sentence):
    sentences = [sentence]
    with open(vectors, 'rb') as f:
      train_vectors = pickle.load(f)
    sentence_vectors = train_vectors.transform(sentences)
    with open(model, 'rb') as f:
      model = pickle.load(f)
    predict = model.predict(sentence_vectors)
    if predict[0] == 1:
      return "Positive Sentiment"
    return "Negative Sentiment"

  def save_model(self, model):
    with open('sentiment_model.pkl', 'wb') as f:
      pickle.dump(model, f)
    print("Best model saved as 'sentiment_model.pkl'")
  
  def train(self, df):
    # df = pd.DataFrame(data, columns=['text', 'label'])
    df = df.sample(frac=1)
    train, test = train_test_split(df, test_size=0.2)
    x_train, y_train = train['text'], train['label']
    x_test, y_test = test['text'], test['label']
    #Fit and transform the training data 
    x_train = self.tfidf_vectorizer.fit_transform(x_train)
    with open('vectors.pkl', 'wb') as f:
      pickle.dump(self.tfidf_vectorizer, f)
    print("Train Vector saved as 'vectors.pkl' for prediction")

    #Transform the test set 
    x_test = self.tfidf_vectorizer.transform(x_test)

    accuracy_dict = {}
    categories = [0, 1]
    # train svm and get accuracy score
    self.svm.fit(x_train, y_train)
    y_pred = self.svm.predict(x_test)
    svm_accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict['svm'] = svm_accuracy
    print(f'svm accuracy: {svm_accuracy}')

    # train knn and get accuracy score
    self.knn.fit(x_train, y_train)
    y_pred = self.knn.predict(x_test)
    knn_accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict['knn'] = knn_accuracy
    print(f'knn accuracy: {knn_accuracy}')

    # train logistic regression and get accuracy score
    self.logistic_regression.fit(x_train, y_train)
    y_pred = self.logistic_regression.predict(x_test)
    lr_accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict['lr'] = lr_accuracy
    print(f'logistic regression accuracy: {lr_accuracy}')

    # train random forest and get accuracy score
    self.random_forest.fit(x_train, y_train)
    y_pred = self.random_forest.predict(x_test)
    rf_accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict['rf'] = rf_accuracy
    print(f'random forest accuracy: {rf_accuracy}')

    self.adaboost.fit(x_train, y_train)
    y_pred = self.adaboost.predict(x_test)
    adaboost_accuracy = accuracy_score(y_test, y_pred)
    accuracy_dict['adaboost'] = adaboost_accuracy
    print(f'adaboost accuracy: {adaboost_accuracy}\n')
    print("="*50)
    # saving best model
    accuracy_dict = {k: v for k, v in sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True)}
    accuracy_dict = accuracy_dict.items()
    accuracy_dict = list(accuracy_dict)[0]
    best_model, best_accuracy = accuracy_dict
    if best_model == 'svm':
      print(f'best model: SVM, best accuracy: {best_accuracy}')
      self.save_model(self.svm)
    if best_model == 'lr':
      print(f'best model: Logistic Regression, best accuracy: {best_accuracy}')
      self.save_model(self.logistic_regression)
    if best_model == 'knn':
      print(f'best model: KNN, best accuracy: {best_accuracy}')
      self.save_model(self.knn)

    if best_model == 'rf':
      print(f'best model: Random Forest, best accuracy: {best_accuracy}')
      self.save_model(self.random_forest)
    if best_model == 'adaboost':
      print(f'best model: Adaboost, best accuracy: {best_accuracy}')
      self.save_model(self.adaboost)
    print("="*50)


# if __name__=="__main__":
#     sa = SentimentAnalysis()
#     # x = sa.sentiment('sentiment_model.pkl', 'vectors.pkl', "আমার খুব প্রিয় মডেল আমি খুব ভালো বাসি মিম আপু")
#     # x
#     sa.train(df)

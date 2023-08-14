import pickle
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from bnlp.utils.utils import transform_to_dataset, features

class CRFTaggerTrainer:
    def train(self, model_name, train_data, test_data, average="micro"):

        X_train, y_train = transform_to_dataset(train_data)
        X_test, y_test = transform_to_dataset(test_data)
        print(len(X_train))
        print(len(X_test))

        print("Training Started........")
        print("It will take time according to your dataset size...")
        model = CRF()
        model.fit(X_train, y_train)
        print("Training Finished!")

        print("Evaluating with Test Data...")
        y_pred = model.predict(X_test)
        print("Accuracy is: ")
        print(metrics.flat_accuracy_score(y_test, y_pred))
        print(f"F1 Score({average}) is: ")
        print(metrics.flat_f1_score(y_test, y_pred, average=average))

        pickle.dump(model, open(model_name, "wb"))
        print("Model Saved!")

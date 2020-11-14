import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
import joblib


if __name__ == "__main__":
    df = pd.read_csv("/home/praveen/Desktop/Projects/technocolab_project_2/nlp_chatbot/input/data.csv")
    
    df.type = df.type.apply(lambda x: 1 if x=="greeting" else (2 if x=="question" else 3))

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.type.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_,v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    for fold_ in range(5):
        train_df = df[df.kfold!=fold_].reset_index(drop=True)
        test_df = df[df.kfold==fold_].reset_index(drop=True)

        count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

        count_vec.fit(train_df["query"])

        x_train = count_vec.transform(train_df["query"])
        x_test = count_vec.transform(test_df["query"])

        model = linear_model.LogisticRegression()

        model.fit(x_train, train_df.type)

        preds = model.predict(x_test)

        
        f1score = metrics.f1_score(test_df.type, preds, average='micro')
        precision = metrics.precision_score(test_df.type, preds, average='micro')

        model_path = "/home/praveen/Desktop/Projects/technocolab_project_2/nlp_chatbot/models/log_reg_count_vec_"+str(fold_)+".pkl"
        vectorizer_path = "/home/praveen/Desktop/Projects/technocolab_project_2/nlp_chatbot/models/vectorizer_"+str(fold_)+".pkl"
        joblib.dump(model, model_path)
        joblib.dump(count_vec, vectorizer_path)
       

        
        print(f"Fold : {fold_}")
        print(f"F1 score : {f1score}")
        print(f"Precision : {precision}")
        print("")

    
        



import joblib

count_vec = joblib.load("/home/praveen/Desktop/Projects/technocolab_project_2/nlp_chatbot/models/vectorizer_3.pkl")
model = joblib.load("/home/praveen/Desktop/Projects/technocolab_project_2/nlp_chatbot/models/log_reg_count_vec_3.pkl")


sentence = input()
sentence = count_vec.transform([sentence])
sentence = sentence.toarray()
    
prediction = model.predict(sentence)
if prediction == 1:
    print("Sentence is of type greeting")
elif prediction == 2:
    print("Sentence is a question")
else:
    print("It is a complimentary/closing sentence")









        
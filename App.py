import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet 
import mlflow
from mlflow_utils import log_metrics
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger') 

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Ensure MLflow points to the correct URI
mlflow.set_experiment("Resume_Ranking_Experiment")

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = " "
    for page in pdf.pages:
        text += page.extract_text()
    return text    
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()  
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()  # Get first letter of POS tag
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
schar= string.punctuation
# Preprocessing function
def preprocess_txt(text):
    words = word_tokenize(text)
    filtered_words = [
        lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word))
        for word in words
        if word.lower() not in stop_words and word not in schar
    ]
    return ' '.join(filtered_words) 

     
# def rank_resumes(job_description,resumes):
#     documents = [job_description] + resumes
#     vectorizer =TfidfVectorizer().fit_transform(documents)
#     vectors = vectorizer.toarray()

#     job_description_vector = vectors[0]
#     resume_vectors = vectors[1:]
#     cosine_similarities = cosine_similarity([job_description],resume_vectors).flatten()
#     return cosine_similarities
def rank_resumes(job_description, resumes):
    job_description = preprocess_txt(job_description)

    # Compute TF-IDF vectors for the job description only once
    vectorizer = TfidfVectorizer()
    job_vector = vectorizer.fit_transform([job_description])

    # Calculate similarity of each resume with the job description
    similarity_scores = []
    for resume in resumes:
        resume_vector = vectorizer.transform([preprocess_txt(resume)])
        similarity = cosine_similarity(job_vector, resume_vector).flatten()[0]
        similarity_scores.append(similarity)

    # Sort resumes by similarity scores in descending order
    ranked_resumes = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    return ranked_resumes


st.title("Resume Ranking System")
st.header("Job Description")
job_description=st.text_area("Enter the job  description")
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type = ['pdf'],accept_multiple_files =True)


if uploaded_files and job_description:
    st.header("Ranking Resumes")
    resumes =[]
    for file in uploaded_files:
        text =extract_text_from_pdf(file)
        resumes.append(text)
    # resume = preprocess_txt(resumes)
    # job_description=preprocess_txt(job_description)
    with mlflow.start_run():
        # Log the job description
        mlflow.log_param("job_description", job_description)
        # Rank resumes and log the results
        scores =rank_resumes(job_description,resumes)
        results = pd.DataFrame({
        "Rank": [i + 1 for i in range(len(scores))] ,    
        "Resume": [file.name for file in uploaded_files], 
        "Score": [score for _, score in scores]
        })
   

        results =results.sort_values(by = "Score",ascending=False)
        st.write(results) 
        # Log parameters, metrics, and artifacts with MLflow
        avg_score = results["Score"].mean()
        mlflow.log_param("job_description", job_description)
        mlflow.log_metric("avg_similarity_score", avg_score)

        # Save the results to a CSV and log it as an artifact
        results_csv = "resume_ranking_results.csv"
        results.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)

        st.success("MLflow tracking completed!")
  
import streamlit as st
import pdfplumber
# import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
# ---------------------------------------
# Functions
# ---------------------------------------
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def calculate_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


def extract_keywords(text, top_n=15):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.toarray()[0]
    keyword_scores = dict(zip(feature_names, scores))

    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:top_n]]

def find_missing_keywords(resume_text, job_text):
    job_keywords = set(extract_keywords(job_text, 20))
    resume_keywords = set(extract_keywords(resume_text, 30))

    missing = job_keywords - resume_keywords
    return list(missing)

# ----------------------------------------
# User Interface
# ----------------------------------------
st.title("AI Resume Analyzer")
st.write("Upload your resume and paste a job description to get a match score!")

resume_file = st.file_uploader("Upload yor Resume (PDF only)", type=["pdf"])

job_description = st.text_area(
    "Paste the Job Description",
    placeholder="Enter the job description here..."
)

if st.button("Analyze"):
    if resume_file and job_description.strip():

        # Extract resume text
        resume_text = extract_text_from_pdf(resume_file)

        # Calculate similarity score
        score = calculate_match_score(resume_text, job_description)

        st.success(f"Your Job Match Score is: **{score}%**")

        missing_keywords = find_missing_keywords(resume_text, job_description)
        
        st.subheader("Suggestions to Improve:")
        
        if missing_keywords:
            st.write("Consider adding or emphasizing the following keywords from the job description:")
            for keyword in missing_keywords:
                st.write(f"- {keyword}")
                
        else:
            st.success("Great job! Your resume already matches most of the job requirements.")

    else:
        st.error("Please upload a resume and enter a job description.")


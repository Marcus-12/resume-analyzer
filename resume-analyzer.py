import streamlit as st
import pdfplumber
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
# ---------------------------------------
# Functions
# ---------------------------------------
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def calculate_match_score(resume_text, job_description):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)


# ----------------------------------------
# UI
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

        st.subheader("Suggestion to Improve:")
        st.write("- Add more keywords from the job description.")
        st.write("- Highlight relevant skills and experience.")
        st.write("- Tailor your resume for each job.")

        st.subheader("Extracted Resume Text Preview:")
        st.text_area("", resume_text, height=300)

    else:
        st.error("Please upload a resume and enter a job description.")
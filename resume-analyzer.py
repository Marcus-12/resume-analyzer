import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer # A powerful tool for converting a collection of raw documents into a matrix.
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------
# Streamlit page config
# ---------------------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# ---------------------------------------
# Functions
# ---------------------------------------

def extract_text_from_pdf(pdf_file):
    """Extract text safely from PDF"""
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def calculate_match_score(resume_text, job_description):
    """Compute similarity score using TF-IDF + cosine similarity"""
    vectorizer = TfidfVectorizer() 
    vectors = vectorizer.fit_transform([resume_text, job_description])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(score * 100, 2)

def extract_keywords(text, top_n=25):
    """Extract top N keywords from text"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keyword_scores = dict(zip(feature_names, scores))
    sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [word for word, score in sorted_keywords[:top_n]]

def find_missing_keywords(resume_text, job_text):
    """Return keywords in job description missing from resume"""
    job_keywords = set(extract_keywords(job_text, 30))
    resume_keywords = set(extract_keywords(resume_text, 50))
    missing = job_keywords - resume_keywords
    return list(missing)

# --- Skill grouping ---
SKILL_CATEGORIES = {
    "Technical / Hard Skills": [
        "java","python","sql","javascript","html","css","api","rest",
        "spring","backend","frontend","oop","data structures","algorithms",
        "machine learning","nlp","excel","financial modeling"
    ],
    "Tools & Technologies": [
        "git","github","docker","aws","azure","figma","tableau","power bi",
        "jira","intellij","vscode","mysql","postgresql"
    ],
    "Soft Skills": [
        "communication","teamwork","leadership","problem solving",
        "adaptability","time management","critical thinking","creativity"
    ],
    "Domain Knowledge": [
        "marketing","finance","healthcare","data analysis","cloud","devops"
    ]
}

def classify_skill(skill):
    skill = skill.lower()
    for category, skills in SKILL_CATEGORIES.items():
        if skill in skills:
            return category
    # fallback
    if len(skill.split()) > 1:
        return "Domain Knowledge"
    return "Technical / Hard Skills"

def group_skills(missing_keywords):
    grouped = {category: [] for category in SKILL_CATEGORIES}
    for skill in missing_keywords:
        category = classify_skill(skill)
        grouped[category].append(skill)
    return grouped

def generate_sample_bullets(grouped_skills):
    bullets = []
    for category, skills in grouped_skills.items():
        for skill in skills:
            bullets.append(f"- Demonstrated experience with **{skill}** in relevant projects or work experience.")
    return bullets

# --- Job role detection ---
ROLE_KEYWORDS = {
    "Software Developer": ["java","python","javascript","api","spring","backend","frontend"],
    "Data Analyst": ["excel","sql","power bi","tableau","data analysis","statistics"],
    "Designer": ["figma","ux","ui","prototyping","design","adobe"],
    "Marketing": ["seo","campaign","marketing","analytics","content","branding"],
    "Finance": ["financial","forecast","budget","audit","risk","finance"]
}

def detect_job_role(job_description):
    job_description = job_description.lower()
    scores = {}
    for role, keywords in ROLE_KEYWORDS.items():
        scores[role] = sum(1 for kw in keywords if kw in job_description)
    best_role = max(scores, key=scores.get)
    if scores[best_role] == 0:
        return "General / Unknown Role"
    return best_role

# --- Skill coverage ---
def calculate_skill_coverage(resume_text, job_text, grouped_skills):
    coverage = {}
    resume_keywords = set(extract_keywords(resume_text, 50))
    job_keywords = set(extract_keywords(job_text, 30))
    for category, skills in grouped_skills.items():
        total_skills = set([s for s in job_keywords if classify_skill(s) == category])
        if len(total_skills) == 0:
            coverage[category] = 100
        else:
            matched = total_skills & resume_keywords
            coverage[category] = round(len(matched)/len(total_skills) * 100, 2)
    return coverage

# ----------------------------------------
# Streamlit UI
# ----------------------------------------
st.title("AI Resume Analyzer")
st.write("Upload your resume and paste a job description to get a match score, auto-detected role, skill coverage, and actionable suggestions!")

resume_file = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
job_description = st.text_area("Paste the Job Description", placeholder="Enter the job description here...")

if st.button("Analyze"):
    if resume_file and job_description.strip():

        # Extract resume
        resume_text = extract_text_from_pdf(resume_file)

        # Match score
        score = calculate_match_score(resume_text, job_description)
        st.success(f"**Job Match Score:** {score}%")

        # Detect role
        detected_role = detect_job_role(job_description)
        st.info(f"**Detected Role:** {detected_role}")

        # Missing keywords & grouping
        missing_keywords = find_missing_keywords(resume_text, job_description)
        grouped = group_skills(missing_keywords)
        bullets = generate_sample_bullets(grouped)

        # Skill coverage
        coverage = calculate_skill_coverage(resume_text, job_description, grouped)
        st.subheader("Skill Coverage by Category")
        for category, percent in coverage.items():
            st.write(f"{category}: {percent}%")
            st.progress(percent / 100)

        # Suggestions
        st.subheader(" Suggestions to Improve Your Resume")
        if bullets:
            for bullet in bullets:
                st.write(bullet)
        else:
            st.success("Excellent! Your resume already aligns well with this role ")

        # Resume preview
        st.subheader(" Extracted Resume Text Preview")
        st.text_area("", resume_text, height=300)

    else:
        st.error("Please upload a resume and enter a job description.")

# 🌟 AI Resume Analyzer

An AI-powered web application that analyzes a resume against a job description, calculates a match score, detects the job role, identifies missing skills, and provides actionable suggestions to improve the resume.

This project simulates how modern **Applicant Tracking Systems (ATS)** and recruiters evaluate resumes.

---

## 🚀 Features

### ✅ Resume–Job Match Score
- Uses **TF-IDF** and **Cosine Similarity**
- Outputs a percentage score showing how well the resume matches the job description

### 🎯 Auto-Detect Job Role
- Automatically identifies the most likely job role (e.g. Software Developer, Data Analyst, Designer, Marketing, Finance)
- Based on keyword analysis of the job description

### 📊 Skill Coverage by Category
Displays skill coverage percentages with progress bars for:
- Technical / Hard Skills
- Tools & Technologies
- Soft Skills
- Domain Knowledge

### 📝 Actionable Resume Suggestions
- Identifies **missing or weak skills**
- Groups them into meaningful categories
- Generates **sample resume bullet points** to guide improvements

### 📄 Resume Text Preview
- Extracts and displays text from uploaded PDF resumes
- Helps users verify parsing accuracy

---

## 🧠 How It Works

1. The user uploads a **PDF resume**
2. The user pastes a **job description**
3. The system:
   - Extracts text from the resume
   - Computes similarity score
   - Extracts keywords using TF-IDF
   - Detects missing skills
   - Groups skills dynamically
   - Calculates skill coverage percentages
4. The user receives:
   - Match score
   - Detected job role
   - Skill coverage visualization
   - Personalized improvement suggestions

---

## 🛠 Technologies Used

- **Python**
- **Streamlit** – UI framework
- **scikit-learn** – TF-IDF & cosine similarity
- **pdfplumber** – PDF text extraction
- **Natural Language Processing (NLP)** techniques

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/ai-resume-analyzer.git
cd ai-resume-analyzer

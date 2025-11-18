from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import io
import re
from collections import Counter
import google.generativeai as genai
import os


# -----------------------------------------
# GEMINI CONFIG (STEP 4)
# -----------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # <-- your key
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL =  "gemini-1.5-flash-latest"


# -----------------------------------------
# GLOBALS & KEYWORD LISTS
# -----------------------------------------

STOPWORDS = set([
    "the", "and", "with", "your", "for", "are", "was", "were", "you",
    "but", "this", "that", "from", "have", "has", "had", "his", "her",
    "its", "their", "our", "they", "them", "him", "she", "he", "who",
    "why", "what", "when", "how", "which"
])

HARD_SKILLS = [
    "python", "java", "c++", "javascript", "react", "node", "sql", "mysql",
    "postgresql", "mongodb", "aws", "docker", "kubernetes", "tensorflow",
    "pytorch", "nlp", "machine learning", "deep learning", "excel",
    "power bi", "tableau", "html", "css", "django", "flask"
]

SOFT_SKILLS = [
    "teamwork", "communication", "leadership", "problem solving",
    "critical thinking", "adaptability", "creativity", "time management"
]

JOB_TITLES = [
    "software engineer", "developer", "data analyst",
    "data scientist", "web developer", "backend developer",
    "frontend developer", "machine learning engineer"
]


# -----------------------------------------
# WRITING ANALYSIS LISTS (STEP 3)
# -----------------------------------------
ACTION_VERBS = [
    "built", "created", "developed", "managed", "led", "optimized",
    "designed", "launched", "improved", "implemented", "reduced",
    "increased", "analyzed", "solved", "configured", "directed",
    "automated", "maintained", "trained", "organized"
]

WEAK_PHRASES = [
    "responsible for", "helped with", "worked on", "assisted with",
    "participated in", "involved in"
]

PASSIVE_VERBS = ["was", "were", "is", "are", "been", "be", "being"]


# -----------------------------------------
# FASTAPI INIT
# -----------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------
# FILE EXTRACTORS
# -----------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    document = docx.Document(io.BytesIO(file_bytes))
    lines = [para.text for para in document.paragraphs]
    return "\n".join(lines).strip()


# -----------------------------------------
# FORMATTING ANALYSIS ENGINE  (STEP 2)
# -----------------------------------------
def analyze_formatting(text, file_bytes, filename):
    formatting_score = 20
    issues = []

    if len(text) < 200:
        issues.append("Resume text seems too short. ATS may not read it well.")
        formatting_score -= 2

    if any(line.isupper() and len(line) > 5 for line in text.split("\n")):
        issues.append("Too many ALL CAPS lines detected.")
        formatting_score -= 1

    if re.search(r"[^\x00-\x7F]", text):
        issues.append("Non-ASCII characters detected. ATS may not parse them.")
        formatting_score -= 2

    if "\t" in text:
        issues.append("Tabs detected — may indicate multi-column layout.")
        formatting_score -= 3

    if re.search(r"\s{3,}", text):
        issues.append("Multiple consecutive spaces detected — avoid manual alignment.")
        formatting_score -= 2

    pages = len(text.split("\f"))
    if pages > 2:
        issues.append(f"Resume is too long ({pages} pages). Keep it under 2 pages.")
        formatting_score -= 3

    # ---- PDF checks ----
    if filename.endswith(".pdf"):
        try:
            pdf = pdfplumber.open(io.BytesIO(file_bytes))
            table_found = any(page.extract_tables() for page in pdf.pages)

            if table_found:
                issues.append("Tables detected — ATS may fail to read table content.")
                formatting_score -= 4

            image_count = sum(len(page.images) for page in pdf.pages)
            if image_count > 0:
                issues.append("Images/icons detected. ATS cannot read images.")
                formatting_score -= 3

        except Exception:
            issues.append("PDF may be scanned or image-based.")
            formatting_score -= 4

    # ---- DOCX checks ----
    if filename.endswith(".docx"):
        try:
            doc = docx.Document(io.BytesIO(file_bytes))

            if len(doc.tables) > 0:
                issues.append("Tables detected in DOCX — ATS cannot read tables well.")
                formatting_score -= 4

            if len(doc.inline_shapes) > 0:
                issues.append("Images/icons detected in DOCX.")
                formatting_score -= 3

            for section in doc.sections:
                header_text = section.header.paragraphs[0].text.strip()
                footer_text = section.footer.paragraphs[0].text.strip()
                if header_text:
                    issues.append("Header contains text — ATS often ignores header content.")
                    formatting_score -= 2
                if footer_text:
                    issues.append("Footer contains text — ATS often ignores footer content.")
                    formatting_score -= 2

        except Exception:
            issues.append("DOCX parsing issue.")
            formatting_score -= 3

    formatting_score = max(0, min(formatting_score, 20))
    return formatting_score, issues


# -----------------------------------------
# ADVANCED KEYWORD ENGINE (STEP 1)
# -----------------------------------------
def normalize_text(t: str):
    """
    Simple tokenizer for ATS keyword matching.
    No NLTK, so it works on Render without extra corpora.
    """
    t = t.lower()
    # keep only a–z, 0–9 and spaces
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    words = [
        w
        for w in t.split()
        if w not in STOPWORDS and len(w) > 2
    ]
    return words


# -----------------------------------------
# WRITING ANALYSIS ENGINE (STEP 3)
# -----------------------------------------
def analyze_writing(text):
    writing_score = 30
    issues = []

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    bullets = [l for l in lines if l.startswith(("-", "•", "*"))]

    # 1. ACTION VERBS
    action_count = sum(1 for word in text.lower().split() if word in ACTION_VERBS)
    if action_count < 5:
        issues.append("Few action verbs found. Use stronger verbs like Built, Led, Created.")
        writing_score -= 4

    # 2. ACHIEVEMENTS (% / numbers)
    if not re.search(r"\d+%|\d{2,}", text):
        issues.append("Add measurable achievements (numbers or percentages).")
        writing_score -= 4

    # 3. WEAK PHRASES
    weak_found = [p for p in WEAK_PHRASES if p in text.lower()]
    if weak_found:
        issues.append(f"Weak phrases detected: {', '.join(weak_found)}.")
        writing_score -= 3

    # 4. REPETITION
    words = [w for w in text.lower().split() if len(w) > 3]
    freq = Counter(words)
    repeated = [word for word, count in freq.items() if count > 5]
    if repeated:
        issues.append(f"Overused words: {', '.join(repeated[:5])}.")
        writing_score -= 3

    # 5. PASSIVE VOICE
    passive_hits = sum(1 for w in text.lower().split() if w in PASSIVE_VERBS)
    if passive_hits > 10:
        issues.append("Too much passive voice detected.")
        writing_score -= 3

    # 6. BULLET LENGTH QUALITY
    for b in bullets:
        if len(b.split()) > 25:
            issues.append("Some bullet points are too long. Keep under 25 words.")
            writing_score -= 2
            break

    writing_score = max(0, min(writing_score, 30))
    return writing_score, issues


# -----------------------------------------
# GEMINI HELPER (STEP 4)
# -----------------------------------------
def gemini_generate(prompt: str) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"


# -----------------------------------------
# MAIN ATS ENDPOINT
# -----------------------------------------
@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form(default="")
):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    filename = file.filename.lower()

    # Extract text
    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text.")

    # -------- SECTION DETECTION --------
    lower = text.lower()
    sections = {
        "summary": any(x in lower for x in ["summary", "objective", "profile"]),
        "skills": any(x in lower for x in ["skills", "technical skills"]),
        "experience": any(x in lower for x in ["experience", "work experience", "employment"]),
        "education": any(x in lower for x in ["education", "academics", "qualification"]),
    }

    structure_score = 0
    if sections["summary"]:
        structure_score += 5
    if sections["skills"]:
        structure_score += 10
    if sections["experience"]:
        structure_score += 10
    if sections["education"]:
        structure_score += 5

    # -------- FORMATTING (STEP 2) --------
    formatting_score, formatting_issues = analyze_formatting(text, file_bytes, filename)

    # -------- KEYWORDS (STEP 1) --------
    resume_words = normalize_text(text)
    job_words = normalize_text(job_description)

    resume_set = set(resume_words)
    job_set = set(job_words)

    matched_keywords = []
    keyword_score = 0

    for word in job_set:
        if word in HARD_SKILLS and word in resume_set:
            keyword_score += 4
            matched_keywords.append(word)
        elif word in SOFT_SKILLS and word in resume_set:
            keyword_score += 2
            matched_keywords.append(word)
        elif word in JOB_TITLES and word in resume_set:
            keyword_score += 3
            matched_keywords.append(word)
        elif word in resume_set:
            keyword_score += 1
            matched_keywords.append(word)

    keyword_score = min(keyword_score, 50)

    # -------- WRITING ANALYSIS (STEP 3) --------
    writing_score, writing_issues = analyze_writing(text)

    # -------- FINAL ATS SCORE --------
    ats_score = structure_score + formatting_score + keyword_score + writing_score
    ats_score = min(100, ats_score)

    # -------- Suggestions --------
    suggestions = []
    if not sections["summary"]:
        suggestions.append("Add a professional summary at the top.")
    if not sections["skills"]:
        suggestions.append("Include a skills section.")
    if not sections["experience"]:
        suggestions.append("Add a detailed experience section.")
    if not sections["education"]:
        suggestions.append("Include an education section.")

    if job_description.strip():
        suggestions.append(
            f"Matched {len(matched_keywords)} job keywords. Add more relevant keywords truthfully."
        )

    return {
        "filename": file.filename,
        "char_count": len(text),
        "sections_found": sections,
        "structure_score": structure_score,
        "formatting_score": formatting_score,
        "formatting_issues": formatting_issues,
        "keyword_score": keyword_score,
        "matched_keywords": matched_keywords,
        "writing_score": writing_score,
        "writing_issues": writing_issues,
        "ats_score": ats_score,
        "suggestions": suggestions,
        "preview": text[:800],
        "job_description": job_description,
    }


# -----------------------------------------
# AI ENDPOINTS (STEP 4)
# -----------------------------------------
@app.post("/ai/rewrite-summary")
async def rewrite_summary(
    summary: str = Form(...),
    job_description: str = Form(default="")
):
    prompt = f"""
Rewrite this resume summary to be highly professional, ATS-friendly, keyword-optimized,
and achievement-focused.

Resume Summary:
{summary}

Job Description (for keyword matching):
{job_description}

Return ONLY the improved summary, no explanation.
"""
    result = gemini_generate(prompt)
    return {"improved_summary": result}


@app.post("/ai/rewrite-bullet")
async def rewrite_bullet(
    bullet: str = Form(...),
    job_description: str = Form(default="")
):
    prompt = f"""
Rewrite this resume bullet point to be stronger, achievement-based, quantified,
and ATS-friendly. Use a single bullet sentence.

Original bullet:
{bullet}

Job Description:
{job_description}

Return ONLY the improved bullet, starting with a strong action verb.
"""
    result = gemini_generate(prompt)
    return {"improved_bullet": result}


@app.post("/ai/improve-section")
async def improve_section(
    section_text: str = Form(...),
    section_name: str = Form(...)
):
    prompt = f"""
Improve this resume section called '{section_name}'.
Make it:
- ATS-friendly
- Action-driven
- Clear
- Concise
- Focused on achievements

Section content:
{section_text}

Return ONLY the improved section text.
"""
    result = gemini_generate(prompt)
    return {"improved_section": result}


@app.post("/ai/keyword-optimize")
async def keyword_optimize(
    resume_text: str = Form(...),
    job_description: str = Form(...)
):
    prompt = f"""
Analyze the resume and job description.

1. List important job-related keywords that are MISSING from the resume.
2. Suggest exact sentences or bullet points to add to the resume to include these keywords naturally.

Resume:
{resume_text}

Job Description:
{job_description}

Return in this format:
Missing Keywords:
- ...
Suggestions:
- ...
"""
    result = gemini_generate(prompt)
    return {"keyword_optimization": result}


@app.post("/ai/full-ats-improve")
async def full_ats_improve(
    resume_text: str = Form(...),
    job_description: str = Form(...)
):
    prompt = f"""
You are an ATS optimization assistant.

Improve the resume below based on the job description by:
- Improving the summary
- Strengthening experience bullets
- Adding measurable achievements
- Adding relevant keywords
- Keeping it ATS-friendly (no tables, no images, no fancy formatting)

Resume:
{resume_text}

Job Description:
{job_description}

Return in this JSON-like text structure (BUT as plain text, not actual JSON):

Improved Summary:
...

Improved Experience:
...

Improved Skills:
...

Added Keywords:
- ...
- ...

Final Optimized Resume:
...
"""
    result = gemini_generate(prompt)
    return {"full_improvement": result}


# -----------------------------------------
# ROOT TEST ROUTE
# -----------------------------------------
@app.get("/")
def read_root():
    return {"message": "ATSPro backend running with Steps 1–4 (ATS + Gemini AI)"}




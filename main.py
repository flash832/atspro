from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import io
import re
from collections import Counter
import os
import google.generativeai as genai

# -----------------------------------------
# GEMINI CONFIG — NEW API (WORKS WITH 1.5 FLASH)
# -----------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise Exception("GEMINI_API_KEY environment variable not found.")

genai.configure(api_key=GEMINI_API_KEY)

# IMPORTANT → use new v1 model path
GEMINI_MODEL = "models/gemini-1.5-flash"


def gemini_generate(prompt):
    """Safe wrapper for Gemini API calls."""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 800}
        )
        return response.text
    except Exception as e:
        return f"AI Error: {str(e)}"


# -----------------------------------------
# FASTAPI INIT
# -----------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# -----------------------------------------
# FILE EXTRACTORS
# -----------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except:
        raise HTTPException(400, "Unable to parse PDF. It may be scanned.")
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in document.paragraphs]).strip()
    except:
        raise HTTPException(400, "Unable to parse DOCX file.")


# -----------------------------------------
# UTILITY CLEANERS
# -----------------------------------------

STOPWORDS = set([
    "the", "and", "with", "your", "for", "are", "was", "were", "you",
    "this", "that", "from", "have", "has", "had", "who", "what"
])


def normalize_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [w for w in t.split() if w not in STOPWORDS and len(w) > 2]


# -----------------------------------------
# ANALYSIS ENGINE
# -----------------------------------------

ACTION_VERBS = [
    "built", "created", "developed", "managed", "led", "optimized",
    "designed", "launched", "improved", "implemented"
]

WEAK_PHRASES = [
    "responsible for", "helped with", "worked on", "involved in"
]

PASSIVE = ["was", "were", "is", "are", "be", "been"]


def analyze_writing(text):
    score = 30
    issues = []

    if sum(1 for w in text.lower().split() if w in ACTION_VERBS) < 3:
        issues.append("Too few strong action verbs.")
        score -= 5

    if not re.search(r"\d+%|\d{2,}", text):
        issues.append("No measurable achievements (numbers/metrics).")
        score -= 4

    weak = [p for p in WEAK_PHRASES if p in text.lower()]
    if weak:
        issues.append(f"Weak phrases detected: {', '.join(weak)}")
        score -= 3

    if sum(1 for w in text.lower().split() if w in PASSIVE) > 8:
        issues.append("Too much passive voice.")
        score -= 3

    return max(score, 0), issues


def analyze_formatting(text):
    score = 20
    issues = []

    if len(text) < 200:
        score -= 2
        issues.append("Resume appears too short.")

    if "\t" in text:
        score -= 3
        issues.append("Tabs detected — ATS may break formatting.")

    if re.search(r"\s{3,}", text):
        score -= 2
        issues.append("Extra spaces detected (avoid manual alignment).")

    return score, issues


# -----------------------------------------
# MAIN ATS ENDPOINT
# -----------------------------------------

@app.post("/upload-resume")
async def upload_resume(file: UploadFile = File(...), job_description: str = Form("")):

    file_bytes = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(400, "Only PDF or DOCX allowed.")

    if not text.strip():
        raise HTTPException(400, "Could not extract text.")

    # SECTION DETECTION
    lower = text.lower()
    sections = {
        "summary": "summary" in lower or "objective" in lower,
        "skills": "skills" in lower,
        "experience": "experience" in lower,
        "education": "education" in lower
    }

    structure_score = (
        (5 if sections["summary"] else 0) +
        (10 if sections["skills"] else 0) +
        (10 if sections["experience"] else 0) +
        (5 if sections["education"] else 0)
    )

    formatting_score, formatting_issues = analyze_formatting(text)
    writing_score, writing_issues = analyze_writing(text)

    # KEYWORD MATCHING
    resume_words = set(normalize_text(text))
    job_words = set(normalize_text(job_description))

    matched = list(resume_words.intersection(job_words))
    keyword_score = min(len(matched) * 3, 40)

    # FINAL ATS SCORE
    ats_score = min(100, structure_score + formatting_score + writing_score + keyword_score)

    return {
        "filename": file.filename,
        "preview": text[:800],
        "ats_score": ats_score,
        "sections_found": sections,
        "structure_score": structure_score,
        "formatting_score": formatting_score,
        "formatting_issues": formatting_issues,
        "writing_score": writing_score,
        "writing_issues": writing_issues,
        "keyword_score": keyword_score,
        "matched_keywords": matched,
        "suggestions": [
            "Add summary section" if not sections["summary"] else "",
            "Add skills section" if not sections["skills"] else "",
            "Add job keywords naturally" if job_description else ""
        ]
    }


# -----------------------------------------
# AI ENDPOINTS
# -----------------------------------------

@app.post("/ai/rewrite-summary")
async def rewrite_summary(summary: str = Form(...), job_description: str = Form("")):
    prompt = f"""
Rewrite this resume summary to be more ATS-friendly, keyword-rich and impactful.

SUMMARY:
{summary}

JOB DESCRIPTION:
{job_description}

Return ONLY the improved summary.
"""
    return {"improved_summary": gemini_generate(prompt)}


@app.post("/ai/rewrite-bullet")
async def rewrite_bullet(bullet: str = Form(...), job_description: str = Form("")):
    prompt = f"""
Rewrite this bullet point to start with a strong action verb and include measurable results.

BULLET:
{bullet}

JOB DESCRIPTION:
{job_description}

Return only the improved bullet.
"""
    return {"improved_bullet": gemini_generate(prompt)}


@app.post("/ai/improve-section")
async def improve_section(section_text: str = Form(...), section_name: str = Form(...)):
    prompt = f"""
Improve this resume section called {section_name}.

SECTION:
{section_text}

Make it more ATS-friendly, achievement-based, and concise.
Return only the improved text.
"""
    return {"improved_section": gemini_generate(prompt)}


@app.post("/ai/full-ats-improve")
async def full_ats_improve(resume_text: str = Form(...), job_description: str = Form(...)):
    prompt = f"""
Improve the following resume using ATS rules, strong action verbs, and job keywords.

RESUME:
{resume_text}

JOB:
{job_description}

Return:
- Improved Summary
- Improved Experience Bullets
- Improved Skills
- Suggested Keywords
- Final Optimized Resume
"""
    return {"full_improvement": gemini_generate(prompt)}


@app.get("/")
def root():
    return {"message": "ATSPro backend running successfully (Gemini 1.5 Flash enabled)!"}

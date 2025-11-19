from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import io
import re
from collections import Counter
import os

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
    except Exception:
        raise HTTPException(400, "Unable to parse PDF. It may be scanned.")
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in document.paragraphs]).strip()
    except Exception:
        raise HTTPException(400, "Unable to parse DOCX file.")


# -----------------------------------------
# UTILITY CLEANERS
# -----------------------------------------

STOPWORDS = set([
    "the", "and", "with", "your", "for", "are", "was", "were", "you",
    "this", "that", "from", "have", "has", "had", "who", "what", "his",
    "her", "their", "our", "they", "them", "she", "he"
])


def normalize_text(t: str):
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return [w for w in t.split() if w not in STOPWORDS and len(w) > 2]


# -----------------------------------------
# ANALYSIS ENGINE
# -----------------------------------------

ACTION_VERBS = [
    "built", "created", "developed", "managed", "led", "optimized",
    "designed", "launched", "improved", "implemented", "delivered",
    "increased", "reduced", "automated", "configured"
]

WEAK_PHRASES = [
    "responsible for", "helped with", "worked on", "involved in", "assisted with"
]

PASSIVE = ["was", "were", "is", "are", "be", "been", "being"]


def analyze_writing(text: str):
    score = 30
    issues = []

    # action verbs
    if sum(1 for w in text.lower().split() if w in ACTION_VERBS) < 3:
        issues.append("Too few strong action verbs (e.g. Built, Led, Created).")
        score -= 5

    # numbers / metrics
    if not re.search(r"\d+%|\d{2,}", text):
        issues.append("No measurable achievements (add numbers or percentages).")
        score -= 4

    # weak phrases
    weak = [p for p in WEAK_PHRASES if p in text.lower()]
    if weak:
        issues.append(f"Weak phrases detected: {', '.join(weak)}.")
        score -= 3

    # passive voice
    if sum(1 for w in text.lower().split() if w in PASSIVE) > 8:
        issues.append("Too much passive voice — use active verbs.")
        score -= 3

    # long paragraphs
    paragraphs = [p for p in text.split("\n") if p.strip()]
    if any(len(p.split()) > 120 for p in paragraphs):
        issues.append("Some paragraphs are very long — break into shorter bullets.")
        score -= 3

    return max(score, 0), issues


def analyze_formatting(text: str):
    score = 20
    issues = []

    if len(text) < 200:
        score -= 2
        issues.append("Resume appears very short; add more detail if possible.")

    if "\t" in text:
        score -= 3
        issues.append("Tabs detected — may break ATS parsing. Use single-column layout.")

    if re.search(r"\s{3,}", text):
        score -= 2
        issues.append("Multiple consecutive spaces detected — avoid manual spacing.")

    if re.search(r"[^\x00-\x7F]", text):
        score -= 2
        issues.append("Non-ASCII characters detected — some ATS may misread them.")

    return score, issues


# -----------------------------------------
# SIMPLE LOCAL "AI" HELPERS (NO EXTERNAL API)
# -----------------------------------------

def simple_summary_rewrite(summary: str, job_description: str) -> str:
    """
    Rule-based "AI" to make the summary more ATS-friendly:
    - removes first person
    - adds some generic keywords based on job description
    - makes tone more impact-focused
    """
    base = summary.strip()
    if not base:
        return "Results-driven professional with proven ability to deliver measurable impact across projects."

    # remove first person phrases
    base = re.sub(r"\bI am\b", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\bI\b", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\bmy\b", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\bme\b", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\s{2,}", " ", base).strip()

    # pull a few keywords from job description
    jd_words = normalize_text(job_description)
    freq = Counter(jd_words)
    top_keywords = [w for w, c in freq.most_common(6) if len(w) > 3]

    kw_part = ""
    if top_keywords:
        kw_part = " Key strengths include: " + ", ".join(top_keywords[:6]) + "."

    return (
        "Results-driven professional with a strong track record of delivering high-quality solutions. "
        + base.rstrip(".")
        + "."
        + kw_part
    )


def simple_bullet_rewrite(bullet: str, job_description: str) -> str:
    text = bullet.strip().lstrip("-•* ").strip()
    if not text:
        return "Led key initiatives that delivered measurable business impact."

    # ensure starts with action verb
    words = text.split()
    first = words[0].lower()
    if first not in ACTION_VERBS:
        text = "Led " + text[0].lower() + text[1:]

    # hint to add numbers if missing
    if not re.search(r"\d+%|\d{2,}", text):
        text += " (added measurable impact, e.g. 20% improvement in efficiency)."

    return "• " + text


def simple_section_improve(section_text: str, section_name: str) -> str:
    lines = [l.strip() for l in section_text.split("\n") if l.strip()]
    improved = []
    for line in lines:
        # treat each line as bullet if not already
        if not line.startswith(("-", "•", "*")) and len(line.split()) > 3:
            line = "• " + line
        # shorten very long lines
        if len(line.split()) > 30:
            line = " ".join(line.split()[:30]) + " ..."
        improved.append(line)
    header = section_name.strip().title() or "Section"
    return header + "\n" + "\n".join(improved)


def simple_full_ats_improve(resume_text: str, job_description: str) -> str:
    summary = simple_summary_rewrite(resume_text[:500], job_description)
    writing_score, writing_issues = analyze_writing(resume_text)
    formatting_score, formatting_issues = analyze_formatting(resume_text)

    issues_list = ""
    if formatting_issues or writing_issues:
        all_issues = ["- " + i for i in (formatting_issues + writing_issues)]
        issues_list = "\n".join(all_issues)
    else:
        issues_list = "None detected."

    return f"""
Improved Summary:
{summary}

Formatting & Writing Issues Detected:
{issues_list}

Suggested Next Steps:
- Turn dense paragraphs into short bullets starting with action verbs.
- Add 2–3 quantified achievements (numbers, percentages, time saved).
- Mirror key skills from the job description honestly in your Skills section.
""".strip()


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
        raise HTTPException(400, "Could not extract text from the file.")

    # SECTION DETECTION
    lower = text.lower()
    sections = {
        "summary": "summary" in lower or "objective" in lower,
        "skills": "skills" in lower or "technical skills" in lower,
        "experience": "experience" in lower or "work experience" in lower,
        "education": "education" in lower or "academics" in lower,
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

    suggestions = []
    if not sections["summary"]:
        suggestions.append("Add a clear 'Summary' or 'Profile' section at the top.")
    if not sections["skills"]:
        suggestions.append("Include a dedicated 'Skills' section with tools, languages, and frameworks.")
    if not sections["experience"]:
        suggestions.append("Add a 'Experience' section with bullet points for each role.")
    if not sections["education"]:
        suggestions.append("Include your 'Education' details (degree, university, year).")
    if job_description:
        suggestions.append("Mirror the most relevant job description keywords in your Skills and Experience sections.")

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
        "suggestions": suggestions
    }


# -----------------------------------------
# AI-LIKE ENDPOINTS (LOCAL LOGIC)
# -----------------------------------------

@app.post("/ai/rewrite-summary")
async def rewrite_summary(summary: str = Form(...), job_description: str = Form("")):
    improved = simple_summary_rewrite(summary, job_description)
    return {"improved_summary": improved}


@app.post("/ai/rewrite-bullet")
async def rewrite_bullet(bullet: str = Form(...), job_description: str = Form("")):
    improved = simple_bullet_rewrite(bullet, job_description)
    return {"improved_bullet": improved}


@app.post("/ai/improve-section")
async def improve_section(section_text: str = Form(...), section_name: str = Form(...)):
    improved = simple_section_improve(section_text, section_name)
    return {"improved_section": improved}


@app.post("/ai/full-ats-improve")
async def full_ats_improve(resume_text: str = Form(...), job_description: str = Form(...)):
    improved = simple_full_ats_improve(resume_text, job_description)
    return {"full_improvement": improved}


# -----------------------------------------
# ROOT
# -----------------------------------------

@app.get("/")
def root():
    return {"message": "ATSPro backend running (local AI, no external API)."}

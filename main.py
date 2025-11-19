from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import docx
import io
import re
from collections import Counter

# =========================================
# FASTAPI INIT
# =========================================

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# TEXT HELPERS
# =========================================

STOPWORDS = {
    "the", "and", "with", "your", "for", "are", "was", "were", "you",
    "this", "that", "from", "have", "has", "had", "who", "what", "when",
    "how", "which", "their", "our", "they", "them", "his", "her"
}

ACTION_VERBS = [
    "Led", "Delivered", "Improved", "Built", "Created", "Developed",
    "Optimized", "Designed", "Launched", "Implemented", "Managed"
]

WEAK_PHRASES = [
    "responsible for",
    "helped with",
    "worked on",
    "involved in",
    "assisted with"
]

PASSIVE = ["was", "were", "is", "are", "be", "been", "being"]


def normalize_words(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if w not in STOPWORDS and len(w) > 2]


# =========================================
# FILE EXTRACTORS
# =========================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += (page.extract_text() or "") + "\n"
    except Exception:
        raise HTTPException(400, "Unable to parse PDF. File might be scanned.")
    return text.strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        document = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in document.paragraphs).strip()
    except Exception:
        raise HTTPException(400, "Unable to parse DOCX file.")


# =========================================
# ANALYSIS ENGINES (ATS STYLE)
# =========================================

def analyze_formatting(text: str):
    score = 20
    issues = []

    if len(text) < 200:
        score -= 2
        issues.append("Resume appears very short; aim for at least ~1 page.")

    if "\t" in text:
        score -= 3
        issues.append("Tabs detected. Use simple single-column layout.")

    if re.search(r"\s{3,}", text):
        score -= 2
        issues.append("Multiple spaces detected – avoid manual spacing/alignment.")

    if re.search(r"[^\x00-\x7F]", text):
        score -= 2
        issues.append("Non-ASCII characters detected; ATS may not read them correctly.")

    return max(score, 0), issues


def analyze_writing(text: str):
    score = 30
    issues = []

    # Action verbs
    action_hits = sum(1 for w in text.split() if w.lower() in {v.lower() for v in ACTION_VERBS})
    if action_hits < 4:
        score -= 5
        issues.append("Use more strong action verbs (Led, Delivered, Built, Optimized...).")

    # Numbers / achievements
    if not re.search(r"\d+%|\d{2,}", text):
        score -= 4
        issues.append("Add measurable achievements (numbers, %, users, revenue, etc.).")

    # Weak phrases
    weak_found = [p for p in WEAK_PHRASES if p in text.lower()]
    if weak_found:
        score -= 4
        issues.append(f"Weak phrases detected: {', '.join(weak_found)}. Use direct, impact-focused language.")

    # Passive voice (rough)
    passive_hits = sum(1 for w in text.split() if w.lower() in PASSIVE)
    if passive_hits > 10:
        score -= 3
        issues.append("Too much passive voice. Prefer active (Built, Led, Delivered) sentences.")

    # Repetition
    words = [w.lower() for w in re.findall(r"[a-zA-Z]{4,}", text)]
    freq = Counter(words)
    repeated = [w for w, c in freq.items() if c > 6]
    if repeated:
        score -= 2
        issues.append("Some words are over-used: " + ", ".join(repeated[:6]) + ".")

    return max(score, 0), issues


def detect_sections(text: str):
    """
    Very simple heuristic: split into lines and assign them into
    SUMMARY / SKILLS / EXPERIENCE / EDUCATION / OTHER buckets.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    sections = {
        "header": [],
        "summary": [],
        "skills": [],
        "experience": [],
        "education": [],
        "other": [],
    }

    current = "other"
    for i, line in enumerate(lines):
        low = line.lower()

        if "summary" in low or "objective" in low:
            current = "summary"
            continue
        if "skill" in low and len(low) < 40:
            current = "skills"
            continue
        if any(k in low for k in ["experience", "employment", "work history"]):
            current = "experience"
            continue
        if any(k in low for k in ["education", "academic", "qualification"]):
            current = "education"
            continue

        # assume top 2–3 lines are header if no current section yet
        if i < 3 and current == "other":
            sections["header"].append(line)
        else:
            sections[current].append(line)

    return sections


# =========================================
# "AI" REWRITERS – PURE RULE-BASED
# =========================================

def simple_summary_rewrite(resume_text: str, job_description: str) -> str:
    # remove first person pronouns
    clean = re.sub(r"\b(I|my|me|mine)\b", "", resume_text, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip()

    # pick first ~80–100 words as base
    words = clean.split()
    base = " ".join(words[:90])

    # extract top keywords from JD
    jd_words = normalize_words(job_description)
    common = [w for w, c in Counter(jd_words).most_common(15)]
    common_text = ", ".join(common) if common else ""

    summary = (
        "Results-driven professional with a strong track record across projects and teams. "
        "Skilled in delivering reliable, production-ready solutions and collaborating with cross-functional stakeholders. "
    )

    if common_text:
        summary += f"Key strengths aligned with the role include: {common_text}."

    if base:
        summary += " " + base

    return summary.strip()


def simple_bullet_rewrite(bullet: str, job_description: str = "") -> str:
    text = bullet.strip().lstrip("•-").strip()
    if not text:
        return ""

    # remove weak beginnings
    low = text.lower()
    for phrase in WEAK_PHRASES:
        if low.startswith(phrase):
            text = text[len(phrase):].lstrip(" ,.-")
            break

    # ensure starts with action verb
    if not any(text.startswith(v) for v in ACTION_VERBS):
        text = f"{ACTION_VERBS[0]} {text}"

    # ensure compact
    text = re.sub(r"\s+", " ", text).strip()

    # small hint if no numbers present
    if not re.search(r"\d+%|\d{2,}", text):
        text += " (add a measurable result, e.g., 20% improvement in efficiency)."

    return f"• {text}"


def simple_section_improve(section_text: str, section_name: str) -> str:
    lines = [l.strip() for l in section_text.splitlines() if l.strip()]
    improved = []

    for line in lines:
        if len(line.split()) > 6:
            improved.append(simple_bullet_rewrite(line))
        else:
            improved.append(line)

    return "\n".join(improved)


def auto_fix_resume(resume_text: str, job_description: str) -> dict:
    """
    Main engine that:
    - parses sections
    - rewrites summary
    - cleans skills
    - rewrites experience bullets
    - returns final resumed text
    """
    sections = detect_sections(resume_text)

    # SUMMARY
    base_for_summary = " ".join(sections["summary"]) or resume_text
    improved_summary = simple_summary_rewrite(base_for_summary, job_description)

    # SKILLS
    skills_raw = " ".join(sections["skills"])
    tokens = re.split(r"[,\n;/•|]", skills_raw)
    skills_clean = []
    for t in tokens:
        s = t.strip()
        if len(s) < 2:
            continue
        skills_clean.append(s)
    # unique preserve order
    seen = set()
    skills_unique = []
    for s in skills_clean:
        sl = s.lower()
        if sl not in seen:
            seen.add(sl)
            skills_unique.append(s)
    skills_block = "\n".join(f"• {s}" for s in skills_unique) if skills_unique else ""

    # EXPERIENCE
    exp_lines = sections["experience"]
    exp_block_lines = []
    for line in exp_lines:
        if not line.strip():
            continue
        exp_block_lines.append(simple_bullet_rewrite(line))
    experience_block = "\n".join(l for l in exp_block_lines if l)

    # EDUCATION
    education_block = "\n".join(sections["education"]).strip()

    # HEADER
    header_block = "\n".join(sections["header"]).strip()

    final_resume_parts = []
    if header_block:
        final_resume_parts.append(header_block)

    final_resume_parts.append("SUMMARY")
    final_resume_parts.append(improved_summary)

    if skills_block:
        final_resume_parts.append("")
        final_resume_parts.append("SKILLS")
        final_resume_parts.append(skills_block)

    if experience_block:
        final_resume_parts.append("")
        final_resume_parts.append("EXPERIENCE")
        final_resume_parts.append(experience_block)

    if education_block:
        final_resume_parts.append("")
        final_resume_parts.append("EDUCATION")
        final_resume_parts.append(education_block)

    final_resume = "\n".join(final_resume_parts).strip()

    return {
        "summary": improved_summary,
        "skills_block": skills_block,
        "experience_block": experience_block,
        "education_block": education_block,
        "final_resume": final_resume,
    }


# =========================================
# MAIN ATS ENDPOINT
# =========================================

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    job_description: str = Form("")
):
    file_bytes = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files are supported.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from the resume.")

    # SECTION FLAGS
    lower = text.lower()
    sections_found = {
        "summary": any(k in lower for k in ["summary", "objective", "profile"]),
        "skills": "skills" in lower,
        "experience": any(k in lower for k in ["experience", "employment", "work history"]),
        "education": any(k in lower for k in ["education", "academic", "qualification"]),
    }

    structure_score = 0
    if sections_found["summary"]:
        structure_score += 5
    if sections_found["skills"]:
        structure_score += 10
    if sections_found["experience"]:
        structure_score += 10
    if sections_found["education"]:
        structure_score += 5

    formatting_score, formatting_issues = analyze_formatting(text)
    writing_score, writing_issues = analyze_writing(text)

    # KEYWORDS vs JD
    resume_words = set(normalize_words(text))
    jd_words = set(normalize_words(job_description))
    matched_keywords = sorted(list(resume_words.intersection(jd_words)))
    keyword_score = min(len(matched_keywords) * 3, 40)

    ats_score = min(100, structure_score + formatting_score + writing_score + keyword_score)

    # NEW: auto-fixed full resume
    auto_fix = auto_fix_resume(text, job_description or "")

    suggestions = []
    if not sections_found["summary"]:
        suggestions.append("Add a clear SUMMARY section at the top.")
    if not sections_found["skills"]:
        suggestions.append("Add a SKILLS section with key tools, languages and technologies.")
    if not sections_found["experience"]:
        suggestions.append("Add an EXPERIENCE section with bullet-based achievements.")
    if not sections_found["education"]:
        suggestions.append("Add an EDUCATION section with degree, institution and year.")
    if job_description:
        suggestions.append("Include more job-specific keywords truthfully based on the job description.")

    return {
        "filename": file.filename,
        "preview": text[:800],
        "raw_text": text,
        "ats_score": ats_score,
        "sections_found": sections_found,
        "structure_score": structure_score,
        "formatting_score": formatting_score,
        "formatting_issues": formatting_issues,
        "writing_score": writing_score,
        "writing_issues": writing_issues,
        "keyword_score": keyword_score,
        "matched_keywords": matched_keywords,
        "suggestions": suggestions,
        # auto fixed resume:
        "auto_final_resume": auto_fix["final_resume"],
        "auto_summary": auto_fix["summary"],
    }


# =========================================
# OPTIONAL "AI" ENDPOINTS (BUTTONS)
# =========================================

@app.post("/ai/rewrite-summary")
async def rewrite_summary(
    summary: str = Form(...),
    job_description: str = Form("")
):
    return {
        "improved_summary": simple_summary_rewrite(summary, job_description)
    }


@app.post("/ai/rewrite-bullet")
async def rewrite_bullet_endpoint(
    bullet: str = Form(...),
    job_description: str = Form("")
):
    return {
        "improved_bullet": simple_bullet_rewrite(bullet, job_description)
    }


@app.post("/ai/improve-section")
async def improve_section_endpoint(
    section_text: str = Form(...),
    section_name: str = Form("Section")
):
    return {
        "improved_section": simple_section_improve(section_text, section_name)
    }


@app.post("/ai/full-ats-improve")
async def full_ats_improve_endpoint(
    resume_text: str = Form(...),
    job_description: str = Form("")
):
    auto = auto_fix_resume(resume_text, job_description)
    return {
        "improved_summary": auto["summary"],
        "improved_experience": auto["experience_block"],
        "improved_skills": auto["skills_block"],
        "improved_education": auto["education_block"],
        "final_optimized_resume": auto["final_resume"],
    }


# =========================================
# ROOT TEST ROUTE
# =========================================

@app.get("/")
def root():
    return {"message": "ATSPro backend running with rule-based ATS + auto-fix!"}

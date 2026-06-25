import logging
import uuid
import io
import PyPDF2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import List

from database import engine, SessionLocal, Base, DBCandidate
from parser import ResumeParser

# --- CREATE DATABASE TABLES ---
Base.metadata.create_all(bind=engine)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- PYDANTIC MODELS ---
class ParsedCandidate(BaseModel):
    full_name: str
    current_role: str
    skills: list[str]
    years_experience: int
    industry: str
    recent_activity_score: float = Field(..., ge=0.0, le=1.0)
    id: str = "" # Added to send the generated ID back to the frontend

class BulkAPIResponse(BaseModel):
    status: str
    processed_count: int
    candidates: List[ParsedCandidate]

app = FastAPI(title="TalentAegis Bio-Neural Engine", version="1.2.0 (Batch Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ai_parser = ResumeParser()

def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "".join(page.extract_text() + "\n" for page in pdf.pages if page.extract_text())
        if not text.strip():
            raise ValueError("PDF contains no readable text.")
        return text
    except Exception as e:
        logger.error(f"PDF Extraction Failed: {e}")
        raise ValueError(f"Invalid PDF document. {e}")

@app.get("/candidates/")
def get_all_candidates(db: Session = Depends(get_db)):
    candidates = db.query(DBCandidate).order_by(DBCandidate.recent_activity_score.desc()).all()
    return {"status": "success", "history": candidates}

# --- BULK UPLOAD ENDPOINT ---
@app.post("/upload-resumes/", response_model=BulkAPIResponse)
async def upload_resumes(
    files: List[UploadFile] = File(...), # <-- Now accepts an array of files!
    target_role: str = Form("Not Specified"),
    required_skills: str = Form("Not Specified"),
    target_experience: str = Form("0"),
    db: Session = Depends(get_db)
):
    logger.info(f"Incoming BATCH request -> Files: {len(files)} | Role: {target_role}")
    
    processed_candidates = []

    # Loop through every uploaded file
    for file in files:
        try:
            logger.info(f"Processing Organism: {file.filename}...")
            contents = await file.read()
            pdf_text = extract_text_from_pdf(contents) 
            
            raw_parsed_data = ai_parser.parse_resume(
                pdf_text=pdf_text, 
                target_role=target_role, 
                required_skills=required_skills, 
                target_experience=target_experience
            )

            validated_data = ParsedCandidate(**raw_parsed_data)
            candidate_id = f"TAGS-{str(uuid.uuid4().int)[:6]}"
            validated_data.id = candidate_id # Assign ID
            
            # Save individual candidate to Vault
            db_candidate = DBCandidate(
                id=candidate_id,
                full_name=validated_data.full_name,
                current_role=validated_data.current_role,
                skills=validated_data.skills,
                years_experience=validated_data.years_experience,
                industry=validated_data.industry,
                recent_activity_score=validated_data.recent_activity_score
            )
            db.add(db_candidate)
            db.commit()          
            db.refresh(db_candidate) 
            
            processed_candidates.append(validated_data)
            logger.info(f"Success. {validated_data.full_name} saved to Vault.")

        except Exception as e:
            logger.error(f"Failed to process {file.filename}: {e}")
            # We don't crash the whole batch if one file fails! We log it and move to the next.
            continue

    if not processed_candidates:
        raise HTTPException(status_code=400, detail="Failed to process any of the uploaded files. Check formats.")

    return BulkAPIResponse(
        status="success",
        processed_count=len(processed_candidates),
        candidates=processed_candidates
    )
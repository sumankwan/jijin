from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from models import Base, Company, Document
from database import SessionLocal, engine

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Ayoda Capital Group API")

# Configure CORS for future mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/companies/")
def get_companies(db: Session = Depends(get_db)):
    companies = db.query(Company).all()
    return companies

@app.get("/companies/{company_id}")
def get_company(company_id: int, db: Session = Depends(get_db)):
    company = db.query(Company).filter(Company.id == company_id).first()
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return company

@app.get("/companies/{company_id}/documents")
def get_company_documents(company_id: int, db: Session = Depends(get_db)):
    documents = db.query(Document).filter(Document.company_id == company_id).all()
    return documents

@app.post("/companies/{company_id}/documents")
async def upload_document(
    company_id: int,
    file: UploadFile = File(...),
    document_type: str = None,
    db: Session = Depends(get_db)
):
    # Verify company exists
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{company_id}_{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create document record
    document = Document(
        company_id=company_id,
        name=file.filename,
        document_type=document_type or "Unknown",
        file_path=file_path,
        file_size=os.path.getsize(file_path),
        mime_type=file.content_type
    )
    
    db.add(document)
    db.commit()
    db.refresh(document)
    
    return document

@app.get("/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(Document).filter(Document.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document

@app.on_event("startup")
def insert_dummy_companies():
    db = SessionLocal()
    try:
        if db.query(Company).count() == 0:
            parent = Company(
                name="Ayoda Capital Group",
                ownership_percentage=100,
                parent_id=None,
                direktur_utama="John Doe",
                direktur="Alice, Bob",
                komisaris_utama="Jane Smith",
                komisaris="Charlie, Dave",
                modal=1000000000
            )
            db.add(parent)
            db.commit()
            db.refresh(parent)
            sub1 = Company(
                name="Bumi Sentosa Jaya",
                ownership_percentage=80,
                parent_id=parent.id,
                direktur_utama="Eve",
                direktur="Frank, Grace",
                komisaris_utama="Heidi",
                komisaris="Ivan, Judy",
                modal=500000000
            )
            sub2 = Company(
                name="Jaya Abadi Semesta",
                ownership_percentage=60,
                parent_id=parent.id,
                direktur_utama="Ken",
                direktur="Leo, Mallory",
                komisaris_utama="Nina",
                komisaris="Oscar, Peggy",
                modal=300000000
            )
            sub3 = Company(
                name="Koninis Fajar Mineral",
                ownership_percentage=70,
                parent_id=parent.id,
                direktur_utama="Quentin",
                direktur="Ruth, Steve",
                komisaris_utama="Trudy",
                komisaris="Uma, Victor",
                modal=400000000
            )
            db.add_all([sub1, sub2, sub3])
            db.commit()
    except IntegrityError:
        db.rollback()
    finally:
        db.close() 
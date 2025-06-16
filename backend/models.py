from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Company(Base):
    __tablename__ = 'companies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    ownership_percentage = Column(Integer)
    parent_id = Column(Integer, ForeignKey('companies.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    direktur_utama = Column(String)
    direktur = Column(String)
    komisaris_utama = Column(String)
    komisaris = Column(String)
    modal = Column(Integer)
    
    # Correct self-referential relationship
    parent = relationship("Company", remote_side=[id], backref=backref("subsidiaries"))
    documents = relationship("Document", back_populates="company")

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey('companies.id'))
    name = Column(String, nullable=False)
    document_type = Column(String, nullable=False)  # e.g., 'Akte Pendirian', 'NIB', etc.
    file_path = Column(String, nullable=False)  # Path to stored PDF
    file_size = Column(Integer)  # Size in bytes
    mime_type = Column(String, default='application/pdf')
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    company = relationship("Company", back_populates="documents") 
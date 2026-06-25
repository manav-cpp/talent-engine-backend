from sqlalchemy import create_engine, Column, Integer, String, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Create a local SQLite file named 'talentaegis.db'
SQLALCHEMY_DATABASE_URL = "sqlite:///./talentaegis.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. Define the exact shape of your Database Table
class DBCandidate(Base):
    __tablename__ = "candidates"

    id = Column(String, primary_key=True, index=True)
    full_name = Column(String, index=True)
    current_role = Column(String)
    skills = Column(JSON)  # SQLite allows us to save the array of skills as JSON!
    years_experience = Column(Integer)
    industry = Column(String)
    recent_activity_score = Column(Float)
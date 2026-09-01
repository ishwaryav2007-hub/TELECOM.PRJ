from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime

from database import Base


class OptimizationResult(Base):
    __tablename__ = "optimization_results"

    id = Column(Integer, primary_key=True, index=True)

    load = Column(Float)
    esmode = Column(Float)
    txpower = Column(Float)

    hour = Column(Integer)
    dayofweek = Column(Integer)

    current_energy = Column(Float)
    optimized_energy = Column(Float)

    saving = Column(Float)
    saving_percent = Column(Float)

    recommended_esmode = Column(Float)
    recommended_txpower = Column(Float)

    load_category = Column(String)
    recommendation = Column(String)

    created_at = Column(DateTime, default=datetime.utcnow)
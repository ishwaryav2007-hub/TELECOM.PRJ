from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import engine, get_db, Base
from .models import OptimizationResult

from data.optimizer import optimize_tower


# Create database tables
Base.metadata.create_all(bind=engine)


# Create FastAPI application
app = FastAPI(
    title="Telecom Energy Optimization API",
    description="AI-based telecom tower energy optimization backend",
    version="1.0"
)


# ==========================================
# INPUT MODEL
# ==========================================

class OptimizationRequest(BaseModel):

    load: float
    esmode: float
    txpower: float
    hour: int
    dayofweek: int


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Telecom Energy Optimizer Backend is running!"
    }


# ==========================================
# OPTIMIZE
# ==========================================

@app.post("/optimize")
def optimize(
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):

    # --------------------------------------
    # CALL ML OPTIMIZER
    # --------------------------------------

    result = optimize_tower(

        load=request.load,

        current_esmode=request.esmode,

        current_txpower=request.txpower,

        hour=request.hour,

        dayofweek=request.dayofweek
    )


    # --------------------------------------
    # SAVE RESULT TO DATABASE
    # --------------------------------------

    record = OptimizationResult(

        # Input
        load=request.load,
        esmode=request.esmode,
        txpower=request.txpower,
        hour=request.hour,
        dayofweek=request.dayofweek,

        # AI result
        current_energy=result["current_energy"],
        optimized_energy=result["optimized_energy"],
        saving=result["saving"],
        saving_percent=result["saving_percent"],

        recommended_esmode=result[
            "recommended_ESMODE"
        ],

        recommended_txpower=result[
            "recommended_TXpower"
        ],

        load_category=result[
            "load_category"
        ],

        candidate_count=result[
            "candidate_count"
        ],

        historical_observations=result[
            "historical_observations"
        ],

        evidence_strength=result[
            "evidence_strength"
        ],

        recommendation=result[
            "recommendation"
        ]
    )


    db.add(record)

    db.commit()

    db.refresh(record)


    # --------------------------------------
    # RETURN RESULT
    # --------------------------------------

    return result


# ==========================================
# HISTORY
# ==========================================

@app.get("/history")
def get_history(
    db: Session = Depends(get_db)
):

    records = (
        db.query(OptimizationResult)
        .order_by(
            OptimizationResult.id.desc()
        )
        .all()
    )

    return records
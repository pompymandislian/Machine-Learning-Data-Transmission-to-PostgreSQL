from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from sqlalchemy.orm import sessionmaker
from database import engine, Base
from models import UserPredict
import logging

# Load the model
with open("logistic_regression_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Define the database session class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define the FastAPI app
app = FastAPI()

# Create all tables in the database
Base.metadata.create_all(bind=engine)

# Define input model for the prediction endpoint
class UserPredictInput(BaseModel):
    """Model representing input data for creating a new user."""
    transaction: int
    age: int
    tenure: int
    num_pages_visited: int
    has_credit_card: bool
    items_in_cart: int

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add a file handler to save logs to a file
file_handler = logging.FileHandler("prediction_logs.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Prediction endpoint
@app.post('/predict/')
async def predict_purchase(user_input: UserPredictInput):
    try:
        # Perform prediction using the loaded model
        prediction = loaded_model.predict([[user_input.transaction, 
                                            user_input.age, 
                                            user_input.tenure, 
                                            user_input.num_pages_visited, 
                                            user_input.has_credit_card, 
                                            user_input.items_in_cart]])

        # Assuming the prediction is binary (e.g., 0 or 1)
        purchase_prediction = bool(prediction[0])

        # Log prediction and user input
        logger.info(f"Received prediction request: {user_input.dict()}, Predicted purchase: {purchase_prediction}")

        # Save the prediction to the database
        db = SessionLocal()
        db_prediction = UserPredict(
            transaction=user_input.transaction,
            age=user_input.age,
            tenure=user_input.tenure,
            num_pages_visited=user_input.num_pages_visited,
            has_credit_card=user_input.has_credit_card,
            items_in_cart=user_input.items_in_cart,
            purchase_prediction=purchase_prediction
        )
        db.add(db_prediction)
        db.commit()
        db.close()

        return { "purchase_prediction": purchase_prediction}
    except Exception as e:
        # Rollback the transaction if there's an error
        db.rollback()
        logger.error(f"Error predicting purchase: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

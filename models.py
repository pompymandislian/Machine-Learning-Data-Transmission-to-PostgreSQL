from sqlalchemy import Boolean, Column, Integer, String
from database import Base

class UserPredict(Base):
    __tablename__ = "user_predicts"

    id = Column(Integer, primary_key=True, index=True)
    transaction = Column(Integer)
    age = Column(Integer)
    tenure = Column(Integer)
    num_pages_visited = Column(Integer)
    has_credit_card = Column(Boolean)
    items_in_cart = Column(Integer)
    purchase_prediction = Column(Boolean)

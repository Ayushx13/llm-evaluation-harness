from enum import Enum
from pydantic import BaseModel, Field

class Category(str, Enum):
    payment = "payment"
    delivery = "delivery"
    account = "account"
    product_defect = "product_defect"
    refund = "refund"
    other = "other"

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class TicketOutput(BaseModel):
    category: Category
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_span: str   # exact text from input that justifies the category



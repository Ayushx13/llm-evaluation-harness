from enum import StrEnum
from pydantic import BaseModel, Field


# StrEnum keeps enum members behaving like their underlying string values.
# This ensures serialized outputs use labels such as "payment" and "high"
# instead of enum representations such as "Category.payment".
class Category(StrEnum):
    payment = "payment"
    delivery = "delivery"
    account = "account"
    product_defect = "product_defect"
    refund = "refund"
    other = "other"


class Priority(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class TicketOutput(BaseModel):
    category: Category
    priority: Priority
    confidence: float = Field(ge=0.0, le=1.0)

    # Evidence must be copied verbatim from the input ticket.
    # The evaluation harness checks this span against the original input
    # to measure whether the model's supporting evidence is grounded.
    evidence_span: str = Field(min_length=1)
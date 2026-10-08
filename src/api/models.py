"""Simple API for the saved credit scoring pipeline."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Amount = Annotated[float, Field(allow_inf_nan=False)]
Payment = Annotated[float, Field(ge=0, allow_inf_nan=False)]
RepaymentStatus = Annotated[int, Field(ge=-3)]


class CreditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit_bal: Payment
    sex: Literal[1, 2]
    education: Literal[0, 1, 2, 3, 4, 5, 6]
    marriage: Literal[0, 1, 2, 3]
    age: Annotated[int, Field(ge=0)]
    pay_0: RepaymentStatus
    pay_2: RepaymentStatus
    pay_3: RepaymentStatus
    pay_4: RepaymentStatus
    pay_5: RepaymentStatus
    pay_6: RepaymentStatus
    bill_amt1: Amount
    bill_amt2: Amount
    bill_amt3: Amount
    bill_amt4: Amount
    bill_amt5: Amount
    bill_amt6: Amount
    pay_amt1: Payment
    pay_amt2: Payment
    pay_amt3: Payment
    pay_amt4: Payment
    pay_amt5: Payment
    pay_amt6: Payment


class CreditResponse(BaseModel):
    prediction: Literal[0, 1]
    default_probability: float = Field(ge=0, le=1)

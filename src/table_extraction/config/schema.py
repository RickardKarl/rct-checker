from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------- Group ----------


class Group(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str
    label: str
    sample_size: int


# ---------- Value Entry ----------


class ValueEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str
    original: str

    mean: float | None
    median: float | None
    count: int | None

    IQR_lower: float | None
    IQR_upper: float | None

    CI95_lower: float | None = Field(alias="95CI_lower")
    CI95_upper: float | None = Field(alias="95CI_upper")

    sd: float | None

    pvalue: float | None


# ---------- Row ----------


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")

    variable: str
    variable_type: Literal["Continuous", "Categorical"]
    level: str | None
    values: list[ValueEntry]


# ---------- Root Schema ----------


class PaperTable1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table1_exists: bool
    groups: list[Group] = Field(min_length=2)
    rows: list[Row]

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DatasetSplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trainRatio: float = Field(default=0.70, gt=0.0)
    validationRatio: float = Field(default=0.15, gt=0.0)
    testRatio: float = Field(default=0.15, gt=0.0)
    purgeEnabled: bool = True
    embargoBars: int = Field(default=0, ge=0)
    minimumTrainRows: int = Field(default=1, ge=1)
    minimumValidationRows: int = Field(default=1, ge=1)
    minimumTestRows: int = Field(default=1, ge=1)

    @field_validator("trainRatio", "validationRatio", "testRatio")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("split ratios must be greater than zero and no greater than one")
        return value

    @model_validator(mode="after")
    def validate_ratio_sum(self) -> "DatasetSplitConfig":
        total = self.trainRatio + self.validationRatio + self.testRatio
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "split ratios must sum to 1.0; "
                f"received {self.trainRatio} + {self.validationRatio} + {self.testRatio}"
            )
        return self

from pydantic import BaseModel, Field

from app.models.provider import ModelProvider


class ModelOption(BaseModel):
    id: str
    label: str
    provider: ModelProvider
    group: str
    status: str = Field(default="ready")


class PresetOption(BaseModel):
    id: str
    label: str


class ModelCatalogResponse(BaseModel):
    default_model: str
    default_preset: str = Field(default="general")
    models: list[ModelOption]
    presets: list[PresetOption] = Field(default_factory=list)

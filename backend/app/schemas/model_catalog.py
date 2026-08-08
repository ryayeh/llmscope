from pydantic import BaseModel, Field

from app.models.provider import ModelProvider
from app.schemas.provider_capabilities import ProviderCapabilitiesDetail


class ModelOption(BaseModel):
    id: str
    label: str
    provider: ModelProvider
    group: str
    status: str = Field(default="ready")
    capabilities: ProviderCapabilitiesDetail


class ProviderOption(BaseModel):
    id: ModelProvider
    label: str
    status: str = Field(default="ready")
    status_message: str | None = None
    recommended_models: list[str] = Field(default_factory=list)
    capabilities: ProviderCapabilitiesDetail


class PresetOption(BaseModel):
    id: str
    label: str


class ModelCatalogResponse(BaseModel):
    default_provider: ModelProvider = Field(default=ModelProvider.OPENAI)
    default_model: str
    default_preset: str = Field(default="general")
    providers: list[ProviderOption] = Field(default_factory=list)
    models: list[ModelOption]
    presets: list[PresetOption] = Field(default_factory=list)

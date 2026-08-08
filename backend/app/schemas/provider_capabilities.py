from pydantic import BaseModel, Field


class ProviderCapabilitiesDetail(BaseModel):
    supports_logprobs: bool
    supports_entropy: bool
    supports_attention: bool
    supports_exact_continuation: bool
    supports_streaming: bool
    supports_branching: bool = False
    supports_continuation: bool = False
    minimum_output_tokens: int = Field(..., ge=1)

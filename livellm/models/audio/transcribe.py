from pydantic import BaseModel, Field, field_validator
from typing import Tuple, TypeAlias, Optional, Union
from ..common import BaseRequest
import base64

File: TypeAlias = Tuple[str, Union[bytes, str], str] # (filename, file_content, content_type)

class TranscribeRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    file: File = Field(..., description="The file to transcribe")
    language: Optional[str] = Field(default=None, description="The language to transcribe")
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
    
    @field_validator('file', mode='before')
    @classmethod
    def decode_base64_file(cls, v) -> File:
        """
        Validates and processes the file field.
        
        Accepts two formats:
        1. Tuple[str, bytes, str] - Direct bytes (from multipart/form-data)
        2. Tuple[str, str, str] - Base64 encoded string (from JSON)
        
        Returns: Tuple[str, bytes, str] with decoded bytes
        """
        if not isinstance(v, (tuple, list)) or len(v) != 3:
            raise ValueError("file must be a tuple/list of 3 elements: (filename, content, content_type)")
        
        filename, content, content_type = v
        
        # If content is already bytes, return as-is
        if isinstance(content, bytes):
            try:
                encoded_content = base64.b64encode(content).decode("utf-8") # base64 encode the content
                return (filename, encoded_content, content_type)
            except Exception as e:
                raise ValueError(f"Failed to encode base64 content: {str(e)}")
        
        # If content is a string, assume it's base64 encoded
        elif isinstance(content, str):
            # assume it's already base64 encoded
            return (filename, content, content_type)


class TranscribeResponse(BaseModel):
    text: str = Field(..., description="The text of the transcription")
    language: Optional[str] = Field(default=None, description="The language of the transcription")

"""Output schema models for structured output support."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class PropertyDef(BaseModel):
    """Definition of a property in the output schema."""
    type: Union[str, List[str]] = Field(..., description="Property type: string, integer, number, boolean, array, object, null")
    description: Optional[str] = Field(default=None, description="Description of the property")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values for the property")
    default: Optional[Any] = Field(default=None, description="Default value")
    # String constraints
    minLength: Optional[int] = Field(default=None, description="Minimum string length")
    maxLength: Optional[int] = Field(default=None, description="Maximum string length")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for string validation")
    # Number constraints
    minimum: Optional[float] = Field(default=None, description="Minimum number value")
    maximum: Optional[float] = Field(default=None, description="Maximum number value")
    exclusiveMinimum: Optional[float] = Field(default=None, description="Exclusive minimum number value")
    exclusiveMaximum: Optional[float] = Field(default=None, description="Exclusive maximum number value")
    # Array constraints
    items: Optional[Union["PropertyDef", Dict[str, Any]]] = Field(default=None, description="Schema for array items")
    minItems: Optional[int] = Field(default=None, description="Minimum array length")
    maxItems: Optional[int] = Field(default=None, description="Maximum array length")
    uniqueItems: Optional[bool] = Field(default=None, description="Whether array items must be unique")
    # Object constraints
    properties: Optional[Dict[str, Union["PropertyDef", Dict[str, Any]]]] = Field(default=None, description="Nested object properties")
    required: Optional[List[str]] = Field(default=None, description="Required properties for nested objects")
    additionalProperties: Optional[Union[bool, "PropertyDef", Dict[str, Any]]] = Field(default=None, description="Schema for additional properties")

    class Config:
        extra = "allow"  # Allow additional JSON Schema properties


class OutputSchema(BaseModel):
    """
    Schema definition for structured output.
    
    This model represents a JSON Schema that the AI model must follow when generating responses.
    When provided, the agent will return a JSON string matching the specified schema.
    
    Example:
        schema = OutputSchema(
            title="Person",
            description="A person's information",
            properties={
                "name": PropertyDef(type="string", description="The person's name"),
                "age": PropertyDef(type="integer", minimum=0, maximum=150),
            },
            required=["name", "age"]
        )
    """
    title: str = Field(..., description="Name of the schema, used as the output tool name")
    description: Optional[str] = Field(default=None, description="Description to help the model understand what to output")
    properties: Dict[str, Union[PropertyDef, Dict[str, Any]]] = Field(..., description="Dictionary of property definitions")
    required: Optional[List[str]] = Field(default=None, description="List of required property names")
    additionalProperties: Optional[Union[bool, PropertyDef, Dict[str, Any]]] = Field(default=None, description="Whether extra properties are allowed")

    class Config:
        extra = "allow"  # Allow additional JSON Schema properties

    @classmethod
    def from_pydantic(cls, model: type[BaseModel]) -> "OutputSchema":
        """
        Create an OutputSchema from a Pydantic BaseModel class.
        
        Args:
            model: A Pydantic BaseModel class to convert to OutputSchema.
            
        Returns:
            An OutputSchema instance representing the model's schema.
            
        Example:
            class Person(BaseModel):
                name: str
                age: int
            
            schema = OutputSchema.from_pydantic(Person)
        """
        json_schema = model.model_json_schema()
        
        # Extract the main properties
        title = json_schema.get("title", model.__name__)
        description = json_schema.get("description")
        properties = json_schema.get("properties", {})
        required = json_schema.get("required")
        
        # Handle $defs for nested models (Pydantic generates these for complex models)
        defs = json_schema.get("$defs", {})
        if defs:
            # Inline the definitions into properties
            properties = cls._resolve_refs(properties, defs)
        
        return cls(
            title=title,
            description=description,
            properties=properties,
            required=required,
        )
    
    @classmethod
    def _resolve_refs(cls, obj: Any, defs: Dict[str, Any]) -> Any:
        """Recursively resolve $ref references in the schema."""
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"]
                # Extract the definition name from "#/$defs/ModelName"
                if ref_path.startswith("#/$defs/"):
                    def_name = ref_path[len("#/$defs/"):]
                    if def_name in defs:
                        # Return the resolved definition (also resolve any nested refs)
                        return cls._resolve_refs(defs[def_name], defs)
                return obj
            else:
                return {k: cls._resolve_refs(v, defs) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [cls._resolve_refs(item, defs) for item in obj]
        else:
            return obj



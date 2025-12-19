# LLM API Change Guide

This guide explains how to modify and manage APIs in TensorRT LLM, focusing on the high-level LLM API.

## Overview

TensorRT LLM provides multiple API levels:

1. **LLM API** - The highest-level API (e.g., the `LLM` class)
2. **PyExecutor API** - The mid-level API (e.g., the `PyExecutor` class)

This guide focuses on the LLM API, which is the primary interface for most users.

## API Types and Stability Guarantees

TensorRT LLM classifies APIs into two categories:

### 1. Committed APIs
- **Stable** and guaranteed to remain consistent across releases
- No breaking changes without major version updates
- Schema stored in: `tests/unittest/api_stability/references_committed/`

### 2. Non-committed APIs
- Under active development and may change between releases
- Marked with a `status` field in the docstring:
  - `prototype` - Early experimental stage
  - `beta` - More stable but still subject to change
  - `deprecated` - Scheduled for removal
- Schema stored in: `tests/unittest/api_stability/references/`
- See [API status documentation](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html) for complete details

## API Schema Management

All API schemas are:
- Stored as YAML files in the codebase
- Protected by unit tests in `tests/unittest/api_stability/`
- Automatically validated to ensure consistency 

## Modifying LLM Constructor Arguments

The LLM class accepts numerous configuration parameters for models, runtime, and other components. These are managed through a Pydantic dataclass called `LlmArgs`.

### Architecture

- The LLM's `__init__` method parameters map directly to `LlmArgs` fields
- `LlmArgs` is an alias for `TorchLlmArgs` (defined in `tensorrt_llm/llmapi/llm_args.py`)
- All arguments are validated and type-checked through Pydantic

### Adding a New Argument

Follow these steps to add a new constructor argument:

#### 1. Add the field to `TorchLlmArgs`

```python
garbage_collection_gen0_threshold: int = Field(
    default=20000,
    description=(
        "Threshold for Python garbage collection of generation 0 objects. "
        "Lower values trigger more frequent garbage collection."
    ),
    status="beta"  # Required for non-committed arguments
)
```

**Field requirements:**
- **Type annotation**: Required for all fields
- **Default value**: Recommended unless the field is mandatory
- **Description**: Clear explanation of the parameter's purpose
- **Status**: Required for non-committed arguments (`prototype`, `beta`, etc.)

#### 2. Update the API schema

Add the field to the appropriate schema file:

- **Non-committed arguments**: `tests/unittest/api_stability/references/llm_args.yaml`
  ```yaml
  garbage_collection_gen0_threshold:
    type: int
    default: 20000
    status: beta  # Must match the status in code
  ```

- **Committed arguments**: `tests/unittest/api_stability/references_committed/llm_args.yaml`
  ```yaml
  garbage_collection_gen0_threshold:
    type: int
    default: 20000
    # No status field for committed arguments
  ```

#### 3. Run validation tests

```bash
python -m pytest tests/unittest/api_stability/test_llm_api.py
```

## Modifying LLM Class Methods

Public methods in the LLM class constitute the API surface. All changes must be properly documented and tracked.

### Implementation Details

- The actual implementation is in the `_TorchLLM` class ([llm.py](https://github.com/NVIDIA/TensorRT-LLM/blob/release/1.0/tensorrt_llm/llmapi/llm.py))
- Public methods (not starting with `_`) are automatically exposed as APIs

### Adding a New Method

Follow these steps to add a new API method:

#### 1. Implement the method in `_TorchLLM`

For non-committed APIs, use the `@set_api_status` decorator:

```python
@set_api_status("beta")
def generate_with_streaming(
    self, 
    prompts: List[str], 
    **kwargs
) -> Iterator[GenerationOutput]:
    """Generate text with streaming output.
    
    Args:
        prompts: Input prompts for generation
        **kwargs: Additional generation parameters
        
    Returns:
        Iterator of generation outputs
    """
    # Implementation here
    pass
```

For committed APIs, no decorator is needed:

```python
def generate(self, prompts: List[str], **kwargs) -> GenerationOutput:
    """Generate text from prompts."""
    # Implementation here
    pass
```

#### 2. Update the API schema

Add the method to the appropriate `llm.yaml` file:

**Non-committed API** (`tests/unittest/api_stability/references/llm.yaml`):
```yaml
generate_with_streaming:
  status: beta  # Must match @set_api_status
  parameters:
    - name: prompts
      type: List[str]
    - name: kwargs
      type: dict
  returns: Iterator[GenerationOutput]
```

**Committed API** (`tests/unittest/api_stability/references_committed/llm.yaml`):
```yaml
generate:
  parameters:
    - name: prompts
      type: List[str]
    - name: kwargs
      type: dict
  returns: GenerationOutput
```

### Modifying Existing Methods

When modifying existing methods:

1. **Non-breaking changes** (adding optional parameters):
   - Update the method signature
   - Update the schema file
   - No status change needed

2. **Breaking changes** (changing required parameters, return types):
   - Only allowed for non-committed APIs
   - Consider deprecation path for beta APIs
   - Update documentation with migration guide

### Best Practices

1. **Documentation**: Always include comprehensive docstrings
2. **Type hints**: Use proper type annotations for all parameters and returns
3. **Testing**: Add unit tests for new methods
4. **Examples**: Provide usage examples in the docstring
5. **Validation**: Run API stability tests before submitting changes

### Running Tests

Validate your changes:

```bash
# Run API stability tests
python -m pytest tests/unittest/api_stability/

# Run specific test for LLM API
python -m pytest tests/unittest/api_stability/test_llm_api.py -v
```

## Common Workflows

### Promoting an API from Beta to Committed

1. Remove the `@set_api_status("beta")` decorator from the method
2. Move the schema entry from `tests/unittest/api_stability/references/` to `tests/unittest/api_stability/references_committed/`
3. Remove the `status` field from the schema
4. Update any documentation referring to the API's beta status

### Deprecating an API

1. Add `@set_api_status("deprecated")` to the method
2. Update the schema with `status: deprecated`
3. Add deprecation warning in the method:
   ```python
   import warnings
   warnings.warn(
       "This method is deprecated and will be removed in v2.0. "
       "Use new_method() instead.",
       DeprecationWarning,
       stacklevel=2
   )
   ```
4. Document the migration path

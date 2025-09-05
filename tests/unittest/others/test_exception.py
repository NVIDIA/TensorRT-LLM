import sys

from tensorrt_llm.bindings.exceptions import RequestSpecificException


def test_basic_exception_creation():
    """Test basic exception creation and catching."""
    try:
        raise RequestSpecificException("Test error message")
    except RequestSpecificException as e:
        assert isinstance(e, RequestSpecificException)
        assert "Test error message" in str(e)


def test_exception_inheritance():
    """Test that exception properly inherits from base Exception."""
    try:
        raise RequestSpecificException("Test inheritance")
    except Exception as e:  # Should catch base Exception
        assert isinstance(e, RequestSpecificException)
    except RequestSpecificException as e:  # Should also catch specific type
        assert isinstance(e, RequestSpecificException)


def test_exception_attributes_exist():
    """Test that exception has the expected attributes."""
    try:
        raise RequestSpecificException("Test attributes")
    except RequestSpecificException as e:
        # Check that attributes exist (they might be None if not set by C++)
        assert hasattr(e, 'request_id'), "request_id attribute missing"
        assert hasattr(e, 'error_code'), "error_code attribute missing"


def test_exception_traceback():
    """Test that exception provides proper traceback information."""
    try:
        raise RequestSpecificException("Test traceback")
    except RequestSpecificException as e:
        # Check that we can get traceback info
        exc_type, exc_value, exc_traceback = sys.exc_info()
        assert exc_type == RequestSpecificException
        assert exc_value == e
        assert exc_traceback is not None


def test_exception_equality():
    """Test exception equality and identity."""
    try:
        raise RequestSpecificException("Test equality")
    except RequestSpecificException as e1:
        try:
            raise RequestSpecificException("Test equality")
        except RequestSpecificException as e2:
            # Exceptions should not be equal even with same message
            assert e1 != e2, "Different exceptions should not be equal"
            assert e1 is not e2, "Different exceptions should not be identical"


def test_exception_context():
    """Test exception context and chaining."""
    try:
        try:
            raise RequestSpecificException("Inner exception")
        except RequestSpecificException as inner:
            raise RequestSpecificException("Outer exception") from inner
    except RequestSpecificException as outer:
        assert outer.__cause__ is not None
        assert isinstance(outer.__cause__, RequestSpecificException)


def test_cpp_exception_translation():
    """Test that C++ exceptions are properly translated to Python."""
    try:
        # This would normally be triggered by C++ code
        # For now, we'll test the Python exception creation
        raise RequestSpecificException("Test C++ translation")
    except RequestSpecificException as e:
        # Check that the exception was properly created
        assert isinstance(e, RequestSpecificException)


def test_exception_multiple_instances():
    """Test creating multiple exception instances."""
    exceptions = []
    for i in range(5):
        try:
            raise RequestSpecificException(f"Exception {i}")
        except RequestSpecificException as e:
            exceptions.append(e)

    assert len(exceptions) == 5
    for i, e in enumerate(exceptions):
        assert isinstance(e, RequestSpecificException)
        assert f"Exception {i}" in str(e)

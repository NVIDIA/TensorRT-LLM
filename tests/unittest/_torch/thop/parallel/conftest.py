import os

import pytest


@pytest.fixture(autouse=True, scope='function')
def set_torchinductor_compile_threads():
    """
    Fixture to set TORCHINDUCTOR_COMPILE_THREADS=1 for tests in this directory.
    """
    # --- Setup Phase ---
    # Save the original value if it exists
    original_value = os.environ.get('TORCHINDUCTOR_COMPILE_THREADS')

    # Set the desired value for the test
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

    # Let the test run with the new environment variable
    yield

    # --- Teardown Phase ---
    # Restore the original environment state after the test is done
    if original_value is None:
        # If the variable didn't exist before, remove it
        del os.environ['TORCHINDUCTOR_COMPILE_THREADS']
    else:
        # Otherwise, restore its original value
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = original_value

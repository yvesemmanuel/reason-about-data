# Tests for Reason-About-Data

This directory contains tests for the various components of the Reason-About-Data application.

## Test Structure

- `conftest.py` - Contains shared fixtures for use across test files
- `test_qa_service.py` - Tests for the QA service module
- `test_document_service.py` - Tests for the document service module

## Running Tests

### Running All Tests

To run all tests, execute the following command from the project root:

```bash
pytest
```

### Running Specific Test Files

To run tests from a specific file:

```bash
pytest tests/test_qa_service.py
```

### Running Tests with Verbosity

For more detailed output:

```bash
pytest -v
```

### Running Tests with Coverage

To generate a coverage report:

```bash
pytest --cov=services tests/
```

For a more detailed HTML coverage report:

```bash
pytest --cov=services --cov-report=html tests/
```

This will create a `htmlcov` directory with an interactive HTML report.

## Writing Tests

When adding new tests, follow these guidelines:

1. Use descriptive test names that explain what the test is checking
2. Each test should focus on a single functionality
3. Use fixtures from `conftest.py` when possible to minimize test setup duplication
4. Mock external dependencies to ensure tests are fast and reliable
5. Test both success and error cases
6. Add appropriate docstrings to test classes and methods 
import pytest
from src.validation import validate_mark, validate_upload
from src.image_processing import ImageError


@pytest.mark.parametrize("v,exp", [(None, None), ("", None), (0, 0), ("7", 7), (10, 10)])
def test_valid_marks(v, exp):
    assert validate_mark(v) == exp


@pytest.mark.parametrize("v", [11, -1, "abc", 3.7 if False else "x"])
def test_invalid_marks(v):
    with pytest.raises(ValueError):
        validate_mark(v)


def test_upload_rules():
    validate_upload("a.JPG", 1000)
    with pytest.raises(ImageError):
        validate_upload("a.pdf", 1000)
    with pytest.raises(ImageError):
        validate_upload("a.png", 100 * 1024 * 1024)

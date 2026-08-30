# test pii
from app.pii.scrubber import scrub


def test_redacts_email():
    result = scrub("Contact me at jane.doe@example.com please")
    assert "[REDACTED:email]" in result.text
    assert "jane.doe@example.com" not in result.text
    assert result.has_pii


def test_redacts_phone():
    result = scrub("Call 555-123-4567 tomorrow")
    assert "[REDACTED:phone]" in result.text


def test_redacts_ssn():
    result = scrub("SSN is 123-45-6789")
    assert "[REDACTED:ssn]" in result.text


def test_redacts_valid_credit_card_only():
    # Valid Luhn test number
    result = scrub("Card: 4111 1111 1111 1111")
    assert "[REDACTED:credit_card]" in result.text


def test_does_not_flag_random_digit_string_as_card():
    result = scrub("Order number 123456789012")
    assert "[REDACTED:credit_card]" not in result.text


def test_clean_text_untouched():
    result = scrub("What's the weather like today?")
    assert result.text == "What's the weather like today?"
    assert not result.has_pii


def test_empty_text():
    result = scrub("")
    assert result.text == ""
    assert not result.has_pii

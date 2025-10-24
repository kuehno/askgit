def sanitize_text_for_postgres(text: str | None) -> str | None:
    """Remove NUL bytes from text to make it compatible with PostgreSQL.

    PostgreSQL text fields cannot contain NUL (0x00) bytes. This function
    removes them from the input text to prevent database errors.

    Args:
        text: The text to sanitize, or None.

    Returns:
        Sanitized text with NUL bytes removed, or None if input was None.
    """
    if text is None:
        return None
    return text.replace("\x00", "")

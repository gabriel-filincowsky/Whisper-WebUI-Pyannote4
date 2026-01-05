import os
from pathlib import Path

def get_huggingface_token():
    """
    Retrieve Hugging Face authentication token.
    Required for accessing pyannote models.
    
    Returns
    -------
    str
        Hugging Face token
        
    Raises
    ------
    ValueError
        If token cannot be found and user doesn't provide one
    """
    # Check environment variable
    token = os.getenv('HF_TOKEN')
    if token:
        return token

    # Check token file
    token_file = Path.home() / '.huggingface' / 'token'
    if token_file.exists():
        return token_file.read_text().strip()

    # Return None if not found - let the calling code handle prompting
    return None

HF_TOKEN = get_huggingface_token()

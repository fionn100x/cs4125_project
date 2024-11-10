#Any extra functionality that need to be reused will go here# FILE: utils.py
import ast
import json

def string2any(s: str) -> dict:
    if not isinstance(s, str):
        return s

    # Remove whitespace
    s = s.strip()

    # Handle empty string
    if not s:
        return {}

    # Try to evaluate as Python literal
    try:
        # Handle dict-like strings
        if s.startswith('{') and s.endswith('}'):
            return json.loads(s.replace("'", '"'))
        
        # Try evaluating as Python literal
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # If not a valid Python literal, try common string conversions
        s_lower = s.lower()
        if s_lower == 'true':
            return True
        elif s_lower == 'false':
            return False
        elif s_lower == 'null' or s_lower == 'none':
            return None
        
        # Try number conversion
        try:
            if '.' in s:
                return float(s)
            return int(s)
        except ValueError:
            # Return original string if no conversion possible
            return s

    # If all conversions fail, return empty dict
    return {}
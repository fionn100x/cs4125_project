# Utility methods for reuse

def string2any(value):
    # Convert string to appropriate Python type (placeholder)
    try:
        return eval(value)
    except:
        return value

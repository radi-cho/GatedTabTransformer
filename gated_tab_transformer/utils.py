def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val
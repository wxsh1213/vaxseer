def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ('false', 'no', '0'):
        return False
    elif v.lower() in ('true', 'yes', '1'):
        return True
    else:
        return ValueError("Unknown value: %s. Excepting (false, no, 0) or (true, yes, 1)." % v)
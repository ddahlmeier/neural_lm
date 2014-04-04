
def uniq(sequence, preserve_order = False):
    if preserve_order:
        seen = set()
        uniq_values = [x for x in sequence if not x in seen and not seen.add(x)]
    else:
        uniq_values = [x for x in set(sequence)]
    return uniq_values



"""Validate scan-specific processing roles independently of display controls."""


def signal_indices(count, cell, requested=None):
    if type(cell) is not int or not 0<=cell<count:
        raise ValueError('Cell channel does not exist in this scan.')
    if requested is None:return [i for i in range(count) if i!=cell]
    if any(type(i) is not int or not 0<=i<count for i in requested):
        raise ValueError('A selected signal channel does not exist in this scan.')
    if len(requested)!=len(set(requested)):
        raise ValueError('A signal channel was selected more than once.')
    selected=[i for i in requested if i!=cell]
    if not selected:raise ValueError('Select at least one signal channel different from the cell channel.')
    return selected

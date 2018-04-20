
def mvectorize(func, palloc):
    """Memory preserving vectorization."""

    def a(x):
        """apply on vector x and store in s."""
        for i, v in enumerate(x):
            palloc[i] = func(v)

        return palloc 
    
    return a




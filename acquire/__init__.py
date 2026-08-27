"""Independent data-acquisition engine.

Collects publicly available football data, stores it raw and normalised, and
stops there. It does not import model code, compute features, or write model
inputs — the modelling engine reads these tables, never the other way round.

    SOURCE -> RAW -> PARSE -> NORMALISE -> RESOLVE -> VALIDATE -> DATABASE
                                                                      |
                                                        (the model reads it)

The only dependency on ``fpl_engine`` is ``db.connect`` for the SQLite handle,
so that one database serves both rather than a second, unrelated one. No model
logic is imported and none may be.
"""
__version__ = "0.1"

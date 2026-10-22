# Contributing

Start from the documented support matrix. Changes to scientific behavior require
a stated model, upstream source/citation review, an R reference generator, and
explicit numerical acceptance criteria. Preserve failed cases rather than
silently relaxing tolerances. Add unit or invariant tests for meaningful behavior,
especially boundaries, identifiers, and model nesting.

```sh
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m build
```

Keep Python runtime installation independent of R. R is required only to
regenerate reference evidence. Record source and dataset licenses, retain
attribution, and submit changes under the project's GPL-3.0-or-later terms.
Claims of expanded support must include validation and independent review.

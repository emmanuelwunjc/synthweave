"""Optional connectors to external data sources.

Nothing here is imported by `synthweave/__init__.py`. A connector fetches a
plain `pandas.DataFrame` and hands it to the public API the same way any
other real data would arrive, e.g. `sw.Empirical(fetch_pums(...))`. Kept
separate from the core namespace because it does network I/O and the core
package does not.

`faker_names` (`sw.connectors.faker_names.Name`, `.SSN`) is not imported
here, deliberately: it needs the optional `pii` extra
(`pip install "synthweave[pii]"`), and this package must stay importable
without it. Import it directly:
`from synthweave.connectors.faker_names import Name, SSN`.
"""

from .acs_pums import fetch_pums

__all__ = ["fetch_pums"]

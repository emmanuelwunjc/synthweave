# Security

synthweave has no network-facing service and no auth surface; the closest
thing to an attack surface is the data connectors (Census/PUMS, SSA,
GeoNames, Faker), which parse third-party input files.

To report a vulnerability, use
[GitHub's private vulnerability reporting](https://github.com/emmanuelwunjc/synthweave/security/advisories/new)
rather than a public issue.

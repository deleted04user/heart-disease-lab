# Reserved cohort transparency

The reserved artifact is the untouched **raw file within this project**, not a
blinded clinical trial cohort. Official public UCI documentation was consulted
to identify Long Beach VA and contains historical aggregate cohort counts.
No schema, labels, distributions, missingness, predictions or performance have
been computed from the reserved bytes by this project.

The declared experiment protocol is preserved byte-for-byte at the SHA-256
recorded in `reports/reserved_cohort.json`. The download occurred only after
that protocol was written. The ordinary data loader rejects both the reserved
path and renamed copies matching the recorded hash. The raw file is excluded
from Git and release archives. A future confirmation run requires an explicit
protocol-controlled release of those bytes; no current model is evaluated on it.

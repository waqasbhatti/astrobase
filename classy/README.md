### CLASSY: a variable star classifier

Expected features:

`classy-process`:

- run on a directory containing light curves to generate LSP pkls and
  variability info
- ingest these into a classy-db.sqlite file if set to do so, otherwise, ingest
  them into a postgres table defined in classy.conf


`classy-server`:

- replicate the current checkplot functionality
- add in objectinfo from HAT data server
- add in stamps from HAT data server
- add in ability to redo LSP, AoV-harm, and BLS
- classify variables automatically by RF machine learning

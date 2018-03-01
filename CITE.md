## Citing astrobase

[Released versions](https://github.com/waqasbhatti/astrobase/releases) of
**astrobase** are archived at [the Zenodo
repository](https://doi.org/10.5281/zenodo.1011188). Zenodo provides a DOI that
can be cited for each specific version. The following `bibtex` entry for
**astrobase** v0.3.8 may be useful as a template. You can substitute in values
of `month`, `year`, `version`, `doi`, and `url` for the version of `astrobase`
you used for your publication.

```tex
@misc{wbhatti_astrobase,
  author       = {Waqas Bhatti and
                  Luke G. Bouma and
                  Joshua Wallace},
  title        = {\texttt{astrobase}},
  month        = feb,
  year         = 2018,
  version      = {0.3.8},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.1185231},
  url          = {https://doi.org/10.5281/zenodo.1185231}
}
```

Alternatively, the following `bibtex` entry can be used for all versions of
**astrobase** (the DOI will always resolve to the latest version):

```tex
@misc{wbhatti_astrobase,
  author       = {Waqas Bhatti and
                  Luke G. Bouma and
                  Joshua Wallace},
  title        = {\texttt{astrobase}},
  month        = oct,
  year         = 2017,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.1011188},
  url          = {https://doi.org/10.5281/zenodo.1011188}
}
```

Also see this [AAS Journals note on citing repositories](https://github.com/AASJournals/Tutorials/blob/master/Repositories/CitingRepositories.md).

## Period-finder algorithms

If you use any of the period-finder methods implemented by
**[astrobase.periodbase](astrobase/periodbase)**, please also make sure to cite
their respective papers as well.

- the generalized Lomb-Scargle algorithm from Zechmeister & Kurster
  ([2008](http://adsabs.harvard.edu/abs/2009A%26A...496..577Z))
- the phase dispersion minimization algorithm from Stellingwerf
  ([1978](http://adsabs.harvard.edu/abs/1978ApJ...224..953S),
  [2011](http://adsabs.harvard.edu/abs/2011rrls.conf...47S))
- the AoV and AoV-multiharmonic algorithms from Schwarzenberg-Czerny
  ([1989](http://adsabs.harvard.edu/abs/1989MNRAS.241..153S),
  [1996](http://adsabs.harvard.edu/abs/1996ApJ...460L.107S))
- the BLS algorithm from Kovacs et
  al. ([2002](http://adsabs.harvard.edu/abs/2002A%26A...391..369K))
- the ACF period-finding algorithm from McQuillan et
  al. ([2013a](http://adsabs.harvard.edu/abs/2013MNRAS.432.1203M),
  [2014](http://adsabs.harvard.edu/abs/2014ApJS..211...24M))

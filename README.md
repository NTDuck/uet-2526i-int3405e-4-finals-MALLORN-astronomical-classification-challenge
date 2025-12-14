# uet-2526i-int3405e-4-finals-MALLORN-astronomical-classification-challenge
> https://www.kaggle.com/competitions/mallorn-astronomical-classification-challenge/

## Prerequisites
- Python 3.14
- Poetry 2.2.1

## How to install
```cmd
$ poetry install
$ poetry run pre-commit install
```

```cmd
$ poetry run mallorn ingest
$ poetry run mallorn featurize
```

## How to run
...

## Key takeaways
- The continuum of TDEs is well described by a thermal blackbody.
- At face value, they do appear to have a light curve that follows the general shape of the theoretical TDE fallback rate. In fact, when one fits a t^-5/3 power law to the light curve on its decline from peak, there is a strong correlation with the time of peak since the inferred time of disruption, delta_t = t_peak - t_D, and the estimated black hole mass, where delta_t ^^ M^1/2_BH, as would be expected for the fallback timescale. (partial TDEs fits t^-9/4 though...)

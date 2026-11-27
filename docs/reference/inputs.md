# Input columns

DECIMA reads any table astropy reads (FITS, CSV, ECSV, VOTable, HDF5, Parquet), or an astropy `Table` or pandas `DataFrame`. It looks for the column names below. If your table uses other names, map them with a **columns file** (`decima columns` writes one for you, see [Make your columns file](../how-to/config-files.ipynb)); the file maps *DECIMA's name → your name*.

```json
{"teff": "TEFF", "teff_error": "e_teff", "Ks": "ks_m"}
```

Names are case-sensitive: `Ks`, not `ks`. Use NaN (or a masked value) for anything not measured, not a flag such as `-9999` or `99`: `decima check` warns about those.

<!-- BEGIN GENERATED: inputs -->
<!-- written by docs/gen_reference.py: edit the code, not this table -->

### Required

| Column | Unit | Meaning |
|---|---|---|
| `parallax` | mas | Gaia parallax (zero-point corrected, or see `parallax_zero_point`). |
| `parallax_error` | mas | Uncertainty on parallax. |
| `teff` | K | Spectroscopic effective temperature. |
| `teff_error` | K | Uncertainty on T_eff. Used as given (no floor). |
| `feh` | dex | Spectroscopic surface [Fe/H]. |
| `feh_error` | dex | Uncertainty on [Fe/H]. Used as given (no floor). |
| `Ks` | mag | 2MASS Ks apparent magnitude (required: it gives the pseudo-magnitude). |
| `Ks_error` | mag | Uncertainty on 2MASS Ks, floored at `twomass_err_floor`; a magnitude without one gets `missing_err`. |

### Position (ra and dec, or L, alpha and beta)

| Column | Unit | Meaning |
|---|---|---|
| `ra` | deg | Right ascension (ICRS), for the Bailer-Jones+21 distance prior. |
| `dec` | deg | Declination (ICRS), for the Bailer-Jones+21 distance prior. |
| `L` | pc | Per-star Bailer-Jones+21 prior length scale; with `alpha` and `beta` it replaces the lookup from ra, dec. |
| `alpha` |  | Per-star Bailer-Jones+21 prior shape parameter (with `L`, `beta`). |
| `beta` |  | Per-star Bailer-Jones+21 prior shape parameter (with `L`, `alpha`). |

### Optional

| Column | Unit | Meaning |
|---|---|---|
| `id` |  | Star identifier. Without it the summary's `id` is the 0-based row of the input table. |
| `alpha_fe` | dex | Spectroscopic [alpha/Fe]. Missing values are filled with `alpha_fe_fill` (default 0) +- `alpha_fe_err_fill` (default 0.2). |
| `alpha_fe_error` | dex | Uncertainty on [alpha/Fe]. Used as given (no floor). |
| `logg` | dex | Spectroscopic log g (cgs). Used for the pseudo-magnitude inflation, and in stage 2 only with `use_logg`. |
| `logg_error` | dex | Uncertainty on log g. |
| `G` | mag | Gaia G apparent magnitude. NaN or masked: the band is skipped for that star. |
| `BP` | mag | Gaia BP apparent magnitude. NaN or masked: the band is skipped for that star. |
| `RP` | mag | Gaia RP apparent magnitude. NaN or masked: the band is skipped for that star. |
| `J` | mag | 2MASS J apparent magnitude. NaN or masked: the band is skipped for that star. |
| `H` | mag | 2MASS H apparent magnitude. NaN or masked: the band is skipped for that star. |
| `G_error` | mag | Uncertainty on Gaia G, floored at `gaia_err_floor`; a magnitude without one gets `missing_err`. |
| `BP_error` | mag | Uncertainty on Gaia BP, floored at `gaia_err_floor`; a magnitude without one gets `missing_err`. |
| `RP_error` | mag | Uncertainty on Gaia RP, floored at `gaia_err_floor`; a magnitude without one gets `missing_err`. |
| `J_error` | mag | Uncertainty on 2MASS J, floored at `twomass_err_floor`; a magnitude without one gets `missing_err`. |
| `H_error` | mag | Uncertainty on 2MASS H, floored at `twomass_err_floor`; a magnitude without one gets `missing_err`. |

### Parallax zero-point (only with `parallax_zero_point=True`)

| Column | Unit | Meaning |
|---|---|---|
| `nu_eff_used_in_astrometry` | 1/um | Gaia archive column; only for `parallax_zero_point` (which also needs `G`). |
| `pseudocolour` | 1/um | Gaia archive column; only for `parallax_zero_point`. |
| `ecl_lat` | deg | Gaia archive column (ecliptic latitude); only for `parallax_zero_point`. |
| `astrometric_params_solved` |  | Gaia archive column (31 = 5-parameter, 95 = 6-parameter solution); only for `parallax_zero_point`. |

<!-- END GENERATED: inputs -->

## Missing values and rejected stars

- **Missing magnitude:** NaN or masked. That band is skipped for that star.
- **Magnitude without an error:** gets the error `missing_err` (0.5 mag by default; `0` uses the error floor instead).
- **Error floors:** G, BP, RP errors are floored at `gaia_err_floor`, J, H, Ks at `twomass_err_floor`. Spectroscopic errors are used as given.
- **Missing [α/Fe]** (or its error): filled with `alpha_fe_fill` ± `alpha_fe_err_fill` (0 ± 0.2 by default).
- **Rejected stars:** a star is not run, and is listed in `rejected` with the reason, if it has no value for a required column (`no teff`, `no Ks`, …), `parallax_error <= 0`, or no position (`no ra, dec (or L, alpha, beta)`). Only the first reason is recorded. Negative parallaxes are allowed.
- **Row numbers:** rejected stars are left out of the summary, so its rows do not line up with the input rows. Map an `id` column to match them up; without one, the summary's `id` is the 0-based input row.

## Parallax

Used as given, or with `parallax_zero_point=True` DECIMA subtracts the Lindegren+21 zero-point itself ([details](settings.md#the-parallax-zero-point)). Give DECIMA one or the other: do not correct a parallax twice.

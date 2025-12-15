import warnings
from pathlib import Path

import feets
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from typer import Typer

from .augment import load_meta_aug_df as _load_meta_aug_df
from .augment import load_obs_aug_df as _load_obs_aug_df
from .common import AUG_DIR, FEATS_DIR, DfType, mkdir, mksb


warnings.filterwarnings("ignore", category=OptimizeWarning)

typer = Typer()


@typer.command()
def featurize(df_type: DfType, aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
    with mksb() as sb:
        with sb.job("Loading augmented datasets"):
            meta_df = _load_meta_aug_df(df_type, aug_dir)
            obs_df = _load_obs_aug_df(df_type, aug_dir)

            feats_df = meta_df.drop(columns=["SpecType", "English Translation", "split"])

        with sb.job("Merging augmented datasets"):
            mrg_df = obs_df.merge(meta_df, left_on="object_id", right_index=True)
            mrg_df.sort_values(["object_id", "filter", "mjd"], inplace=True)
            mrg_df.dropna(subset=["flux", "flux_err"], inplace=True)

        with sb.job("Extracting per-filter features"):
            per_filt_feats = mrg_df.groupby(["object_id", "filter"])[["mjd", "flux", "flux_err"]].apply(_extract_per_filt).unstack("filter")
            per_filt_feats.columns = [f"{feat}_{filt}" for feat, filt in per_filt_feats.columns]

            feats_df = feats_df.join(per_filt_feats)

        with sb.job("Extracting multi-filter features"):
            multi_filt_feats = mrg_df.groupby("object_id")[["filter", "mjd", "flux", "flux_err"]].apply(_extract_multi_filt)

            feats_df = feats_df.join(multi_filt_feats)

        with sb.job("Saving featurized dataset"):
            feats_df.to_parquet(mkdir(feats_dir) / f"{df_type}_feats.parquet")


def _extract_per_filt(mrg_df: pd.DataFrame) -> pd.Series:
    mjd = mrg_df["mjd"].to_numpy(dtype=np.float64)
    flux = mrg_df["flux"].to_numpy(dtype=np.float64)
    flux_err = mrg_df["flux_err"].to_numpy(dtype=np.float64)

    bazin_feats = _extract_bazin_fit(mjd, flux, flux_err)
    villar_feats = _extract_villar_fit(mjd, flux, flux_err)
    power_law_feats = _extract_power_law(mjd, flux, flux_err)
    stoch_feats = _extract_stochasticity(flux, flux_err)

    chi2_rat = power_law_feats["TDE_ReducedChi2"] / (bazin_feats["BazinFit_ReducedChi2"] + np.finfo(float).eps)
    chi2_rat_feats = pd.Series([chi2_rat], index=["Likelihood_ReducedChi2_Ratio_TDE_SN"])

    return pd.concat([bazin_feats, villar_feats, power_law_feats, stoch_feats, chi2_rat_feats])


def _extract_bazin_fit(mjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> pd.Series:
    try:
        feats = feets.extractors.BazinFit().extract(time=mjd, flux=flux, flux_error=flux_err)
        return pd.Series(feats)
    except:
        return pd.Series(np.nan, index=["BazinFit_Amplitude", "BazinFit_Baseline", "BazinFit_ReferenceTime", "BazinFit_RiseTime", "BazinFit_FallTime", "BazinFit_ReducedChi2"])


def _extract_villar_fit(mjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> pd.Series:
    try:
        feats = feets.extractors.VillarFit().extract(time=mjd, flux=flux, flux_error=flux_err)
        return pd.Series(feats)
    except:
        return pd.Series(
            np.nan,
            index=[
                "VillarFit_Amplitude",
                "VillarFit_Baseline",
                "VillarFit_ReferenceTime",
                "VillarFit_RiseTime",
                "VillarFit_FallTime",
                "VillarFit_PlateauRelAmplitude",
                "VillarFit_PlateauDuration",
                "VillarFit_ReducedChi2",
            ],
        )


def _extract_power_law(mjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray) -> pd.Series:
    if len(mjd) < _POWER_LAW_MIN_NROWS:
        return pd.Series(np.nan, index=["TDE_Alpha", "TDE_RiseTime", "TDE_ReducedChi2"])

    pk_idx = np.argmax(flux)
    mjd_pk = mjd[pk_idx]
    flux_pk = flux[pk_idx]

    dcy_mask = mjd >= mjd_pk
    mjd_dcy = mjd[dcy_mask]
    flux_dcy = flux[dcy_mask]
    flux_err_dcy = flux_err[dcy_mask]

    try:
        (mjd_0, mjd_pk, flux_pk, α, β), _ = curve_fit(
            f=_tde_dcy,
            xdata=mjd_dcy,
            ydata=flux_dcy,
            sigma=flux_err_dcy,
            p0=[mjd_pk - 20, mjd_pk, flux_pk, 1.67, 0],
            bounds=(
                [-np.inf, mjd_pk - 1, 0, 0.1, -np.inf],
                [mjd_dcy.min(), mjd_pk + 1, np.inf, 5.0, np.inf],
            ),
            max_nfev=5000,
        )

        mjd_rise = mjd_pk - mjd_0
        ε = flux_dcy - _tde_dcy(mjd_dcy, mjd_0, mjd_pk, flux_pk, α, β)
        chi2 = np.sum(np.pow(ε / (flux_err_dcy + np.finfo(float).eps), 2) / (len(mjd_dcy) - _POWER_LAW_MIN_NROWS + 1))

        return pd.Series([α, mjd_rise, chi2], index=["TDE_Alpha", "TDE_RiseTime", "TDE_ReducedChi2"])
    except:
        return pd.Series(np.nan, index=["TDE_Alpha", "TDE_RiseTime", "TDE_ReducedChi2"])


def _tde_dcy(mjd, mjd_0, mjd_pk, flux_pk, α, β):
    if mjd_pk <= mjd_0:
        return np.full_like(mjd, np.inf)
    return flux_pk * np.pow((mjd - mjd_0) / (mjd_pk - mjd_0), -α) + β


_POWER_LAW_MIN_NROWS = 4


def _extract_stochasticity(flux: np.ndarray, flux_err: np.ndarray) -> pd.Series:
    if len(flux) < _STOCHASTICITY_MIN_NROWS:
        return pd.Series(np.nan, index=["VonNeumannRatio", "ExcessVariance"])

    vn_rat = np.mean(np.pow(np.diff(flux), 2) / (np.var(flux) + np.finfo(float).eps))
    excess_var = (np.var(flux) - np.mean(np.pow(flux_err, 2))) / (np.pow(np.mean(flux), 2) + np.finfo(float).eps)

    return pd.Series([vn_rat, excess_var], index=["VonNeumannRatio", "ExcessVariance"])


_STOCHASTICITY_MIN_NROWS = 3


def _extract_multi_filt(mrg_df: pd.DataFrame) -> pd.Series:
    filts = ["g", "r", "i"]
    gp_fluxes = {}

    mjd_min, mjd_max = mrg_df["mjd"].min(), mrg_df["mjd"].max()
    mjd_grid = np.linspace(mjd_min, mjd_max)

    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=20.0, length_scale_bounds=(1, 1e2)) + WhiteKernel(
        noise_level=1.0, noise_level_bounds=(np.finfo(float).eps, 10)
    )

    for filt in filts:
        filt_df = mrg_df[mrg_df["filter"] == filt]
        if len(filt_df) < _MULTI_FILT_MIN_NROWS:
            gp_fluxes[filt] = None
            continue

        X = filt_df[["mjd"]].to_numpy() - mjd_min
        y = filt_df["flux"].to_numpy()

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            gp_reg = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=1)
            gp_reg.fit(X, y)
            gp_fluxes[filt] = gp_reg.predict((mjd_grid - mjd_min).reshape(-1, 1))
        except:
            gp_fluxes[filt] = None

    bb_evo_feats = _extract_bb_evo(mjd_grid, gp_fluxes)
    col_evo_feats = _extract_col_evo(mjd_grid, gp_fluxes)

    return pd.concat([bb_evo_feats, col_evo_feats])


def _extract_bb_evo(mjd_grid: np.ndarray, fluxes: dict) -> pd.Series:
    temps = []

    for idx in range(len(mjd_grid)):
        x_wave = []
        y_flux = []
        for band in ["g", "r", "i"]:
            if fluxes[band] is not None:
                x_wave.append(_EFF_WLS[band])
                y_flux.append(fluxes[band][idx])

        if len(x_wave) >= 3:
            try:
                popt, _ = curve_fit(_planck, x_wave, y_flux, p0=[20000, 1e10], bounds=([1000, 0], [100000, np.inf]), maxfev=500)
                temps.append(popt[0])
            except:
                pass

    if not temps:
        return pd.Series(np.nan, index=["Blackbody_Temp_Mean", "Blackbody_Temp_Max", "Blackbody_Temp_Slope"])

    slope = 0
    if len(temps) > 2:
        slope = np.polyfit(np.arange(len(temps)), temps, 1)[0]

    return pd.Series([np.mean(temps), np.max(temps), slope], index=["Blackbody_Temp_Mean", "Blackbody_Temp_Max", "Blackbody_Temp_Slope"])


def _extract_col_evo(mjd_grid: np.ndarray, fluxes: dict) -> pd.Series:
    idx = ["Color_Slope_g_r", "Color_Mean_g_r"]

    if fluxes["g"] is None or fluxes["r"] is None:
        return pd.Series(np.nan, index=idx)

    flux_g = fluxes["g"]
    flux_r = fluxes["r"]

    valid = flux_r != 0
    if np.sum(valid) < 3:
        return pd.Series(np.nan, index=idx)

    # Using offset magnitude: -2.5 * log10((g + C) / (r + C)) to handle negatives
    C = 100
    color_curve = -2.5 * np.log10(np.abs((flux_g + C) / (flux_r + C)))

    slope = np.polyfit(mjd_grid, color_curve, 1)[0]

    return pd.Series([slope, np.mean(color_curve)], index=idx)


def _planck(lam, T, A):
    c2 = 1.4388e8  # hc/k
    val = np.clip(c2 / (lam * T), -100, 100)
    return A / (np.power(lam, 5) * (np.exp(val) - 1))


_MULTI_FILT_MIN_NROWS = 3
_EFF_WLS = {
    "u": 3685.0,
    "g": 4802.0,
    "r": 6231.0,
    "i": 7542.0,
    "z": 8690.0,
    "y": 9736.0,
}


def load_feats_df(df_type: DfType, feats_dir: Path = FEATS_DIR, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(feats_dir / f"{df_type}_feats.parquet", **kwargs)

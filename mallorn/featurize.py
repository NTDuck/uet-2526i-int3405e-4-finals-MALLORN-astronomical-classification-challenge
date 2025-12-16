import warnings
from pathlib import Path

import feets
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import kurtosis, skew
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from typer import Typer

from .augment import load_meta_aug_df as _load_meta_aug_df
from .augment import load_obs_aug_df as _load_obs_aug_df
from .common import AUG_DIR, FEATS_DIR, DfType, mkdir, mkpb, mksb


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

typer = Typer()


@typer.command(name="featurize-all")
def featurize_all(aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
    with mksb() as sb:
        with sb.job("Calling `featurize(df_type='train')`"):
            featurize(df_type="train", aug_dir=aug_dir, feats_dir=feats_dir)

        with sb.job("Calling `featurize(df_type='test')`"):
            featurize(df_type="test", aug_dir=aug_dir, feats_dir=feats_dir)


@typer.command()
def featurize(df_type: DfType, aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
    with mksb() as sb:
        with sb.job("Loading augmented datasets"):
            meta_df = _load_meta_aug_df(df_type, aug_dir)
            obs_df = _load_obs_aug_df(df_type, aug_dir)

            feats_df = meta_df.drop(columns=["SpecType", "English Translation", "split"], errors="ignore")

        with sb.job("Merging augmented datasets"):
            mrg_df = obs_df.merge(meta_df[["z", "μ"]], left_on="object_id", right_index=True)
            mrg_df.sort_values(["object_id", "filter", "mjd"], inplace=True)
            mrg_df.dropna(subset=["flux", "flux_err", "flux_lupt"], inplace=True)

        with mkpb(total=len(meta_df) * 6) as pb:  # 6 filters

            def _extract_per_filt_(df: pd.DataFrame) -> pd.Series:
                with pb.job(f"Extracting per-filter features for `{df['object_id'].iloc[0]}`"):
                    return _extract_per_filt(df)

            per_filt_feats = mrg_df.groupby(["object_id", "filter"]).apply(_extract_per_filt_).unstack(["filter"])
            per_filt_feats.columns = [f"{filt}_{feat}" for filt, feat in per_filt_feats.columns]

            feats_df = feats_df.join(per_filt_feats)

        with mkpb(total=len(meta_df)) as pb:

            def _extract_multi_filt_(df: pd.DataFrame) -> pd.Series:
                with pb.job(f"Extracting multi-filter features for `{df['object_id'].iloc[0]}`"):
                    return _extract_multi_filt(df)

            multi_filt_feats = mrg_df.groupby("object_id").apply(_extract_multi_filt_)
            feats_df = feats_df.join(multi_filt_feats)

        with sb.job(f"Saving featurized dataset to {(feats_dir / f'{df_type}_feats.parquet').as_posix()}"):
            feats_df.to_parquet(mkdir(feats_dir) / f"{df_type}_feats.parquet")


def _extract_per_filt(df: pd.DataFrame) -> pd.Series:
    z = df["z"].iloc[0]
    μ = df["μ"].iloc[0]

    mjd = df["mjd"].to_numpy(dtype=np.float64)
    flux = df["flux"].to_numpy(dtype=np.float64)
    flux_err = df["flux_err"].to_numpy(dtype=np.float64)
    flux_lupt = df["flux_lupt"].to_numpy(dtype=np.float64)

    stats = _extract_stats(mjd, flux, flux_lupt, z, μ)

    bazin_fit = _extract_bazin_fit(mjd, flux, flux_err, z)
    viller_fit = _extract_villar_fit(mjd, flux, flux_err, z)
    power_law = _extract_power_law(mjd, flux, flux_err, z)
    stoch = _extract_stoch(flux, flux_err)

    chi2_rat = power_law["TDE_ReducedChi2"] / (bazin_fit["BazinFit_ReducedChi2"] + np.finfo(np.float64).eps)
    chi2_feat = pd.Series([chi2_rat], index=["Likelihood_ReducedChi2_Ratio_TDE_SN"])

    return pd.concat([stats, bazin_fit, viller_fit, power_law, stoch, chi2_feat])


def _extract_stats(mjd: np.ndarray, flux: np.ndarray, flux_lupt: np.ndarray, z: float, μ: float) -> pd.Series:
    if len(mjd) < _STATS_MIN_NROWS:
        return pd.Series(
            np.nan,
            index=[
                "Flux_Mean",
                "Flux_Std",
                "Flux_Skew",
                "Flux_Kurt",
                "Flux_Q10",
                "Flux_Q90",
                "Flux_Max_Med_Ratio",
                "Time_Span_Rest",
                "Rise_Fall_Ratio_Simple",
                "FWHM_Rest",
                "FluxLupt_Mean",
                "FluxLupt_Std",
                "FluxLupt_Skew",
                "FluxLupt_Kurt",
                "AbsFluxLupt_Mean",
                "AbsFluxLupt_Peak",
            ],
        )

    flux_mean = np.mean(flux)
    flux_median = np.median(flux)
    flux_max = np.max(flux)

    mjd_span_rest = (mjd.max() - mjd.min()) / (1 + z)

    half_max = (flux_max + np.min(flux)) / 2
    fwhm_mask = flux >= half_max
    if np.any(fwhm_mask):
        fwhm_rest = (mjd[fwhm_mask].max() - mjd[fwhm_mask].min()) / (1 + z)
    else:
        fwhm_rest = np.nan

    flux_pk_idx = np.argmax(flux)
    if 0 < flux_pk_idx < len(flux) - 1:
        mjd_rise = (mjd[flux_pk_idx] - mjd[0]) / (1 + z)
        mjd_fall = (mjd[-1] - mjd[flux_pk_idx]) / (1 + z)
        rise_fall_rat = mjd_rise / (mjd_fall + np.finfo(np.float64).eps)
    else:
        rise_fall_rat = np.nan

    flux_lupt_mean = np.mean(flux_lupt)

    abs_flux_lupt_mean = flux_lupt_mean - μ
    abs_flux_lupt_peak = np.max(flux_lupt) - μ

    return pd.Series(
        {
            "Flux_Mean": flux_mean,
            "Flux_Std": np.std(flux, ddof=1),
            "Flux_Skew": skew(flux),
            "Flux_Kurt": kurtosis(flux),
            "Flux_Q10": np.percentile(flux, 10),
            "Flux_Q90": np.percentile(flux, 90),
            "Flux_Max_Med_Ratio": flux_max / (flux_median + np.finfo(np.float64).eps),
            "Time_Span_Rest": mjd_span_rest,
            "Rise_Fall_Ratio_Simple": rise_fall_rat,
            "FWHM_Rest": fwhm_rest,
            "FluxLupt_Mean": flux_lupt_mean,
            "FluxLupt_Std": np.std(flux_lupt, ddof=1),
            "FluxLupt_Skew": skew(flux_lupt),
            "FluxLupt_Kurt": kurtosis(flux_lupt),
            "AbsFluxLupt_Mean": abs_flux_lupt_mean,
            "AbsFluxLupt_Peak": abs_flux_lupt_peak,
        }
    )


_STATS_MIN_NROWS = 3


def _extract_bazin_fit(mjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, z: float) -> pd.Series:
    try:
        # Normalize flux with linear scaling
        N = np.max(np.abs(flux)) + np.finfo(np.float64).eps
        feats = feets.extractors.BazinFit().extract(time=mjd, flux=flux / N, flux_error=flux_err / N)

        # Re-scale
        feats["BazinFit_RiseTime"] /= 1 + z
        feats["BazinFit_FallTime"] /= 1 + z
        feats["BazinFit_Amplitude"] *= N
        feats["BazinFit_Baseline"] *= N

        return pd.Series(feats)

    except:
        return pd.Series(np.nan, index=["BazinFit_Amplitude", "BazinFit_Baseline", "BazinFit_ReferenceTime", "BazinFit_RiseTime", "BazinFit_FallTime", "BazinFit_ReducedChi2"])


def _extract_villar_fit(mjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, z: float) -> pd.Series:
    try:
        # Normalize flux with linear scaling
        N = np.max(np.abs(flux)) + np.finfo(np.float64).eps
        feats = feets.extractors.VillarFit().extract(time=mjd, flux=flux / N, flux_error=flux_err / N)

        # Re-scale
        feats["VillarFit_RiseTime"] /= 1 + z
        feats["VillarFit_FallTime"] /= 1 + z
        feats["VillarFit_PlateauDuration"] /= 1 + z
        feats["VillarFit_Amplitude"] *= N
        feats["VillarFit_Baseline"] *= N

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


def _extract_power_law(mjd: np.ndarray, flux: np.ndarray, flux_err: np.ndarray, z: float) -> pd.Series:
    if len(mjd) < _POWER_LAW_MIN_NROWS:
        return pd.Series(np.nan, index=["TDE_Alpha", "TDE_RiseTime", "TDE_ReducedChi2"])

    flux_pk_idx = np.argmax(flux)
    mjd_pk = mjd[flux_pk_idx]
    flux_pk = flux[flux_pk_idx]

    dcy_mask = mjd >= mjd_pk
    mjd_dcy = mjd[dcy_mask]
    flux_dcy = flux[dcy_mask]
    flux_err_dcy = flux_err[dcy_mask]

    if len(mjd_dcy) < _POWER_LAW_MIN_NROWS:
        return pd.Series(np.nan, index=["TDE_Alpha", "TDE_RiseTime", "TDE_ReducedChi2"])

    try:
        p0 = [mjd_pk - 10, mjd_pk, flux_pk, 1.67, 0]
        bounds = ([-np.inf, mjd_pk - 1, 0, 0.1, -np.inf], [mjd_dcy.min(), mjd_pk + 1, np.inf, 5.0, np.inf])

        (mjd_0, mjd_pk_fit, flux_pk_fit, α, β), _ = curve_fit(f=_tde_dcy, xdata=mjd_dcy, ydata=flux_dcy, sigma=flux_err_dcy, p0=p0, bounds=bounds, max_nfev=5000)

        mjd_rise = (mjd_pk_fit - mjd_0) / (1 + z)
        ε = flux_dcy - _tde_dcy(mjd_dcy, mjd_0, mjd_pk_fit, flux_pk_fit, α, β)
        chi2 = np.sum(np.pow((ε / (flux_err_dcy + np.finfo(np.float64).eps)), 2)) / (len(mjd_dcy) - _POWER_LAW_MIN_NROWS - 1)

        return pd.Series(
            {
                "TDE_Alpha": α,
                "TDE_RiseTime": mjd_rise,
                "TDE_ReducedChi2": chi2,
            }
        )

    except:
        return pd.Series(np.nan, index=["TDE_Alpha", "TDE_RiseTime", "TDE_ReducedChi2"])


_POWER_LAW_MIN_NROWS = 3


def _tde_dcy(mjd, mjd_0, mjd_pk, flux_pk, α, β):
    with np.errstate(invalid="ignore", divide="ignore"):
        val = flux_pk * np.power((mjd - mjd_0) / (mjd_pk - mjd_0), -α) + β
    val = np.where(mjd < mjd_pk, np.inf, val)
    return val


def _extract_stoch(flux: np.ndarray, flux_err: np.ndarray) -> pd.Series:
    if len(flux) < _STOCH_MIN_NROWS:
        return pd.Series(np.nan, index=["VonNeumannRatio", "ExcessVariance"])

    vn_rat = np.mean(np.pow(np.diff(flux), 2)) / (np.var(flux) + np.finfo(np.float64).eps)
    excess_var = (np.var(flux) - np.mean(flux_err**2)) / (np.pow(np.mean(flux), 2) + np.finfo(np.float64).eps)

    return pd.Series(
        {
            "VonNeumannRatio": vn_rat,
            "ExcessVariance": excess_var,
        }
    )


_STOCH_MIN_NROWS = 3


def _extract_multi_filt(df: pd.DataFrame) -> pd.Series:
    z = df["z"].iloc[0]
    μ = df["μ"].iloc[0]

    filts = ["u", "g", "r", "i"]
    gp_fluxes = {}

    mjd_min, mjd_max = df["mjd"].min(), df["mjd"].max()
    mjd_grid = np.linspace(mjd_min, mjd_max, num=50)

    # kernel = ConstantKernel(1.0, (1e-3, 1e5)) * RBF(20.0, (1, 200)) + WhiteKernel(1.0, (1e-5, 100))
    kernel = ConstantKernel(1.0, (1e-3, 1e10)) * Matern(length_scale=20.0, length_scale_bounds=(1e-1, 1000), nu=1.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))

    for filt in filts:
        filt_df = df[df["filter"] == filt]
        if len(filt_df) < _MULTI_FILT_MIN_NROWS:
            gp_fluxes[filt] = None
            continue

        X = filt_df[["mjd"]].to_numpy() - mjd_min
        y = filt_df["flux"].to_numpy()

        try:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0)
            gp.fit(X, y)
            gp_fluxes[filt] = gp.predict((mjd_grid - mjd_min).reshape(-1, 1))
        except:
            gp_fluxes[filt] = None

    bb_evo = _extract_bb_evo(mjd_grid, gp_fluxes, z)
    col_evo = _extract_col_evo(mjd_grid, gp_fluxes, z)
    abs_mag = _extract_abs_mag(df, μ)

    return pd.concat([bb_evo, col_evo, abs_mag])


_MULTI_FILT_MIN_NROWS = 3


def _extract_bb_evo(mjd_grid: np.ndarray, fluxes: dict, z: float) -> pd.Series:
    temps = []
    filts = ["u", "g", "r", "i"]
    wls_rest = np.array([_EFF_WLS[filt] for filt in filts]) / (1 + z)

    for idx in range(len(mjd_grid)):
        y_flux = []
        valid_wls = []
        for j, filt in enumerate(filts):
            if fluxes[filt] is not None:
                valid_wls.append(wls_rest[j])
                y_flux.append(fluxes[filt][idx])

        if len(y_flux) >= _MULTI_FILT_MIN_NROWS and all(y > 0 for y in y_flux):
            try:
                popt, _ = curve_fit(_planck, valid_wls, y_flux, p0=[20000, 1e-15], bounds=([1000, 0], [100000, np.inf]), maxfev=100)
                temps.append(popt[0])
            except:
                pass

    if not temps:
        return pd.Series(np.nan, index=["BB_Temp_Mean", "BB_Temp_Slope"])

    slope = 0
    if len(temps) > _BB_EVO_MIN_NROWS:
        mjd_rest = (np.arange(len(temps)) * (mjd_grid[1] - mjd_grid[0])) / (1 + z)
        slope = np.polyfit(mjd_rest, temps, 1)[0]

    return pd.Series(
        {
            "BB_Temp_Mean": np.mean(temps),
            "BB_Temp_Slope": slope,
        }
    )


_BB_EVO_MIN_NROWS = 5


def _extract_col_evo(mjd_grid: np.ndarray, fluxes: dict, z: float) -> pd.Series:
    filt_pairs = [("g", "r"), ("u", "r")]
    feats = {}

    for filt0, filt1 in filt_pairs:
        if fluxes[filt0] is None or fluxes[filt1] is None:
            feats[f"Color_Slope_{filt0}_{filt1}"] = np.nan
            feats[f"Color_Mean_{filt0}_{filt1}"] = np.nan
            continue

        f1 = fluxes[filt0]
        f2 = fluxes[filt1]

        valid = (f1 > 0) & (f2 > 0)
        if np.sum(valid) < _COL_EVO_MIN_NROWS:
            feats[f"Color_Slope_{filt0}_{filt1}"] = np.nan
            feats[f"Color_Mean_{filt0}_{filt1}"] = np.nan
            continue

        color = -2.5 * np.log10(f1[valid] / f2[valid])
        t_valid = mjd_grid[valid]
        t_rest = (t_valid - t_valid[0]) / (1 + z)

        feats[f"Color_Slope_{filt0}_{filt1}"] = np.polyfit(t_rest, color, 1)[0]
        feats[f"Color_Mean_{filt0}_{filt1}"] = np.mean(color)

    return pd.Series(feats)


_COL_EVO_MIN_NROWS = 5


def _extract_abs_mag(df: pd.DataFrame, μ: float) -> pd.Series:
    if df.empty or μ <= 0:
        return pd.Series(np.nan, index=[f"AbsMag_{filt}" for filt in ["g", "r"]] + [f"AbsLupt_{filt}" for filt in ["g", "r"]])

    feats = {}

    for filt in ["g", "r"]:
        filt_df = df[df["filter"] == filt]

        flux = filt_df["flux"]
        if flux.empty or flux.max() <= 0:
            feats[f"AbsMag_{filt}"] = np.nan
        else:
            m_app = -2.5 * np.log10(flux.max()) + 23.9
            feats[f"AbsMag_{filt}"] = m_app - μ

        flux_lupt = filt_df["flux_lupt"]
        if flux_lupt.empty:
            feats[f"AbsLupt_{filt}"] = np.nan
        else:
            feats[f"AbsLupt_{filt}"] = flux_lupt.max() - μ

    return pd.Series(feats)


def _planck(λ, T, A):
    c2 = 1.4388e8
    val = np.clip(c2 / (λ * T), -100, 100)
    return A / (np.pow(λ, 5) * (np.exp(val) - 1))


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

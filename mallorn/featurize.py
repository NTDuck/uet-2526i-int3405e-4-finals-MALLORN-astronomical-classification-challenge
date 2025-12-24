import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from typer import Typer

# Import data loaders from augment.py (Internal)
from .augment import load_meta_aug_df as _load_meta_aug_df
from .augment import load_obs_aug_df as _load_obs_aug_df

# Import common utils
from .common import AUG_DIR, FEATS_DIR, DfType, mkdir, mkpb, mksb


# Suppress noisy GP warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

typer = Typer()

# =============================================================================
# CLI COMMANDS
# =============================================================================


@typer.command(name="featurize-all")
def featurize_all(aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
    """Run featurization for both train and test sets."""
    with mksb() as sb:
        with sb.job("Calling `featurize(df_type='train')`"):
            featurize(df_type="train", aug_dir=aug_dir, feats_dir=feats_dir)
        with sb.job("Calling `featurize(df_type='test')`"):
            featurize(df_type="test", aug_dir=aug_dir, feats_dir=feats_dir)


@typer.command()
def featurize(df_type: DfType, aug_dir: Path = AUG_DIR, feats_dir: Path = FEATS_DIR):
    """Extract features from augmented data using 2D Gaussian Processes."""
    with mksb() as sb:
        with sb.job("Loading augmented datasets"):
            meta_df = _load_meta_aug_df(df_type, aug_dir)
            obs_df = _load_obs_aug_df(df_type, aug_dir)

            # Start feature dataframe with metadata
            feats_df = meta_df.drop(columns=["SpecType", "English Translation", "split"], errors="ignore")

        with sb.job("Merging & Cleaning Observations"):
            # Merge redshift/distmod for context if needed, though GP is mostly photometric
            mrg_df = obs_df.merge(meta_df[["z", "μ"]], left_on="object_id", right_index=True)
            mrg_df.sort_values(["object_id", "filter", "mjd"], inplace=True)

            # Critical Cleanup: Remove invalid measurements before GP
            mrg_df = mrg_df.dropna(subset=["flux", "flux_err"])
            mrg_df = mrg_df[mrg_df["flux_err"] > 0]

        # --- 2D Gaussian Process Feature Extraction ---
        # Group by object_id and apply the extraction logic
        print(f"\n   [Info] Extracting 2D GP features for {len(meta_df)} objects...")
        print("   [Info] This is computationally intensive. Please wait.")

        with mkpb(total=len(meta_df)) as pb:

            def _process_group(df: pd.DataFrame) -> pd.Series:
                with pb.job(f"GP Extract: {df['object_id'].iloc[0]}"):
                    return _extract_gp_features(df)

            # Parallelization is handled by Pandas/Typer/Common internals or external runners
            # For simplicity here, we use direct apply.
            gp_feats = mrg_df.groupby("object_id").apply(_process_group)

            # FIX: Ensure result is a DataFrame with features as columns
            # If gp_feats is a MultiIndex Series (object_id, feature), unstack it.
            if isinstance(gp_feats, pd.Series):
                gp_feats = gp_feats.unstack()

            # Join extracted features back to metadata
            feats_df = feats_df.join(gp_feats)

        # Save result
        out_path = mkdir(feats_dir) / f"{df_type}_feats.parquet"
        with sb.job(f"Saving featurized dataset to {out_path}"):
            feats_df.to_parquet(out_path)


# =============================================================================
# EXTERNAL UTILITIES
# =============================================================================


def load_feats_df(df_type: DfType, feats_dir: Path = FEATS_DIR, **kwargs) -> pd.DataFrame:
    """Helper to load the featurized dataframe."""
    return pd.read_parquet(feats_dir / f"{df_type}_feats.parquet", **kwargs)


# =============================================================================
# FEATURE EXTRACTION LOGIC (The "Avocado" Method)
# =============================================================================

_EFF_WLS = {"u": 3641.0, "g": 4704.0, "r": 6155.0, "i": 7504.0, "z": 8695.0, "y": 10056.0}


def _extract_gp_features(df: pd.DataFrame) -> pd.Series:
    """
    Fits a 2D Gaussian Process (Time x Wavelength) and extracts features.
    Ref: Boone 2019 (Avocado)
    """
    # 1. Data Prep
    if len(df) < 5:
        return pd.Series(dtype=float)

    mjd = df["mjd"].values
    flux = df["flux"].values
    flux_err = df["flux_err"].values

    # Normalize Time
    t_start = mjd.min()  # pyright: ignore[reportAttributeAccessIssue]
    t_norm = mjd - t_start

    # Normalize Wavelength
    wls = df["filter"].map(_EFF_WLS).values
    wl_min, wl_max = 3000.0, 10000.0
    wl_norm = (wls - wl_min) / (wl_max - wl_min)  # pyright: ignore[reportOperatorIssue]

    # 2. GP Fit
    X = np.column_stack([t_norm, wl_norm])
    y = flux
    # Add floor to error to prevent overfitting spikes
    alpha = np.maximum(flux_err, 0.01 * np.max(np.abs(flux))) ** 2  # pyright: ignore[reportArgumentType, reportCallIssue]

    # Kernel: Time Scale (~20d) * Wavelength Scale (~0.2) + Noise
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e4)) * RBF(length_scale=[20.0, 0.2], length_scale_bounds=[(1.0, 200.0), (0.05, 0.5)])  # pyright: ignore[reportArgumentType]
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e3))
    )

    gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True, n_restarts_optimizer=0)

    feat_dict = {}
    filts = ["u", "g", "r", "i", "z", "y"]

    try:
        gp.fit(X, y)  # pyright: ignore[reportArgumentType]

        # 3. Prediction Grid
        # Create a dense grid to find peaks/shapes accurately
        t_grid = np.linspace(0, t_norm.max(), 50)

        peak_times = {}
        peak_fluxes = {}
        curves = {}

        for f in filts:
            wl_val = (_EFF_WLS[f] - wl_min) / (wl_max - wl_min)
            X_pred = np.column_stack([t_grid, np.full_like(t_grid, wl_val)])
            y_pred, y_std = gp.predict(X_pred, return_std=True)  # pyright: ignore[reportAssignmentType]

            curves[f] = y_pred

            idx_max = np.argmax(y_pred)
            peak_val = y_pred[idx_max]
            peak_t = t_grid[idx_max]

            peak_fluxes[f] = peak_val
            peak_times[f] = peak_t

            # Per-Band Features
            feat_dict[f"{f}_Peak_Flux"] = peak_val
            feat_dict[f"{f}_Peak_Time"] = peak_t
            feat_dict[f"{f}_Peak_Uncertainty"] = y_std[idx_max]
            feat_dict[f"{f}_Rise_Time"] = peak_t - t_grid[0]  # Approx

        # 4. Cross-Band / Physical Features

        # Spectral Slope at Peak (Proxy for Temperature)
        # We take the median peak time to define "event peak"
        valid_peaks = [t for f, t in peak_times.items() if peak_fluxes[f] > 0]
        if valid_peaks:
            t_global_peak = np.median(valid_peaks)

            sed_x = []
            sed_y = []
            for f in filts:
                wl_val = (_EFF_WLS[f] - wl_min) / (wl_max - wl_min)
                # Predict flux at global peak time for this band
                f_at_peak = gp.predict([[t_global_peak, wl_val]])[0]
                if f_at_peak > 0:
                    sed_x.append(np.log10(_EFF_WLS[f]))
                    sed_y.append(np.log10(f_at_peak))

            if len(sed_x) >= 3:
                slope, _ = np.polyfit(sed_x, sed_y, 1)
                feat_dict["Spectral_Slope"] = slope  # TDEs are blue -> expect specific slope range

        # Blue Excess (u - g)
        if "u" in curves and "g" in curves:
            u = curves["u"]
            g = curves["g"]
            mask = (u > 0) & (g > 0)
            if mask.any():
                be = -2.5 * np.log10(u[mask] / g[mask])
                feat_dict["Blue_Excess_Mean"] = np.mean(be)
                feat_dict["Blue_Excess_Min"] = np.min(be)  # Bluest point
                feat_dict["Blue_Excess_Max"] = np.max(be)

        # Granular Color Evolution (Early / Mid / Late)
        # TDEs cool -> redden over time. SNe behave differently.
        for b1, b2 in [("g", "r"), ("u", "r")]:
            if b1 in curves and b2 in curves:
                y1, y2 = curves[b1], curves[b2]
                mask = (y1 > 0) & (y2 > 0)
                if mask.sum() > 10:
                    color = -2.5 * np.log10(y1[mask] / y2[mask])
                    n = len(color)
                    feat_dict[f"Color_{b1}_{b2}_Early"] = np.mean(color[: n // 3])
                    feat_dict[f"Color_{b1}_{b2}_Mid"] = np.mean(color[n // 3 : 2 * n // 3])
                    feat_dict[f"Color_{b1}_{b2}_Late"] = np.mean(color[2 * n // 3 :])
                    feat_dict[f"Color_{b1}_{b2}_Slope"] = feat_dict[f"Color_{b1}_{b2}_Late"] - feat_dict[f"Color_{b1}_{b2}_Early"]

    except Exception:
        pass  # Return NaNs on failure

    return pd.Series(feat_dict)

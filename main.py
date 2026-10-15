"""
🐄 Bovine SNP Platform v6.0 — Tout-en-un
Pipeline complet + analyses avancées + binômes + auto-MDS + GWAS 10cM.
"""

import gzip
import json
import os
import warnings
import zipfile
from collections import Counter
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🐄 Bovine SNP Platform v6.0",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_THRESHOLDS = {
    "geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6, "het_sd": 3.0,
    "king_cutoff": 0.354, "ld_r2": 0.2,
}

HWE_EXACT_MAX_SNP = 20_000

ARS_UCD12_LENGTHS = {
    "1": 158_534_110, "2": 136_231_102, "3": 121_005_158, "4": 120_000_166,
    "5": 120_089_699, "6": 117_806_340, "7": 110_682_743, "8": 113_384_748,
    "9": 105_708_134, "10": 103_308_737, "11": 107_310_763, "12": 91_163_125,
    "13": 84_246_514, "14": 84_648_346, "15": 85_207_080, "16": 81_726_628,
    "17": 75_176_999, "18": 66_059_976, "19": 64_089_169, "20": 72_042_983,
    "21": 71_599_096, "22": 61_416_492, "23": 52_531_573, "24": 62_384_193,
    "25": 42_959_610, "26": 51_680_158, "27": 46_772_073, "28": 46_333_854,
    "29": 51_319_414, "X": 139_009_144, "Y": 50_927_933, "MT": 16_338,
}
BOVINE_AUTOSOMES = [str(i) for i in range(1, 30)]

POPULATION_PAIRS_DEFAULT = [
    ("LJR", "YBS", "Cluster isolé en haut à droite"),
    ("EZP", "FGN", "Extrême gauche du plan"),
    ("EBG", "ELN", "Partie supérieure gauche"),
    ("NAR", "NBD", "Sommet de la diagonale"),
    ("PTP", "SHO", "Partie intermédiaire diagonale"),
    ("NKA", "PSH", "Extrémité basse/droite diagonale"),
    ("AND", "PRS", "Étirées sur la diagonale centrale"),
]


# ============================================================
# CACHE + CONVERSION
# ============================================================

def _hash_ndarray(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return ("empty",)
    return (x.shape, str(x.dtype), float(np.nansum(x)),
            float(np.nansum(np.abs(x))), float(np.nansum(x * x)))


HASH_FUNCS = {np.ndarray: _hash_ndarray}


def cache_data(func=None, **kw):
    def _decorate(f):
        return st.cache_data(show_spinner=False, hash_funcs=HASH_FUNCS, **kw)(f)
    if func is None:
        return _decorate
    return _decorate(func)


def to_np_str(arr):
    return np.asarray(list(arr), dtype=object)


def step_header(numero, titre, ce_qu_on_fait, pourquoi, attendu):
    with st.expander(f"📖 **Étape {numero} — {titre}**", expanded=False):
        st.markdown(f"**🎯 Ce qu'on fait :** {ce_qu_on_fait}")
        st.markdown(f"**❓ Pourquoi :** {pourquoi}")
        st.markdown(f"**✅ Ce qu'on attend :** {attendu}")


def impute_mean(gt):
    gt2 = gt.astype(np.float32, copy=True)
    col_mean = np.nanmean(gt2, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(gt2)
    if not nan_mask.any():
        return gt2
    gt2[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    return gt2


def _autosome_weights():
    lens = np.array([ARS_UCD12_LENGTHS[c] for c in BOVINE_AUTOSOMES],
                    dtype=float)
    return lens / lens.sum()


def _chr_sort_key(chrom):
    s = str(chrom).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit():
        return (0, int(s), "")
    special = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    if s in special:
        return (1, special[s], "")
    return (2, 0, s)


def _detect_encoding(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return "gzip"
    if len(raw) >= 2 and raw[:2] == b"\x6c\x1b":
        return "plink_binary"
    return "text"


# ============================================================
# PARSING
# ============================================================

def parse_map(map_bytes):
    enc = _detect_encoding(map_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(map_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .map.gz corrompu : {e}")
    elif enc == "plink_binary":
        raise ValueError("❌ Le fichier .map est en binaire PLINK.")
    else:
        text = map_bytes.decode("utf-8", errors="replace")

    rows, rejected = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            rejected += 1
            continue
        try:
            cm = float(parts[2]) if parts[2] not in ("0", ".", "NA") else 0.0
        except ValueError:
            cm = 0.0
        try:
            bp = int(float(parts[3]))
        except (ValueError, TypeError):
            rejected += 1
            continue
        rows.append({"CHR": str(parts[0]), "SNP": str(parts[1]),
                     "CM": cm, "BP": bp})

    if not rows:
        raise ValueError("Fichier .map vide ou invalide.")
    return pd.DataFrame(rows), rejected


def parse_ped(ped_bytes, n_snp_map):
    """Parser PED robuste avec détection auto du header."""
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(ped_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .ped.gz corrompu : {e}")
    elif enc == "plink_binary":
        raise ValueError("❌ Ce fichier est un PLINK binaire (.bed) renommé.")
    else:
        text = ped_bytes.decode("utf-8", errors="replace")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Fichier .ped vide.")

    header_skipped = False
    if len(lines) >= 2:
        n0 = len(lines[0].split())
        n1 = len(lines[1].split())
        is_header = False
        if n0 < 7 and n1 >= 7:
            is_header = True
        elif n1 > 0 and n0 < n1 * 0.5:
            is_header = True
        else:
            first_words = [w.lower().strip('"\'').strip()
                           for w in lines[0].split()[:6]]
            header_kw = {'id', 'iid', 'fid', 'sample', 'individual',
                         'breed', 'race', 'population', 'animal',
                         'snp', 'marker', 'chr', 'chromosome',
                         'pedigree', 'family'}
            if any(w in header_kw for w in first_words):
                is_header = True
        if is_header:
            lines = lines[1:]
            header_skipped = True
            st.info(f"ℹ️ Header détecté et ignoré "
                    f"(ligne 1 = {n0} cols, données = {n1} cols).")

    if not lines:
        raise ValueError("Fichier .ped vide après suppression du header.")

    parts0 = lines[0].split()
    n_cols_first = len(parts0)
    n_snp_ped = max(0, (n_cols_first - 6) // 2)

    if n_cols_first < 7:
        peek_lines = []
        for i, ln in enumerate(lines[:5]):
            trunc = ln[:180] + (" [...]" if len(ln) > 180 else "")
            peek_lines.append(f"Ligne {i+1} ({len(ln.split())} cols) : {trunc}")
        peek_str = "\n".join(peek_lines)
        if "\t" in lines[0]:
            sep_hint = "Tabulations (\\t)"
        elif "," in lines[0]:
            sep_hint = "Virgules (,) — convertissez avec sed 's/,/ /g'"
        else:
            sep_hint = "Espaces"
        raise ValueError(
            f"❌ Format PED invalide : {n_cols_first} colonnes.\n\n"
            f"**Séparateur détecté :** {sep_hint}\n\n"
            f"**Aperçu :**\n```\n{peek_str}\n```\n\n"
            f"Format attendu : 6 + 2 × N_SNP colonnes par ligne.")

    if 0 < n_snp_ped < 10 and len(lines) > 100:
        raise ValueError("⚠️ Fichier probablement transposé.")

    if n_snp_ped != n_snp_map and n_snp_ped > 0:
        st.warning(f"⚠️ Désalignement PED/MAP : "
                   f"{n_snp_ped} vs {n_snp_map} SNPs.")
        n_snp_eff = min(n_snp_map, n_snp_ped)
    else:
        n_snp_eff = n_snp_map

    expected_cols_eff = 6 + 2 * n_snp_eff
    fids, iids, geno_rows = [], [], []
    rejected_short = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        if len(parts) < expected_cols_eff:
            rejected_short += 1
            continue
        fids.append(str(parts[0]))
        iids.append(str(parts[1]))
        geno_rows.append(parts[6:6 + 2 * n_snp_eff])

    n_ind = len(iids)
    if n_ind == 0:
        raise ValueError(f"❌ Aucun individu chargé.")

    if rejected_short > 0:
        st.warning(f"⚠️ {rejected_short} ligne(s) ignorée(s).")

    allele_counts = [Counter() for _ in range(n_snp_eff)]
    for row in geno_rows:
        for j in range(n_snp_eff):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            c = allele_counts[j]
            c[a1] += 1
            c[a2] += 1

    minor = [None] * n_snp_eff
    for j, c in enumerate(allele_counts):
        if len(c) >= 2:
            minor[j] = min(c, key=c.get)

    gt = np.full((n_ind, n_snp_eff), np.nan, dtype=np.float32)
    for i, row in enumerate(geno_rows):
        for j in range(n_snp_eff):
            m = minor[j]
            if m is None:
                continue
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            gt[i, j] = (1 if a1 == m else 0) + (1 if a2 == m else 0)

    ind_df = pd.DataFrame({"FID": fids, "IID": iids})

    if header_skipped:
        st.success(f"✅ Parsing réussi : {n_ind} individus × "
                   f"{n_snp_eff} SNPs.")

    return gt, ind_df, rejected_short


def parse_vcf(vcf_bytes):
    is_gz = len(vcf_bytes) >= 2 and vcf_bytes[:2] == b"\x1f\x8b"
    if is_gz:
        try:
            text = gzip.decompress(vcf_bytes).decode("utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .vcf.gz corrompu : {e}")
    else:
        text = vcf_bytes.decode("utf-8", errors="replace")

    samples, rows = [], []
    n_skipped, n_multi = 0, 0

    for line in text.splitlines():
        line = line.rstrip("\n\r")
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
            continue
        if line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 10:
            n_skipped += 1
            continue

        chrom, pos_s, vid, ref, alt = (parts[0], parts[1], parts[2],
                                        parts[3], parts[4])
        if "," in alt:
            n_multi += 1
            continue
        try:
            pos = int(pos_s)
        except ValueError:
            n_skipped += 1
            continue

        fmt = parts[8].split(":")
        if "GT" not in fmt:
            n_skipped += 1
            continue
        gt_idx = fmt.index("GT")

        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(parts[9:]):
            fields = s.split(":")
            if gt_idx >= len(fields):
                continue
            gt_str = fields[gt_idx]
            sep = "|" if "|" in gt_str else "/"
            alleles = gt_str.split(sep)
            if any(a in (".", "") for a in alleles):
                continue
            try:
                gts[k] = sum(int(a) for a in alleles)
            except ValueError:
                continue

        rows.append({"CHR": str(chrom), "SNP": str(vid or f"{chrom}:{pos}"),
                     "CM": 0.0, "BP": pos, "A1": str(ref), "A2": str(alt),
                     "GT": gts})

    if not rows:
        raise ValueError("Aucun variant biallélique trouvé.")

    n_ind, n_snp = len(samples), len(rows)
    gt = np.zeros((n_ind, n_snp), dtype=np.float32)
    for j, r in enumerate(rows):
        gt[:, j] = r["GT"]

    for j in range(n_snp):
        col = gt[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue
        if valid.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(col), np.nan, 2.0 - col)

    snp_df = pd.DataFrame({
        "CHR": [str(r["CHR"]) for r in rows],
        "SNP": [str(r["SNP"]) for r in rows],
        "CM": 0.0,
        "BP": [int(r["BP"]) for r in rows],
        "A1": [str(r["A1"]) for r in rows],
        "A2": [str(r["A2"]) for r in rows]})
    ind_df = pd.DataFrame({"FID": [str(s) for s in samples],
                            "IID": [str(s) for s in samples]})
    return gt, ind_df, snp_df, n_skipped, n_multi


# ============================================================
# DÉMO
# ============================================================

def generate_demo_data(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng = np.random.default_rng(seed)
    pool = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
            "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]
    n_pop = max(1, min(n_pop, len(pool)))
    pop_names = pool[:n_pop]
    per_pop = n_ind // n_pop
    remainder = n_ind - per_pop * n_pop
    counts = [per_pop + (1 if i < remainder else 0) for i in range(n_pop)]
    n_ind_eff = sum(counts)
    p_anc = rng.beta(0.6, 0.6, n_snp)
    drift = 0.15
    gt = np.zeros((n_ind_eff, n_snp), dtype=np.float32)
    ind_rows = []
    k = 0
    for pop, n_i in zip(pop_names, counts):
        p_k = np.clip(p_anc + rng.normal(0, drift, n_snp), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(n_snp) < p_k).astype(np.float32)
            a2 = (rng.random(n_snp) < p_k).astype(np.float32)
            gt[k] = a1 + a2
            ind_rows.append({"FID": str(pop), "IID": f"{pop}_{i+1:03d}"})
            k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    ind_df = pd.DataFrame(ind_rows)
    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c]) for c in chr_names],
                   dtype=np.int64)
    snp_df = pd.DataFrame({
        "CHR": [str(c) for c in chr_names],
        "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": 0.0, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    snp_df = (snp_df.sort_values(["_k", "BP"]).drop(columns="_k")
              .reset_index(drop=True))
    return gt, ind_df, snp_df


# ============================================================
# MÉTRIQUES QC
# ============================================================

def missingness_per_ind(gt):
    return np.isnan(gt).mean(axis=1)


def missingness_per_snp(gt):
    return np.isnan(gt).mean(axis=0)


def allele_freq(gt):
    return np.nanmean(gt, axis=0) / 2.0


def maf(gt):
    p = allele_freq(gt)
    return np.minimum(p, 1.0 - p)


def heterozygosity(gt):
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt == 1, axis=1)


def hwe_exact_p(n_het, n_hom1, n_hom2):
    n = n_het + n_hom1 + n_hom2
    if n == 0:
        return np.nan
    if n_het == 0 and (n_hom1 == 0 or n_hom2 == 0):
        return 1.0
    rare = 2 * min(n_hom1, n_hom2) + n_het
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2:
        mid += 1
    probs = np.zeros(rare + 1)
    probs[mid] = 1.0
    mysum = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = probs[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        mysum += probs[ch + 2]
        ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = (probs[ch] * ch * (ch - 1)
                         / (4 * (chr_ + 1) * (chc + 1)))
        mysum += probs[ch - 2]
        ch -= 2; chr_ += 1; chc += 1
    p_obs = probs[n_het] if n_het < len(probs) else 0.0
    return float(min(probs[probs <= p_obs + 1e-7].sum() / mysum, 1.0))


@cache_data
def hwe_pvalues(gt):
    _, n_snp = gt.shape
    pvals = np.full(n_snp, np.nan, dtype=np.float64)
    nan_mask = np.isnan(gt)
    n0 = ((gt == 0) & ~nan_mask).sum(axis=0)
    n1 = ((gt == 1) & ~nan_mask).sum(axis=0)
    n2 = ((gt == 2) & ~nan_mask).sum(axis=0)
    n_valid = n0 + n1 + n2

    if n_snp <= HWE_EXACT_MAX_SNP:
        for j in range(n_snp):
            if n_valid[j] < 5:
                continue
            pvals[j] = hwe_exact_p(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0, n_valid, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_het = 2.0 * p * (1.0 - p) * n_valid
            valid = (exp_het >= 5) & np.isfinite(exp_het)
            chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
            pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals


# ============================================================
# NETTOYAGE PED
# ============================================================

def extract_fid_from_iid(iid):
    for sep in ("_", "-", "."):
        if sep in iid:
            return iid.split(sep)[0]
    return iid


def reconstruct_fid_from_iid(ind_df):
    df = ind_df.copy()
    unique_fids = df["FID"].nunique() if "FID" in df.columns else 0
    if unique_fids <= 1:
        df["FID"] = df["IID"].astype(str).apply(extract_fid_from_iid)
        st.info(f"🔧 FID reconstruit → **{df['FID'].nunique()}** races.")
    return df


def filter_autosomes(gt, snp_df, keep_auto_only=True):
    chr_str = snp_df["CHR"].astype(str).str.upper().str.replace("CHR", "")
    if keep_auto_only:
        keep = chr_str.isin([str(i) for i in range(1, 30)])
    else:
        keep = chr_str.isin([str(i) for i in range(1, 30)] + ["X", "Y", "MT"])
    n_before = len(snp_df)
    gt2 = gt[:, keep.values]
    snp2 = snp_df[keep].reset_index(drop=True)
    if keep_auto_only:
        st.info(f"🧬 Autosomes 1–29 : {n_before - len(snp2)} SNPs exclus → "
                f"**{len(snp2)}** conservés.")
    return gt2, snp2


@cache_data
def ld_pruning(gt, window=50, step=5, r2_thr=0.2):
    n_snp = gt.shape[1]
    if n_snp < 3:
        return np.ones(n_snp, dtype=bool)
    keep = np.ones(n_snp, dtype=bool)
    X = impute_mean(gt); X = X - X.mean(axis=0)
    i = 0
    while i < n_snp:
        win_end = min(i + step, n_snp)
        while win_end < n_snp and (win_end - i) < window:
            j = win_end
            if keep[j] and keep[i]:
                a, b = X[:, i], X[:, j]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    r = np.corrcoef(a, b)[0, 1]
                    if r * r > r2_thr:
                        keep[j] = False
            win_end += 1
        i += step
    return keep


# ============================================================
# QC DOUBLE ÉTAT
# ============================================================

def _align_shapes(gt, ind_df, snp_df):
    n_gt, m_gt = gt.shape
    n_ind, n_snp = min(n_gt, len(ind_df)), min(m_gt, len(snp_df))
    if (n_gt, m_gt) != (len(ind_df), len(snp_df)):
        st.warning(f"⚠️ Alignement : gt={gt.shape} → ({n_ind},{n_snp})")
    return (gt[:n_ind, :n_snp],
            ind_df.iloc[:n_ind].reset_index(drop=True),
            snp_df.iloc[:n_snp].reset_index(drop=True))


def apply_qc_filters(gt, ind_df, snp_df, params):
    """QC cascade. Retourne (gt_filt, ind_filt, snp_filt, stats,
    gt_snp, ind_snp, snp_snp)."""
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    trace = {}

    keep = missingness_per_snp(gt) <= params["geno"]
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["geno_exclus"] = int(m0 - gt.shape[1])

    keep = missingness_per_ind(gt) <= params["mind"]
    gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
    trace["mind_exclus"] = int(n0 - gt.shape[0])

    if gt.shape[1] == 0 or gt.shape[0] == 0:
        raise ValueError("Tous SNPs/individus exclus (missingness).")

    m_before = gt.shape[1]
    m = maf(gt)
    keep = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["maf_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous SNPs exclus par MAF.")

    m_before = gt.shape[1]
    pv = hwe_pvalues(gt)
    keep = np.isnan(pv) | (pv >= 1e-6)
    gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    if gt.shape[1] > 0:
        pv2 = hwe_pvalues(gt)
        keep = np.isnan(pv2) | (pv2 >= params["hwe"])
        gt = gt[:, keep]; snp_df = snp_df[keep].reset_index(drop=True)
    trace["hwe_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous SNPs exclus par HWE.")

    gt_snp = gt.copy()
    ind_snp = ind_df.copy()
    snp_snp = snp_df.copy()

    het = heterozygosity(gt)
    keep = np.ones(len(het), dtype=bool)
    for fid in ind_df["FID"].unique():
        mask = (ind_df["FID"] == fid).values
        sub = het[mask]
        if len(sub) < 5 or np.nanstd(sub) < 1e-9:
            continue
        z = (sub - np.nanmean(sub)) / np.nanstd(sub)
        sub_keep = np.abs(z) <= params["het_sd"]
        idx = np.where(mask)[0]
        keep[idx[~sub_keep]] = False
    n_before = gt.shape[0]
    gt = gt[keep]; ind_df = ind_df[keep].reset_index(drop=True)
    trace["het_exclus"] = int(n_before - gt.shape[0])

    if gt.shape[0] == 0:
        raise ValueError("Tous individus exclus (hétérozygotie).")

    stats = {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]), "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]),
        "n_ind_snp": int(gt_snp.shape[0]),
        "n_snp_snp": int(gt_snp.shape[1]),
        "trace": trace,
    }
    return (gt, ind_df, snp_df, stats, gt_snp, ind_snp, snp_snp)


# ============================================================
# POPULATION GENETICS
# ============================================================

@cache_data
def fst_per_snp(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    pops = np.unique(pop_labels)
    if len(pops) < 2:
        return np.full(gt.shape[1], np.nan)
    fst = np.full(gt.shape[1], np.nan, dtype=np.float64)
    masks = {p: (pop_labels == p) for p in pops}
    for j in range(gt.shape[1]):
        pl, nl = [], []
        for p in pops:
            vals = gt[masks[p], j]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 3:
                continue
            pl.append(vals.mean() / 2.0); nl.append(len(vals))
        if len(pl) < 2:
            continue
        p_arr, n_arr = np.asarray(pl), np.asarray(nl)
        p_bar = np.average(p_arr, weights=n_arr)
        h_s = np.average(2.0 * p_arr * (1.0 - p_arr), weights=n_arr)
        h_t = 2.0 * p_bar * (1.0 - p_bar)
        if h_t > 1e-9:
            fst[j] = (h_t - h_s) / h_t
    return fst


@cache_data
def fst_pairwise(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    pops = sorted(np.unique(pop_labels))
    K = len(pops)
    matrix = np.full((K, K), np.nan); np.fill_diagonal(matrix, 0.0)
    masks = {p: (pop_labels == p) for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            g1 = gt[masks[pops[i]]]; g2 = gt[masks[pops[j]]]
            p1 = np.nanmean(g1, 0) / 2.0; p2 = np.nanmean(g2, 0) / 2.0
            n1 = (~np.isnan(g1)).sum(0); n2 = (~np.isnan(g2)).sum(0)
            denom = np.where(n1 + n2 > 0, n1 + n2, np.nan)
            p_bar = (p1 * n1 + p2 * n2) / denom
            hs = (2 * p1 * (1 - p1) * n1 + 2 * p2 * (1 - p2) * n2) / denom
            ht = 2 * p_bar * (1 - p_bar)
            with np.errstate(divide="ignore", invalid="ignore"):
                fst_j = (ht - hs) / ht
            matrix[i, j] = matrix[j, i] = float(np.nanmean(fst_j))
    return matrix, pops


@cache_data
def pca_analysis(gt, n_components=10):
    X = impute_mean(gt); X = X - X.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X), pca.explained_variance_ratio_ * 100.0


@cache_data
def mds_analysis(gt, n_components=5):
    X = impute_mean(gt); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(n_components=n_components, dissimilarity="precomputed",
                      random_state=42, n_init=1, max_iter=300,
                      normalized_stress=False).fit_transform(D)


@cache_data
def ld_decay(gt, snp_bp, max_kb=1000, max_snp=1500, seed=42):
    n_snp = gt.shape[1]
    if n_snp < 2:
        return pd.DataFrame(columns=["dist_kb", "r2"])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
           if n_snp > max_snp else np.arange(n_snp))
    gt_sub = impute_mean(gt[:, idx]); bp_sub = snp_bp[idx].astype(np.float64)
    X = gt_sub - gt_sub.mean(0)
    std = gt_sub.std(0); std[std < 1e-8] = np.nan
    X = X / std
    C = (X.T @ X) / X.shape[0]; R2 = C ** 2
    iu, ju = np.triu_indices(len(idx), k=1)
    dist_kb = (bp_sub[ju] - bp_sub[iu]) / 1000.0
    mask = (dist_kb > 0) & (dist_kb <= max_kb)
    return pd.DataFrame({"dist_kb": dist_kb[mask],
                         "r2": R2[iu[mask], ju[mask]]})


@cache_data
def king_kinship(gt):
    n_ind = gt.shape[0]
    het = np.nansum(gt == 1, axis=1).astype(np.float64)
    K = np.zeros((n_ind, n_ind))
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            both_het = np.nansum((gt[i] == 1) & (gt[j] == 1))
            denom = het[i] + het[j]
            phi = (denom - 2 * both_het) / denom if denom > 0 else 0.0
            K[i, j] = K[j, i] = phi
    np.fill_diagonal(K, 0.5)
    return K


def apply_king_cutoff(K, ind_df, cutoff=0.354):
    n = K.shape[0]
    to_remove = np.zeros(n, dtype=bool)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if K[i, j] > cutoff and not to_remove[i] and not to_remove[j]]
    for i, j in pairs:
        to_remove[j] = True
    return ~to_remove, int(to_remove.sum())


@cache_data
def admixture_nmf(gt, K=3, seed=42, max_iter=500):
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    model = NMF(n_components=K, init="nndsvda",
                random_state=seed, max_iter=max_iter)
    W = model.fit_transform(X)
    s = W.sum(1, keepdims=True); s[s == 0] = 1.0
    return W / s, model.components_


@cache_data
def admix_cv_error(gt, K_range=(2, 6), n_reps=3, holdout=0.05, seed=42):
    rng = np.random.default_rng(seed)
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    n, m = X.shape
    mask = rng.random((n, m)) < holdout
    X_train = X.copy(); X_train[mask] = 0.0
    results = {}
    for K in range(K_range[0], K_range[1]):
        errs = []
        for rep in range(n_reps):
            model = NMF(n_components=K, init="nndsvda",
                        random_state=seed + rep, max_iter=300)
            W = model.fit_transform(X_train); H = model.components_
            err = np.mean((X[mask] - (W @ H)[mask]) ** 2)
            errs.append(err)
        results[K] = {"mean": float(np.mean(errs)),
                      "std": float(np.std(errs))}
    return results


@cache_data
def estimate_ne_historical(gt, _snp_df, max_snp=2000):
    snp_df = _snp_df
    bp = snp_df["BP"].astype(np.float64).values
    chr_arr = snp_df["CHR"].astype(str).values
    n_snp = gt.shape[1]
    if n_snp > max_snp:
        idx = np.linspace(0, n_snp - 1, max_snp).astype(int)
        gt = gt[:, idx]; bp = bp[idx]; chr_arr = chr_arr[idx]
    X = impute_mean(gt)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r2_list, dist_list = [], []
    n = X.shape[1]
    for i in range(0, n - 1):
        for j in range(i + 1, min(i + 50, n)):
            if chr_arr[i] != chr_arr[j]:
                continue
            d = abs(bp[j] - bp[i]) / 1e6
            if d < 0.001 or d > 50:
                continue
            r = np.corrcoef(X[:, i], X[:, j])[0, 1]
            r2_list.append(r * r); dist_list.append(d)
    if not dist_list:
        return pd.DataFrame({"GenAgo": [], "Ne": []})
    r2 = np.array(r2_list); dist = np.array(dist_list)
    bins = np.logspace(np.log10(dist.min() + 1e-6),
                       np.log10(dist.max()), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    r2b = np.array([
        r2[(dist >= bins[k]) & (dist < bins[k + 1])].mean()
        if ((dist >= bins[k]) & (dist < bins[k + 1])).sum() > 0 else np.nan
        for k in range(len(bins) - 1)])
    m = np.isfinite(r2b)
    if m.sum() < 3:
        return pd.DataFrame({"GenAgo": [], "Ne": []})
    c_morgans = bc[m] * 0.01
    r2v = r2b[m]
    ne = np.where(r2v > 1e-4, 1.0 / (4 * c_morgans * r2v), np.nan)
    gen_ago = bc[m] * 100
    df = pd.DataFrame({"GenAgo": gen_ago, "Ne": ne}).dropna()
    df = df[df["Ne"] > 0].sort_values("GenAgo")
    if len(df) > 3:
        df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df


@cache_data
def reynolds_distance(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    pops = sorted(np.unique(pop_labels)); K = len(pops)
    p_hat = {}
    for p in pops:
        p_hat[p] = np.nanmean(gt[pop_labels == p], 0) / 2.0
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            pi, pj = p_hat[pops[i]], p_hat[pops[j]]
            num = np.nansum((pi - pj) ** 2)
            denom = np.nansum(pi * (1 - pj) + pj * (1 - pi))
            if denom > 0:
                D[i, j] = D[j, i] = num / denom
    return D, pops


def nj_tree_newick(D, labels):
    try:
        from scipy.cluster.hierarchy import linkage, to_tree
        Dm = squareform(D, checks=False)
        Z = linkage(Dm, method="average")
        tree = to_tree(Z)
        def rec(node):
            if node.is_leaf():
                return f"{labels[node.id]}:{node.dist:.4f}"
            return (f"({rec(node.get_left())},"
                    f"{rec(node.get_right())}):{node.dist:.4f}")
        return rec(tree)
    except Exception as e:
        return f"Erreur NJ : {e}"


@cache_data
def detect_roh(gt, snp_df_json, min_snps=30, min_kb=500.0):
    snp_df = pd.read_json(snp_df_json)
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    n_ind = gt.shape[0]
    rohs, froh_bp = [], np.zeros(n_ind)
    genome_bp_total = 0.0
    for chrom in pd.unique(chr_arr):
        idxs = np.where(chr_arr == chrom)[0]
        if len(idxs) < min_snps:
            continue
        order = np.argsort(bp[idxs]); idxs = idxs[order]
        chrom_span = bp[idxs[-1]] - bp[idxs[0]]
        if chrom_span < 1:
            continue
        genome_bp_total += chrom_span
        for i in range(n_ind):
            col = gt[i, idxs]; start_k = None
            for k in range(len(idxs)):
                is_hom = (col[k] == 0) or (col[k] == 2)
                if is_hom:
                    if start_k is None:
                        start_k = k
                else:
                    if start_k is not None:
                        n_run = k - start_k
                        length_kb = (bp[idxs[k - 1]]
                                     - bp[idxs[start_k]]) / 1000.0
                        if n_run >= min_snps and length_kb >= min_kb:
                            rohs.append({"IID_idx": i, "CHR": str(chrom),
                                         "start_bp": int(bp[idxs[start_k]]),
                                         "end_bp": int(bp[idxs[k - 1]]),
                                         "n_snp": int(n_run),
                                         "length_kb": float(length_kb)})
                            froh_bp[i] += (bp[idxs[k - 1]]
                                            - bp[idxs[start_k]])
                        start_k = None
            if start_k is not None:
                n_run = len(idxs) - start_k
                length_kb = (bp[idxs[-1]] - bp[idxs[start_k]]) / 1000.0
                if n_run >= min_snps and length_kb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": str(chrom),
                                 "start_bp": int(bp[idxs[start_k]]),
                                 "end_bp": int(bp[idxs[-1]]),
                                 "n_snp": int(n_run),
                                 "length_kb": float(length_kb)})
                    froh_bp[i] += bp[idxs[-1]] - bp[idxs[start_k]]
    froh = froh_bp / genome_bp_total if genome_bp_total > 0 else froh_bp
    roh_df = (pd.DataFrame(rohs) if rohs else
              pd.DataFrame(columns=["IID_idx", "CHR", "start_bp",
                                    "end_bp", "n_snp", "length_kb"]))
    return roh_df, froh


@cache_data
def selection_signatures(gt, _snp_df, _pop_labels):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    fst = fst_per_snp(gt, pop_labels)
    hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    df = snp_df.copy()
    df["FST"] = fst; df["HOM"] = hom
    hom_std = df["HOM"].std()
    df["SCORE"] = (df["FST"].fillna(0) *
                   (df["HOM"] - df["HOM"].mean()) /
                   (hom_std if hom_std > 1e-9 else 1.0))
    return df.sort_values("SCORE", ascending=False)


# ============================================================
# ANALYSES AVANCÉES
# ============================================================

@cache_data
def nucleotide_diversity_pi(gt):
    n_ind, n_snp = gt.shape
    pi_per_snp = np.full(n_snp, np.nan)
    for j in range(n_snp):
        col = gt[:, j]
        valid = col[~np.isnan(col)]
        n = len(valid)
        if n < 2:
            continue
        p = valid.mean() / 2.0
        pi_per_snp[j] = 2.0 * p * (1.0 - p) * n / (n - 1)
    return pi_per_snp, float(np.nanmean(pi_per_snp))


@cache_data
def watterson_theta(gt):
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    S = int(np.sum((p > 0) & (p < 1)))
    a_n = float(np.sum(1.0 / np.arange(1, n_ind)))
    theta = S / (a_n * n_snp) if (a_n > 0 and n_snp > 0) else 0.0
    return {"S": S, "a_n": round(a_n, 4), "theta_W": round(theta, 6)}


@cache_data
def site_frequency_spectrum(gt, unfolded=True):
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    p = p[(p > 0) & (p < 1)]
    if not unfolded:
        p = np.minimum(p, 1.0 - p)
    n_bins = min(20, len(p))
    if n_bins < 2:
        return pd.DataFrame({"freq": [], "count": []})
    hist, edges = np.histogram(p, bins=n_bins,
                                range=(0, 1 if unfolded else 0.5))
    centers = (edges[:-1] + edges[1:]) / 2
    return pd.DataFrame({"freq": centers, "count": hist})


@cache_data
def tajimas_d(gt):
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0 or n_snp == 0:
        return {"D": np.nan, "S": 0}
    pi_total = 0.0
    for j in range(n_snp):
        col = gt[:, j]
        valid = col[~np.isnan(col)]
        n = len(valid)
        if n < 2:
            continue
        pj = valid.mean() / 2.0
        pi_total += 2.0 * pj * (1 - pj) * n / (n - 1)
    pi = pi_total / n_snp
    a1 = float(np.sum(1.0 / np.arange(1, n_ind)))
    a2 = float(np.sum(1.0 / np.arange(1, n_ind) ** 2))
    theta_w = S / (a1 * n_snp) if (a1 > 0 and n_snp > 0) else 0.0
    n = n_ind
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n * n + n + 3) / (9 * n * (n - 1))
    c1 = b1 - 1.0 / a1
    c2 = b2 - (n + 2) / (a1 * n) + a2 / (a1 ** 2)
    e1 = c1 / a1
    e2 = c2 / (a1 ** 2 + a2)
    var_d = e1 * S + e2 * S * (S - 1)
    if var_d <= 0:
        return {"D": np.nan, "S": S}
    D = (pi - theta_w) / np.sqrt(var_d)
    pval = 2 * (1 - stats.norm.cdf(abs(D)))
    return {"D": round(float(D), 4),
            "p_value": round(float(pval), 4),
            "pi": round(float(pi), 6),
            "theta_W": round(float(theta_w), 6), "S": S,
            "interpretation": ("Goulot" if D > 2
                               else "Expansion" if D < -2 else "Neutre")}


@cache_data
def fu_li_d(gt):
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0:
        return {"D_star": np.nan, "S": 0}
    thr = 1.0 / (2 * n_ind)
    n_singletons = int(np.sum((p > 0) & (p <= thr * 1.5)))
    a1 = float(np.sum(1.0 / np.arange(1, n_ind)))
    a2 = float(np.sum(1.0 / np.arange(1, n_ind) ** 2))
    eta_e = S / a1 if a1 > 0 else 0
    if a1 > 0 and a2 > 0 and n_ind > 2:
        b1 = (n_ind + 1) / (3 * (n_ind - 1))
        b2 = 2 * (n_ind * n_ind + n_ind + 3) / (9 * n_ind * (n_ind - 1))
        c1 = b1 - 1.0 / a1
        c2 = b2 - (n_ind + 2) / (a1 * n_ind) + a2 / (a1 ** 2)
        v_d = (c1 / a1) * eta_e + (c2 / (a1 ** 2 + a2)) * eta_e * (eta_e - 1)
        D_star = (eta_e - n_singletons) / np.sqrt(v_d) if v_d > 0 else np.nan
    else:
        D_star = np.nan
    return {"D_star": round(float(D_star), 4)
            if np.isfinite(D_star) else np.nan,
            "S": S, "n_singletons": n_singletons}


@cache_data
def f_statistics(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    pops = np.unique(pop_labels)
    if len(pops) < 2:
        return {"F_IS": np.nan, "F_IT": np.nan, "F_ST": np.nan}
    p_list, n_list = [], []
    for p in pops:
        sub = gt[pop_labels == p]
        p_list.append(np.nanmean(sub, axis=0) / 2.0)
        n_list.append((~np.isnan(sub)).sum(axis=0))
    p_arr = np.array(p_list); n_arr = np.array(n_list)
    F_IS_list = []
    for pop in pops:
        sub = gt[pop_labels == pop]
        n_ij = (~np.isnan(sub)).sum(axis=0)
        p_ij = np.nanmean(sub, axis=0) / 2.0
        h_obs = np.nanmean(sub == 1, axis=0)
        h_exp = 2 * p_ij * (1 - p_ij)
        mask = (n_ij > 1) & (h_exp > 0.01)
        if mask.sum() > 0:
            F_IS_list.append(1.0 - np.nanmean(h_obs[mask] / h_exp[mask]))
    F_IS = float(np.nanmean(F_IS_list)) if F_IS_list else np.nan
    p_mean = np.nanmean(p_arr, axis=0)
    h_s = np.nanmean(2 * p_arr * (1 - p_arr), axis=0)
    h_t = 2 * p_mean * (1 - p_mean)
    valid = (h_t > 1e-6) & ((n_arr > 0).sum(axis=0) >= 2)
    F_ST = (float(np.nanmean((h_t[valid] - h_s[valid]) / h_t[valid]))
            if valid.sum() > 0 else np.nan)
    F_IT = (1 - (1 - F_IS) * (1 - F_ST)
            if (np.isfinite(F_IS) and np.isfinite(F_ST) and (1 - F_ST) > 0)
            else np.nan)
    return {"F_IS": round(F_IS, 4) if np.isfinite(F_IS) else np.nan,
            "F_IT": round(F_IT, 4) if np.isfinite(F_IT) else np.nan,
            "F_ST": round(F_ST, 4) if np.isfinite(F_ST) else np.nan}


@cache_data
def f_grm_inbreeding(gt):
    X = impute_mean(gt)
    p = np.clip(X.mean(axis=0) / 2.0, 1e-3, 1 - 1e-3)
    Z = X - 2 * p
    Zn = Z / np.sqrt(2 * p * (1 - p))
    diag = np.nanmean(Zn * Zn, axis=1)
    return (diag - 1.0) / 2.0


@cache_data
def allelic_richness(gt, _pop_labels, n_rarefaction=10):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    results = {}
    for p in np.unique(pop_labels):
        sub = gt[pop_labels == p]
        n_sub = min(sub.shape[0], n_rarefaction)
        ar_list = []
        for j in range(min(sub.shape[1], 500)):
            col = sub[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) < n_sub or n_sub < 1:
                continue
            sample = np.random.choice(valid, n_sub, replace=False)
            ar_list.append(len(np.unique(sample)))
        results[str(p)] = (round(float(np.mean(ar_list)), 3)
                           if ar_list else 0.0)
    return results


@cache_data
def f3_statistic(gt, _pop_labels, pop_A, pop_B, pop_C):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    masks = {p: (pop_labels == p) for p in [pop_A, pop_B, pop_C]}
    pA = np.nanmean(gt[masks[pop_A]], axis=0) / 2.0
    pB = np.nanmean(gt[masks[pop_B]], axis=0) / 2.0
    pC = np.nanmean(gt[masks[pop_C]], axis=0) / 2.0
    valid = np.isfinite(pA) & np.isfinite(pB) & np.isfinite(pC)
    if valid.sum() == 0:
        return np.nan, np.nan
    vals = (pA[valid] - pC[valid]) * (pB[valid] - pC[valid])
    f3 = float(np.mean(vals))
    boots = [np.mean(vals[np.random.choice(len(vals), len(vals),
                                            replace=True)])
             for _ in range(200)]
    return f3, float(np.std(boots))


@cache_data
def d_statistic_abbababa(gt, _pop_labels, pop1, pop2, pop3, pop4):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    masks = {p: (pop_labels == p) for p in [pop1, pop2, pop3, pop4]}
    p1 = np.nanmean(gt[masks[pop1]], axis=0) / 2.0
    p2 = np.nanmean(gt[masks[pop2]], axis=0) / 2.0
    p3 = np.nanmean(gt[masks[pop3]], axis=0) / 2.0
    p4 = np.nanmean(gt[masks[pop4]], axis=0) / 2.0
    valid = np.isfinite(p1) & np.isfinite(p2) & np.isfinite(p3) & np.isfinite(p4)
    if valid.sum() < 10:
        return {"D": np.nan, "Z": np.nan, "ABBA": 0, "BABA": 0}
    abba = (1 - p1[valid]) * p2[valid] * p3[valid] * (1 - p4[valid])
    baba = p1[valid] * (1 - p2[valid]) * p3[valid] * (1 - p4[valid])
    ABBA = float(np.sum(abba)); BABA = float(np.sum(baba))
    denom = ABBA + BABA
    D = (ABBA - BABA) / denom if denom > 0 else 0.0
    n_blocks = min(20, len(abba) // 5)
    if n_blocks < 2:
        return {"D": round(D, 4), "Z": np.nan,
                "ABBA": round(ABBA, 2), "BABA": round(BABA, 2)}
    block_size = len(abba) // n_blocks
    d_jack = []
    for i in range(n_blocks):
        mask = np.ones(len(abba), dtype=bool)
        mask[i * block_size:(i + 1) * block_size] = False
        a_j = np.sum(abba[mask]); b_j = np.sum(baba[mask])
        if a_j + b_j > 0:
            d_jack.append((a_j - b_j) / (a_j + b_j))
    if len(d_jack) > 1:
        d_jack = np.array(d_jack)
        se = np.sqrt((len(d_jack) - 1) / len(d_jack)
                     * np.sum((d_jack - np.mean(d_jack)) ** 2))
        Z = D / se if se > 0 else np.nan
    else:
        Z = np.nan
    return {"D": round(D, 4),
            "Z": round(float(Z), 3) if np.isfinite(Z) else np.nan,
            "ABBA": round(ABBA, 2), "BABA": round(BABA, 2),
            "interpretation": ("Introgression" if (np.isfinite(Z)
                                                    and abs(Z) > 3)
                               else "Pas d'introgression")}


@cache_data
def pbs_statistic(gt, _pop_labels, pop_test, pop_ref1, pop_ref2):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    def _fst_pair(g1, g2):
        p1 = np.nanmean(g1, axis=0) / 2.0
        p2 = np.nanmean(g2, axis=0) / 2.0
        p_bar = (p1 + p2) / 2.0
        h_s = (2 * p1 * (1 - p1) + 2 * p2 * (1 - p2)) / 2.0
        h_t = 2 * p_bar * (1 - p_bar)
        with np.errstate(divide="ignore", invalid="ignore"):
            fst = (h_t - h_s) / h_t
        return np.where(np.isfinite(fst), fst, 0.0)
    fst1 = _fst_pair(gt[pop_labels == pop_test], gt[pop_labels == pop_ref1])
    fst2 = _fst_pair(gt[pop_labels == pop_test], gt[pop_labels == pop_ref2])
    fst3 = _fst_pair(gt[pop_labels == pop_ref1], gt[pop_labels == pop_ref2])
    t1 = -np.log(1 - np.clip(fst1, 0, 0.99))
    t2 = -np.log(1 - np.clip(fst2, 0, 0.99))
    t3 = -np.log(1 - np.clip(fst3, 0, 0.99))
    return (t1 + t2 - t3) / 2.0


@cache_data
def mantel_test(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    pops = sorted(np.unique(pop_labels))
    if len(pops) < 3:
        return {"r": np.nan, "p": np.nan}
    gen_dist = []
    for i in range(len(pops)):
        for j in range(i + 1, len(pops)):
            p1 = np.nanmean(gt[pop_labels == pops[i]], axis=0) / 2.0
            p2 = np.nanmean(gt[pop_labels == pops[j]], axis=0) / 2.0
            gen_dist.append(np.nanmean((p1 - p2) ** 2))
    gen_dist = np.array(gen_dist)
    geo_dist = np.random.uniform(0, 10, len(gen_dist))
    r, _ = stats.pearsonr(gen_dist, geo_dist)
    r_perm = []
    for _ in range(500):
        idx = np.random.permutation(len(gen_dist))
        r_p, _ = stats.pearsonr(gen_dist[idx], geo_dist)
        r_perm.append(r_p)
    p_mantel = float(np.mean(np.abs(r_perm) >= abs(r)))
    return {"r": round(float(r), 4), "p": round(p_mantel, 4)}


@cache_data
def amova(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    if len(np.unique(pop_labels)) < 2:
        return {}
    X = impute_mean(gt)
    n_ind = X.shape[0]
    if n_ind > 200:
        idx = np.random.choice(n_ind, 200, replace=False)
        X = X[idx]; pop_labels = pop_labels[idx]
    D = squareform(pdist(X, metric="cityblock"))
    n = D.shape[0]
    ss_total = np.sum(D ** 2) / (2 * n)
    ss_inter = 0.0
    for p in np.unique(pop_labels):
        mask = pop_labels == p
        n_p = mask.sum()
        if n_p < 2:
            continue
        other = ~mask
        if other.sum() == 0:
            continue
        d_mean_p_other = D[np.ix_(mask, other)].mean()
        tri = np.triu_indices(n_p, 1)
        d_mean_p_p = D[np.ix_(mask, mask)][tri].mean() if n_p > 1 else 0
        ss_inter += n_p * (d_mean_p_other - d_mean_p_p) ** 2
    pct_inter = (ss_inter / ss_total) * 100 if ss_total > 0 else 0
    return {"var_intra_%": round(float(100 - pct_inter), 2),
            "var_inter_%": round(float(pct_inter), 2)}


@cache_data
def fst_sliding_window(gt, _snp_df, _pop_labels, window_size=50):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    fst = fst_per_snp(gt, pop_labels)
    n_snp = len(fst)
    results = []
    for i in range(0, n_snp - window_size + 1, window_size // 2):
        win = fst[i:i + window_size]
        if np.sum(np.isfinite(win)) < window_size // 2:
            continue
        results.append({
            "CHR": str(snp_df.iloc[i]["CHR"]),
            "start_bp": int(snp_df.iloc[i]["BP"]),
            "end_bp": int(snp_df.iloc[i + window_size - 1]["BP"]),
            "FST_mean": float(np.nanmean(win)),
            "n_snp": int(np.sum(np.isfinite(win)))})
    return pd.DataFrame(results)


@cache_data
def ld_by_breed(gt, _snp_df, _pop_labels, max_kb=500, max_snp=1000, seed=42):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    results = {}
    bp = snp_df["BP"].astype(np.float64).values
    chr_arr = snp_df["CHR"].astype(str).values
    for p in np.unique(pop_labels):
        sub = gt[pop_labels == p]
        if sub.shape[0] < 5:
            continue
        rng = np.random.default_rng(seed)
        n_snp = sub.shape[1]
        idx = (np.sort(rng.choice(n_snp, max_snp, replace=False))
               if n_snp > max_snp else np.arange(n_snp))
        X = impute_mean(sub[:, idx])
        bp_sub = bp[idx]; chr_sub = chr_arr[idx]
        r2_list, dist_list = [], []
        for i in range(len(idx) - 1):
            for j in range(i + 1, min(i + 30, len(idx))):
                if chr_sub[i] != chr_sub[j]:
                    continue
                d = abs(bp_sub[j] - bp_sub[i]) / 1000.0
                if d < 1 or d > max_kb:
                    continue
                r = np.corrcoef(X[:, i], X[:, j])[0, 1]
                if np.isfinite(r):
                    r2_list.append(r * r); dist_list.append(d)
        if r2_list:
            results[str(p)] = pd.DataFrame({"dist_kb": dist_list,
                                             "r2": r2_list})
    return results


@cache_data
def ld_heatmap_chromosome(gt, _snp_df, chrom, max_snp=100):
    snp_df = _snp_df
    mask = snp_df["CHR"].astype(str) == str(chrom)
    idx = np.where(mask)[0]
    if len(idx) < 2:
        return None, None
    if len(idx) > max_snp:
        idx = idx[np.linspace(0, len(idx) - 1, max_snp).astype(int)]
    X = impute_mean(gt[:, idx])
    X = X - X.mean(axis=0)
    std = X.std(axis=0); std[std < 1e-8] = np.nan
    X = X / std
    C = (X.T @ X) / X.shape[0]
    R2 = C ** 2
    return R2, snp_df.iloc[idx]["BP"].values


@cache_data
def detect_inversions_ld(gt, _snp_df, _pop_labels, min_snps=30):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    fst = fst_per_snp(gt, pop_labels)
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    candidates = []
    for chrom in np.unique(chr_arr):
        mask = chr_arr == chrom
        idx = np.where(mask)[0]
        if len(idx) < min_snps:
            continue
        order = np.argsort(bp[idx]); idx = idx[order]
        win_size = min(50, len(idx) // 3)
        if win_size < 5:
            continue
        for i in range(0, len(idx) - win_size, win_size // 2):
            win = idx[i:i + win_size]
            win_bp = bp[win]
            span_mb = (win_bp[-1] - win_bp[0]) / 1e6
            if span_mb < 5:
                continue
            win_fst = fst[win]
            if np.nanmean(win_fst) > 0.2 and np.nanstd(win_fst) < 0.1:
                candidates.append({
                    "CHR": str(chrom),
                    "start_bp": int(win_bp[0]),
                    "end_bp": int(win_bp[-1]),
                    "span_mb": round(span_mb, 2),
                    "FST_mean": round(float(np.nanmean(win_fst)), 4),
                    "n_snp": len(win)})
    return pd.DataFrame(candidates) if candidates else pd.DataFrame()


@cache_data
def ibd_segments(gt, _snp_df, min_snps=30, min_kb=500):
    snp_df = _snp_df
    n_ind = gt.shape[0]
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    if n_ind > 80:
        idx = np.random.choice(n_ind, 80, replace=False)
        gt = gt[idx]; n_ind = 80
    ibd_list = []
    for chrom in np.unique(chr_arr):
        mask = chr_arr == chrom
        idx_chr = np.where(mask)[0]
        if len(idx_chr) < min_snps:
            continue
        order = np.argsort(bp[idx_chr])
        idx_chr = idx_chr[order]
        for i in range(n_ind):
            for j in range(i + 1, n_ind):
                col_i = gt[i, idx_chr]; col_j = gt[j, idx_chr]
                ibs = ((col_i == col_j) & ~np.isnan(col_i)
                       & ~np.isnan(col_j))
                start = None
                for k in range(len(ibs)):
                    if ibs[k]:
                        if start is None:
                            start = k
                    else:
                        if start is not None and (k - start) >= min_snps:
                            length_kb = (bp[idx_chr[k - 1]]
                                          - bp[idx_chr[start]]) / 1000.0
                            if length_kb >= min_kb:
                                ibd_list.append({
                                    "i": int(i), "j": int(j),
                                    "CHR": str(chrom),
                                    "start_kb": float(
                                        bp[idx_chr[start]] / 1000),
                                    "length_kb": float(length_kb),
                                    "n_snp": int(k - start)})
                        start = None
    return pd.DataFrame(ibd_list) if ibd_list else pd.DataFrame()


@cache_data
def kinship_network(gt, threshold=0.15):
    K = king_kinship(gt)
    n = K.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] > threshold:
                edges.append({"i": i, "j": j, "weight": float(K[i, j])})
    return pd.DataFrame(edges) if edges else pd.DataFrame()


@cache_data
def bottleneck_detection(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    results = {}
    for p in np.unique(pop_labels):
        sub = gt[pop_labels == p]
        if sub.shape[0] < 5:
            continue
        h_obs = np.nanmean(sub == 1, axis=0)
        p_mean = np.nanmean(sub, axis=0) / 2.0
        h_exp = 2 * p_mean * (1 - p_mean)
        valid = (np.isfinite(h_obs) & np.isfinite(h_exp) & (h_exp > 0.01))
        if valid.sum() < 10:
            results[str(p)] = {"test": "insufficient_data"}
            continue
        diff = h_obs[valid] - h_exp[valid]
        try:
            _, pval = stats.wilcoxon(diff, alternative="greater")
            results[str(p)] = {
                "mean_diff": round(float(diff.mean()), 4),
                "p_value": round(float(pval), 4),
                "interpretation": ("Goulot probable" if pval < 0.05
                                   else "Pas de goulot")}
        except Exception:
            results[str(p)] = {"test": "error"}
    return results


@cache_data
def ihs_approximation(gt, _snp_df, window=50):
    snp_df = _snp_df
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    results = []
    for chrom in np.unique(chr_arr):
        idx = np.where(chr_arr == chrom)[0]
        if len(idx) < window:
            continue
        order = np.argsort(bp[idx]); idx = idx[order]
        for i in range(0, len(idx) - window, window // 2):
            win = idx[i:i + window]
            sub = gt[:, win]
            hom = np.nanmean((sub == 0) | (sub == 2))
            results.append({
                "CHR": str(chrom),
                "start_bp": int(bp[win[0]]),
                "end_bp": int(bp[win[-1]]),
                "iHS_approx": float(hom),
                "n_snp": len(win)})
    return pd.DataFrame(results) if results else pd.DataFrame()


@cache_data
def xp_ehh_approximation(gt, _snp_df, _pop_labels, pop1, pop2, window=30):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    m1 = gt[pop_labels == pop1]; m2 = gt[pop_labels == pop2]
    if m1.shape[0] < 3 or m2.shape[0] < 3:
        return pd.DataFrame()
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    results = []
    for chrom in np.unique(chr_arr):
        idx = np.where(chr_arr == chrom)[0]
        if len(idx) < window:
            continue
        order = np.argsort(bp[idx]); idx = idx[order]
        for i in range(0, len(idx) - window, window // 2):
            win = idx[i:i + window]
            hom1 = np.nanmean((m1[:, win] == 0) | (m1[:, win] == 2))
            hom2 = np.nanmean((m2[:, win] == 0) | (m2[:, win] == 2))
            xpehh = (np.log(hom1 / hom2)
                     if (hom2 > 0.01 and hom1 > 0.01) else np.nan)
            results.append({
                "CHR": str(chrom),
                "start_bp": int(bp[win[0]]),
                "end_bp": int(bp[win[-1]]),
                "XP_EHH_approx": (float(xpehh)
                                  if np.isfinite(xpehh) else np.nan),
                "hom1": float(hom1), "hom2": float(hom2)})
    return pd.DataFrame(results) if results else pd.DataFrame()


@cache_data
def clr_approximation(gt, _snp_df, window=50):
    snp_df = _snp_df
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    p_all = np.nanmean(gt, axis=0) / 2.0
    p_all = p_all[np.isfinite(p_all) & (p_all > 0) & (p_all < 1)]
    if len(p_all) < 10:
        return pd.DataFrame()
    results = []
    for chrom in np.unique(chr_arr):
        idx = np.where(chr_arr == chrom)[0]
        if len(idx) < window:
            continue
        order = np.argsort(bp[idx]); idx = idx[order]
        for i in range(0, len(idx) - window, window // 2):
            win = idx[i:i + window]
            p_local = np.nanmean(gt[:, win], axis=0) / 2.0
            p_local = p_local[np.isfinite(p_local) & (p_local > 0)
                              & (p_local < 1)]
            if len(p_local) < 5:
                continue
            lr = float(np.var(p_local) - np.var(p_all))
            results.append({
                "CHR": str(chrom),
                "start_bp": int(bp[win[0]]),
                "end_bp": int(bp[win[-1]]),
                "CLR_approx": round(lr, 5)})
    return pd.DataFrame(results) if results else pd.DataFrame()


@cache_data
def fay_wu_h(gt):
    n_ind, n_snp = gt.shape
    p = np.nanmean(gt, axis=0) / 2.0
    p = p[np.isfinite(p)]
    S = int(np.sum((p > 0) & (p < 1)))
    if S == 0:
        return {"H": np.nan, "S": 0}
    pi = 0.0
    for j in range(n_snp):
        col = gt[:, j]
        valid = col[~np.isnan(col)]
        n = len(valid)
        if n < 2:
            continue
        pj = valid.mean() / 2.0
        pi += 2 * pj * (1 - pj) * n / (n - 1)
    pi /= n_snp
    theta_h = 0.0
    for j in range(n_snp):
        col = gt[:, j]
        valid = col[~np.isnan(col)]
        n = len(valid)
        if n < 2:
            continue
        n_derived_hom = np.sum(valid == 2)
        if n_derived_hom > 0:
            theta_h += 2 * (n_derived_hom ** 2) / (n * (n - 1))
    theta_h /= n_snp
    return {"H": round(float(pi - theta_h), 6),
            "pi": round(float(pi), 6),
            "theta_H": round(float(theta_h), 6), "S": S}


# ============================================================
# BINÔMES
# ============================================================

@cache_data
def fst_per_snp_pair(gt, _pop_labels, pop_A, pop_B):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    mask_A = (pop_labels == pop_A); mask_B = (pop_labels == pop_B)
    if mask_A.sum() < 3 or mask_B.sum() < 3:
        return pd.DataFrame()
    gt_A = gt[mask_A]; gt_B = gt[mask_B]
    p_A = np.nanmean(gt_A, axis=0) / 2.0
    p_B = np.nanmean(gt_B, axis=0) / 2.0
    n_A = (~np.isnan(gt_A)).sum(axis=0)
    n_B = (~np.isnan(gt_B)).sum(axis=0)
    denom = np.where(n_A + n_B > 0, n_A + n_B, np.nan)
    p_bar = (p_A * n_A + p_B * n_B) / denom
    h_s = (2 * p_A * (1 - p_A) + 2 * p_B * (1 - p_B)) / 2.0
    h_t = 2 * p_bar * (1 - p_bar)
    with np.errstate(divide="ignore", invalid="ignore"):
        fst = (h_t - h_s) / h_t
    fst = np.clip(np.where(np.isfinite(fst), fst, np.nan), 0, 1)
    return pd.DataFrame({"FST": fst, "P_A": p_A, "P_B": p_B,
                         "P_bar": p_bar, "n_A": n_A, "n_B": n_B})


def detect_fst_outliers(fst_df, snp_df, quantile=0.99, min_fst=0.15):
    if fst_df.empty:
        return pd.DataFrame()
    df = pd.concat([snp_df.reset_index(drop=True),
                    fst_df.reset_index(drop=True)], axis=1)
    df = df.dropna(subset=["FST"])
    if df.empty:
        return pd.DataFrame()
    q_threshold = float(np.nanquantile(df["FST"], quantile))
    threshold = max(q_threshold, min_fst)
    df["is_outlier"] = df["FST"] >= threshold
    df["threshold"] = threshold
    return (df[df["is_outlier"]].sort_values("FST", ascending=False)
            .reset_index(drop=True))


@cache_data
def xp_ehh_pair(gt, _snp_df, _pop_labels, pop_A, pop_B, window=50):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    mask_A = (pop_labels == pop_A); mask_B = (pop_labels == pop_B)
    if mask_A.sum() < 3 or mask_B.sum() < 3:
        return pd.DataFrame()
    gt_A = gt[mask_A]; gt_B = gt[mask_B]
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    results = []
    for chrom in np.unique(chr_arr):
        idx_chr = np.where(chr_arr == chrom)[0]
        if len(idx_chr) < window:
            continue
        idx_chr = idx_chr[np.argsort(bp[idx_chr])]
        for i in range(0, len(idx_chr) - window, window // 2):
            win = idx_chr[i:i + window]
            hom_A = np.nanmean((gt_A[:, win] == 0) | (gt_A[:, win] == 2))
            hom_B = np.nanmean((gt_B[:, win] == 0) | (gt_B[:, win] == 2))
            xpehh = (float(np.log(hom_A / hom_B))
                     if (hom_A > 0.01 and hom_B > 0.01) else np.nan)
            results.append({
                "CHR": str(chrom),
                "start_bp": int(bp[win[0]]),
                "end_bp": int(bp[win[-1]]),
                "mid_bp": int((bp[win[0]] + bp[win[-1]]) / 2),
                "XP_EHH": xpehh,
                "hom_A": float(hom_A), "hom_B": float(hom_B),
                "n_snp": len(win)})
    if not results:
        return pd.DataFrame()
    return (pd.DataFrame(results)
            .sort_values("XP_EHH", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True))


@cache_data
def analyze_binome_complete(gt, _snp_df, _pop_labels, pop_A, pop_B,
                             quantile=0.99, min_fst=0.15):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    result = {"pop_A": pop_A, "pop_B": pop_B,
              "n_ind_A": int((pop_labels == pop_A).sum()),
              "n_ind_B": int((pop_labels == pop_B).sum())}
    fst_df = fst_per_snp_pair(gt, pop_labels, pop_A, pop_B)
    if fst_df.empty:
        result["status"] = "insufficient_data"
        return result
    outliers = detect_fst_outliers(fst_df, snp_df, quantile, min_fst)
    result["n_snp_analyzed"] = int(len(fst_df))
    result["n_snp_outliers"] = int(len(outliers))
    result["fst_mean"] = float(fst_df["FST"].mean())
    result["fst_median"] = float(fst_df["FST"].median())
    result["fst_max"] = float(fst_df["FST"].max())
    result["outliers_df"] = outliers
    xpehh_df = xp_ehh_pair(gt, snp_df, pop_labels, pop_A, pop_B)
    result["xpehh_df"] = xpehh_df
    if not xpehh_df.empty:
        result["n_windows_xpehh"] = int(len(xpehh_df))
    result["status"] = "success"
    return result


# ============================================================
# AUTO-DÉTECTION BINÔMES MDS
# ============================================================

def auto_detect_binomes_from_mds(mds_coords, pop_labels,
                                   min_dist_quantile=0.15,
                                   max_dist_quantile=0.85):
    import itertools
    mds_coords = np.asarray(mds_coords)
    pop_labels = np.asarray(pop_labels, dtype=object)
    pops = sorted(np.unique(pop_labels))
    centroids, counts = {}, {}
    for p in pops:
        mask = (pop_labels == p)
        if mask.sum() < 3:
            continue
        centroids[p] = mds_coords[mask, :2].mean(axis=0)
        counts[p] = int(mask.sum())
    if len(centroids) < 2:
        return pd.DataFrame()
    pairs = []
    for a, b in itertools.combinations(centroids.keys(), 2):
        d = float(np.linalg.norm(centroids[a] - centroids[b]))
        pairs.append({"pop_A": a, "pop_B": b, "distance_MDS": d,
                      "n_ind_A": counts[a], "n_ind_B": counts[b]})
    df = pd.DataFrame(pairs)
    if df.empty:
        return df
    q_low = df["distance_MDS"].quantile(min_dist_quantile)
    q_high = df["distance_MDS"].quantile(max_dist_quantile)
    def _cat(d):
        if d <= q_low:
            return "🟢 Proche (races sœurs)"
        elif d >= q_high:
            return "🔴 Éloignée (isolée)"
        return "🟡 Moyenne"
    df["catégorie"] = df["distance_MDS"].apply(_cat)
    return df.sort_values("distance_MDS").reset_index(drop=True)


def predict_fst_binome(distance_mds, pop_A, pop_B):
    d = float(distance_mds)
    fst_pred = min(0.35, 0.02 + 1.2 * d)
    fst_low = max(0.0, fst_pred * 0.6)
    fst_high = min(0.5, fst_pred * 1.4)
    if fst_pred < 0.08:
        interp = "🟢 Faible — Races très proches"
    elif fst_pred < 0.15:
        interp = "🟡 Modérée — Différenciation classique"
    elif fst_pred < 0.25:
        interp = "🟠 Forte — Races nettement distinctes"
    else:
        interp = "🔴 Très forte — Quasi-espèces distinctes"
    return {"pop_A": pop_A, "pop_B": pop_B,
            "distance_MDS": round(d, 4),
            "FST_prédit": round(fst_pred, 4),
            "FST_IC_95": f"[{round(fst_low, 4)}, {round(fst_high, 4)}]",
            "interprétation": interp}


def score_binome_priority(binomes_df):
    df = binomes_df.copy()
    df["score_distance"] = df["distance_MDS"] / (
        df["distance_MDS"].max() + 1e-9)
    df["ratio_taille"] = (np.minimum(df["n_ind_A"], df["n_ind_B"]) /
                          np.maximum(df["n_ind_A"], df["n_ind_B"]))
    df["score_puissance"] = (np.sqrt(df["n_ind_A"] * df["n_ind_B"]) /
                              np.sqrt(df["n_ind_A"].max() *
                                      df["n_ind_B"].max()))
    df["score_priorité"] = (0.5 * df["score_distance"] +
                             0.2 * df["ratio_taille"] +
                             0.3 * df["score_puissance"])
    df = df.sort_values("score_priorité", ascending=False).reset_index(drop=True)
    df["rang"] = df.index + 1
    return df


# ============================================================
# GTF + ANNOTATION
# ============================================================

def load_genes_from_gtf(gtf_path):
    genes = []
    opener = gzip.open if gtf_path.endswith(".gz") else open
    with opener(gtf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom = parts[0].replace("chr", "")
            gene_id, gene_name = None, None
            for attr in parts[8].split(";"):
                attr = attr.strip()
                if attr.startswith("gene_id"):
                    gene_id = attr.split('"')[1]
                elif attr.startswith("gene_name"):
                    gene_name = attr.split('"')[1]
            genes.append({"CHR": str(chrom), "start": int(parts[3]),
                          "end": int(parts[4]), "strand": parts[6],
                          "gene_id": gene_id, "gene_name": gene_name})
    return pd.DataFrame(genes)


def extract_snp_list_with_genes(outliers_df, genes_df, flank_kb=500):
    if outliers_df.empty:
        return pd.DataFrame()
    if genes_df is None or genes_df.empty:
        return outliers_df.copy()
    rows = []
    flank_bp = flank_kb * 1000
    for _, snp in outliers_df.iterrows():
        chrom = str(snp["CHR"]).replace("CHR", "").strip()
        bp = int(snp["BP"])
        genes_chr = genes_df[genes_df["CHR"].astype(str) == chrom]
        if genes_chr.empty:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None})
            continue
        candidates = []
        for _, gene in genes_chr.iterrows():
            if gene["start"] <= bp <= gene["end"]:
                candidates.append((gene["gene_name"], 0, "intragenic"))
            else:
                dist = min(abs(bp - gene["start"]), abs(bp - gene["end"]))
                if dist <= flank_bp:
                    region = ("upstream" if bp < gene["start"]
                              else "downstream")
                    candidates.append((gene["gene_name"], dist, region))
        if candidates:
            best = min(candidates, key=lambda x: x[1])
            rows.append({**snp.to_dict(), "gene_name": best[0],
                         "distance": best[1], "region": best[2]})
        else:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None})
    return pd.DataFrame(rows)


def compare_genes_across_binomes(binome_results):
    all_genes = {}
    for name, result in binome_results.items():
        if result.get("status") != "success":
            continue
        if "annotated_df" not in result:
            continue
        annotated = result["annotated_df"]
        with_genes = annotated.dropna(subset=["gene_name"])
        all_genes[name] = set(with_genes["gene_name"].unique())
    if len(all_genes) < 2:
        return pd.DataFrame(), {}, {}
    union_genes = set().union(*all_genes.values())
    intersection_genes = set.intersection(*all_genes.values())
    specific_genes = {name: genes - intersection_genes
                      for name, genes in all_genes.items()}
    rows = []
    for gene in union_genes:
        row = {"gene": gene}
        for name in all_genes:
            row[name] = "✅" if gene in all_genes[name] else "❌"
        row["partagé"] = "✅" if gene in intersection_genes else "❌"
        rows.append(row)
    return pd.DataFrame(rows), all_genes, specific_genes


# ============================================================
# GWAS LOCAL 10 cM
# ============================================================

def load_phenotypes(pheno_bytes, key_col="IID"):
    try:
        df = pd.read_csv(pheno_bytes)
        if key_col not in df.columns:
            st.error(f"❌ Colonne '{key_col}' absente.")
            return None
        df[key_col] = df[key_col].astype(str)
        return df
    except Exception as e:
        st.error(f"❌ Erreur chargement : {e}")
        return None


def local_gwas_10cm(gt, snp_df, ind_df, phenotypes,
                     target_chr, target_bp, phenotype_name,
                     pop_covariate=True, window_cm=10):
    window_bp = window_cm * 1_000_000
    chr_str = snp_df["CHR"].astype(str)
    mask = ((chr_str == str(target_chr)) &
            (snp_df["BP"] >= target_bp - window_bp) &
            (snp_df["BP"] <= target_bp + window_bp))
    idx_window = np.where(mask)[0]
    if len(idx_window) == 0:
        return pd.DataFrame()
    ind_df = ind_df.copy()
    ind_df["IID"] = ind_df["IID"].astype(str)
    pheno = phenotypes.set_index("IID")
    common_iids = [iid for iid in ind_df["IID"] if iid in pheno.index]
    if len(common_iids) < 10:
        return pd.DataFrame()
    ind_mask = ind_df["IID"].isin(common_iids).values
    gt_sub = gt[ind_mask][:, idx_window]
    ind_sub = ind_df[ind_mask].reset_index(drop=True)
    pheno_vals = []
    for iid in ind_sub["IID"]:
        try:
            val = pheno.loc[iid, phenotype_name]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            pheno_vals.append(float(val))
        except Exception:
            pheno_vals.append(np.nan)
    pheno_sub = np.array(pheno_vals)
    pop_labels_sub = ind_sub["FID"].astype(str).values
    unique_pops = sorted(set(pop_labels_sub))
    pop_dummies = np.zeros((len(pop_labels_sub), max(0, len(unique_pops) - 1)))
    for i, p in enumerate(pop_labels_sub):
        for j, up in enumerate(unique_pops[1:]):
            if p == up:
                pop_dummies[i, j] = 1.0
    results = []
    snp_window = snp_df.iloc[idx_window]
    for j in range(len(idx_window)):
        geno = gt_sub[:, j]
        valid = ~np.isnan(geno) & ~np.isnan(pheno_sub)
        if valid.sum() < 10:
            continue
        geno_v = geno[valid]
        pheno_v = pheno_sub[valid]
        if pop_covariate and pop_dummies.shape[1] > 0:
            X = np.column_stack([np.ones(valid.sum()), geno_v,
                                  pop_dummies[valid]])
        else:
            X = np.column_stack([np.ones(valid.sum()), geno_v])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, pheno_v, rcond=None)
            y_pred = X @ beta
            ss_res = np.sum((pheno_v - y_pred) ** 2)
            ss_tot = np.sum((pheno_v - np.mean(pheno_v)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            df_resid = valid.sum() - X.shape[1]
            if df_resid > 0:
                mse = ss_res / df_resid
                try:
                    cov_matrix = np.linalg.inv(X.T @ X)
                    se_beta = np.sqrt(mse * cov_matrix[1, 1])
                    t_stat = beta[1] / se_beta
                    pval = 2 * (1 - stats.t.cdf(abs(t_stat), df_resid))
                except np.linalg.LinAlgError:
                    pval, t_stat = np.nan, np.nan
            else:
                pval, t_stat = np.nan, np.nan
            results.append({
                "SNP": snp_window.iloc[j]["SNP"],
                "CHR": snp_window.iloc[j]["CHR"],
                "BP": int(snp_window.iloc[j]["BP"]),
                "beta": float(beta[1]),
                "t_stat": (float(t_stat) if np.isfinite(t_stat)
                           else np.nan),
                "p_value": (float(pval) if np.isfinite(pval)
                            else np.nan),
                "r2": float(r2),
                "n_obs": int(valid.sum())})
        except Exception:
            continue
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res["p_bonferroni"] = np.minimum(1.0,
                                              df_res["p_value"] * len(df_res))
        df_res["neg_log10_p"] = -np.log10(
            df_res["p_value"].clip(lower=1e-300))
        df_res = df_res.sort_values("p_value")
    return df_res


def test_morphometric_effect(gwas_results, phenotype_name):
    if gwas_results.empty:
        return {}
    n_sig = int((gwas_results["p_value"] < 0.05).sum())
    n_bonf = int((gwas_results["p_bonferroni"] < 0.05).sum())
    min_p = float(gwas_results["p_value"].min())
    top_snp = gwas_results.iloc[0]
    if n_bonf > 0:
        interp = (f"🔴 {n_bonf} SNP significatifs après Bonferroni "
                  f"pour **{phenotype_name}**")
    elif n_sig > 0:
        interp = f"🟡 {n_sig} SNP avec p < 0.05 (à confirmer)"
    else:
        interp = f"🟢 Aucun SNP significatif pour **{phenotype_name}**"
    return {"phenotype": phenotype_name,
            "n_snp_testés": len(gwas_results),
            "n_snp_p_005": n_sig, "n_snp_bonferroni": n_bonf,
            "p_value_min": round(min_p, 6), "top_snp": top_snp["SNP"],
            "top_beta": round(float(top_snp["beta"]), 4),
            "interprétation": interp}


# ============================================================
# INTERPRÉTATION IA
# ============================================================

def interpret_qc(qc_stats):
    pct_ind = qc_stats["excluded_ind"] / max(qc_stats["n_ind_init"], 1) * 100
    pct_snp = qc_stats["excluded_snp"] / max(qc_stats["n_snp_init"], 1) * 100
    if pct_ind < 2 and pct_snp < 10:
        v, e = "🟢 EXCELLENTE", "Qualité de génotypage très élevée."
    elif pct_ind < 5 and pct_snp < 25:
        v, e = "🟡 ACCEPTABLE", "Qualité correcte."
    else:
        v, e = "🔴 FAIBLE", "Exclusion élevée."
    return {"verdict": v, "explanation": e,
            "metrics": {"ind_exclus_%": round(pct_ind, 2),
                        "snp_exclus_%": round(pct_snp, 2)},
            "recommendation": ("Passez à l'étape suivante."
                               if "🟢" in v else "Vérifiez la qualité.")}


def interpret_maf(maf_values):
    m = maf_values[np.isfinite(maf_values)]
    mm = float(m.mean()) if len(m) else 0.0
    pct = float((m < 0.05).mean() * 100) if len(m) else 0.0
    if mm > 0.25:
        v, e = "🟢 Spectre riche", "Bonne diversité."
    elif mm > 0.15:
        v, e = "🟡 Diversité modérée", "Population moyenne."
    else:
        v, e = "🔴 Diversité faible", "Population consanguine."
    return {"verdict": v, "explanation": e,
            "metrics": {"MAF_moyen": round(mm, 4),
                        "pct_MAF<0.05": round(pct, 2)},
            "recommendation": f"Filtre MAF 0.05 élimine {pct:.1f}%."}


def interpret_fst(fst_values):
    m = float(np.nanmean(fst_values))
    if m < 0.05:
        v, e = "🟢 Faible", "Fort flux génique."
    elif m < 0.15:
        v, e = "🟡 MODÉRÉE", "Races distinctes."
    elif m < 0.25:
        v, e = "🟠 FORTE", "Races isolées."
    else:
        v, e = "🔴 TRÈS FORTE", "Quasi-espèces."
    return {"verdict": v, "explanation": e,
            "metrics": {"FST_moyen": round(m, 4)},
            "recommendation": "Utiliser PC1-PC10 comme covariables."}


def interpret_admixture(cv_results):
    ks = sorted(cv_results.keys())
    errors = [cv_results[k]["mean"] for k in ks]
    drops = [errors[i] - errors[i + 1] for i in range(len(errors) - 1)]
    k_opt = ks[int(np.argmax(drops)) + 1] if drops else ks[0]
    return {"verdict": f"🟢 K optimal = {k_opt}",
            "explanation": "Minimise la cross-entropy.",
            "metrics": {f"K={k}": round(cv_results[k]["mean"], 4)
                        for k in ks},
            "recommendation": f"Relancer avec K={k_opt}."}


def interpret_roh(froh):
    m = float(np.nanmean(froh))
    if m < 0.02:
        v, e = "🟢 Consanguinité faible", "Bonne diversité."
    elif m < 0.10:
        v, e = "🟡 Consanguinité modérée", "Niveau normal."
    else:
        v, e = "🔴 Consanguinité élevée", "Risque de dépression."
    return {"verdict": v, "explanation": e,
            "metrics": {"FROH_moyen": round(m, 4),
                        "FROH_max": round(float(np.nanmax(froh)), 4)},
            "recommendation": ("Croiser avec lignées extérieures"
                               if m > 0.10 else "Population saine.")}


def interpret_with_llm(context, api_key=None,
                       model="llama-3.3-70b-versatile"):
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ Pas de clé API Groq."
    try:
        import requests
        prompt = f"""Tu es un généticien expert en bioinformatique bovine.
Analyse ces résultats : (1) Synthèse, (2) Interprétation biologique,
(3) Points d'alerte, (4) Recommandations. Français, ≤ 400 mots.

{json.dumps(context, indent=2, default=str)[:6000]}"""
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system",
                 "content": "Expert génétique bovine."},
                {"role": "user", "content": prompt}],
                "temperature": 0.3, "max_tokens": 1200},
            timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Erreur LLM : {e}"


def render_ai_panel(module, data, key_suffix=""):
    st.divider()
    st.markdown(f"### 🤖 Interprétation IA — {module}")
    interp_map = {"QC": interpret_qc, "MAF": interpret_maf,
                  "FST": interpret_fst, "Admixture": interpret_admixture,
                  "ROH": interpret_roh}
    if module in interp_map:
        result = interp_map[module](data)
        st.success(f"**{result['verdict']}** — {result['explanation']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Métriques**"); st.json(result["metrics"])
        with c2:
            st.markdown("**✅ Recommandation**")
            st.info(result["recommendation"])
    with st.expander("🧠 Analyse LLM approfondie (Groq gratuit)"):
        st.caption("Clé gratuite : https://console.groq.com/keys")
        k = st.text_input("Clé API Groq", type="password",
                          key=f"groq_{module}_{key_suffix}")
        if st.button(f"🚀 Analyser {module}",
                     key=f"btn_llm_{module}_{key_suffix}"):
            ctx = (data if isinstance(data, dict)
                   else {"raw": str(data)[:2000]})
            with st.spinner("Analyse LLM..."):
                st.markdown(interpret_with_llm(ctx, api_key=k))


# ============================================================
# EXPORT
# ============================================================

def _dosage_to_plink_bits(col):
    n = len(col); bits = np.zeros(n, dtype=np.uint8)
    bits[np.isnan(col)] = 0b01
    bits[col == 0] = 0b11; bits[col == 1] = 0b10; bits[col == 2] = 0b00
    return bits


def build_plink_bed(gt):
    n_ind, n_snp = gt.shape
    n_bytes = (n_ind + 3) // 4
    buf = bytearray([0x6C, 0x1B, 0x01])
    for j in range(n_snp):
        bits = _dosage_to_plink_bits(gt[:, j])
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for k in range(n_ind):
            packed[k // 4] |= (bits[k] & 0x03) << (2 * (k % 4))
        buf.extend(packed.tobytes())
    return bytes(buf)


def build_plink_bim(snp_df):
    lines = []
    has_a1 = "A1" in snp_df.columns
    has_a2 = "A2" in snp_df.columns
    for _, r in snp_df.iterrows():
        a1 = str(r["A1"]) if has_a1 else "A"
        a2 = str(r["A2"]) if has_a2 else "G"
        if a1 in ("nan", ""): a1 = "A"
        if a2 in ("nan", "", a1): a2 = "G" if a1 == "A" else "A"
        lines.append(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t"
                     f"{int(r['BP'])}\t{a1}\t{a2}")
    return "\n".join(lines) + "\n"


def build_plink_fam(ind_df):
    return "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9"
                     for _, r in ind_df.iterrows()) + "\n"


def build_vcf_output(gt, ind_df, snp_df, project="BovineSNP"):
    n_ind, n_snp = gt.shape
    has_a1 = "A1" in snp_df.columns
    has_a2 = "A2" in snp_df.columns
    header = ["##fileformat=VCFv4.2",
              f"##source=BovineSNPPlatform-{project}",
              f"##fileDate={datetime.now():%Y%m%d}",
              ('##FORMAT=<ID=GT,Number=1,Type=String,'
               'Description="Genotype">')]
    samples = ind_df["IID"].astype(str).tolist()
    header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                  + "\t".join(samples))
    body = []
    for j in range(n_snp):
        r = snp_df.iloc[j]
        ref = str(r["A1"]) if has_a1 else "A"
        alt = str(r["A2"]) if has_a2 else "G"
        if ref in ("nan", ""): ref = "A"
        if alt in ("nan", "", ref): alt = "G" if ref == "A" else "A"
        gts = []
        for i in range(n_ind):
            v = gt[i, j]
            gts.append("./." if np.isnan(v) else
                       "0/0" if v == 0 else
                       "0/1" if v == 1 else "1/1")
        body.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t{ref}\t{alt}"
                    f"\t.\tPASS\t.\tGT\t" + "\t".join(gts))
    return "\n".join(header + body) + "\n"


def build_plink_zip(gt, ind_df, snp_df, prefix="bovine_qc"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", build_plink_bed(gt))
        zf.writestr(f"{prefix}.bim", build_plink_bim(snp_df))
        zf.writestr(f"{prefix}.fam", build_plink_fam(ind_df))
    return bio.getvalue()


# ============================================================
# VISUALISATION
# ============================================================

def plot_hist(values, title, xlabel, color="#3498db"):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=values, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Fréquence", height=380,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missingness / individu",
                                        "Missingness / SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60,
                               marker_color="skyblue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60,
                               marker_color="coral"), row=1, col=2)
    fig.update_layout(height=400, showlegend=False,
                      margin=dict(l=40, r=20, t=60, b=40))
    return fig


def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Population",
                     title=f"PCA — PC1 ({var_pct[0]:.1f}%) vs "
                           f"PC2 ({var_pct[1]:.1f}%)", height=550)
    fig.update_traces(marker=dict(size=10,
                                   line=dict(width=1, color="white")))
    return fig


def plot_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS (IBS)", height=550)
    fig.update_traces(marker=dict(size=10,
                                   line=dict(width=1, color="white")))
    return fig


def plot_manhattan(fst, chr_col, threshold_q=0.999):
    df = pd.DataFrame({"FST": np.asarray(fst),
                       "CHR": np.asarray(chr_col)})
    df = df.dropna(subset=["FST"]).reset_index(drop=True)
    if df.empty:
        return None
    order = df["CHR"].map(_chr_sort_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df["_k2"] = order.map(lambda t: t[2])
    df = df.sort_values(["_k0", "_k1", "_k2"]).reset_index(drop=True)
    cumulative, ticks, labels = 0, [], []
    cum_pos = np.zeros(len(df), dtype=int)
    for chrom in df["CHR"].unique():
        sub_idx = df.index[df["CHR"] == chrom]
        s, e = cumulative, cumulative + len(sub_idx)
        cum_pos[s:e] = np.arange(s, e)
        ticks.append((s + e) / 2); labels.append(str(chrom))
        cumulative = e
    df["x"] = cum_pos
    fig = go.Figure()
    for idx_chr, chrom in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == chrom]
        color = "#2c3e50" if idx_chr % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"], mode="markers",
                                 marker=dict(size=5, color=color),
                                 showlegend=False, hoverinfo="skip"))
    q_upper = float(np.nanquantile(fst, threshold_q))
    fig.add_hline(y=q_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Top {100*(1-threshold_q):.1f}% "
                                  f"= {q_upper:.4f}",
                  annotation_position="top right")
    fig.update_layout(title="Manhattan Plot — FST/SNP",
                      xaxis_title="Chromosome", yaxis_title="FST",
                      height=500, margin=dict(l=40, r=20, t=60, b=40))
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty:
        return None
    d = ld_df.copy(); d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb
    agg = d.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(agg, x="bin", y="r2",
                  labels={"bin": "Distance (kb)", "r2": "r² moyen"},
                  title="LD decay", height=450)
    fig.update_traces(line=dict(color="royalblue", width=3))
    return fig


def plot_kinship_heatmap(G, labels):
    labels = [str(x) for x in labels]
    n = len(labels)
    seen, uniq = {}, []
    for lab in labels:
        seen[lab] = seen.get(lab, 0) + 1
        uniq.append(lab if seen[lab] == 1 else f"{lab}#{seen[lab]}")
    vmin = float(np.nanpercentile(G, 1))
    vmax = float(np.nanpercentile(G, 99))
    if not np.isfinite(vmin) or vmin == vmax:
        vmin, vmax = -0.3, 0.5
    custom = np.stack([np.repeat(uniq, n).reshape(n, n),
                       np.tile(uniq, n).reshape(n, n)], axis=-1)
    fig = go.Figure(data=go.Heatmap(
        z=G, colorscale="RdBu", zmid=0.0, zmin=vmin, zmax=vmax,
        colorbar=dict(title="Kinship"),
        hovertemplate="%{customdata[0]} × %{customdata[1]}<br>"
                      "φ = %{z:.3f}<extra></extra>", customdata=custom))
    fig.update_layout(title="Matrice de parenté (KING robust)", height=650,
                      xaxis=dict(showticklabels=False),
                      yaxis=dict(showticklabels=False,
                                 autorange="reversed"))
    return fig


def plot_admixture(Q, labels, pop_labels, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
    df["IID"] = labels; df["Pop"] = pop_labels
    df["_dom"] = Q.argmax(1)
    df["_po"] = pd.Categorical(pop_labels,
                                categories=sorted(set(pop_labels)))
    df = df.sort_values(["_po", "_dom"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    palette = (px.colors.qualitative.Set2
               + px.colors.qualitative.Set3)[:K]
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(
            x=df["x"], y=df[f"K{k+1}"], name=f"Composante {k+1}",
            marker_color=palette[k % len(palette)],
            hovertemplate="Ind %{customdata}<br>K" + str(k + 1)
                          + " = %{y:.2f}<extra></extra>",
            customdata=df["IID"]))
    fig.update_layout(barmode="stack",
                      title=f"Proportions d'ancestralité (K={K})",
                      xaxis_title="Individus triés",
                      yaxis_title="Proportion", height=500)
    return fig, df


def plot_fst_pairwise(matrix, pops):
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=pops, y=pops, colorscale="Viridis",
        colorbar=dict(title="FST"),
        text=np.round(matrix, 4), texttemplate="%{text}",
        hovertemplate="%{y} vs %{x}<br>FST = %{z:.4f}<extra></extra>"))
    fig.update_layout(title="FST pairwise", height=550)
    return fig


def plot_roh_histogram(froh, labels):
    df = pd.DataFrame({"FROH": froh, "Population": labels})
    return px.histogram(df, x="FROH", color="Population", nbins=50,
                        title="Distribution de FROH", height=450)


def plot_ne_curves(ne_dict):
    fig = go.Figure()
    palette = (px.colors.qualitative.Set2 + px.colors.qualitative.Set3)
    for i, (race, df) in enumerate(ne_dict.items()):
        if df is None or df.empty:
            continue
        fig.add_trace(go.Scatter(x=df["GenAgo"], y=df["Ne"],
                                 mode="lines+markers", name=race,
                                 line=dict(
                                     color=palette[i % len(palette)],
                                     width=3)))
    fig.update_layout(title="Ne historique",
                      xaxis_title="Générations passées",
                      yaxis_title="Ne (relatif)", height=550)
    return fig


def plot_cv_error(cv_results):
    ks = sorted(cv_results.keys())
    means = [cv_results[k]["mean"] for k in ks]
    stds = [cv_results[k]["std"] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=means, mode="lines+markers",
                             error_y=dict(type="data", array=stds),
                             line=dict(color="crimson", width=3),
                             marker=dict(size=12)))
    k_opt = ks[int(np.argmin(means))]
    fig.add_vline(x=k_opt, line_dash="dash", line_color="green",
                  annotation_text=f"K optimal = {k_opt}")
    fig.update_layout(title="Cross-Entropy Error",
                      xaxis_title="K", yaxis_title="CV Error", height=450)
    return fig


def plot_sfs(sfs_df, unfolded=True):
    if sfs_df.empty:
        return None
    fig = px.bar(sfs_df, x="freq", y="count",
                 title=f"SFS ({'déplié' if unfolded else 'replié'})",
                 labels={"freq": "Fréquence", "count": "Nb SNPs"})
    fig.update_traces(marker_color="steelblue")
    return fig


def plot_pi_distribution(pi_arr):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pi_arr, nbinsx=60,
                                marker_color="seagreen"))
    fig.update_layout(title="Distribution de π",
                      xaxis_title="π par SNP",
                      yaxis_title="Fréquence", height=400)
    return fig


def plot_ld_by_breed(ld_dict, bin_kb=20):
    if not ld_dict:
        return None
    fig = go.Figure()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    for i, (breed, df) in enumerate(ld_dict.items()):
        d = df.copy()
        d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb
        agg = d.groupby("bin")["r2"].mean().reset_index()
        fig.add_trace(go.Scatter(x=agg["bin"], y=agg["r2"],
                                 mode="lines+markers", name=breed,
                                 line=dict(
                                     color=palette[i % len(palette)],
                                     width=3)))
    fig.update_layout(title="LD decay par race",
                      xaxis_title="Distance (kb)",
                      yaxis_title="r² moyen", height=500)
    return fig


def plot_ld_heatmap(R2):
    if R2 is None:
        return None
    fig = go.Figure(data=go.Heatmap(z=R2, colorscale="Hot",
                                     colorbar=dict(title="r²")))
    fig.update_layout(title="Heatmap LD (r²)", height=600)
    return fig


def plot_kinship_network(edges_df, ind_df):
    if edges_df.empty:
        return None
    fig = go.Figure()
    n = len(ind_df)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos_x = np.cos(angles) * 10
    pos_y = np.sin(angles) * 10
    for _, row in edges_df.iterrows():
        i, j = int(row["i"]), int(row["j"])
        fig.add_trace(go.Scatter(x=[pos_x[i], pos_x[j]],
                                 y=[pos_y[i], pos_y[j]], mode="lines",
                                 line=dict(width=0.5, color="#888"),
                                 showlegend=False, hoverinfo="none"))
    pops = ind_df["FID"].astype(str).values
    unique_pops = sorted(set(pops))
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Set3
    color_map = {p: palette[i % len(palette)]
                 for i, p in enumerate(unique_pops)}
    node_colors = [color_map[p] for p in pops]
    fig.add_trace(go.Scatter(
        x=pos_x, y=pos_y, mode="markers",
        marker=dict(size=10, color=node_colors,
                    line=dict(width=1, color="white")),
        text=ind_df["IID"].astype(str).values,
        hovertemplate="%{text}<extra></extra>", showlegend=False))
    fig.update_layout(title="Réseau de parenté", height=600,
                      showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False,
                                 visible=False),
                      yaxis=dict(showgrid=False, zeroline=False,
                                 visible=False))
    return fig


def plot_ibd_heatmap(ibd_df, n_ind):
    if ibd_df.empty:
        return None
    mat = np.zeros((n_ind, n_ind))
    for _, row in ibd_df.iterrows():
        i, j = int(row["i"]), int(row["j"])
        mat[i, j] += row["length_kb"]
        mat[j, i] += row["length_kb"]
    fig = go.Figure(data=go.Heatmap(
        z=mat, colorscale="Blues",
        colorbar=dict(title="Longueur IBD (kb)")))
    fig.update_layout(title="Matrice IBD", height=650,
                      xaxis=dict(showticklabels=False),
                      yaxis=dict(showticklabels=False,
                                 autorange="reversed"))
    return fig


def plot_gwas_manhattan_local(gwas_df, chr_name, target_bp):
    if gwas_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gwas_df["BP"], y=gwas_df["neg_log10_p"], mode="markers",
        marker=dict(size=8, color=gwas_df["neg_log10_p"],
                    colorscale="Hot", showscale=False),
        hovertemplate="SNP: %{text}<br>BP: %{x}<br>"
                      "-log10(p): %{y:.2f}<extra></extra>",
        text=gwas_df["SNP"]))
    fig.add_hline(y=-np.log10(0.05), line_dash="dash",
                  line_color="orange", annotation_text="p=0.05")
    fig.add_hline(y=-np.log10(0.05 / len(gwas_df)), line_dash="dash",
                  line_color="red", annotation_text="Bonferroni")
    fig.add_vline(x=target_bp, line_dash="dot", line_color="blue",
                  annotation_text="SNP cible")
    fig.update_layout(title=f"GWAS local — Chr {chr_name}",
                      xaxis_title="Position (bp)",
                      yaxis_title="-log10(p)", height=450)
    return fig


# ============================================================
# RAPPORT
# ============================================================

def build_report_html(config, stats, figures=None, interpretations=None):
    figures = figures or {}
    interpretations = interpretations or {}
    first, blocks = True, []
    for title, fig in figures.items():
        if fig is None:
            continue
        html = fig.to_html(full_html=False,
                           include_plotlyjs="cdn" if first else False)
        first = False
        blocks.append(f"<h2>{title}</h2>{html}")
    interp_html = ""
    for module, interp in interpretations.items():
        interp_html += f"""
        <div style="background:#eafaf1;padding:12px;border-radius:8px;margin:10px 0;">
          <h3>{module}</h3>
          <p><b>{interp.get('verdict','')}</b> — {interp.get('explanation','')}</p>
          <p><b>Métriques :</b> {interp.get('metrics', {})}</p>
          <p><b>Recommandation :</b> {interp.get('recommendation','')}</p>
        </div>"""
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="UTF-8">
<title>Rapport Bovine SNP Platform v6.0</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; color: #222; }}
 h1 {{ color: #1a5276; border-bottom: 3px solid #1a5276; padding-bottom: 8px; }}
 h2 {{ color: #2471a3; margin-top: 28px; }}
 .summary {{ background: #f4f6f7; padding: 16px; border-radius: 8px; }}
</style></head><body>
<h1>🐄 Rapport Bovine SNP Platform v6.0</h1>
<p><b>Projet :</b> {config.get('project_name','N/A')} —
   <b>Date :</b> {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="summary"><h2>Résumé exécutif</h2><ul>
 <li>Individus analysés : <b>{stats['n_ind_final']}</b> (sur {stats['n_ind_init']})</li>
 <li>SNPs retenus : <b>{stats['n_snp_final']}</b> (sur {stats['n_snp_init']})</li>
 <li>Individus exclus : <b>{stats['excluded_ind']}</b></li>
 <li>SNPs exclus : <b>{stats['excluded_snp']}</b></li>
</ul></div>
{interp_html}
{''.join(blocks)}
<hr><p style="font-size:0.85em;color:#666">
Rapport généré par Bovine SNP Platform v6.0.</p>
</body></html>
"""


# ============================================================
# ÉTAT
# ============================================================

_STATE_KEYS = ["gt", "ind_df", "snp_df",
               "gt_filt", "ind_filt", "snp_filt", "qc_stats",
               "gt_snp", "ind_snp", "snp_snp",
               "gt_pruned", "snp_pruned",
               "pca_scores", "pca_var", "mds_coords", "king_matrix",
               "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
               "admixture_Q", "admixture_K", "cv_results",
               "roh_df", "froh", "ne_dict", "reynolds_D",
               "reynolds_pops", "selection_df", "run_requested",
               "_plink_zip", "_vcf_str", "_report_html",
               "_binome_result", "_all_binomes", "_custom_pairs",
               "_genes_df", "_annotated",
               "_pi_arr", "_pi_mean", "_fgrm", "_pbs", "_fst_win",
               "_ld_breed", "_inv", "_ibd", "_net", "_ihs", "_xpehh",
               "_gwas_results"]


def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None


def invalidate_downstream():
    for k in ["pca_scores", "pca_var", "mds_coords", "king_matrix",
              "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
              "admixture_Q", "admixture_K", "cv_results",
              "roh_df", "froh", "ne_dict", "reynolds_D",
              "reynolds_pops", "selection_df", "gt_pruned",
              "snp_pruned", "_binome_result", "_all_binomes",
              "_gwas_results"]:
        st.session_state[k] = None


def reset_all_derived():
    for k in (["gt_filt", "ind_filt", "snp_filt", "qc_stats",
               "gt_snp", "ind_snp", "snp_snp"] +
              ["pca_scores", "pca_var", "mds_coords", "king_matrix",
               "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
               "admixture_Q", "admixture_K", "cv_results",
               "roh_df", "froh", "ne_dict", "reynolds_D",
               "reynolds_pops", "selection_df", "gt_pruned",
               "snp_pruned", "_binome_result", "_all_binomes",
               "_gwas_results"]):
        st.session_state[k] = None


def has_data():
    return st.session_state.gt is not None


def has_qc():
    return (st.session_state.gt_filt is not None
            and st.session_state.qc_stats is not None
            and st.session_state.ind_filt is not None
            and st.session_state.snp_filt is not None)


def has_pruned():
    return (st.session_state.gt_pruned is not None
            and st.session_state.snp_pruned is not None)


def has_snp_state():
    return (st.session_state.gt_snp is not None
            and st.session_state.snp_snp is not None)


# ============================================================
# HELPERS BINÔMES
# ============================================================

def _run_binome_analysis(gt, snp_df, pop_labels, pop_A, pop_B,
                          quantile, min_fst):
    try:
        with st.spinner(f"Analyse {pop_A} vs {pop_B}..."):
            result = analyze_binome_complete(
                gt, snp_df, pop_labels, pop_A, pop_B, quantile, min_fst)
            st.session_state["_binome_result"] = result
        if result.get("status") == "success":
            st.success(f"✅ {pop_A} vs {pop_B} : "
                       f"{result['n_snp_outliers']} SNP outliers")
        else:
            st.error(f"❌ {result.get('status', 'Erreur')}")
    except Exception as e:
        st.error(f"❌ {e}")
        import traceback
        st.code(traceback.format_exc())


def _display_binome_results(pop_A, pop_B):
    result = st.session_state.get("_binome_result")
    if not result or result.get("status") != "success":
        return
    if result.get("pop_A") != pop_A or result.get("pop_B") != pop_B:
        return
    st.divider()
    st.markdown(f"### 📊 Résultats — {pop_A} vs {pop_B}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ind. " + pop_A, result["n_ind_A"])
    c2.metric("Ind. " + pop_B, result["n_ind_B"])
    c3.metric("SNPs analysés", f"{result['n_snp_analyzed']:,}")
    c4.metric("SNP outliers", result["n_snp_outliers"])
    c5.metric("FST moyen", f"{result['fst_mean']:.4f}")
    c1, c2, c3 = st.columns(3)
    c1.metric("FST médian", f"{result['fst_median']:.4f}")
    c2.metric("FST max", f"{result['fst_max']:.4f}")
    c3.metric("Fenêtres XP-EHH", result.get("n_windows_xpehh", 0))
    outliers = result["outliers_df"]
    if not outliers.empty:
        st.dataframe(
            outliers[["CHR", "SNP", "BP", "FST", "P_A", "P_B"]].head(100),
            use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=outliers["FST"], nbinsx=40,
                                    marker_color="crimson"))
        fig.update_layout(title=f"Distribution FST — {pop_A} vs {pop_B}",
                          xaxis_title="FST", yaxis_title="Nb SNPs",
                          height=400)
        st.plotly_chart(fig, use_container_width=True)
        csv_out = outliers.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇ SNP outliers ({pop_A} vs {pop_B})", data=csv_out,
            file_name=f"outliers_{pop_A}_vs_{pop_B}.csv",
            mime="text/csv", use_container_width=True,
            key=f"dl_out_{pop_A}_{pop_B}")
    xpehh_df = result.get("xpehh_df", pd.DataFrame())
    if not xpehh_df.empty:
        st.markdown("### ⚡ XP-EHH")
        st.dataframe(xpehh_df.head(30), use_container_width=True)
        fig = px.scatter(xpehh_df, x="mid_bp", y="XP_EHH", color="CHR",
                         title=f"XP-EHH — {pop_A} vs {pop_B}", height=450)
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def _display_all_binomes_summary():
    if not st.session_state.get("_all_binomes"):
        return
    st.divider()
    st.markdown("### 📊 Tableau de synthèse global")
    summary = []
    for name, r in st.session_state["_all_binomes"].items():
        if r.get("status") == "success":
            summary.append({
                "Binôme": name.replace("_vs_", " ↔ "),
                "Ind. A": r["n_ind_A"], "Ind. B": r["n_ind_B"],
                "SNPs": r["n_snp_analyzed"],
                "Outliers": r["n_snp_outliers"],
                "FST moyen": round(r["fst_mean"], 4),
                "FST max": round(r["fst_max"], 4)})
    if not summary:
        return
    sum_df = pd.DataFrame(summary).sort_values("FST moyen", ascending=False)
    st.dataframe(sum_df, use_container_width=True)
    fig = px.bar(sum_df, x="Binôme", y="FST moyen",
                 color="FST moyen", color_continuous_scale="Reds",
                 title="FST moyen par binôme", height=450)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    csv_sum = sum_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Tableau (CSV)", data=csv_sum,
                       file_name="binomes_synthese.csv", mime="text/csv",
                       use_container_width=True, key="dl_summary_flex")


# ============================================================
# INTERFACE
# ============================================================

def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v6.0")
    st.caption("Pipeline + analyses avancées + binômes + GWAS 10cM")

    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :",
                        ["Demo", "Upload PED/MAP", "Upload VCF"], index=0)

        if mode == "Demo":
            c1, c2 = st.columns(2)
            n_ind = c1.number_input("Individus", 20, 1000, 150, 10)
            n_snp = c2.number_input("SNPs", 100, 10000, 800, 100)
            n_pop = st.slider("Populations", 2, 10, 4)
            if st.button("🎲 Générer le jeu de démo",
                         use_container_width=True):
                with st.spinner("Génération..."):
                    gt, ind_df, snp_df = generate_demo_data(
                        int(n_ind), int(n_snp), int(n_pop))
                    ind_df = reconstruct_fid_from_iid(ind_df)
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind_df
                    st.session_state.snp_df = snp_df
                    reset_all_derived()
                st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")

        elif mode == "Upload PED/MAP":
            st.info("Formats : .ped, .ped.gz, .map, .map.gz")
            ped_f = st.file_uploader("Fichier .ped",
                                     type=["ped", "gz", "txt"])
            map_f = st.file_uploader("Fichier .map",
                                     type=["map", "gz", "txt"])
            if ped_f and map_f:
                if st.button("📥 Charger PED + MAP",
                             use_container_width=True):
                    try:
                        with st.spinner("Parsing .map..."):
                            map_df, _ = parse_map(map_f.read())
                        with st.spinner(
                                f"Parsing .ped "
                                f"({ped_f.size/1024/1024:.1f} Mo)..."):
                            gt, ind_df, _ = parse_ped(ped_f.read(),
                                                       len(map_df))
                        if len(map_df) != gt.shape[1]:
                            map_df = map_df.iloc[:gt.shape[1]].reset_index(
                                drop=True)
                        ind_df = reconstruct_fid_from_iid(ind_df)
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = map_df
                        reset_all_derived()
                        st.success(f"✅ {gt.shape[0]} ind × "
                                   f"{gt.shape[1]} SNPs")
                    except Exception as e:
                        st.error(f"❌ {e}")

        else:
            vcf_f = st.file_uploader("Fichier .vcf",
                                     type=["vcf", "gz", "txt"])
            if vcf_f is not None:
                if st.button("📥 Charger VCF",
                             use_container_width=True):
                    try:
                        with st.spinner("Parsing VCF..."):
                            gt, ind_df, snp_df, _, _ = parse_vcf(
                                vcf_f.read())
                        ind_df = reconstruct_fid_from_iid(ind_df)
                        st.session_state.gt = gt
                        st.session_state.ind_df = ind_df
                        st.session_state.snp_df = snp_df
                        reset_all_derived()
                        st.success(f"✅ {gt.shape[0]} ind × "
                                   f"{gt.shape[1]} variants")
                    except Exception as e:
                        st.error(f"❌ {e}")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("Missingness SNP", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["geno"], 0.01)
        mind = st.slider("Missingness individu", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["mind"], 0.01)
        maf_thr = st.slider("MAF minimal", 0.0, 0.5,
                            DEFAULT_THRESHOLDS["maf"], 0.01)
        hwe_thr = st.number_input("HWE p-value",
                                  value=DEFAULT_THRESHOLDS["hwe"],
                                  format="%.0e")
        het_sd = st.slider("Hétérozygotie ±σ", 1.0, 5.0,
                           DEFAULT_THRESHOLDS["het_sd"], 0.1)
        ld_r2 = st.slider("LD Pruning r²", 0.05, 0.5,
                          DEFAULT_THRESHOLDS["ld_r2"], 0.05)
        king_cutoff = st.slider("KING cutoff", 0.1, 0.5,
                                DEFAULT_THRESHOLDS["king_cutoff"], 0.01)
        st.divider()
        if st.button("🚀 Pipeline complet", type="primary",
                     use_container_width=True):
            if not has_data():
                st.error("Chargez d'abord des données.")
            else:
                st.session_state.run_requested = True
        st.divider()
        st.caption("v6.0 — Double état + auto-MDS + GWAS")

    if not has_data():
        st.info("👉 Générez un jeu de démo ou importez PED/MAP ou VCF.")
        st.markdown("""
        ### 🧬 Pipeline v6.0 — 13 onglets

        | Onglet | Contenu |
        |---|---|
        | 🏠 Aperçu | Statistiques |
        | 🧹 QC | Filtres cascade (double état) |
        | ✂️ LD Pruning | Indep-pairwise |
        | 🧬 Structure | PCA, MDS, KING |
        | 🎨 Admixture | NMF + CV |
        | 📈 Démographie | LD decay, ROH, Ne |
        | 🔍 Sélection | FST, Manhattan |
        | 🌳 Phylogénie | Reynolds + NJ |
        | 📤 Export | PLINK ZIP, VCF |
        | 📄 Rapport | HTML |
        | 🔬 Analyses Avancées | 25+ analyses |
        | 🧬 Binômes | 5 modes (dont auto-MDS) |
        | 🧪 **SNP & Gènes (GWAS)** | Extraction + GWAS 10cM |
        """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "✂️ LD Pruning",
                    "🧬 Structure", "🎨 Admixture", "📈 Démographie",
                    "🔍 Sélection", "🌳 Phylogénie", "📤 Export",
                    "📄 Rapport", "🔬 Analyses Avancées", "🧬 Binômes",
                    "🧪 SNP & Gènes (GWAS)"])

    # ============ TAB 1 : Aperçu ============
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        st.subheader("Individus")
        st.dataframe(ind_df.head(20), use_container_width=True)
        st.subheader("SNPs")
        st.dataframe(snp_df.head(5), use_container_width=True)

    # ============ TAB 2 : QC ============
    with tabs[1]:
        st.subheader("🧹 Contrôle Qualité")
        if st.button("🧬 Filtrer autosomes 1-29", use_container_width=True):
            try:
                gt_a, snp_a = filter_autosomes(st.session_state.gt,
                                               st.session_state.snp_df)
                st.session_state.gt = gt_a
                st.session_state.snp_df = snp_a
                st.success(f"✅ Autosomes → {gt_a.shape[1]} SNPs")
            except Exception as e:
                st.error(f"❌ {e}")
        if st.button("▶ Lancer le QC complet", type="primary",
                     use_container_width=True):
            try:
                with st.spinner("QC..."):
                    params = {"geno": geno, "mind": mind, "maf": maf_thr,
                              "hwe": hwe_thr, "het_sd": het_sd}
                    (gt_f, ind_f, snp_f, qc_stats,
                     gt_s, ind_s, snp_s) = apply_qc_filters(
                        st.session_state.gt, st.session_state.ind_df,
                        st.session_state.snp_df, params)
                    st.session_state.gt_filt = gt_f
                    st.session_state.ind_filt = ind_f
                    st.session_state.snp_filt = snp_f
                    st.session_state.gt_snp = gt_s
                    st.session_state.ind_snp = ind_s
                    st.session_state.snp_snp = snp_s
                    st.session_state.qc_stats = qc_stats
                    invalidate_downstream()
                st.success(
                    f"✅ QC terminé.\n"
                    f"• Structure : {gt_f.shape[0]} ind × "
                    f"{gt_f.shape[1]:,} SNP\n"
                    f"• SNP différenciés : {gt_s.shape[0]} ind × "
                    f"{gt_s.shape[1]:,} SNP")
            except Exception as e:
                st.error(f"❌ {e}")
        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind. finaux", s["n_ind_final"])
            c2.metric("SNPs finaux", s["n_snp_final"])
            c3.metric("Exclus (ind)", s["excluded_ind"])
            c4.metric("Exclus (SNPs)", s["excluded_snp"])
            gt_full = st.session_state.gt
            st.plotly_chart(plot_missingness_dashboard(
                missingness_per_ind(gt_full),
                missingness_per_snp(gt_full)),
                use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_hist(maf(gt_full), "Spectre MAF",
                                          "MAF", "#2ecc71"),
                                use_container_width=True)
            with c2:
                st.plotly_chart(plot_hist(heterozygosity(gt_full),
                                          "Hétérozygotie",
                                          "HET", "#9b59b6"),
                                use_container_width=True)
            render_ai_panel("QC", s, "tab_qc")

    # ============ TAB 3 : LD Pruning ============
    with tabs[2]:
        st.subheader("✂️ LD Pruning")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if st.button("▶ Lancer le LD Pruning", type="primary",
                         use_container_width=True):
                try:
                    with st.spinner("LD pruning..."):
                        keep = ld_pruning(st.session_state.gt_filt,
                                          window=50, step=5, r2_thr=ld_r2)
                        st.session_state.gt_pruned = (
                            st.session_state.gt_filt[:, keep])
                        st.session_state.snp_pruned = (
                            st.session_state.snp_filt[keep]
                            .reset_index(drop=True))
                    st.success(f"✅ {st.session_state.gt_pruned.shape[1]} "
                               f"SNPs conservés.")
                except Exception as e:
                    st.error(f"❌ {e}")
            if has_pruned():
                with st.spinner("LD decay..."):
                    ld_df = ld_decay(
                        st.session_state.gt_pruned,
                        st.session_state.snp_pruned["BP"].values,
                        max_kb=1000, max_snp=1000)
                    st.session_state.ld_df = ld_df
                fig = plot_ld_decay(ld_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 4 : Structure ============
    with tabs[3]:
        st.subheader("🧬 Structure")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            if st.button("▶ PCA + MDS + KING", type="primary",
                         use_container_width=True):
                try:
                    with st.spinner("PCA..."):
                        s_, v_ = pca_analysis(gt_use, 10)
                        st.session_state.pca_scores = s_
                        st.session_state.pca_var = v_
                    with st.spinner("MDS..."):
                        st.session_state.mds_coords = mds_analysis(gt_use, 5)
                    with st.spinner("KING..."):
                        st.session_state.king_matrix = king_kinship(gt_use)
                    st.success("✅ Terminé.")
                except Exception as e:
                    st.error(f"❌ {e}")
            labels = st.session_state.ind_filt["FID"].values
            if st.session_state.pca_scores is not None:
                st.plotly_chart(plot_pca(st.session_state.pca_scores,
                                         st.session_state.pca_var, labels),
                                use_container_width=True)
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plot_mds(st.session_state.mds_coords,
                                         labels),
                                use_container_width=True)
            if st.session_state.king_matrix is not None:
                id_labels = (st.session_state.ind_filt["FID"].astype(str)
                             + "_"
                             + st.session_state.ind_filt["IID"].astype(str)
                             ).values
                st.plotly_chart(plot_kinship_heatmap(
                    st.session_state.king_matrix, id_labels),
                    use_container_width=True)

    # ============ TAB 5 : Admixture ============
    with tabs[4]:
        st.subheader("🎨 Admixture")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            c1, c2 = st.columns(2)
            K_max = c1.slider("K max", 3, 10, 6)
            n_reps = c2.slider("Répétitions CV", 1, 5, 3)
            if st.button("▶ CV error", use_container_width=True):
                try:
                    with st.spinner("CV..."):
                        cv = admix_cv_error(gt_use, K_range=(2, K_max + 1),
                                            n_reps=int(n_reps))
                        st.session_state.cv_results = cv
                    st.success("✅ CV terminée.")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.cv_results is not None:
                st.plotly_chart(plot_cv_error(st.session_state.cv_results),
                                use_container_width=True)
            K = st.slider("K", 2, 10, 4)
            if st.button("▶ Admixture", use_container_width=True):
                try:
                    with st.spinner(f"NMF K={K}..."):
                        Q, _ = admixture_nmf(gt_use, K=int(K))
                        st.session_state.admixture_Q = Q
                        st.session_state.admixture_K = K
                    st.success(f"✅ Q : {Q.shape}")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.admixture_Q is not None:
                Q = st.session_state.admixture_Q
                K = st.session_state.admixture_K
                pops = st.session_state.ind_filt["FID"].values
                iids = st.session_state.ind_filt["IID"].values
                fig, df_s = plot_admixture(Q, iids, pops, K)
                st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 6 : Démographie ============
    with tabs[5]:
        st.subheader("📈 Démographie")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            sub1, sub2 = st.tabs(["LD decay", "ROH"])
            with sub1:
                if st.button("▶ LD decay", key="btn_ld"):
                    with st.spinner("LD decay..."):
                        st.session_state.ld_df = ld_decay(
                            gt_use, st.session_state.snp_filt["BP"].values,
                            max_kb=1000, max_snp=1000)
                    st.success("✅ calculé.")
                if (st.session_state.ld_df is not None
                        and not st.session_state.ld_df.empty):
                    st.plotly_chart(plot_ld_decay(st.session_state.ld_df),
                                    use_container_width=True)
            with sub2:
                if st.button("▶ ROH", key="btn_roh"):
                    with st.spinner("ROH..."):
                        snp_json = st.session_state.snp_filt[
                            ["CHR", "BP"]].to_json()
                        roh_df, froh = detect_roh(gt_use, snp_json)
                        st.session_state.roh_df = roh_df
                        st.session_state.froh = froh
                    st.success(f"✅ {len(roh_df)} ROH")
                if st.session_state.froh is not None:
                    labels = st.session_state.ind_filt["FID"].values
                    st.plotly_chart(plot_roh_histogram(
                        st.session_state.froh, labels),
                        use_container_width=True)

    # ============ TAB 7 : Sélection ============
    with tabs[6]:
        st.subheader("🔍 Sélection")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if has_snp_state():
                gt_use = st.session_state.gt_snp
                snp_use = st.session_state.snp_snp
                ind_use = st.session_state.ind_snp
            else:
                gt_use = (st.session_state.gt_pruned if has_pruned()
                          else st.session_state.gt_filt)
                snp_use = (st.session_state.snp_pruned if has_pruned()
                           else st.session_state.snp_filt)
                ind_use = st.session_state.ind_filt
            pop_labels_np = to_np_str(ind_use["FID"].values)
            if st.button("▶ FST par SNP", use_container_width=True):
                try:
                    with st.spinner("FST..."):
                        st.session_state.fst = fst_per_snp(gt_use,
                                                            pop_labels_np)
                    st.success("✅ FST calculé.")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.fst is not None:
                fst_c = st.session_state.fst[
                    np.isfinite(st.session_state.fst)]
                if len(fst_c) > 0:
                    fig = plot_manhattan(st.session_state.fst,
                                         snp_use["CHR"].values)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

    # ============ TAB 8 : Phylogénie ============
    with tabs[7]:
        st.subheader("🌳 Phylogénie")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            pop_labels_np = to_np_str(
                st.session_state.ind_filt["FID"].values)
            if st.button("▶ Reynolds + NJ", type="primary"):
                try:
                    with st.spinner("Calcul..."):
                        D, pops = reynolds_distance(gt_use, pop_labels_np)
                        st.session_state.reynolds_D = D
                        st.session_state.reynolds_pops = pops
                    st.success("✅ Calculé.")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.reynolds_D is not None:
                fig = go.Figure(data=go.Heatmap(
                    z=st.session_state.reynolds_D,
                    x=st.session_state.reynolds_pops,
                    y=st.session_state.reynolds_pops,
                    colorscale="Blues",
                    text=np.round(st.session_state.reynolds_D, 3),
                    texttemplate="%{text}"))
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                newick = nj_tree_newick(st.session_state.reynolds_D,
                                        st.session_state.reynolds_pops)
                st.code(newick, language="text")
                st.download_button("⬇ Newick",
                                   newick.encode("utf-8"),
                                   file_name="cattle_nj.nwk",
                                   mime="text/plain")

    # ============ TAB 9 : Export ============
    with tabs[8]:
        st.subheader("📤 Export")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            ind_use = st.session_state.ind_filt
            snp_use = (st.session_state.snp_pruned if has_pruned()
                       else st.session_state.snp_filt)
            prefix = st.text_input("Préfixe", "bovine_qc")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📦 PLINK ZIP"):
                    try:
                        st.session_state["_plink_zip"] = build_plink_zip(
                            gt_use, ind_use, snp_use, prefix=prefix)
                        st.success("✅ Prêt.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.get("_plink_zip"):
                    st.download_button("⬇ .zip",
                                       data=st.session_state["_plink_zip"],
                                       file_name=f"{prefix}.zip",
                                       mime="application/zip")
            with c2:
                if st.button("📄 VCF"):
                    try:
                        st.session_state["_vcf_str"] = build_vcf_output(
                            gt_use, ind_use, snp_use)
                        st.success("✅ Prêt.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.get("_vcf_str"):
                    st.download_button(
                        "⬇ .vcf",
                        data=st.session_state["_vcf_str"].encode("utf-8"),
                        file_name=f"{prefix}.vcf", mime="text/plain")

    # ============ TAB 10 : Rapport ============
    with tabs[9]:
        st.subheader("📄 Rapport HTML")
        if not has_qc():
            st.warning("⚠️ Lancez au moins le QC.")
        else:
            project_name = st.text_input("Projet", "Cattle_Project")
            interps = {}
            if st.session_state.qc_stats:
                interps["QC"] = interpret_qc(st.session_state.qc_stats)
            if has_qc():
                interps["MAF"] = interpret_maf(maf(st.session_state.gt_filt))
            if st.session_state.fst is not None:
                fst_c = st.session_state.fst[
                    np.isfinite(st.session_state.fst)]
                if len(fst_c):
                    interps["FST"] = interpret_fst(fst_c)
            if st.button("📄 Générer HTML"):
                figures = {}
                if st.session_state.pca_scores is not None:
                    figures["PCA"] = plot_pca(
                        st.session_state.pca_scores,
                        st.session_state.pca_var,
                        st.session_state.ind_filt["FID"].values)
                html = build_report_html(
                    {"project_name": project_name},
                    st.session_state.qc_stats,
                    figures=figures, interpretations=interps)
                st.session_state["_report_html"] = html
                st.success("✅ Prêt.")
            if st.session_state.get("_report_html"):
                st.download_button(
                    "⬇ Télécharger",
                    data=st.session_state["_report_html"].encode("utf-8"),
                    file_name=f"rapport_{datetime.now():%Y%m%d}.html",
                    mime="text/html")

    # ============ TAB 11 : ANALYSES AVANCÉES ============
    with tabs[10]:
        st.subheader("🔬 Analyses Avancées")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            ind_use = st.session_state.ind_filt
            pop_labels_np = to_np_str(ind_use["FID"].values)
            sub1, sub2 = st.tabs(["🧬 Diversité", "🌍 Structure"])
            with sub1:
                if st.button("▶ π", key="btn_pi"):
                    with st.spinner("π..."):
                        pi_arr, pi_mean = nucleotide_diversity_pi(gt_use)
                        st.session_state["_pi_arr"] = pi_arr
                    st.success(f"✅ π = **{pi_mean:.6f}**")
                if st.session_state.get("_pi_arr") is not None:
                    st.plotly_chart(plot_pi_distribution(
                        st.session_state["_pi_arr"]),
                        use_container_width=True)
                if st.button("▶ θ_W", key="btn_tw"):
                    with st.spinner("θ_W..."):
                        res = watterson_theta(gt_use)
                    st.json(res)
                if st.button("▶ Tajima's D", key="btn_td"):
                    with st.spinner("Tajima..."):
                        res = tajimas_d(gt_use)
                    st.json(res)
                if st.button("▶ F-statistics", key="btn_fstats"):
                    with st.spinner("F-stat..."):
                        res = f_statistics(gt_use, pop_labels_np)
                    st.json(res)
                if st.button("▶ F_GRM", key="btn_fgrm"):
                    with st.spinner("F_GRM..."):
                        F = f_grm_inbreeding(gt_use)
                        st.session_state["_fgrm"] = F
                    st.success(f"✅ F_GRM = **{np.nanmean(F):.4f}**")
            with sub2:
                pops = sorted(np.unique(pop_labels_np))
                with st.expander("🔬 F3"):
                    c1, c2, c3 = st.columns(3)
                    pA = c1.selectbox("A", pops, key="f3_A")
                    pB = c2.selectbox("B", pops,
                                      index=1 if len(pops) > 1 else 0,
                                      key="f3_B")
                    pC = c3.selectbox("C", pops,
                                      index=2 if len(pops) > 2 else 0,
                                      key="f3_C")
                    if st.button("▶ F3", key="btn_f3"):
                        with st.spinner("F3..."):
                            f3, se = f3_statistic(gt_use, pop_labels_np,
                                                   pA, pB, pC)
                        st.success(f"F3 = **{f3:.5f}** ± {se:.5f}")
                with st.expander("🔬 D-stat"):
                    c1, c2, c3, c4 = st.columns(4)
                    p1 = c1.selectbox("P1", pops, key="d_p1")
                    p2 = c2.selectbox("P2", pops,
                                      index=1 if len(pops) > 1 else 0,
                                      key="d_p2")
                    p3 = c3.selectbox("P3", pops,
                                      index=2 if len(pops) > 2 else 0,
                                      key="d_p3")
                    p4 = c4.selectbox("P4", pops,
                                      index=3 if len(pops) > 3 else 0,
                                      key="d_p4")
                    if st.button("▶ D-stat", key="btn_dstat"):
                        with st.spinner("D..."):
                            res = d_statistic_abbababa(
                                gt_use, pop_labels_np, p1, p2, p3, p4)
                        st.json(res)
                with st.expander("🔬 AMOVA"):
                    if st.button("▶ AMOVA", key="btn_amova"):
                        with st.spinner("AMOVA..."):
                            res = amova(gt_use, pop_labels_np)
                        st.json(res)

    # ============ TAB 12 : BINÔMES ============
    with tabs[11]:
        st.subheader("🧬 Binômes")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if has_snp_state():
                gt_use = st.session_state.gt_snp
                snp_use = st.session_state.snp_snp
                ind_use = st.session_state.ind_snp
                st.success(f"📌 gt_snp (sans outliers het) : "
                           f"**{gt_use.shape[0]}** ind × "
                           f"**{gt_use.shape[1]:,}** SNPs")
            else:
                gt_use = (st.session_state.gt_pruned if has_pruned()
                          else st.session_state.gt_filt)
                snp_use = (st.session_state.snp_pruned if has_pruned()
                           else st.session_state.snp_filt)
                ind_use = st.session_state.ind_filt
                st.warning("⚠️ Fallback (gt_snp non dispo).")
            pop_labels_np = to_np_str(ind_use["FID"].values)
            available_pops = sorted(set(pop_labels_np))

            st.markdown("### 🎯 Mode")
            mode = st.radio(
                "Choisissez :",
                ["🧠 Auto-détection MDS",
                 "📋 Prédéfinis (MDS manuels)",
                 "🖱️ Manuelle libre",
                 "🌐 Toutes les paires",
                 "➕ Personnalisée"],
                horizontal=False, label_visibility="collapsed",
                key="binome_mode")

            if mode == "🧠 Auto-détection MDS":
                st.markdown("#### 🧠 Auto-détection")
                if st.session_state.mds_coords is None:
                    st.warning("⚠️ Lancez PCA + MDS d'abord.")
                else:
                    mds_c = st.session_state.mds_coords
                    pops_mds = to_np_str(
                        st.session_state.ind_filt["FID"].values)
                    with st.spinner("Détection..."):
                        binomes = auto_detect_binomes_from_mds(
                            mds_c, pops_mds,
                            min_dist_quantile=0.15,
                            max_dist_quantile=0.85)
                    if binomes.empty:
                        st.error("Impossible.")
                    else:
                        predictions = []
                        for _, row in binomes.iterrows():
                            pred = predict_fst_binome(
                                row["distance_MDS"],
                                row["pop_A"], row["pop_B"])
                            pred["catégorie"] = row["catégorie"]
                            pred["n_ind_A"] = row["n_ind_A"]
                            pred["n_ind_B"] = row["n_ind_B"]
                            predictions.append(pred)
                        pred_df = pd.DataFrame(predictions)
                        pred_df = score_binome_priority(pred_df)
                        st.markdown("##### 🏆 TOP 10 prioritaires")
                        top10 = pred_df.head(10).copy()
                        top10["Binôme"] = (top10["pop_A"] + " ↔ "
                                           + top10["pop_B"])
                        cols = ["rang", "Binôme", "distance_MDS",
                                "FST_prédit", "catégorie",
                                "interprétation", "score_priorité"]
                        cols = [c for c in cols if c in top10.columns]
                        st.dataframe(top10[cols],
                                     use_container_width=True)
                        fig = px.scatter(pred_df,
                                         x="distance_MDS", y="FST_prédit",
                                         color="catégorie",
                                         hover_data=["pop_A", "pop_B"],
                                         title="Distance MDS vs FST prédit",
                                         height=450)
                        st.plotly_chart(fig, use_container_width=True)
                        categories = pred_df["catégorie"].unique().tolist()
                        cat_choice = st.multiselect(
                            "Filtrer :", categories, default=categories,
                            key="binome_cat_filter")
                        filtered = pred_df[
                            pred_df["catégorie"].isin(cat_choice)]
                        if not filtered.empty:
                            pair_options = [
                                f"{r['pop_A']} ↔ {r['pop_B']}  "
                                f"[{r['catégorie']}, FST≈{r['FST_prédit']:.3f}]"
                                for _, r in filtered.iterrows()]
                            selected = st.selectbox(
                                "Choisir :", pair_options, key="auto_sel")
                            pop_A = selected.split(" ↔ ")[0].strip()
                            pop_B = selected.split(" ↔ ")[1].split()[0].strip()
                            c1, c2 = st.columns(2)
                            with c1:
                                q_auto = st.slider(
                                    "Quantile", 0.90, 0.9999, 0.99, 0.001,
                                    format="%.4f", key="q_auto")
                            with c2:
                                fst_auto = st.slider(
                                    "FST min", 0.0, 0.5, 0.15, 0.01,
                                    key="mf_auto")
                            if st.button("▶ Analyser", type="primary",
                                         use_container_width=True,
                                         key="btn_auto"):
                                _run_binome_analysis(
                                    gt_use, snp_use, pop_labels_np,
                                    pop_A, pop_B, q_auto, fst_auto)
                            _display_binome_results(pop_A, pop_B)

            elif mode == "📋 Prédéfinis (MDS manuels)":
                valid_pairs = []
                for A, B, desc in POPULATION_PAIRS_DEFAULT:
                    ok = A in available_pops and B in available_pops
                    valid_pairs.append({
                        "Binôme": f"{A} ↔ {B}", "Description": desc,
                        "OK": "✅" if ok else "⚠️",
                        "pop_A": A, "pop_B": B, "ok": ok})
                st.dataframe(pd.DataFrame([
                    {k: p[k] for k in ["Binôme", "Description", "OK"]}
                    for p in valid_pairs]), use_container_width=True)
                pair_options = [p["Binôme"]
                                for p in valid_pairs if p["ok"]]
                if pair_options:
                    selected = st.selectbox("Binôme", pair_options,
                                             key="predef_sel")
                    c1, c2 = st.columns(2)
                    with c1:
                        q_predef = st.slider("Quantile", 0.90, 0.999, 0.99,
                                              0.001, format="%.3f",
                                              key="q_predef")
                    with c2:
                        fst_predef = st.slider("FST min", 0.0, 0.5, 0.15,
                                                0.01, key="mf_predef")
                    pop_A, pop_B = selected.split(" ↔ ")
                    pop_A, pop_B = pop_A.strip(), pop_B.strip()
                    if st.button("▶ Analyser", type="primary",
                                 use_container_width=True,
                                 key="btn_predef"):
                        _run_binome_analysis(
                            gt_use, snp_use, pop_labels_np,
                            pop_A, pop_B, q_predef, fst_predef)
                    _display_binome_results(pop_A, pop_B)

            elif mode == "🖱️ Manuelle libre":
                c1, c2 = st.columns(2)
                with c1:
                    pop_A = st.selectbox("Pop A", available_pops,
                                          key="manual_popA")
                with c2:
                    options_B = [p for p in available_pops if p != pop_A]
                    pop_B = st.selectbox("Pop B", options_B,
                                          key="manual_popB")
                c1, c2 = st.columns(2)
                with c1:
                    q_man = st.slider("Quantile", 0.90, 0.999, 0.99,
                                      0.001, format="%.3f", key="q_man")
                with c2:
                    fst_man = st.slider("FST min", 0.0, 0.5, 0.15,
                                        0.01, key="mf_man")
                if st.button(f"▶ {pop_A} vs {pop_B}", type="primary",
                             use_container_width=True, key="btn_manual"):
                    _run_binome_analysis(gt_use, snp_use, pop_labels_np,
                                          pop_A, pop_B, q_man, fst_man)
                _display_binome_results(pop_A, pop_B)

            elif mode == "🌐 Toutes les paires":
                n_pops = len(available_pops)
                n_pairs = n_pops * (n_pops - 1) // 2
                st.info(f"**{n_pops} pops** → **{n_pairs} paires**")
                c1, c2 = st.columns(2)
                with c1:
                    q_all = st.slider("Quantile", 0.90, 0.999, 0.99,
                                      0.001, format="%.3f", key="q_all")
                with c2:
                    fst_all = st.slider("FST min", 0.0, 0.5, 0.15,
                                        0.01, key="mf_all")
                if st.button(f"🚀 Analyser les {n_pairs} paires",
                             type="primary", use_container_width=True,
                             key="btn_all"):
                    try:
                        all_results = {}
                        prog = st.progress(0)
                        import itertools
                        pairs = list(itertools.combinations(
                            available_pops, 2))
                        for i, (A, B) in enumerate(pairs):
                            r = analyze_binome_complete(
                                gt_use, snp_use, pop_labels_np,
                                A, B, q_all, fst_all)
                            all_results[f"{A}_vs_{B}"] = r
                            prog.progress((i + 1) / len(pairs))
                        st.session_state["_all_binomes"] = all_results
                        st.success(f"✅ {len(all_results)} paires.")
                    except Exception as e:
                        st.error(f"❌ {e}")
                _display_all_binomes_summary()

            else:  # Personnalisée
                if st.session_state.get("_custom_pairs") is None:
                    st.session_state["_custom_pairs"] = []
                c1, c2, c3 = st.columns([2, 2, 3])
                with c1:
                    new_A = st.selectbox("Pop A", available_pops,
                                          key="custom_new_A")
                with c2:
                    options_B = [p for p in available_pops if p != new_A]
                    new_B = st.selectbox("Pop B", options_B,
                                          key="custom_new_B")
                with c3:
                    label = st.text_input("Label",
                                           value=f"{new_A}-{new_B}",
                                           key="custom_label")
                if st.button("➕ Ajouter", key="btn_add_custom"):
                    pair = {"pop_A": new_A, "pop_B": new_B,
                            "label": label}
                    existing = [(p["pop_A"], p["pop_B"])
                                for p in
                                st.session_state["_custom_pairs"]]
                    if (new_A, new_B) not in existing:
                        st.session_state["_custom_pairs"].append(pair)
                        st.success(f"✅ {label}")
                custom_pairs = st.session_state.get("_custom_pairs", [])
                if custom_pairs:
                    st.dataframe(pd.DataFrame(custom_pairs),
                                 use_container_width=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        q_cus = st.slider("Quantile", 0.90, 0.999, 0.99,
                                          0.001, format="%.3f",
                                          key="q_cus")
                    with c2:
                        fst_cus = st.slider("FST min", 0.0, 0.5, 0.15,
                                            0.01, key="mf_cus")
                    if st.button(f"🚀 Analyser {len(custom_pairs)}",
                                 type="primary",
                                 use_container_width=True,
                                 key="btn_run_custom"):
                        try:
                            all_results = {}
                            prog = st.progress(0)
                            for i, p in enumerate(custom_pairs):
                                r = analyze_binome_complete(
                                    gt_use, snp_use, pop_labels_np,
                                    p["pop_A"], p["pop_B"],
                                    q_cus, fst_cus)
                                all_results[p["label"]] = r
                                prog.progress((i+1)/len(custom_pairs))
                            st.session_state["_all_binomes"] = all_results
                            st.success(f"✅ {len(all_results)}.")
                        except Exception as e:
                            st.error(f"❌ {e}")
                    _display_all_binomes_summary()

            # ===== GTF + ANNOTATION =====
            st.divider()
            st.markdown("### 🧬 GTF + annotation")
            gtf_file = st.file_uploader("GTF (.gtf/.gtf.gz)",
                                         type=["gtf", "gz"],
                                         key="binome_gtf_flex")
            flank_kb = st.slider("Fenêtre kb", 10, 5000, 500, 10,
                                  key="flank_kb_flex")
            if gtf_file is not None:
                if st.button("📥 Charger", key="btn_load_gtf"):
                    try:
                        with st.spinner("Chargement..."):
                            tmp_path = "/tmp/annotations.gtf"
                            if gtf_file.name.endswith(".gz"):
                                tmp_path += ".gz"
                            with open(tmp_path, "wb") as f:
                                f.write(gtf_file.getbuffer())
                            genes_df = load_genes_from_gtf(tmp_path)
                            st.session_state["_genes_df"] = genes_df
                        st.success(f"✅ {len(genes_df):,} gènes.")
                    except Exception as e:
                        st.error(f"❌ {e}")

    # ============ TAB 13 : SNP & GÈNES (GWAS) ============
    with tabs[12]:
        st.subheader("🧪 SNP & Gènes (GWAS)")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if has_snp_state():
                gt_use = st.session_state.gt_snp
                snp_use = st.session_state.snp_snp
                ind_use = st.session_state.ind_snp
            else:
                gt_use = (st.session_state.gt_pruned if has_pruned()
                          else st.session_state.gt_filt)
                snp_use = (st.session_state.snp_pruned if has_pruned()
                           else st.session_state.snp_filt)
                ind_use = st.session_state.ind_filt

            sub1, sub2, sub3 = st.tabs([
                "🧬 Extraction SNP", "📊 GWAS 10 cM",
                "🔬 Comparaison"])

            with sub1:
                all_binomes = st.session_state.get("_all_binomes") or {}
                if not all_binomes:
                    st.info("💡 Lancez d'abord un binôme.")
                else:
                    pair_key = st.selectbox(
                        "Binôme :", list(all_binomes.keys()),
                        key="snp_gene_pair")
                    result = all_binomes.get(pair_key)
                    if result and result.get("status") == "success":
                        c1, c2, c3 = st.columns(3)
                        c1.metric("SNP outliers", result["n_snp_outliers"])
                        c2.metric("FST moyen", f"{result['fst_mean']:.4f}")
                        c3.metric("FST max", f"{result['fst_max']:.4f}")
                        if st.session_state.get("_genes_df") is not None:
                            if st.button("📋 Extraire SNP + gènes",
                                         type="primary",
                                         use_container_width=True,
                                         key="btn_extract_snp"):
                                try:
                                    with st.spinner("Extraction..."):
                                        annotated = extract_snp_list_with_genes(
                                            result["outliers_df"],
                                            st.session_state["_genes_df"],
                                            flank_kb=500)
                                        result["annotated_df"] = annotated
                                        st.session_state["_all_binomes"][pair_key] = result
                                    st.success(f"✅ {len(annotated)} SNP.")
                                except Exception as e:
                                    st.error(f"❌ {e}")
                            if "annotated_df" in result:
                                df = result["annotated_df"]
                                with_genes = df.dropna(subset=["gene_name"])
                                c1, c2, c3 = st.columns(3)
                                c1.metric("SNP totaux", len(df))
                                c2.metric("SNP avec gène",
                                          len(with_genes))
                                c3.metric("Gènes uniques",
                                          (with_genes["gene_name"].nunique()
                                           if not with_genes.empty else 0))
                                cols_show = ["CHR", "SNP", "BP", "FST",
                                             "gene_name", "distance",
                                             "region"]
                                cols_show = [c for c in cols_show
                                             if c in df.columns]
                                st.dataframe(df[cols_show].head(200),
                                             use_container_width=True)
                                csv = df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "⬇ CSV", data=csv,
                                    file_name=f"snp_genes_{pair_key}.csv",
                                    mime="text/csv",
                                    use_container_width=True)
                        else:
                            st.info("💡 Chargez le GTF.")

            with sub2:
                all_binomes = st.session_state.get("_all_binomes") or {}
                if not all_binomes:
                    st.info("💡 Lancez un binôme.")
                else:
                    pheno_file = st.file_uploader(
                        "CSV phénotypes : IID,hauteur_garrot,...",
                        type=["csv"], key="pheno_upload")
                    if pheno_file is not None:
                        phenotypes = load_phenotypes(
                            pheno_file.read(), key_col="IID")
                        if phenotypes is not None:
                            st.success(f"✅ {len(phenotypes)} ind.")
                            pheno_cols = [c for c in phenotypes.columns
                                          if c != "IID"]
                            pheno_name = st.selectbox(
                                "Phénotype :", pheno_cols,
                                key="gwas_pheno_choice")
                            pair_key_gwas = st.selectbox(
                                "Binôme :", list(all_binomes.keys()),
                                key="gwas_pair_choice")
                            result = all_binomes[pair_key_gwas]
                            if result.get("status") == "success":
                                candidates = result["outliers_df"]
                                if not candidates.empty:
                                    top_n = st.slider(
                                        "Top N :", 1, 20, 5,
                                        key="gwas_top_n")
                                    top_snps = candidates.head(top_n)
                                    st.dataframe(
                                        top_snps[["CHR", "SNP", "BP",
                                                  "FST"]],
                                        use_container_width=True)
                                    window_cm = st.slider(
                                        "Fenêtre cM ≈ Mb :", 1, 20, 10,
                                        key="gwas_window")
                                    if st.button("▶ GWAS",
                                                 type="primary",
                                                 use_container_width=True,
                                                 key="btn_gwas"):
                                        try:
                                            all_gwas = {}
                                            prog = st.progress(0)
                                            for i, (_, snp) in enumerate(top_snps.iterrows()):
                                                with st.spinner(
                                                    f"GWAS {snp['SNP']}..."):
                                                    gwas_res = local_gwas_10cm(
                                                        gt_use, snp_use,
                                                        ind_use,
                                                        phenotypes,
                                                        snp["CHR"], snp["BP"],
                                                        pheno_name,
                                                        window_cm=window_cm)
                                                if not gwas_res.empty:
                                                    all_gwas[snp["SNP"]] = gwas_res
                                                prog.progress((i+1)/len(top_snps))
                                            prog.empty()
                                            st.session_state["_gwas_results"] = all_gwas
                                            st.success(f"✅ {len(all_gwas)}.")
                                        except Exception as e:
                                            st.error(f"❌ {e}")
                                    if st.session_state.get("_gwas_results"):
                                        all_gwas = st.session_state["_gwas_results"]
                                        st.markdown("### 📊 Résumé")
                                        summary = []
                                        for snp_name, df in all_gwas.items():
                                            t = test_morphometric_effect(
                                                df, pheno_name)
                                            t["SNP_cible"] = snp_name
                                            summary.append(t)
                                        if summary:
                                            st.dataframe(
                                                pd.DataFrame(summary),
                                                use_container_width=True)
                                        sel_snp = st.selectbox(
                                            "SNP :", list(all_gwas.keys()),
                                            key="gwas_view_snp")
                                        if sel_snp:
                                            df_gwas = all_gwas[sel_snp]
                                            st.dataframe(df_gwas,
                                                         use_container_width=True)
                                            snp_row = snp_use[
                                                snp_use["SNP"] == sel_snp]
                                            if not snp_row.empty:
                                                chr_name = str(snp_row.iloc[0]["CHR"])
                                                tbp = int(snp_row.iloc[0]["BP"])
                                                fig = plot_gwas_manhattan_local(
                                                    df_gwas, chr_name, tbp)
                                                if fig:
                                                    st.plotly_chart(fig,
                                                                    use_container_width=True)
                                            csv = df_gwas.to_csv(
                                                index=False).encode("utf-8")
                                            st.download_button(
                                                f"⬇ GWAS {sel_snp}",
                                                data=csv,
                                                file_name=f"gwas_{sel_snp}.csv",
                                                mime="text/csv",
                                                use_container_width=True)
                    else:
                        st.info("📁 Chargez un CSV.")

            with sub3:
                all_binomes = st.session_state.get("_all_binomes") or {}
                if len(all_binomes) < 2:
                    st.info("💡 Lancez au moins 2 binômes.")
                elif st.session_state.get("_genes_df") is None:
                    st.warning("⚠️ Chargez le GTF.")
                else:
                    if st.button("▶ Comparer",
                                 type="primary",
                                 use_container_width=True,
                                 key="btn_compare_genes_v2"):
                        try:
                            with st.spinner("Annotation..."):
                                genes_df = st.session_state["_genes_df"]
                                for name, r in all_binomes.items():
                                    if r.get("status") != "success":
                                        continue
                                    if "annotated_df" not in r:
                                        r["annotated_df"] = extract_snp_list_with_genes(
                                            r["outliers_df"], genes_df,
                                            flank_kb=500)
                                st.session_state["_all_binomes"] = all_binomes
                            comp_df, all_genes, specific = \
                                compare_genes_across_binomes(all_binomes)
                            if comp_df.empty:
                                st.warning("Aucun gène.")
                            else:
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Total", len(comp_df))
                                shared = comp_df[comp_df["partagé"] == "✅"]
                                c2.metric("Partagés", len(shared))
                                c3.metric("Binômes", len(all_genes))
                                if not shared.empty:
                                    st.markdown("#### 🔷 Partagés")
                                    st.dataframe(shared,
                                                 use_container_width=True)
                                st.markdown("#### 🔶 Spécifiques")
                                for name, genes in specific.items():
                                    with st.expander(
                                        f"**{name}** — {len(genes)}"):
                                        if genes:
                                            st.dataframe(pd.DataFrame(
                                                {"gene": sorted(genes)}),
                                                use_container_width=True)
                                        else:
                                            st.info("Aucun.")
                                csv = comp_df.to_csv(
                                    index=False).encode("utf-8")
                                st.download_button(
                                    "⬇ CSV", data=csv,
                                    file_name="comparaison_genes.csv",
                                    mime="text/csv",
                                    use_container_width=True)
                        except Exception as e:
                            st.error(f"❌ {e}")

    # ---------- PIPELINE COMPLET ----------
    if st.session_state.get("run_requested"):
        st.session_state.run_requested = False
        try:
            with st.spinner("🚀 Pipeline..."):
                gt_a, snp_a = filter_autosomes(st.session_state.gt,
                                               st.session_state.snp_df)
                st.session_state.gt = gt_a
                st.session_state.snp_df = snp_a
                params = {"geno": geno, "mind": mind, "maf": maf_thr,
                          "hwe": hwe_thr, "het_sd": het_sd}
                (gt_f, ind_f, snp_f, qc_stats,
                 gt_s, ind_s, snp_s) = apply_qc_filters(
                    gt_a, st.session_state.ind_df, snp_a, params)
                st.session_state.gt_filt = gt_f
                st.session_state.ind_filt = ind_f
                st.session_state.snp_filt = snp_f
                st.session_state.gt_snp = gt_s
                st.session_state.ind_snp = ind_s
                st.session_state.snp_snp = snp_s
                st.session_state.qc_stats = qc_stats
                keep = ld_pruning(gt_f, window=50, step=5, r2_thr=ld_r2)
                st.session_state.gt_pruned = gt_f[:, keep]
                st.session_state.snp_pruned = (snp_f[keep]
                                               .reset_index(drop=True))
                gt_p = st.session_state.gt_pruned
                snp_p = st.session_state.snp_pruned
                pop_labels_pipe = to_np_str(ind_f["FID"].values)
                (st.session_state.pca_scores,
                 st.session_state.pca_var) = pca_analysis(gt_p, 10)
                st.session_state.mds_coords = mds_analysis(gt_p, 5)
                st.session_state.king_matrix = king_kinship(gt_p)
                st.session_state.ld_df = ld_decay(
                    gt_p, snp_p["BP"].values, max_kb=1000, max_snp=1000)
                st.session_state.fst = fst_per_snp(gt_p, pop_labels_pipe)
                mat, pops = fst_pairwise(gt_p, pop_labels_pipe)
                st.session_state.fst_pairwise_matrix = mat
                st.session_state.fst_pops = pops
                cv = admix_cv_error(gt_p, K_range=(2, 6), n_reps=2)
                st.session_state.cv_results = cv
                ks = sorted(cv.keys())
                K_opt = ks[int(np.argmin([cv[k]["mean"] for k in ks]))]
                Q, _ = admixture_nmf(gt_p, K=K_opt)
                st.session_state.admixture_Q = Q
                st.session_state.admixture_K = K_opt
            st.success("✅ Pipeline terminé !")
        except Exception as e:
            st.error(f"❌ {e}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

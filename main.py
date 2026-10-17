"""
🐄 Bovine SNP Platform v6.2 — Version finale
Corrections :
- Fix detect_roh (json.loads au lieu de pd.read_json)
- Convertisseur Axiom intégré
Nouveautés :
- Signatures par population (vs reste)
- Signatures par GROUPES de races (laitier/viande/cornu/rustique)
- Comparaison entre groupes (Groupe A vs Groupe B)
- GWAS 10 cM avec phénotype groupe
- Utilisation STRICTE de gt_snp (sans outliers HET) pour la sélection
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

st.set_page_config(
    page_title="🐄 Bovine SNP Platform v6.2",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_THRESHOLDS = {
    "geno": 0.05, "mind": 0.05, "maf": 0.05, "hwe": 1e-6,
    "het_sd": 3.0, "king_cutoff": 0.354, "ld_r2": 0.2,
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

DEFAULT_GROUPS = {
    "Laitier": [], "Viande": [], "Cornu": [], "Rustique": [],
}


# ============================================================
# CACHE
# ============================================================

def _hash_ndarray(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return ("empty",)
    return (x.shape, str(x.dtype), float(np.nansum(x)),
            float(np.nansum(np.abs(x))), float(np.nansum(x * x)))


HASH_FUNCS = {np.ndarray: _hash_ndarray}


def cache_data(func=None, **kw):
    def _decorate(f):
        return st.cache_data(show_spinner=False,
                              hash_funcs=HASH_FUNCS, **kw)(f)
    if func is None:
        return _decorate
    return _decorate(func)


def to_np_str(arr):
    return np.asarray(list(arr), dtype=object)


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
# CONVERTISSEUR AXIOM
# ============================================================

def _clean_allele(a):
    if a is None:
        return "0"
    a = str(a).strip()
    if a in ("", ".", "-", "NA", "nan", "None", "0", "00"):
        return "0"
    return a


def sample_to_fid_iid(sample_name, fid_parts=2):
    name = str(sample_name).strip()
    for ext in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    name = (name.replace("(", "_").replace(")", "")
            .replace("[", "_").replace("]", "")
            .replace(" ", "_").replace("/", "_"))
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    parts = name.split("_")
    fid = ("_".join(parts[:fid_parts])
           if len(parts) >= fid_parts else parts[0])
    return fid, name


def parse_raw_axiom_file(raw_bytes, filename=""):
    if (filename.endswith(".gz")
            or (len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b")):
        try:
            raw = gzip.decompress(raw_bytes)
        except Exception as e:
            raise ValueError(f"Fichier .gz corrompu : {e}")
    else:
        raw = raw_bytes
    text = raw.decode("utf-8", errors="replace")
    lines = [l.rstrip("\r\n") for l in text.splitlines()]
    lines = [l for l in lines if l.strip()]
    if not lines:
        raise ValueError("Fichier vide.")
    first = lines[0]
    is_header = (first.startswith("#") or first.startswith("Sample")
                 or "Sample Filename" in first[:50]
                 or "Sample ID" in first[:50])
    start = 1 if is_header else 0

    samples, geno_rows = [], []
    for line in lines[start:]:
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
            if len(parts) < 2:
                continue
        sample_name = parts[0].strip()
        cells = [c.strip() for c in parts[1:]]
        alleles = []
        for cell in cells:
            toks = cell.split()
            if len(toks) == 2:
                alleles.append((_clean_allele(toks[0]),
                                _clean_allele(toks[1])))
            elif len(toks) == 1:
                s = toks[0]
                if len(s) == 2 and all(ch in "ACGT0" for ch in s):
                    alleles.append((s[0], s[1]))
                elif "/" in s:
                    a1, a2 = s.split("/", 1)
                    alleles.append((_clean_allele(a1),
                                    _clean_allele(a2)))
                else:
                    alleles.append((_clean_allele(s),
                                    _clean_allele(s)))
            else:
                alleles.append(("0", "0"))
        samples.append(sample_name)
        geno_rows.append(alleles)
    if not samples:
        raise ValueError("Aucun échantillon trouvé.")
    counts = [len(r) for r in geno_rows]
    common = Counter(counts).most_common(1)[0][0]
    for i, r in enumerate(geno_rows):
        if len(r) > common:
            geno_rows[i] = r[:common]
        elif len(r) < common:
            geno_rows[i] = r + [("0", "0")] * (common - len(r))
    return samples, geno_rows, common


def build_ped_map_from_raw(samples, geno_rows, fid_parts=2):
    n_ind = len(samples)
    n_snp = len(geno_rows[0]) if n_ind else 0
    map_text = "\n".join(
        f"1\tSNP_{i+1:06d}\t0\t{i+1}" for i in range(n_snp)) + "\n"
    ped_lines = []
    for i, sample in enumerate(samples):
        fid, iid = sample_to_fid_iid(sample, fid_parts=fid_parts)
        cols = [fid, iid, "0", "0", "0", "-9"]
        for a1, a2 in geno_rows[i]:
            cols.extend([a1, a2])
        ped_lines.append("\t".join(cols))
    return "\n".join(ped_lines) + "\n", map_text, n_snp


def preview_ped_dataframe(samples, geno_rows, max_rows=10, max_snp=20):
    rows = []
    n_snp = len(geno_rows[0]) if geno_rows else 0
    n_show = min(max_snp, n_snp)
    for sample, alleles in zip(samples[:max_rows], geno_rows[:max_rows]):
        fid, iid = sample_to_fid_iid(sample)
        row = {"FID": fid, "IID": iid, "PID": 0, "MID": 0,
               "Sex": 0, "Pheno": -9}
        for j in range(n_show):
            a1, a2 = alleles[j]
            row[f"SNP_{j+1:05d}_A1"] = a1
            row[f"SNP_{j+1:05d}_A2"] = a2
        rows.append(row)
    return pd.DataFrame(rows)


def render_axiom_converter_page():
    st.header("🧬 Convertisseur Axiom → PED/MAP")
    st.markdown("Transforme un fichier Axiom (`A/G`, `T T`, etc.) "
                "en fichiers **PED** et **MAP** compatibles PLINK.")
    st.divider()

    raw_f = st.file_uploader(
        "Fichier Axiom (.txt, .tsv, .csv, .ped, .gz)",
        type=["ped", "txt", "tsv", "csv", "gz"],
        key="conv_raw_file_v6")
    c1, c2 = st.columns(2)
    fid_parts = c1.number_input("Parties du nom pour FID",
                                min_value=1, max_value=5, value=2,
                                step=1, key="conv_fid_parts_v6")
    sep_out = c2.selectbox("Séparateur",
                            ["Tabulation (PLINK)", "Espace"],
                            key="conv_sep_v6")
    if raw_f is not None:
        if st.button("Convertir", type="primary",
                     use_container_width=True, key="conv_btn_v6"):
            try:
                with st.spinner("Parsing..."):
                    samples, geno_rows, n_snp = parse_raw_axiom_file(
                        raw_f.read(), raw_f.name)
                with st.spinner("Génération PED + MAP..."):
                    ped_text, map_text, _ = build_ped_map_from_raw(
                        samples, geno_rows, fid_parts=int(fid_parts))
                if sep_out.startswith("Espace"):
                    ped_text = ped_text.replace("\t", " ")
                    map_text = map_text.replace("\t", " ")
                st.session_state["_conv_samples"] = samples
                st.session_state["_conv_geno"] = geno_rows
                st.session_state["_conv_n_snp"] = n_snp
                st.session_state["_conv_ped"] = ped_text
                st.session_state["_conv_map"] = map_text
                st.success(f"✅ {len(samples)} ind × {n_snp} SNPs.")
            except Exception as e:
                st.error(f"Erreur : {e}")

    if st.session_state.get("_conv_ped"):
        samples = st.session_state["_conv_samples"]
        geno_rows = st.session_state["_conv_geno"]
        n_snp = st.session_state["_conv_n_snp"]
        ped_text = st.session_state["_conv_ped"]
        map_text = st.session_state["_conv_map"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", len(samples))
        c2.metric("SNPs", n_snp)
        c3.metric("Colonnes PED", 6 + 2 * n_snp)
        st.subheader("Aperçu")
        st.dataframe(preview_ped_dataframe(samples, geno_rows),
                     use_container_width=True)
        st.subheader("Téléchargement")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("Fichier .ped",
                               data=ped_text.encode("utf-8"),
                               file_name="converted.ped",
                               mime="text/plain",
                               use_container_width=True,
                               key="dl_ped_v6")
        with c2:
            st.download_button("Fichier .map",
                               data=map_text.encode("utf-8"),
                               file_name="converted.map",
                               mime="text/plain",
                               use_container_width=True,
                               key="dl_map_v6")
        with c3:
            bio = BytesIO()
            with zipfile.ZipFile(bio, "w",
                                  zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("converted.ped", ped_text)
                zf.writestr("converted.map", map_text)
            st.download_button(".zip PED + MAP",
                               data=bio.getvalue(),
                               file_name="converted_ped_map.zip",
                               mime="application/zip",
                               use_container_width=True,
                               key="dl_zip_v6")
        st.info("💡 Rechargez ces fichiers via **Upload PED/MAP**.")


# ============================================================
# PARSING
# ============================================================

def parse_map(map_bytes):
    enc = _detect_encoding(map_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(map_bytes).decode(
                "utf-8", errors="replace")
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
            cm = (float(parts[2])
                  if parts[2] not in ("0", ".", "NA") else 0.0)
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
    enc = _detect_encoding(ped_bytes)
    if enc == "gzip":
        try:
            text = gzip.decompress(ped_bytes).decode(
                "utf-8", errors="replace")
        except Exception as e:
            raise ValueError(f"Fichier .ped.gz corrompu : {e}")
    elif enc == "plink_binary":
        raise ValueError("❌ Ce fichier est un PLINK binaire.")
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
            st.info(f"ℹ️ Header détecté et ignoré.")

    if not lines:
        raise ValueError("Fichier .ped vide après header.")
    parts0 = lines[0].split()
    n_cols_first = len(parts0)
    n_snp_ped = max(0, (n_cols_first - 6) // 2)
    if n_cols_first < 7:
        raise ValueError(
            f"❌ Format PED invalide : {n_cols_first} colonnes.")
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
        st.success(f"✅ {n_ind} ind × {n_snp_eff} SNPs.")
    return gt, ind_df, rejected_short


def parse_vcf(vcf_bytes):
    is_gz = len(vcf_bytes) >= 2 and vcf_bytes[:2] == b"\x1f\x8b"
    if is_gz:
        try:
            text = gzip.decompress(vcf_bytes).decode(
                "utf-8", errors="replace")
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
        rows.append({"CHR": str(chrom),
                     "SNP": str(vid or f"{chrom}:{pos}"),
                     "CM": 0.0, "BP": pos,
                     "A1": str(ref), "A2": str(alt), "GT": gts})
    if not rows:
        raise ValueError("Aucun variant biallélique.")
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
        "CM": 0.0, "BP": [int(r["BP"]) for r in rows],
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
    rem = n_ind - per_pop * n_pop
    counts = [per_pop + (1 if i < rem else 0) for i in range(n_pop)]
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
            ind_rows.append({"FID": str(pop),
                             "IID": f"{pop}_{i+1:03d}"})
            k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    ind_df = pd.DataFrame(ind_rows)
    chr_names = rng.choice(BOVINE_AUTOSOMES, n_snp,
                            p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_UCD12_LENGTHS[c])
                    for c in chr_names], dtype=np.int64)
    snp_df = pd.DataFrame({
        "CHR": [str(c) for c in chr_names],
        "SNP": [f"rs{i:07d}" for i in range(n_snp)],
        "CM": 0.0, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_sort_key)
    snp_df = (snp_df.sort_values(["_k", "BP"])
              .drop(columns="_k").reset_index(drop=True))
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
        p = (n1 + 2 * n2) / (2.0 * np.where(n_valid > 0,
                                              n_valid, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            exp_het = 2.0 * p * (1.0 - p) * n_valid
            valid = (exp_het >= 5) & np.isfinite(exp_het)
            chi2 = (np.abs(n1 - exp_het) - 0.5) ** 2 / exp_het
            pvals[valid] = 1.0 - stats.chi2.cdf(chi2[valid], df=1)
    return pvals


# ============================================================
# NETTOYAGE / FILTRES
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
    chr_str = (snp_df["CHR"].astype(str).str.upper()
               .str.replace("CHR", ""))
    if keep_auto_only:
        keep = chr_str.isin([str(i) for i in range(1, 30)])
    else:
        keep = chr_str.isin([str(i) for i in range(1, 30)]
                            + ["X", "Y", "MT"])
    n_before = len(snp_df)
    gt2 = gt[:, keep.values]
    snp2 = snp_df[keep].reset_index(drop=True)
    if keep_auto_only:
        st.info(f"🧬 Autosomes 1–29 : {n_before - len(snp2)} exclus "
                f"→ **{len(snp2)}** conservés.")
    return gt2, snp2


@cache_data
def ld_pruning(gt, window=50, step=5, r2_thr=0.2):
    n_snp = gt.shape[1]
    if n_snp < 3:
        return np.ones(n_snp, dtype=bool)
    keep = np.ones(n_snp, dtype=bool)
    X = impute_mean(gt)
    X = X - X.mean(axis=0)
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
    gt_snp, ind_snp, snp_snp).
    gt_snp = état AVANT retrait des outliers hétérozygotie (pour sélection).
    """
    gt, ind_df, snp_df = _align_shapes(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    trace = {}

    keep = missingness_per_snp(gt) <= params["geno"]
    gt = gt[:, keep]
    snp_df = snp_df[keep].reset_index(drop=True)
    trace["geno_exclus"] = int(m0 - gt.shape[1])

    keep = missingness_per_ind(gt) <= params["mind"]
    gt = gt[keep]
    ind_df = ind_df[keep].reset_index(drop=True)
    trace["mind_exclus"] = int(n0 - gt.shape[0])

    if gt.shape[1] == 0 or gt.shape[0] == 0:
        raise ValueError("Tous SNPs/individus exclus (missingness).")

    m_before = gt.shape[1]
    m = maf(gt)
    keep = np.isfinite(m) & (m >= params["maf"])
    gt = gt[:, keep]
    snp_df = snp_df[keep].reset_index(drop=True)
    trace["maf_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous SNPs exclus par MAF.")

    m_before = gt.shape[1]
    pv = hwe_pvalues(gt)
    keep = np.isnan(pv) | (pv >= 1e-6)
    gt = gt[:, keep]
    snp_df = snp_df[keep].reset_index(drop=True)
    if gt.shape[1] > 0:
        pv2 = hwe_pvalues(gt)
        keep = np.isnan(pv2) | (pv2 >= params["hwe"])
        gt = gt[:, keep]
        snp_df = snp_df[keep].reset_index(drop=True)
    trace["hwe_exclus"] = int(m_before - gt.shape[1])

    if gt.shape[1] == 0:
        raise ValueError("Tous SNPs exclus par HWE.")

    # État AVANT retrait HET outliers → gt_snp pour la sélection
    gt_snp = gt.copy()
    ind_snp = ind_df.copy()
    snp_snp = snp_df.copy()

    # Retrait des outliers HET (uniquement pour la structure)
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
    gt = gt[keep]
    ind_df = ind_df[keep].reset_index(drop=True)
    trace["het_exclus"] = int(n_before - gt.shape[0])

    if gt.shape[0] == 0:
        raise ValueError("Tous individus exclus.")

    stats_res = {
        "n_ind_init": int(n0), "n_snp_init": int(m0),
        "n_ind_final": int(gt.shape[0]),
        "n_snp_final": int(gt.shape[1]),
        "excluded_ind": int(n0 - gt.shape[0]),
        "excluded_snp": int(m0 - gt.shape[1]),
        "n_ind_snp": int(gt_snp.shape[0]),
        "n_snp_snp": int(gt_snp.shape[1]),
        "trace": trace,
    }
    return (gt, ind_df, snp_df, stats_res,
            gt_snp, ind_snp, snp_snp)


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
            pl.append(vals.mean() / 2.0)
            nl.append(len(vals))
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
    matrix = np.full((K, K), np.nan)
    np.fill_diagonal(matrix, 0.0)
    masks = {p: (pop_labels == p) for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            g1 = gt[masks[pops[i]]]
            g2 = gt[masks[pops[j]]]
            p1 = np.nanmean(g1, 0) / 2.0
            p2 = np.nanmean(g2, 0) / 2.0
            n1 = (~np.isnan(g1)).sum(0)
            n2 = (~np.isnan(g2)).sum(0)
            denom = np.where(n1 + n2 > 0, n1 + n2, np.nan)
            p_bar = (p1 * n1 + p2 * n2) / denom
            hs = (2 * p1 * (1 - p1) * n1
                  + 2 * p2 * (1 - p2) * n2) / denom
            ht = 2 * p_bar * (1 - p_bar)
            with np.errstate(divide="ignore", invalid="ignore"):
                fst_j = (ht - hs) / ht
            matrix[i, j] = matrix[j, i] = float(np.nanmean(fst_j))
    return matrix, pops


@cache_data
def pca_analysis(gt, n_components=10):
    X = impute_mean(gt)
    X = X - X.mean(axis=0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp)
    return pca.fit_transform(X), pca.explained_variance_ratio_ * 100.0


@cache_data
def mds_analysis(gt, n_components=5):
    X = impute_mean(gt)
    n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(n_components=n_components,
                      dissimilarity="precomputed",
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
    gt_sub = impute_mean(gt[:, idx])
    bp_sub = np.asarray(snp_bp)[idx].astype(np.float64)
    X = gt_sub - gt_sub.mean(0)
    std = gt_sub.std(0)
    std[std < 1e-8] = np.nan
    X = X / std
    C = (X.T @ X) / X.shape[0]
    R2 = C ** 2
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
             if K[i, j] > cutoff and not to_remove[i]
             and not to_remove[j]]
    for i, j in pairs:
        to_remove[j] = True
    return ~to_remove, int(to_remove.sum())


@cache_data
def admixture_nmf(gt, K=3, seed=42, max_iter=500):
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    model = NMF(n_components=K, init="nndsvda",
                random_state=seed, max_iter=max_iter)
    W = model.fit_transform(X)
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return W / s, model.components_


@cache_data
def admix_cv_error(gt, K_range=(2, 6), n_reps=3, holdout=0.05, seed=42):
    rng = np.random.default_rng(seed)
    X = np.clip(impute_mean(gt), 0.0, 2.0)
    n, m = X.shape
    mask = rng.random((n, m)) < holdout
    X_train = X.copy()
    X_train[mask] = 0.0
    results = {}
    for K in range(K_range[0], K_range[1]):
        errs = []
        for rep in range(n_reps):
            model = NMF(n_components=K, init="nndsvda",
                        random_state=seed + rep, max_iter=300)
            W = model.fit_transform(X_train)
            H = model.components_
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
        gt = gt[:, idx]
        bp = bp[idx]
        chr_arr = chr_arr[idx]
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
            r2_list.append(r * r)
            dist_list.append(d)
    if not dist_list:
        return pd.DataFrame({"GenAgo": [], "Ne": []})
    r2 = np.array(r2_list)
    dist = np.array(dist_list)
    bins = np.logspace(np.log10(dist.min() + 1e-6),
                       np.log10(dist.max()), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    r2b = np.array([
        r2[(dist >= bins[k]) & (dist < bins[k + 1])].mean()
        if ((dist >= bins[k]) & (dist < bins[k + 1])).sum() > 0
        else np.nan for k in range(len(bins) - 1)])
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
        df["Ne"] = df["Ne"].rolling(3, center=True,
                                     min_periods=1).mean()
    return df


@cache_data
def reynolds_distance(gt, _pop_labels):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    pops = sorted(np.unique(pop_labels))
    K = len(pops)
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


# ⚠️ FIX : json.loads au lieu de pd.read_json
@cache_data
def detect_roh(gt, snp_df_json, min_snps=30, min_kb=500.0):
    snp_df = pd.DataFrame(json.loads(snp_df_json))
    chr_arr = snp_df["CHR"].astype(str).values
    bp = snp_df["BP"].astype(np.int64).values
    n_ind = gt.shape[0]
    rohs, froh_bp = [], np.zeros(n_ind)
    genome_bp_total = 0.0
    for chrom in pd.unique(chr_arr):
        idxs = np.where(chr_arr == chrom)[0]
        if len(idxs) < min_snps:
            continue
        order = np.argsort(bp[idxs])
        idxs = idxs[order]
        chrom_span = bp[idxs[-1]] - bp[idxs[0]]
        if chrom_span < 1:
            continue
        genome_bp_total += chrom_span
        for i in range(n_ind):
            col = gt[i, idxs]
            start_k = None
            for k in range(len(idxs)):
                is_hom = (col[k] == 0) or (col[k] == 2)
                if is_hom:
                    if start_k is None:
                        start_k = k
                else:
                    if start_k is not None:
                        n_run = k - start_k
                        length_kb = ((bp[idxs[k - 1]]
                                      - bp[idxs[start_k]]) / 1000.0)
                        if n_run >= min_snps and length_kb >= min_kb:
                            rohs.append({
                                "IID_idx": i, "CHR": str(chrom),
                                "start_bp": int(bp[idxs[start_k]]),
                                "end_bp": int(bp[idxs[k - 1]]),
                                "n_snp": int(n_run),
                                "length_kb": float(length_kb)})
                            froh_bp[i] += (bp[idxs[k - 1]]
                                           - bp[idxs[start_k]])
                        start_k = None
            if start_k is not None:
                n_run = len(idxs) - start_k
                length_kb = (bp[idxs[-1]]
                             - bp[idxs[start_k]]) / 1000.0
                if n_run >= min_snps and length_kb >= min_kb:
                    rohs.append({
                        "IID_idx": i, "CHR": str(chrom),
                        "start_bp": int(bp[idxs[start_k]]),
                        "end_bp": int(bp[idxs[-1]]),
                        "n_snp": int(n_run),
                        "length_kb": float(length_kb)})
                    froh_bp[i] += bp[idxs[-1]] - bp[idxs[start_k]]
    froh = (froh_bp / genome_bp_total
            if genome_bp_total > 0 else froh_bp)
    roh_df = (pd.DataFrame(rohs) if rohs
              else pd.DataFrame(columns=["IID_idx", "CHR", "start_bp",
                                          "end_bp", "n_snp",
                                          "length_kb"]))
    return roh_df, froh


@cache_data
def selection_signatures(gt, _snp_df, _pop_labels):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    fst = fst_per_snp(gt, pop_labels)
    hom = np.nanmean((gt == 0) | (gt == 2), axis=0)
    df = snp_df.copy()
    df["FST"] = fst
    df["HOM"] = hom
    hom_std = df["HOM"].std()
    df["SCORE"] = (df["FST"].fillna(0)
                   * (df["HOM"] - df["HOM"].mean())
                   / (hom_std if hom_std > 1e-9 else 1.0))
    return df.sort_values("SCORE", ascending=False)


# ============================================================
# ⭐ SIGNATURES PAR GROUPES DE RACES (NOUVEAU)
# ============================================================

@cache_data
def fst_group_vs_rest(gt, _pop_labels, group_pops):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    mask_group = np.isin(pop_labels, list(group_pops))
    mask_rest = ~mask_group
    if mask_group.sum() < 5 or mask_rest.sum() < 5:
        return pd.DataFrame()
    gt_g = gt[mask_group]
    gt_r = gt[mask_rest]
    p_g = np.nanmean(gt_g, axis=0) / 2.0
    p_r = np.nanmean(gt_r, axis=0) / 2.0
    n_g = (~np.isnan(gt_g)).sum(axis=0)
    n_r = (~np.isnan(gt_r)).sum(axis=0)
    denom = np.where(n_g + n_r > 0, n_g + n_r, np.nan)
    p_bar = (p_g * n_g + p_r * n_r) / denom
    h_s = (2 * p_g * (1 - p_g) * n_g
           + 2 * p_r * (1 - p_r) * n_r) / denom
    h_t = 2 * p_bar * (1 - p_bar)
    with np.errstate(divide="ignore", invalid="ignore"):
        fst = (h_t - h_s) / h_t
    fst = np.clip(np.where(np.isfinite(fst), fst, np.nan), 0, 1)
    return pd.DataFrame({"FST": fst, "P_group": p_g, "P_rest": p_r,
                         "n_group": n_g, "n_rest": n_r})


@cache_data
def fst_group_vs_group(gt, _pop_labels, pops_A, pops_B):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    mask_A = np.isin(pop_labels, list(pops_A))
    mask_B = np.isin(pop_labels, list(pops_B))
    if mask_A.sum() < 5 or mask_B.sum() < 5:
        return pd.DataFrame()
    gt_A = gt[mask_A]
    gt_B = gt[mask_B]
    p_A = np.nanmean(gt_A, axis=0) / 2.0
    p_B = np.nanmean(gt_B, axis=0) / 2.0
    n_A = (~np.isnan(gt_A)).sum(axis=0)
    n_B = (~np.isnan(gt_B)).sum(axis=0)
    denom = np.where(n_A + n_B > 0, n_A + n_B, np.nan)
    p_bar = (p_A * n_A + p_B * n_B) / denom
    h_s = (2 * p_A * (1 - p_A) * n_A
           + 2 * p_B * (1 - p_B) * n_B) / denom
    h_t = 2 * p_bar * (1 - p_bar)
    with np.errstate(divide="ignore", invalid="ignore"):
        fst = (h_t - h_s) / h_t
    fst = np.clip(np.where(np.isfinite(fst), fst, np.nan), 0, 1)
    return pd.DataFrame({"FST": fst, "P_A": p_A, "P_B": p_B,
                         "n_A": n_A, "n_B": n_B})


def _extract_outliers_from_fst(fst_df, snp_df, quantile, min_fst):
    if fst_df.empty:
        return pd.DataFrame(), {}
    df = pd.concat([snp_df.reset_index(drop=True),
                    fst_df.reset_index(drop=True)],
                   axis=1).dropna(subset=["FST"])
    if df.empty:
        return pd.DataFrame(), {}
    q_thr = float(np.nanquantile(df["FST"], quantile))
    threshold = max(q_thr, min_fst)
    outliers = (df[df["FST"] >= threshold]
                .sort_values("FST", ascending=False)
                .reset_index(drop=True))
    stats_d = {
        "n_snp_analyzed": int(len(df)),
        "n_snp_outliers": int(len(outliers)),
        "fst_mean": float(df["FST"].mean()),
        "fst_median": float(df["FST"].median()),
        "fst_max": float(df["FST"].max()),
        "threshold": float(threshold),
    }
    return outliers, stats_d


@cache_data
def analyze_selection_by_groups(gt, snp_df, _pop_labels,
                                 groups_dict,
                                 quantile=0.99, min_fst=0.15):
    results = {}
    for group_name, pops_list in groups_dict.items():
        if len(pops_list) < 1:
            continue
        fst_df = fst_group_vs_rest(gt, _pop_labels, pops_list)
        if fst_df.empty:
            results[group_name] = {
                "status": "insufficient_data",
                "group": group_name}
            continue
        outliers, stats_d = _extract_outliers_from_fst(
            fst_df, snp_df, quantile, min_fst)
        all_fst = pd.concat([snp_df.reset_index(drop=True),
                             fst_df.reset_index(drop=True)], axis=1)
        results[group_name] = {
            "group": group_name,
            "pops": list(pops_list),
            "all_fst_df": all_fst,
            "outliers_df": outliers,
            "status": "success",
            **stats_d,
        }
    return results


@cache_data
def analyze_selection_between_groups(gt, snp_df, _pop_labels,
                                      groups_dict, group_A, group_B,
                                      quantile=0.99, min_fst=0.15):
    pops_A = groups_dict.get(group_A, [])
    pops_B = groups_dict.get(group_B, [])
    fst_df = fst_group_vs_group(gt, _pop_labels, pops_A, pops_B)
    if fst_df.empty:
        return {"status": "insufficient_data"}
    outliers, stats_d = _extract_outliers_from_fst(
        fst_df, snp_df, quantile, min_fst)
    all_fst = pd.concat([snp_df.reset_index(drop=True),
                         fst_df.reset_index(drop=True)], axis=1)
    return {
        "group_A": group_A, "group_B": group_B,
        "pops_A": list(pops_A), "pops_B": list(pops_B),
        "all_fst_df": all_fst,
        "outliers_df": outliers,
        "status": "success",
        **stats_d,
    }


def create_group_phenotype(ind_df, groups_dict, group_name):
    df = ind_df.copy()
    df["pheno_group"] = 0.0
    target = groups_dict.get(group_name, [])
    df.loc[df["FID"].astype(str).isin(target), "pheno_group"] = 1.0
    return df


# ============================================================
# BINÔMES
# ============================================================

@cache_data
def fst_per_snp_pair(gt, _pop_labels, pop_A, pop_B):
    pop_labels = np.asarray(_pop_labels, dtype=object)
    mask_A = (pop_labels == pop_A)
    mask_B = (pop_labels == pop_B)
    if mask_A.sum() < 3 or mask_B.sum() < 3:
        return pd.DataFrame()
    gt_A = gt[mask_A]
    gt_B = gt[mask_B]
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
                    fst_df.reset_index(drop=True)],
                   axis=1).dropna(subset=["FST"])
    if df.empty:
        return pd.DataFrame()
    q_thr = float(np.nanquantile(df["FST"], quantile))
    threshold = max(q_thr, min_fst)
    df["is_outlier"] = df["FST"] >= threshold
    df["threshold"] = threshold
    return (df[df["is_outlier"]]
            .sort_values("FST", ascending=False)
            .reset_index(drop=True))


@cache_data
def xp_ehh_pair(gt, _snp_df, _pop_labels, pop_A, pop_B, window=50):
    snp_df = _snp_df
    pop_labels = np.asarray(_pop_labels, dtype=object)
    mask_A = (pop_labels == pop_A)
    mask_B = (pop_labels == pop_B)
    if mask_A.sum() < 3 or mask_B.sum() < 3:
        return pd.DataFrame()
    gt_A = gt[mask_A]
    gt_B = gt[mask_B]
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
            hom_A = np.nanmean((gt_A[:, win] == 0)
                                | (gt_A[:, win] == 2))
            hom_B = np.nanmean((gt_B[:, win] == 0)
                                | (gt_B[:, win] == 2))
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
            .sort_values("XP_EHH", key=lambda s: s.abs(),
                         ascending=False).reset_index(drop=True))


@cache_data
def analyze_binome_complete(gt, _snp_df, _pop_labels,
                             pop_A, pop_B,
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
        pairs.append({"pop_A": a, "pop_B": b,
                      "distance_MDS": d,
                      "n_ind_A": counts[a], "n_ind_B": counts[b]})
    df = pd.DataFrame(pairs)
    if df.empty:
        return df
    q_low = df["distance_MDS"].quantile(min_dist_quantile)
    q_high = df["distance_MDS"].quantile(max_dist_quantile)
    def _cat(d):
        if d <= q_low:
            return "🟢 Proche"
        elif d >= q_high:
            return "🔴 Éloignée"
        return "🟡 Moyenne"
    df["catégorie"] = df["distance_MDS"].apply(_cat)
    return df.sort_values("distance_MDS").reset_index(drop=True)


def predict_fst_binome(distance_mds, pop_A, pop_B):
    d = float(distance_mds)
    fst_pred = min(0.35, 0.02 + 1.2 * d)
    fst_low = max(0.0, fst_pred * 0.6)
    fst_high = min(0.5, fst_pred * 1.4)
    if fst_pred < 0.08:
        interp = "🟢 Faible"
    elif fst_pred < 0.15:
        interp = "🟡 Modérée"
    elif fst_pred < 0.25:
        interp = "🟠 Forte"
    else:
        interp = "🔴 Très forte"
    return {"pop_A": pop_A, "pop_B": pop_B,
            "distance_MDS": round(d, 4),
            "FST_prédit": round(fst_pred, 4),
            "FST_IC_95": f"[{round(fst_low, 4)}, {round(fst_high, 4)}]",
            "interprétation": interp}


def score_binome_priority(binomes_df):
    df = binomes_df.copy()
    df["score_distance"] = (df["distance_MDS"]
                             / (df["distance_MDS"].max() + 1e-9))
    df["ratio_taille"] = (np.minimum(df["n_ind_A"], df["n_ind_B"])
                           / np.maximum(df["n_ind_A"], df["n_ind_B"]))
    df["score_puissance"] = (np.sqrt(df["n_ind_A"] * df["n_ind_B"])
                              / np.sqrt(df["n_ind_A"].max()
                                        * df["n_ind_B"].max()))
    df["score_priorité"] = (0.5 * df["score_distance"]
                             + 0.2 * df["ratio_taille"]
                             + 0.3 * df["score_puissance"])
    df = df.sort_values("score_priorité",
                         ascending=False).reset_index(drop=True)
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
                candidates.append((gene["gene_name"], 0,
                                    "intragenic"))
            else:
                dist = min(abs(bp - gene["start"]),
                            abs(bp - gene["end"]))
                if dist <= flank_bp:
                    region = ("upstream" if bp < gene["start"]
                              else "downstream")
                    candidates.append((gene["gene_name"], dist,
                                        region))
        if candidates:
            best = min(candidates, key=lambda x: x[1])
            rows.append({**snp.to_dict(),
                         "gene_name": best[0],
                         "distance": best[1],
                         "region": best[2]})
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
        row["partagé"] = ("✅" if gene in intersection_genes
                          else "❌")
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
        st.error(f"❌ Erreur : {e}")
        return None


def local_gwas_10cm(gt, snp_df, ind_df, phenotypes,
                     target_chr, target_bp, phenotype_name,
                     pop_covariate=True, window_cm=10):
    window_bp = window_cm * 1_000_000
    chr_str = snp_df["CHR"].astype(str)
    mask = ((chr_str == str(target_chr))
            & (snp_df["BP"] >= target_bp - window_bp)
            & (snp_df["BP"] <= target_bp + window_bp))
    idx_window = np.where(mask)[0]
    if len(idx_window) == 0:
        return pd.DataFrame()
    ind_df = ind_df.copy()
    ind_df["IID"] = ind_df["IID"].astype(str)
    pheno = phenotypes.set_index("IID")
    common_iids = [iid for iid in ind_df["IID"]
                    if iid in pheno.index]
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
    pop_dummies = np.zeros((len(pop_labels_sub),
                             max(0, len(unique_pops) - 1)))
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
                    cov_mat = np.linalg.inv(X.T @ X)
                    se_beta = np.sqrt(mse * cov_mat[1, 1])
                    t_stat = beta[1] / se_beta
                    pval = 2 * (1 - stats.t.cdf(abs(t_stat),
                                                  df_resid))
                except np.linalg.LinAlgError:
                    pval, t_stat = np.nan, np.nan
            else:
                pval, t_stat = np.nan, np.nan
            results.append({
                "SNP": snp_window.iloc[j]["SNP"],
                "CHR": snp_window.iloc[j]["CHR"],
                "BP": int(snp_window.iloc[j]["BP"]),
                "beta": float(beta[1]),
                "t_stat": (float(t_stat)
                            if np.isfinite(t_stat) else np.nan),
                "p_value": (float(pval)
                             if np.isfinite(pval) else np.nan),
                "r2": float(r2),
                "n_obs": int(valid.sum())})
        except Exception:
            continue
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res["p_bonferroni"] = np.minimum(
            1.0, df_res["p_value"] * len(df_res))
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
            "p_value_min": round(min_p, 6),
            "top_snp": top_snp["SNP"],
            "top_beta": round(float(top_snp["beta"]), 4),
            "interprétation": interp}


# ============================================================
# INTERPRÉTATION
# ============================================================

def interpret_qc(qc_stats):
    pct_ind = (qc_stats["excluded_ind"]
               / max(qc_stats["n_ind_init"], 1) * 100)
    pct_snp = (qc_stats["excluded_snp"]
               / max(qc_stats["n_snp_init"], 1) * 100)
    if pct_ind < 2 and pct_snp < 10:
        v, e = "🟢 EXCELLENTE", "Qualité très élevée."
    elif pct_ind < 5 and pct_snp < 25:
        v, e = "🟡 ACCEPTABLE", "Qualité correcte."
    else:
        v, e = "🔴 FAIBLE", "Exclusion élevée."
    return {"verdict": v, "explanation": e,
            "metrics": {"ind_exclus_%": round(pct_ind, 2),
                        "snp_exclus_%": round(pct_snp, 2)},
            "recommendation": ("Continuer." if "🟢" in v
                                else "Vérifier la qualité.")}


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


def interpret_roh(froh):
    m = float(np.nanmean(froh))
    if m < 0.02:
        v, e = "🟢 Faible", "Bonne diversité."
    elif m < 0.10:
        v, e = "🟡 Modérée", "Niveau normal."
    else:
        v, e = "🔴 Élevée", "Risque de dépression."
    return {"verdict": v, "explanation": e,
            "metrics": {"FROH_moyen": round(m, 4),
                        "FROH_max": round(float(np.nanmax(froh)), 4)},
            "recommendation": ("Croiser avec lignées extérieures"
                                if m > 0.10
                                else "Population saine.")}


def render_ai_panel(module, data, key_suffix=""):
    st.divider()
    st.markdown(f"### 🤖 Interprétation IA — {module}")
    interp_map = {"QC": interpret_qc, "MAF": interpret_maf,
                  "FST": interpret_fst, "ROH": interpret_roh}
    if module in interp_map:
        result = interp_map[module](data)
        st.success(f"**{result['verdict']}** — "
                   f"{result['explanation']}")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📊 Métriques**")
            st.json(result["metrics"])
        with c2:
            st.markdown("**✅ Recommandation**")
            st.info(result["recommendation"])


# ============================================================
# EXPORT
# ============================================================

def _dosage_to_plink_bits(col):
    n = len(col)
    bits = np.zeros(n, dtype=np.uint8)
    bits[np.isnan(col)] = 0b01
    bits[col == 0] = 0b11
    bits[col == 1] = 0b10
    bits[col == 2] = 0b00
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
        if a1 in ("nan", ""):
            a1 = "A"
        if a2 in ("nan", "", a1):
            a2 = "G" if a1 == "A" else "A"
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
              f"##source=BovineSNP-{project}",
              f"##fileDate={datetime.now():%Y%m%d}",
              '##FORMAT=<ID=GT,Number=1,Type=String,'
              'Description="Genotype">']
    samples = ind_df["IID"].astype(str).tolist()
    header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\t"
                  "INFO\tFORMAT\t" + "\t".join(samples))
    body = []
    for j in range(n_snp):
        r = snp_df.iloc[j]
        ref = str(r["A1"]) if has_a1 else "A"
        alt = str(r["A2"]) if has_a2 else "G"
        if ref in ("nan", ""):
            ref = "A"
        if alt in ("nan", "", ref):
            alt = "G" if ref == "A" else "A"
        gts = []
        for i in range(n_ind):
            v = gt[i, j]
            gts.append("./." if np.isnan(v)
                       else "0/0" if v == 0
                       else "0/1" if v == 1 else "1/1")
        body.append(f"{r['CHR']}\t{int(r['BP'])}\t{r['SNP']}\t"
                    f"{ref}\t{alt}\t.\tPASS\t.\tGT\t"
                    + "\t".join(gts))
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
    fig.add_trace(go.Histogram(x=values, nbinsx=80,
                                marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Fréquence", height=380)
    return fig


def plot_missingness_dashboard(miss_ind, miss_snp):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Missingness / individu",
                                        "Missingness / SNP"))
    fig.add_trace(go.Histogram(x=miss_ind, nbinsx=60,
                                marker_color="skyblue"),
                  row=1, col=1)
    fig.add_trace(go.Histogram(x=miss_snp, nbinsx=60,
                                marker_color="coral"),
                  row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_pca(scores, var_pct, labels):
    df = pd.DataFrame({"PC1": scores[:, 0],
                       "PC2": scores[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Population",
                     title=f"PCA — PC1 ({var_pct[0]:.1f}%) vs "
                           f"PC2 ({var_pct[1]:.1f}%)", height=550)
    fig.update_traces(marker=dict(size=10,
                                   line=dict(width=1,
                                              color="white")))
    return fig


def plot_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0],
                       "MDS2": coords[:, 1],
                       "Population": labels})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Population",
                     title="MDS (IBS)", height=550)
    fig.update_traces(marker=dict(size=10,
                                   line=dict(width=1,
                                              color="white")))
    return fig


def plot_manhattan(fst, chr_col, threshold_q=0.999,
                    title="Manhattan Plot — FST/SNP"):
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
        ticks.append((s + e) / 2)
        labels.append(str(chrom))
        cumulative = e
    df["x"] = cum_pos
    fig = go.Figure()
    for idx_chr, chrom in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == chrom]
        color = "#2c3e50" if idx_chr % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"],
                                  mode="markers",
                                  marker=dict(size=5, color=color),
                                  showlegend=False, hoverinfo="skip"))
    q_upper = float(np.nanquantile(fst, threshold_q))
    fig.add_hline(y=q_upper, line_dash="dash", line_color="red",
                  annotation_text=f"Top {100*(1-threshold_q):.1f}% "
                                  f"= {q_upper:.4f}",
                  annotation_position="top right")
    fig.update_layout(title=title, xaxis_title="Chromosome",
                      yaxis_title="FST", height=500)
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plot_ld_decay(ld_df, bin_kb=20):
    if ld_df is None or ld_df.empty:
        return None
    d = ld_df.copy()
    d["bin"] = (d["dist_kb"] // bin_kb) * bin_kb
    agg = d.groupby("bin")["r2"].mean().reset_index()
    fig = px.line(agg, x="bin", y="r2",
                  labels={"bin": "Distance (kb)",
                          "r2": "r² moyen"},
                  title="LD decay", height=450)
    fig.update_traces(line=dict(color="royalblue", width=3))
    return fig


def plot_roh_histogram(froh, labels):
    df = pd.DataFrame({"FROH": froh, "Population": labels})
    return px.histogram(df, x="FROH", color="Population",
                        nbins=50,
                        title="Distribution de FROH", height=450)


def plot_cv_error(cv_results):
    ks = sorted(cv_results.keys())
    means = [cv_results[k]["mean"] for k in ks]
    stds = [cv_results[k]["std"] for k in ks]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ks, y=means, mode="lines+markers",
        error_y=dict(type="data", array=stds),
        line=dict(color="crimson", width=3),
        marker=dict(size=12)))
    k_opt = ks[int(np.argmin(means))]
    fig.add_vline(x=k_opt, line_dash="dash",
                  line_color="green",
                  annotation_text=f"K opt = {k_opt}")
    fig.update_layout(title="Cross-Entropy Error",
                      xaxis_title="K", yaxis_title="CV Error",
                      height=450)
    return fig


def plot_admixture(Q, labels, pop_labels, K):
    df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
    df["IID"] = labels
    df["Pop"] = pop_labels
    df["_dom"] = Q.argmax(1)
    df["_po"] = pd.Categorical(
        pop_labels, categories=sorted(set(pop_labels)))
    df = df.sort_values(["_po", "_dom"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    palette = (px.colors.qualitative.Set2
               + px.colors.qualitative.Set3)[:K]
    fig = go.Figure()
    for k in range(K):
        fig.add_trace(go.Bar(
            x=df["x"], y=df[f"K{k+1}"],
            name=f"Composante {k+1}",
            marker_color=palette[k % len(palette)],
            hovertemplate="Ind %{customdata}<br>K"
                          + str(k + 1) + " = %{y:.2f}"
                          "<extra></extra>",
            customdata=df["IID"]))
    fig.update_layout(barmode="stack",
                      title=f"Ancestralité (K={K})",
                      xaxis_title="Individus triés",
                      yaxis_title="Proportion", height=500)
    return fig, df


def plot_fst_pairwise(matrix, pops):
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=pops, y=pops, colorscale="Viridis",
        colorbar=dict(title="FST"),
        text=np.round(matrix, 4), texttemplate="%{text}"))
    fig.update_layout(title="FST pairwise", height=550)
    return fig


def plot_ne_curves(ne_dict):
    fig = go.Figure()
    palette = (px.colors.qualitative.Set2
               + px.colors.qualitative.Set3)
    for i, (race, df) in enumerate(ne_dict.items()):
        if df is None or df.empty:
            continue
        fig.add_trace(go.Scatter(x=df["GenAgo"], y=df["Ne"],
                                  mode="lines+markers", name=race,
                                  line=dict(
                                      color=palette[i
                                                    % len(palette)],
                                      width=3)))
    fig.update_layout(title="Ne historique",
                      xaxis_title="Générations passées",
                      yaxis_title="Ne (relatif)", height=550)
    return fig


def plot_gwas_manhattan_local(gwas_df, chr_name, target_bp):
    if gwas_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gwas_df["BP"], y=gwas_df["neg_log10_p"],
        mode="markers",
        marker=dict(size=8, color=gwas_df["neg_log10_p"],
                    colorscale="Hot", showscale=False),
        hovertemplate="SNP: %{text}<br>BP: %{x}<br>"
                      "-log10(p): %{y:.2f}<extra></extra>",
        text=gwas_df["SNP"]))
    fig.add_hline(y=-np.log10(0.05), line_dash="dash",
                  line_color="orange", annotation_text="p=0.05")
    fig.add_hline(y=-np.log10(0.05 / len(gwas_df)),
                  line_dash="dash", line_color="red",
                  annotation_text="Bonferroni")
    fig.add_vline(x=target_bp, line_dash="dot",
                  line_color="blue",
                  annotation_text="SNP cible")
    fig.update_layout(title=f"GWAS local — Chr {chr_name}",
                      xaxis_title="Position (bp)",
                      yaxis_title="-log10(p)", height=450)
    return fig


# ============================================================
# ÉTAT
# ============================================================

_STATE_KEYS = [
    "gt", "ind_df", "snp_df",
    "gt_filt", "ind_filt", "snp_filt", "qc_stats",
    "gt_snp", "ind_snp", "snp_snp",
    "gt_pruned", "snp_pruned",
    "pca_scores", "pca_var", "mds_coords", "king_matrix",
    "ld_df", "fst", "fst_pairwise_matrix", "fst_pops",
    "admixture_Q", "admixture_K", "cv_results",
    "roh_df", "froh", "ne_dict",
    "reynolds_D", "reynolds_pops", "selection_df",
    "run_requested",
    "_plink_zip", "_vcf_str", "_report_html",
    "_binome_result", "_all_binomes", "_custom_pairs",
    "_genes_df", "_annotated",
    "_group_selection", "_group_pair_result",
    "_group_pair_names", "_group_pair_gt",
    "_group_pair_snp", "_group_pair_ind",
    "_gwas_results",
    "_conv_samples", "_conv_geno", "_conv_n_snp",
    "_conv_ped", "_conv_map",
]


def init_state():
    for k in _STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if "trait_groups" not in st.session_state:
        st.session_state["trait_groups"] = dict(DEFAULT_GROUPS)


def invalidate_downstream():
    for k in ["pca_scores", "pca_var", "mds_coords",
              "king_matrix", "ld_df", "fst",
              "fst_pairwise_matrix", "fst_pops",
              "admixture_Q", "admixture_K", "cv_results",
              "roh_df", "froh", "ne_dict", "reynolds_D",
              "reynolds_pops", "selection_df", "gt_pruned",
              "snp_pruned", "_binome_result", "_all_binomes",
              "_group_selection", "_group_pair_result",
              "_gwas_results"]:
        st.session_state[k] = None


def reset_all_derived():
    for k in (["gt_filt", "ind_filt", "snp_filt", "qc_stats",
               "gt_snp", "ind_snp", "snp_snp"]
              + ["pca_scores", "pca_var", "mds_coords",
                 "king_matrix", "ld_df", "fst",
                 "fst_pairwise_matrix", "fst_pops",
                 "admixture_Q", "admixture_K", "cv_results",
                 "roh_df", "froh", "ne_dict", "reynolds_D",
                 "reynolds_pops", "selection_df", "gt_pruned",
                 "snp_pruned", "_binome_result", "_all_binomes",
                 "_group_selection", "_group_pair_result",
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
# HELPERS UI
# ============================================================

def _run_binome_analysis(gt, snp_df, pop_labels,
                          pop_A, pop_B, quantile, min_fst):
    try:
        with st.spinner(f"Analyse {pop_A} vs {pop_B}..."):
            result = analyze_binome_complete(
                gt, snp_df, pop_labels, pop_A, pop_B,
                quantile, min_fst)
            st.session_state["_binome_result"] = result
        if result.get("status") == "success":
            st.success(f"✅ {pop_A} vs {pop_B} : "
                       f"{result['n_snp_outliers']} SNP outliers")
        else:
            st.error(f"❌ {result.get('status', 'Erreur')}")
    except Exception as e:
        st.error(f"❌ {e}")


def _display_binome_results(pop_A, pop_B):
    result = st.session_state.get("_binome_result")
    if not result or result.get("status") != "success":
        return
    if result.get("pop_A") != pop_A or result.get("pop_B") != pop_B:
        return
    st.divider()
    st.markdown(f"### 📊 {pop_A} vs {pop_B}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ind. " + pop_A, result["n_ind_A"])
    c2.metric("Ind. " + pop_B, result["n_ind_B"])
    c3.metric("SNP analysés", f"{result['n_snp_analyzed']:,}")
    c4.metric("SNP outliers", result["n_snp_outliers"])
    c5.metric("FST moyen", f"{result['fst_mean']:.4f}")
    outliers = result["outliers_df"]
    if not outliers.empty:
        st.dataframe(
            outliers[["CHR", "SNP", "BP", "FST",
                       "P_A", "P_B"]].head(100),
            use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=outliers["FST"],
                                    nbinsx=40,
                                    marker_color="crimson"))
        fig.update_layout(
            title=f"Distribution FST — {pop_A} vs {pop_B}",
            xaxis_title="FST", yaxis_title="Nb SNPs", height=400)
        st.plotly_chart(fig, use_container_width=True)
        csv_out = outliers.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"⬇ SNP outliers ({pop_A} vs {pop_B})",
            data=csv_out,
            file_name=f"outliers_{pop_A}_vs_{pop_B}.csv",
            mime="text/csv", use_container_width=True,
            key=f"dl_out_{pop_A}_{pop_B}")


def _display_all_binomes_summary():
    if not st.session_state.get("_all_binomes"):
        return
    st.divider()
    st.markdown("### 📊 Synthèse globale")
    summary = []
    for name, r in st.session_state["_all_binomes"].items():
        if r.get("status") == "success" and "fst_mean" in r:
            summary.append({
                "Binôme/Groupe": name.replace("_vs_", " ↔ "),
                "SNPs": r.get("n_snp_analyzed", "-"),
                "Outliers": r.get("n_snp_outliers", "-"),
                "FST moyen": round(r.get("fst_mean", 0), 4),
                "FST max": round(r.get("fst_max", 0), 4)})
    if not summary:
        return
    sum_df = pd.DataFrame(summary)
    st.dataframe(sum_df, use_container_width=True)


# ============================================================
# MAIN
# ============================================================

def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v6.2")
    st.caption("Pipeline complet + groupes de races + signatures "
               "par groupe + GWAS + annotation GTF")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio(
            "Source :",
            ["Demo", "Upload PED/MAP", "Upload VCF",
             "Convertisseur Axiom"],
            index=0, key="sb_mode")

        if mode == "Demo":
            c1, c2 = st.columns(2)
            n_ind = c1.number_input("Individus", 20, 1000, 150, 10,
                                    key="sb_n_ind")
            n_snp = c2.number_input("SNPs", 100, 10000, 800, 100,
                                    key="sb_n_snp")
            n_pop = st.slider("Populations", 2, 14, 4,
                              key="sb_n_pop")
            if st.button("🎲 Générer le jeu de démo",
                         use_container_width=True,
                         key="sb_btn_demo"):
                with st.spinner("Génération..."):
                    gt, ind_df, snp_df = generate_demo_data(
                        int(n_ind), int(n_snp), int(n_pop))
                    ind_df = reconstruct_fid_from_iid(ind_df)
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind_df
                    st.session_state.snp_df = snp_df
                    reset_all_derived()
                st.success(f"✅ {gt.shape[0]} ind × "
                           f"{gt.shape[1]} SNPs")

        elif mode == "Upload PED/MAP":
            st.info("Formats : .ped, .ped.gz, .map, .map.gz")
            ped_f = st.file_uploader("Fichier .ped",
                                     type=["ped", "gz", "txt"],
                                     key="sb_ped")
            map_f = st.file_uploader("Fichier .map",
                                     type=["map", "gz", "txt"],
                                     key="sb_map")
            if ped_f and map_f:
                if st.button("📥 Charger PED + MAP",
                             use_container_width=True,
                             key="sb_btn_ped"):
                    try:
                        with st.spinner("Parsing .map..."):
                            map_df, _ = parse_map(map_f.read())
                        with st.spinner(
                                f"Parsing .ped "
                                f"({ped_f.size/1024/1024:.1f} Mo)..."):
                            gt, ind_df, _ = parse_ped(
                                ped_f.read(), len(map_df))
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

        elif mode == "Upload VCF":
            vcf_f = st.file_uploader("Fichier .vcf",
                                     type=["vcf", "gz", "txt"],
                                     key="sb_vcf")
            if vcf_f is not None:
                if st.button("📥 Charger VCF",
                             use_container_width=True,
                             key="sb_btn_vcf"):
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
                         DEFAULT_THRESHOLDS["geno"], 0.01,
                         key="sb_geno")
        mind = st.slider("Missingness individu", 0.0, 0.5,
                         DEFAULT_THRESHOLDS["mind"], 0.01,
                         key="sb_mind")
        maf_thr = st.slider("MAF minimal", 0.0, 0.5,
                            DEFAULT_THRESHOLDS["maf"], 0.01,
                            key="sb_maf")
        hwe_thr = st.number_input("HWE p-value",
                                  value=DEFAULT_THRESHOLDS["hwe"],
                                  format="%.0e", key="sb_hwe")
        het_sd = st.slider("Hétérozygotie ±σ", 1.0, 5.0,
                           DEFAULT_THRESHOLDS["het_sd"], 0.1,
                           key="sb_het_sd")
        ld_r2 = st.slider("LD Pruning r²", 0.05, 0.5,
                          DEFAULT_THRESHOLDS["ld_r2"], 0.05,
                          key="sb_ld_r2")
        king_cutoff = st.slider("KING cutoff", 0.1, 0.5,
                                DEFAULT_THRESHOLDS["king_cutoff"],
                                0.01, key="sb_king")

        # ⭐ GROUPES DE RACES
        st.divider()
        st.header("🐄 Groupes de races")
        st.caption("Regrouper les races par trait partagé "
                   "(laitier / viande / cornu / rustique)")

        if has_data():
            all_pops = sorted(
                st.session_state.ind_df["FID"].astype(str).unique())
            if "trait_groups" not in st.session_state:
                st.session_state["trait_groups"] = dict(DEFAULT_GROUPS)
            with st.expander("Définir les groupes", expanded=False):
                for group_name in ["Laitier", "Viande",
                                    "Cornu", "Rustique"]:
                    selected = st.multiselect(
                        f"Races → {group_name}", all_pops,
                        default=st.session_state[
                            "trait_groups"].get(group_name, []),
                        key=f"grp_{group_name}")
                    st.session_state["trait_groups"][group_name] = \
                        selected

        st.divider()
        if st.button("🚀 Pipeline complet", type="primary",
                     use_container_width=True, key="sb_btn_pipe"):
            if not has_data():
                st.error("Chargez d'abord des données.")
            else:
                st.session_state.run_requested = True

        st.divider()
        st.caption("v6.2 — Groupes + signatures + annotation")

    # ---------- CAS SPÉCIAL AXIOM ----------
    if mode == "Convertisseur Axiom":
        render_axiom_converter_page()
        return

    # ---------- PAS DE DONNÉES ----------
    if not has_data():
        st.info("👉 Générez un jeu de démo, importez PED/MAP/VCF, "
                "ou utilisez le Convertisseur Axiom.")
        st.markdown("""
        ### 🧬 Fonctionnalités v6.2

        | Onglet | Contenu |
        |---|---|
        | 🏠 Aperçu | Statistiques |
        | 🧹 QC | Cascade de filtres |
        | ✂️ LD Pruning | Indep-pairwise |
        | 🧬 Structure | PCA, MDS, KING |
        | 🎨 Admixture | NMF + CV |
        | 📈 Démographie | LD decay, ROH, Ne |
        | 🔍 Sélection | FST, Manhattan |
        | 🌳 Phylogénie | Reynolds + NJ |
        | 📤 Export | PLINK, VCF |
        | 📄 Rapport | HTML |
        | 🧬 **Signatures & Groupes** | **Binômes + Groupes** |
        | 🧪 SNP & Gènes | Annotation GTF + GWAS 10cM |

        ### 🆕 Nouveau dans v6.2
        - ✅ Convertisseur Axiom
        - ✅ Signatures par population vs reste
        - ✅ Signatures par **groupes de races**
        - ✅ Comparaison **Groupe A vs Groupe B**
        - ✅ GWAS 10 cM avec **phénotype groupe**
        - ✅ Fix detect_roh (json.loads)
        - ✅ `gt_snp` utilisé strictement pour la sélection
        """)
        return

    gt = st.session_state.gt
    ind_df = st.session_state.ind_df
    snp_df = st.session_state.snp_df

    tabs = st.tabs([
        "🏠 Aperçu", "🧹 QC", "✂️ LD Pruning",
        "🧬 Structure", "🎨 Admixture", "📈 Démographie",
        "🔍 Sélection", "🌳 Phylogénie", "📤 Export",
        "📄 Rapport", "🧬 Signatures & Groupes",
        "🧪 SNP & Gènes"])

    # ---------- TAB 0 : Aperçu ----------
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        st.subheader("Individus")
        st.dataframe(ind_df.head(20), use_container_width=True)
        st.subheader("SNPs")
        st.dataframe(snp_df.head(5), use_container_width=True)

    # ---------- TAB 1 : QC ----------
    with tabs[1]:
        st.subheader("🧹 Contrôle Qualité")
        if st.button("🧬 Filtrer autosomes 1-29",
                     use_container_width=True,
                     key="qc_btn_auto"):
            try:
                gt_a, snp_a = filter_autosomes(
                    st.session_state.gt, st.session_state.snp_df)
                st.session_state.gt = gt_a
                st.session_state.snp_df = snp_a
                st.success(f"✅ Autosomes → {gt_a.shape[1]} SNPs")
            except Exception as e:
                st.error(f"❌ {e}")

        if st.button("▶ Lancer le QC complet", type="primary",
                     use_container_width=True, key="qc_btn_run"):
            try:
                with st.spinner("QC..."):
                    params = {"geno": geno, "mind": mind,
                              "maf": maf_thr, "hwe": hwe_thr,
                              "het_sd": het_sd}
                    (gt_f, ind_f, snp_f, qc_stats,
                     gt_s, ind_s, snp_s) = apply_qc_filters(
                        st.session_state.gt,
                        st.session_state.ind_df,
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
                    f"• Structure (gt_filt) : {gt_f.shape[0]} ind × "
                    f"{gt_f.shape[1]:,} SNP\n"
                    f"• Sélection (gt_snp, sans HET outliers) : "
                    f"{gt_s.shape[0]} ind × {gt_s.shape[1]:,} SNP")
            except Exception as e:
                st.error(f"❌ {e}")

        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind. finaux", s["n_ind_final"])
            c2.metric("SNPs finaux", s["n_snp_final"])
            c3.metric("Exclus (ind)", s["excluded_ind"])
            c4.metric("Exclus (SNPs)", s["excluded_snp"])
            st.info(f"ℹ️ **gt_snp** disponible : {s['n_ind_snp']} ind "
                    f"× {s['n_snp_snp']:,} SNP (pour sélection)")
            gt_full = st.session_state.gt
            st.plotly_chart(plot_missingness_dashboard(
                missingness_per_ind(gt_full),
                missingness_per_snp(gt_full)),
                use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_hist(maf(gt_full),
                                          "Spectre MAF",
                                          "MAF", "#2ecc71"),
                                use_container_width=True)
            with c2:
                st.plotly_chart(plot_hist(
                    heterozygosity(gt_full),
                    "Hétérozygotie", "HET", "#9b59b6"),
                    use_container_width=True)
            render_ai_panel("QC", s, "tab_qc")

    # ---------- TAB 2 : LD Pruning ----------
    with tabs[2]:
        st.subheader("✂️ LD Pruning")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if st.button("▶ Lancer le LD Pruning", type="primary",
                         use_container_width=True, key="ld_btn"):
                try:
                    with st.spinner("LD pruning..."):
                        keep = ld_pruning(
                            st.session_state.gt_filt,
                            window=50, step=5, r2_thr=ld_r2)
                        st.session_state.gt_pruned = (
                            st.session_state.gt_filt[:, keep])
                        st.session_state.snp_pruned = (
                            st.session_state.snp_filt[keep]
                            .reset_index(drop=True))
                    st.success(
                        f"✅ {st.session_state.gt_pruned.shape[1]} "
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

    # ---------- TAB 3 : Structure ----------
    with tabs[3]:
        st.subheader("🧬 Structure")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            if st.button("▶ PCA + MDS + KING", type="primary",
                         use_container_width=True, key="struct_btn"):
                try:
                    with st.spinner("PCA..."):
                        s_, v_ = pca_analysis(gt_use, 10)
                        st.session_state.pca_scores = s_
                        st.session_state.pca_var = v_
                    with st.spinner("MDS..."):
                        st.session_state.mds_coords = mds_analysis(
                            gt_use, 5)
                    with st.spinner("KING..."):
                        st.session_state.king_matrix = king_kinship(
                            gt_use)
                    st.success("✅ Terminé.")
                except Exception as e:
                    st.error(f"❌ {e}")
            labels = st.session_state.ind_filt["FID"].values
            if st.session_state.pca_scores is not None:
                st.plotly_chart(plot_pca(
                    st.session_state.pca_scores,
                    st.session_state.pca_var, labels),
                    use_container_width=True)
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plot_mds(
                    st.session_state.mds_coords, labels),
                    use_container_width=True)

    # ---------- TAB 4 : Admixture ----------
    with tabs[4]:
        st.subheader("🎨 Admixture")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            c1, c2 = st.columns(2)
            K_max = c1.slider("K max", 3, 10, 6, key="admix_K_max")
            n_reps = c2.slider("Répétitions CV", 1, 5, 3,
                               key="admix_n_reps")
            if st.button("▶ CV error", use_container_width=True,
                         key="admix_btn_cv"):
                try:
                    with st.spinner("CV..."):
                        cv = admix_cv_error(
                            gt_use, K_range=(2, K_max + 1),
                            n_reps=int(n_reps))
                        st.session_state.cv_results = cv
                    st.success("✅ CV terminée.")
                except Exception as e:
                    st.error(f"❌ {e}")
            if st.session_state.cv_results is not None:
                st.plotly_chart(plot_cv_error(
                    st.session_state.cv_results),
                    use_container_width=True)
            K = st.slider("K", 2, 10, 4, key="admix_K")
            if st.button("▶ Admixture", use_container_width=True,
                         key="admix_btn_run"):
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
                fig, _ = plot_admixture(Q, iids, pops, K)
                st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 5 : Démographie ----------
    with tabs[5]:
        st.subheader("📈 Démographie")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_use = (st.session_state.gt_pruned if has_pruned()
                      else st.session_state.gt_filt)
            sub1, sub2 = st.tabs(["LD decay", "ROH"])
            with sub1:
                if st.button("▶ LD decay", key="demo_ld_btn"):
                    with st.spinner("LD decay..."):
                        st.session_state.ld_df = ld_decay(
                            gt_use,
                            st.session_state.snp_filt["BP"].values,
                            max_kb=1000, max_snp=1000)
                    st.success("✅ calculé.")
                if (st.session_state.ld_df is not None
                        and not st.session_state.ld_df.empty):
                    st.plotly_chart(plot_ld_decay(
                        st.session_state.ld_df),
                        use_container_width=True)
            with sub2:
                if st.button("▶ ROH", key="demo_roh_btn"):
                    try:
                        with st.spinner("ROH..."):
                            snp_json = st.session_state.snp_filt[
                                ["CHR", "BP"]].to_json()
                            roh_df, froh = detect_roh(
                                gt_use, snp_json)
                            st.session_state.roh_df = roh_df
                            st.session_state.froh = froh
                        st.success(f"✅ {len(roh_df)} ROH")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.froh is not None:
                    labels = st.session_state.ind_filt["FID"].values
                    st.plotly_chart(plot_roh_histogram(
                        st.session_state.froh, labels),
                        use_container_width=True)

    # ---------- TAB 6 : Sélection (FST global) ----------
    with tabs[6]:
        st.subheader("🔍 Sélection globale (FST par SNP)")
        st.info("ℹ️ Cette analyse utilise **gt_snp** "
                "(sans retrait des outliers HET).")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            if has_snp_state():
                gt_use = st.session_state.gt_snp
                snp_use = st.session_state.snp_snp
                ind_use = st.session_state.ind_snp
            else:
                gt_use = st.session_state.g

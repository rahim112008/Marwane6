"""
🐄 Bovine SNP Platform v7.0 — Version compacte et fonctionnelle
Corrections : detect_roh (json.loads), code complet
Nouveautés  : Groupes de races, signatures par groupe, annotation GTF, GWAS 10cM
"""
import gzip
import json
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
from sklearn.decomposition import PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🐄 Bovine SNP v7", page_icon="🐄",
                    layout="wide", initial_sidebar_state="expanded")

DEFAULTS = {"geno": 0.05, "mind": 0.05, "maf": 0.05,
            "hwe": 1e-6, "het_sd": 3.0, "king": 0.354, "ld_r2": 0.2}

ARS_LEN = {"1": 158_534_110, "2": 136_231_102, "3": 121_005_158,
            "4": 120_000_166, "5": 120_089_699, "6": 117_806_340,
            "7": 110_682_743, "8": 113_384_748, "9": 105_708_134,
            "10": 103_308_737, "11": 107_310_763, "12": 91_163_125,
            "13": 84_246_514, "14": 84_648_346, "15": 85_207_080,
            "16": 81_726_628, "17": 75_176_999, "18": 66_059_976,
            "19": 64_089_169, "20": 72_042_983, "21": 71_599_096,
            "22": 61_416_492, "23": 52_531_573, "24": 62_384_193,
            "25": 42_959_610, "26": 51_680_158, "27": 46_772_073,
            "28": 46_333_854, "29": 51_319_414}
AUTOSOMES = [str(i) for i in range(1, 30)]
DEFAULT_GROUPS = {"Laitier": [], "Viande": [], "Cornu": [], "Rustique": []}


# ============================================================
# CACHE
# ============================================================
def _hash_np(x):
    if not isinstance(x, np.ndarray) or x.size == 0:
        return ("empty",)
    return (x.shape, str(x.dtype), float(np.nansum(x)),
            float(np.nansum(np.abs(x))), float(np.nansum(x * x)))

HASH_FUNCS = {np.ndarray: _hash_np}


def cached(fn=None, **kw):
    def wrap(f):
        return st.cache_data(show_spinner=False,
                             hash_funcs=HASH_FUNCS, **kw)(f)
    return wrap(fn) if fn else wrap


def impute(gt):
    gt2 = gt.astype(np.float32, copy=True)
    cm = np.nanmean(gt2, axis=0)
    cm = np.where(np.isnan(cm), 0.0, cm)
    nm = np.isnan(gt2)
    if nm.any():
        gt2[nm] = np.take(cm, np.where(nm)[1])
    return gt2


def _chr_key(c):
    s = str(c).upper().replace("CHR", "")
    if s.isdigit():
        return (0, int(s))
    special = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    return (1, special.get(s, 999))


def _autosome_weights():
    lens = np.array([ARS_LEN[c] for c in AUTOSOMES], dtype=float)
    return lens / lens.sum()


# ============================================================
# CONVERTISSEUR AXIOM
# ============================================================
def _clean_allele(a):
    if a is None:
        return "0"
    a = str(a).strip()
    return "0" if a in ("", ".", "-", "NA", "nan", "None", "0", "00") else a


def sample_to_fid_iid(name, fid_parts=2):
    n = str(name).strip()
    for ext in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if n.endswith(ext):
            n = n[:-len(ext)]
    n = (n.replace("(", "_").replace(")", "")
         .replace("[", "_").replace("]", "")
         .replace(" ", "_").replace("/", "_"))
    while "__" in n:
        n = n.replace("__", "_")
    n = n.strip("_")
    p = n.split("_")
    return ("_".join(p[:fid_parts]) if len(p) >= fid_parts else p[0]), n


def parse_axiom(raw_bytes, filename=""):
    if (filename.endswith(".gz")
            or (len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b")):
        try:
            raw = gzip.decompress(raw_bytes)
        except Exception as e:
            raise ValueError(f"Fichier .gz corrompu : {e}")
    else:
        raw = raw_bytes
    text = raw.decode("utf-8", errors="replace")
    lines = [l.rstrip("\r\n") for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("Fichier vide.")
    first = lines[0]
    skip = 1 if (first.startswith("#") or first.startswith("Sample")
                 or "Sample ID" in first[:50]) else 0
    samples, geno_rows = [], []
    for line in lines[skip:]:
        parts = line.split("\t")
        if len(parts) < 2:
            parts = line.split()
            if len(parts) < 2:
                continue
        sample, cells = parts[0].strip(), [c.strip() for c in parts[1:]]
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
                    alleles.append((_clean_allele(s), _clean_allele(s)))
            else:
                alleles.append(("0", "0"))
        samples.append(sample)
        geno_rows.append(alleles)
    if not samples:
        raise ValueError("Aucun échantillon.")
    common = Counter([len(r) for r in geno_rows]).most_common(1)[0][0]
    for i, r in enumerate(geno_rows):
        if len(r) > common:
            geno_rows[i] = r[:common]
        elif len(r) < common:
            geno_rows[i] = r + [("0", "0")] * (common - len(r))
    return samples, geno_rows, common


def render_axiom_page():
    st.header("🧬 Convertisseur Axiom → PED/MAP")
    st.markdown("Transforme un fichier Axiom (`A/G`, `T T`) "
                "en **PED** + **MAP** PLINK.")
    st.divider()
    raw_f = st.file_uploader("Fichier Axiom (.txt/.tsv/.csv/.gz)",
                              type=["txt", "tsv", "csv", "ped", "gz"],
                              key="ax_file")
    c1, c2 = st.columns(2)
    fid_parts = c1.number_input("Parties FID", 1, 5, 2, 1, key="ax_fid")
    sep_out = c2.selectbox("Séparateur", ["Tabulation", "Espace"],
                            key="ax_sep")
    if raw_f and st.button("Convertir", type="primary",
                            use_container_width=True, key="ax_btn"):
        try:
            with st.spinner("Parsing..."):
                samples, geno_rows, n_snp = parse_axiom(
                    raw_f.read(), raw_f.name)
            map_text = "\n".join(
                f"1\tSNP_{i+1:06d}\t0\t{i+1}"
                for i in range(n_snp)) + "\n"
            ped_lines = []
            for i, smp in enumerate(samples):
                fid, iid = sample_to_fid_iid(smp, fid_parts=int(fid_parts))
                cols = [fid, iid, "0", "0", "0", "-9"]
                for a1, a2 in geno_rows[i]:
                    cols.extend([a1, a2])
                ped_lines.append("\t".join(cols))
            ped_text = "\n".join(ped_lines) + "\n"
            if sep_out == "Espace":
                ped_text = ped_text.replace("\t", " ")
                map_text = map_text.replace("\t", " ")
            st.session_state["_ax_ped"] = ped_text
            st.session_state["_ax_map"] = map_text
            st.session_state["_ax_n"] = n_snp
            st.session_state["_ax_samples"] = samples
            st.success(f"✅ {len(samples)} ind × {n_snp} SNPs")
        except Exception as e:
            st.error(f"❌ {e}")
    if st.session_state.get("_ax_ped"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", len(st.session_state["_ax_samples"]))
        c2.metric("SNPs", st.session_state["_ax_n"])
        c3.metric("Colonnes", 6 + 2 * st.session_state["_ax_n"])
        st.subheader("Téléchargement")
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ .ped",
                            st.session_state["_ax_ped"].encode("utf-8"),
                            "converted.ped", "text/plain",
                            use_container_width=True, key="dl_ax_ped")
        c2.download_button("⬇ .map",
                            st.session_state["_ax_map"].encode("utf-8"),
                            "converted.map", "text/plain",
                            use_container_width=True, key="dl_ax_map")
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("converted.ped", st.session_state["_ax_ped"])
            zf.writestr("converted.map", st.session_state["_ax_map"])
        c3.download_button("⬇ .zip", bio.getvalue(),
                            "converted.zip", "application/zip",
                            use_container_width=True, key="dl_ax_zip")


# ============================================================
# PARSERS
# ============================================================
def _enc(raw):
    if len(raw) >= 2 and raw[:2] == b"\x1f\x8b":
        return "gzip"
    if len(raw) >= 2 and raw[:2] == b"\x6c\x1b":
        return "binary"
    return "text"


def parse_map(b):
    if _enc(b) == "gzip":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    elif _enc(b) == "binary":
        raise ValueError("❌ .map binaire PLINK")
    else:
        text = b.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        p = line.strip().split()
        if len(p) < 4:
            continue
        try:
            bp = int(float(p[3]))
        except (ValueError, TypeError):
            continue
        try:
            cm = float(p[2]) if p[2] not in ("0", ".", "NA") else 0.0
        except ValueError:
            cm = 0.0
        rows.append({"CHR": str(p[0]), "SNP": str(p[1]),
                     "CM": cm, "BP": bp})
    if not rows:
        raise ValueError("❌ .map vide ou invalide")
    return pd.DataFrame(rows)


def parse_ped(b, n_snp_map):
    if _enc(b) == "gzip":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    elif _enc(b) == "binary":
        raise ValueError("❌ .ped binaire PLINK")
    else:
        text = b.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        raise ValueError("❌ .ped vide")
    if len(lines) >= 2:
        n0, n1 = len(lines[0].split()), len(lines[1].split())
        if (n0 < 7 and n1 >= 7) or (n1 > 0 and n0 < n1 * 0.5):
            lines = lines[1:]
            st.info("ℹ️ Header PED ignoré")
    n_cols = len(lines[0].split())
    if n_cols < 7:
        raise ValueError(f"❌ Format PED : {n_cols} colonnes")
    n_snp_ped = (n_cols - 6) // 2
    n_snp = min(n_snp_map, n_snp_ped) if n_snp_ped > 0 else n_snp_map
    exp_cols = 6 + 2 * n_snp
    fids, iids, genos = [], [], []
    for line in lines:
        p = line.split()
        if len(p) < exp_cols:
            continue
        fids.append(p[0])
        iids.append(p[1])
        genos.append(p[6:6 + 2 * n_snp])
    n_ind = len(iids)
    if n_ind == 0:
        raise ValueError("❌ Aucun individu chargé")
    # Allèle mineur
    counts = [Counter() for _ in range(n_snp)]
    for row in genos:
        for j in range(n_snp):
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 != "0" and a2 != "0":
                counts[j][a1] += 1
                counts[j][a2] += 1
    minor = [min(c, key=c.get) if len(c) >= 2 else None
             for c in counts]
    gt = np.full((n_ind, n_snp), np.nan, dtype=np.float32)
    for i, row in enumerate(genos):
        for j in range(n_snp):
            m = minor[j]
            if m is None:
                continue
            a1, a2 = row[2 * j], row[2 * j + 1]
            if a1 == "0" or a2 == "0":
                continue
            gt[i, j] = (a1 == m) + (a2 == m)
    return gt, pd.DataFrame({"FID": fids, "IID": iids})


def parse_vcf(b):
    if len(b) >= 2 and b[:2] == b"\x1f\x8b":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    else:
        text = b.decode("utf-8", errors="replace")
    samples, rows = [], []
    for line in text.splitlines():
        if not line or line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
            continue
        if line.startswith("#"):
            continue
        p = line.split("\t")
        if len(p) < 10 or "," in p[4]:
            continue
        try:
            pos = int(p[1])
        except ValueError:
            continue
        fmt = p[8].split(":")
        if "GT" not in fmt:
            continue
        gi = fmt.index("GT")
        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(p[9:]):
            f = s.split(":")
            if gi >= len(f):
                continue
            g = f[gi]
            sep = "|" if "|" in g else "/"
            al = g.split(sep)
            if any(a in (".", "") for a in al):
                continue
            try:
                gts[k] = sum(int(a) for a in al)
            except ValueError:
                continue
        rows.append({"CHR": str(p[0]), "SNP": str(p[2] or f"{p[0]}:{pos}"),
                     "CM": 0.0, "BP": pos, "A1": p[3], "A2": p[4],
                     "GT": gts})
    if not rows:
        raise ValueError("❌ Aucun variant biallélique")
    gt = np.zeros((len(samples), len(rows)), dtype=np.float32)
    for j, r in enumerate(rows):
        gt[:, j] = r["GT"]
    for j in range(len(rows)):
        col = gt[:, j]
        v = col[~np.isnan(col)]
        if len(v) and v.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(col), np.nan, 2.0 - col)
    snp_df = pd.DataFrame({
        "CHR": [r["CHR"] for r in rows],
        "SNP": [r["SNP"] for r in rows],
        "CM": 0.0, "BP": [r["BP"] for r in rows],
        "A1": [r["A1"] for r in rows],
        "A2": [r["A2"] for r in rows]})
    ind_df = pd.DataFrame({"FID": samples, "IID": samples})
    return gt, ind_df, snp_df


def demo_data(n_ind=150, n_snp=800, n_pop=4, seed=42):
    rng = np.random.default_rng(seed)
    pool = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
            "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]
    pops = pool[:min(n_pop, len(pool))]
    per, rem = n_ind // len(pops), n_ind % len(pops)
    counts = [per + (1 if i < rem else 0) for i in range(len(pops))]
    p_anc = rng.beta(0.6, 0.6, n_snp)
    gt = np.zeros((sum(counts), n_snp), dtype=np.float32)
    ind_rows = []
    k = 0
    for pop, n_i in zip(pops, counts):
        p_k = np.clip(p_anc + rng.normal(0, 0.15, n_snp), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(n_snp) < p_k).astype(np.float32)
            a2 = (rng.random(n_snp) < p_k).astype(np.float32)
            gt[k] = a1 + a2
            ind_rows.append({"FID": pop, "IID": f"{pop}_{i+1:03d}"})
            k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    chrs = rng.choice(AUTOSOMES, n_snp, p=_autosome_weights())
    bps = np.array([rng.integers(1, ARS_LEN[c]) for c in chrs],
                   dtype=np.int64)
    snp_df = pd.DataFrame({"CHR": chrs,
                           "SNP": [f"rs{i:07d}" for i in range(n_snp)],
                           "CM": 0.0, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_key)
    snp_df = (snp_df.sort_values(["_k", "BP"])
              .drop(columns="_k").reset_index(drop=True))
    return gt, pd.DataFrame(ind_rows), snp_df
    # ============================================================
# MÉTRIQUES QC
# ============================================================
def miss_ind(gt): return np.isnan(gt).mean(axis=1)
def miss_snp(gt): return np.isnan(gt).mean(axis=0)
def allele_freq(gt): return np.nanmean(gt, axis=0) / 2.0
def maf(gt): return np.minimum(allele_freq(gt), 1 - allele_freq(gt))
def het_obs(gt):
    with np.errstate(invalid="ignore"):
        return np.nanmean(gt == 1, axis=1)


def _hwe_exact(n1, n0, n2):
    n = n1 + n0 + n2
    if n == 0: return np.nan
    if n1 == 0 and (n0 == 0 or n2 == 0): return 1.0
    rare = 2 * min(n0, n2) + n1
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    probs = np.zeros(rare + 1); probs[mid] = 1.0; s = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        probs[ch + 2] = probs[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        s += probs[ch + 2]; ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        probs[ch - 2] = probs[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        s += probs[ch - 2]; ch -= 2; chr_ += 1; chc += 1
    p_obs = probs[n1] if n1 < len(probs) else 0
    return float(min(probs[probs <= p_obs + 1e-7].sum() / s, 1.0))


@cached
def hwe_pvals(gt):
    _, n_snp = gt.shape
    pv = np.full(n_snp, np.nan)
    nm = np.isnan(gt)
    n0 = ((gt == 0) & ~nm).sum(0)
    n1 = ((gt == 1) & ~nm).sum(0)
    n2 = ((gt == 2) & ~nm).sum(0)
    nv = n0 + n1 + n2
    for j in range(n_snp):
        if nv[j] < 5: continue
        pv[j] = _hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    return pv


def _align(gt, ind_df, snp_df):
    n = min(gt.shape[0], len(ind_df))
    m = min(gt.shape[1], len(snp_df))
    return (gt[:n, :m], ind_df.iloc[:n].reset_index(drop=True),
            snp_df.iloc[:m].reset_index(drop=True))


def qc_filters(gt, ind_df, snp_df, p):
    gt, ind_df, snp_df = _align(gt, ind_df, snp_df)
    n0, m0 = gt.shape
    tr = {}
    k = miss_snp(gt) <= p["geno"]
    gt, snp_df = gt[:, k], snp_df[k].reset_index(drop=True)
    tr["geno"] = int(m0 - gt.shape[1])
    k = miss_ind(gt) <= p["mind"]
    gt, ind_df = gt[k], ind_df[k].reset_index(drop=True)
    tr["mind"] = int(n0 - gt.shape[0])
    if gt.shape[1] == 0 or gt.shape[0] == 0:
        raise ValueError("Tout exclu (missingness)")
    mb = gt.shape[1]
    m = maf(gt); k = np.isfinite(m) & (m >= p["maf"])
    gt, snp_df = gt[:, k], snp_df[k].reset_index(drop=True)
    tr["maf"] = int(mb - gt.shape[1])
    if gt.shape[1] == 0: raise ValueError("Tout exclu (MAF)")
    mb = gt.shape[1]
    pv = hwe_pvals(gt); k = np.isnan(pv) | (pv >= 1e-6)
    gt, snp_df = gt[:, k], snp_df[k].reset_index(drop=True)
    if gt.shape[1] > 0:
        pv2 = hwe_pvals(gt); k = np.isnan(pv2) | (pv2 >= p["hwe"])
        gt, snp_df = gt[:, k], snp_df[k].reset_index(drop=True)
    tr["hwe"] = int(mb - gt.shape[1])
    if gt.shape[1] == 0: raise ValueError("Tout exclu (HWE)")
    # ⚠️ gt_snp = AVANT retrait HET outliers (pour sélection)
    gt_snp, ind_snp, snp_snp = gt.copy(), ind_df.copy(), snp_df.copy()
    # Retrait des outliers HET (uniquement pour structure)
    het = het_obs(gt)
    keep = np.ones(len(het), dtype=bool)
    for fid in ind_df["FID"].unique():
        m = (ind_df["FID"] == fid).values
        sub = het[m]
        if len(sub) < 5 or np.nanstd(sub) < 1e-9: continue
        z = (sub - np.nanmean(sub)) / np.nanstd(sub)
        idx = np.where(m)[0]
        keep[idx[np.abs(z) > p["het_sd"]]] = False
    nb = gt.shape[0]
    gt, ind_df = gt[keep], ind_df[keep].reset_index(drop=True)
    tr["het"] = int(nb - gt.shape[0])
    if gt.shape[0] == 0: raise ValueError("Tout exclu (HET)")
    return (gt, ind_df, snp_df,
            {"n_ind_init": n0, "n_snp_init": m0,
             "n_ind_final": gt.shape[0], "n_snp_final": gt.shape[1],
             "excluded_ind": n0 - gt.shape[0],
             "excluded_snp": m0 - gt.shape[1],
             "n_ind_snp": gt_snp.shape[0],
             "n_snp_snp": gt_snp.shape[1], "trace": tr},
            gt_snp, ind_snp, snp_snp)


# ============================================================
# POPULATION GENETICS
# ============================================================
def filter_autosomes(gt, snp_df):
    cs = snp_df["CHR"].astype(str).str.upper().str.replace("CHR", "")
    k = cs.isin([str(i) for i in range(1, 30)])
    return gt[:, k.values], snp_df[k].reset_index(drop=True)


def pca_calc(gt, nc=10):
    X = impute(gt); X = X - X.mean(0)
    nc = min(nc, X.shape[0] - 1, X.shape[1])
    p = PCA(nc).fit(X)
    return p.transform(X), p.explained_variance_ratio_ * 100


def mds_calc(gt, nc=5):
    X = impute(gt); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(nc, dissimilarity="precomputed",
                       random_state=42, n_init=1,
                       max_iter=300).fit_transform(D)


def king_matrix(gt):
    n = gt.shape[0]
    het = np.nansum(gt == 1, axis=1).astype(float)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            bh = np.nansum((gt[i] == 1) & (gt[j] == 1))
            d = het[i] + het[j]
            K[i, j] = K[j, i] = (d - 2 * bh) / d if d > 0 else 0
    np.fill_diagonal(K, 0.5)
    return K


# ============================================================
# FST & SIGNATURES
# ============================================================
def fst_per_snp(gt, labels):
    labels = np.asarray(labels, dtype=object)
    pops = np.unique(labels)
    if len(pops) < 2:
        return np.full(gt.shape[1], np.nan)
    out = np.full(gt.shape[1], np.nan)
    masks = {p: labels == p for p in pops}
    for j in range(gt.shape[1]):
        ps, ns = [], []
        for p in pops:
            v = gt[masks[p], j]; v = v[~np.isnan(v)]
            if len(v) < 3: continue
            ps.append(v.mean() / 2.0); ns.append(len(v))
        if len(ps) < 2: continue
        pa, na = np.array(ps), np.array(ns)
        pbar = np.average(pa, weights=na)
        hs = np.average(2 * pa * (1 - pa), weights=na)
        ht = 2 * pbar * (1 - pbar)
        if ht > 1e-9: out[j] = (ht - hs) / ht
    return out


def fst_group_vs_rest(gt, labels, group_pops):
    """FST entre groupe et toutes les autres races."""
    labels = np.asarray(labels, dtype=object)
    mg = np.isin(labels, list(group_pops))
    mr = ~mg
    if mg.sum() < 5 or mr.sum() < 5: return pd.DataFrame()
    p_g = np.nanmean(gt[mg], 0) / 2.0
    p_r = np.nanmean(gt[mr], 0) / 2.0
    n_g = (~np.isnan(gt[mg])).sum(0)
    n_r = (~np.isnan(gt[mr])).sum(0)
    denom = np.where(n_g + n_r > 0, n_g + n_r, np.nan)
    pbar = (p_g * n_g + p_r * n_r) / denom
    hs = (2 * p_g * (1 - p_g) * n_g + 2 * p_r * (1 - p_r) * n_r) / denom
    ht = 2 * pbar * (1 - pbar)
    with np.errstate(divide="ignore", invalid="ignore"):
        fst = (ht - hs) / ht
    fst = np.clip(np.where(np.isfinite(fst), fst, np.nan), 0, 1)
    return pd.DataFrame({"FST": fst, "P_group": p_g, "P_rest": p_r,
                         "n_group": n_g, "n_rest": n_r})


def fst_group_vs_group(gt, labels, pops_A, pops_B):
    """FST entre 2 groupes de races."""
    labels = np.asarray(labels, dtype=object)
    mA = np.isin(labels, list(pops_A))
    mB = np.isin(labels, list(pops_B))
    if mA.sum() < 5 or mB.sum() < 5: return pd.DataFrame()
    p_A = np.nanmean(gt[mA], 0) / 2.0
    p_B = np.nanmean(gt[mB], 0) / 2.0
    n_A = (~np.isnan(gt[mA])).sum(0)
    n_B = (~np.isnan(gt[mB])).sum(0)
    denom = np.where(n_A + n_B > 0, n_A + n_B, np.nan)
    pbar = (p_A * n_A + p_B * n_B) / denom
    hs = (2 * p_A * (1 - p_A) * n_A + 2 * p_B * (1 - p_B) * n_B) / denom
    ht = 2 * pbar * (1 - pbar)
    with np.errstate(divide="ignore", invalid="ignore"):
        fst = (ht - hs) / ht
    fst = np.clip(np.where(np.isfinite(fst), fst, np.nan), 0, 1)
    return pd.DataFrame({"FST": fst, "P_A": p_A, "P_B": p_B,
                         "n_A": n_A, "n_B": n_B})


def extract_outliers(fst_df, snp_df, quantile, min_fst):
    if fst_df.empty: return pd.DataFrame(), {}
    df = pd.concat([snp_df.reset_index(drop=True),
                    fst_df.reset_index(drop=True)], axis=1
                   ).dropna(subset=["FST"])
    if df.empty: return pd.DataFrame(), {}
    q = float(np.nanquantile(df["FST"], quantile))
    thr = max(q, min_fst)
    out = (df[df["FST"] >= thr]
           .sort_values("FST", ascending=False)
           .reset_index(drop=True))
    return out, {"n_snp_analyzed": len(df),
                 "n_snp_outliers": len(out),
                 "fst_mean": float(df["FST"].mean()),
                 "fst_median": float(df["FST"].median()),
                 "fst_max": float(df["FST"].max()),
                 "threshold": float(thr)}


def analyze_group_vs_rest(gt, snp_df, labels, group_name, pops,
                            quantile=0.99, min_fst=0.15):
    fst = fst_group_vs_rest(gt, labels, pops)
    if fst.empty: return {"status": "insufficient"}
    out, st_d = extract_outliers(fst, snp_df, quantile, min_fst)
    return {"status": "success", "group": group_name,
            "pops": list(pops), "all_fst_df":
            pd.concat([snp_df.reset_index(drop=True),
                       fst.reset_index(drop=True)], axis=1),
            "outliers_df": out, **st_d}


def analyze_group_pair(gt, snp_df, labels, groups,
                        gA, gB, quantile=0.99, min_fst=0.15):
    fst = fst_group_vs_group(gt, labels,
                              groups.get(gA, []), groups.get(gB, []))
    if fst.empty: return {"status": "insufficient"}
    out, st_d = extract_outliers(fst, snp_df, quantile, min_fst)
    return {"status": "success", "group_A": gA, "group_B": gB,
            "pops_A": groups.get(gA, []),
            "pops_B": groups.get(gB, []),
            "all_fst_df": pd.concat([snp_df.reset_index(drop=True),
                                     fst.reset_index(drop=True)], axis=1),
            "outliers_df": out, **st_d}


def group_phenotype(ind_df, groups, group_name):
    df = ind_df.copy()
    df["pheno"] = 0.0
    tgt = groups.get(group_name, [])
    df.loc[df["FID"].astype(str).isin(tgt), "pheno"] = 1.0
    return df


# ============================================================
# GTF ANNOTATION
# ============================================================
def load_gtf(path):
    genes = []
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.strip().split("\t")
            if len(p) < 9 or p[2] != "gene": continue
            chrom = p[0].replace("chr", "")
            gname = None
            for attr in p[8].split(";"):
                a = attr.strip()
                if a.startswith("gene_name"):
                    gname = a.split('"')[1]
            genes.append({"CHR": str(chrom),
                          "start": int(p[3]), "end": int(p[4]),
                          "gene_name": gname})
    return pd.DataFrame(genes)


def annotate_snps(outliers_df, genes_df, flank_kb=500):
    if outliers_df.empty or genes_df is None or genes_df.empty:
        return outliers_df.copy() if not outliers_df.empty else pd.DataFrame()
    rows = []
    flank_bp = flank_kb * 1000
    for _, snp in outliers_df.iterrows():
        chrom = str(snp["CHR"]).replace("CHR", "").strip()
        bp = int(snp["BP"])
        gc = genes_df[genes_df["CHR"] == chrom]
        if gc.empty:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None})
            continue
        cands = []
        for _, g in gc.iterrows():
            if g["start"] <= bp <= g["end"]:
                cands.append((g["gene_name"], 0, "intragenic"))
            else:
                d = min(abs(bp - g["start"]), abs(bp - g["end"]))
                if d <= flank_bp:
                    reg = "upstream" if bp < g["start"] else "downstream"
                    cands.append((g["gene_name"], d, reg))
        if cands:
            best = min(cands, key=lambda x: x[1])
            rows.append({**snp.to_dict(), "gene_name": best[0],
                         "distance": best[1], "region": best[2]})
        else:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None})
    return pd.DataFrame(rows)


# ============================================================
# GWAS 10 cM
# ============================================================
def gwas_local(gt, snp_df, ind_df, pheno_df, chr_, bp_,
                pheno_name, window_cm=10):
    wbp = window_cm * 1_000_000
    mask = ((snp_df["CHR"].astype(str) == str(chr_))
            & (snp_df["BP"] >= bp_ - wbp)
            & (snp_df["BP"] <= bp_ + wbp))
    idx = np.where(mask)[0]
    if len(idx) == 0: return pd.DataFrame()
    ind_df = ind_df.copy()
    ind_df["IID"] = ind_df["IID"].astype(str)
    ph = pheno_df.set_index("IID")
    common = [i for i in ind_df["IID"] if i in ph.index]
    if len(common) < 10: return pd.DataFrame()
    im = ind_df["IID"].isin(common).values
    gt_s = gt[im][:, idx]
    ind_s = ind_df[im].reset_index(drop=True)
    pv = []
    for iid in ind_s["IID"]:
        try:
            v = ph.loc[iid, pheno_name]
            if isinstance(v, pd.Series): v = v.iloc[0]
            pv.append(float(v))
        except Exception:
            pv.append(np.nan)
    pv = np.array(pv)
    pops = ind_s["FID"].astype(str).values
    upops = sorted(set(pops))
    dum = np.zeros((len(pops), max(0, len(upops) - 1)))
    for i, p in enumerate(pops):
        for j, u in enumerate(upops[1:]):
            if p == u: dum[i, j] = 1
    results = []
    snp_w = snp_df.iloc[idx]
    for j in range(len(idx)):
        g = gt_s[:, j]
        v = ~np.isnan(g) & ~np.isnan(pv)
        if v.sum() < 10: continue
        gv, pvv = g[v], pv[v]
        X = np.column_stack([np.ones(v.sum()), gv, dum[v]]) \
            if dum.shape[1] > 0 else np.column_stack([np.ones(v.sum()), gv])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, pvv, rcond=None)
            res = pvv - X @ beta
            ssr = np.sum(res ** 2)
            sst = np.sum((pvv - pvv.mean()) ** 2)
            r2 = 1 - ssr / sst if sst > 0 else 0
            df_res = v.sum() - X.shape[1]
            if df_res > 0:
                mse = ssr / df_res
                try:
                    cov = np.linalg.inv(X.T @ X)
                    se = np.sqrt(mse * cov[1, 1])
                    t = beta[1] / se
                    p_val = 2 * (1 - stats.t.cdf(abs(t), df_res))
                except np.linalg.LinAlgError:
                    p_val, t = np.nan, np.nan
            else:
                p_val, t = np.nan, np.nan
            results.append({"SNP": snp_w.iloc[j]["SNP"],
                            "CHR": snp_w.iloc[j]["CHR"],
                            "BP": int(snp_w.iloc[j]["BP"]),
                            "beta": float(beta[1]),
                            "t_stat": float(t) if np.isfinite(t) else np.nan,
                            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                            "r2": float(r2),
                            "n_obs": int(v.sum())})
        except Exception:
            continue
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res["p_bonf"] = np.minimum(1, df_res["p_value"] * len(df_res))
        df_res["neg_log10_p"] = -np.log10(df_res["p_value"].clip(lower=1e-300))
        df_res = df_res.sort_values("p_value")
    return df_res


# ============================================================
# EXPORT
# ============================================================
def _dos_bits(col):
    b = np.zeros(len(col), dtype=np.uint8)
    b[np.isnan(col)] = 0b01
    b[col == 0] = 0b11
    b[col == 1] = 0b10
    b[col == 2] = 0b00
    return b


def build_bed(gt):
    n_ind, n_snp = gt.shape
    n_b = (n_ind + 3) // 4
    buf = bytearray([0x6C, 0x1B, 0x01])
    for j in range(n_snp):
        bits = _dos_bits(gt[:, j])
        p = np.zeros(n_b, dtype=np.uint8)
        for k in range(n_ind):
            p[k // 4] |= (bits[k] & 3) << (2 * (k % 4))
        buf.extend(p.tobytes())
    return bytes(buf)


def build_zip(gt, ind_df, snp_df, prefix="bovine"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{prefix}.bed", build_bed(gt))
        bim = []
        for _, r in snp_df.iterrows():
            a1 = str(r.get("A1", "A")) if "A1" in r else "A"
            a2 = str(r.get("A2", "G")) if "A2" in r else "G"
            if a1 in ("nan", ""): a1 = "A"
            if a2 in ("nan", "", a1): a2 = "G" if a1 == "A" else "A"
            bim.append(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t"
                       f"{int(r['BP'])}\t{a1}\t{a2}")
        zf.writestr(f"{prefix}.bim", "\n".join(bim) + "\n")
        fam = "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9"
                        for _, r in ind_df.iterrows()) + "\n"
        zf.writestr(f"{prefix}.fam", fam)
    return bio.getvalue()


# ============================================================
# PLOTLY
# ============================================================
def plt_hist(vals, title, xlabel, color="#3498db"):
    fig = go.Figure(go.Histogram(x=vals, nbinsx=80, marker_color=color))
    fig.update_layout(title=title, xaxis_title=xlabel,
                      yaxis_title="Fréquence", height=380)
    return fig


def plt_pca(scores, vp, labels):
    df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1],
                       "Pop": labels})
    fig = px.scatter(df, x="PC1", y="PC2", color="Pop",
                     title=f"PCA ({vp[0]:.1f}% / {vp[1]:.1f}%)",
                     height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1,
                                                      color="white")))
    return fig


def plt_mds(coords, labels):
    df = pd.DataFrame({"MDS1": coords[:, 0], "MDS2": coords[:, 1],
                       "Pop": labels})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Pop",
                     title="MDS", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1,
                                                      color="white")))
    return fig


def plt_manhattan(fst, chrs, title="Manhattan — FST/SNP"):
    df = pd.DataFrame({"FST": np.asarray(fst),
                       "CHR": np.asarray(chrs)}).dropna()
    if df.empty: return None
    order = df["CHR"].map(_chr_key)
    df["_k0"] = order.map(lambda t: t[0])
    df["_k1"] = order.map(lambda t: t[1])
    df = df.sort_values(["_k0", "_k1"]).reset_index(drop=True)
    cum, ticks, labels = 0, [], []
    pos = np.zeros(len(df), dtype=int)
    for c in df["CHR"].unique():
        sub = np.where(df["CHR"] == c)[0]
        pos[sub] = np.arange(cum, cum + len(sub))
        ticks.append((cum + cum + len(sub) - 1) / 2)
        labels.append(str(c))
        cum += len(sub)
    df["x"] = pos
    fig = go.Figure()
    for i, c in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == c]
        color = "#2c3e50" if i % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"],
                                  mode="markers",
                                  marker=dict(size=5, color=color),
                                  showlegend=False, hoverinfo="skip"))
    q = float(np.nanquantile(fst, 0.99))
    fig.add_hline(y=q, line_dash="dash", line_color="red",
                  annotation_text=f"Top 1% = {q:.4f}")
    fig.update_layout(title=title, xaxis_title="Chromosome",
                      yaxis_title="FST", height=500)
    fig.update_xaxes(tickvals=ticks, ticktext=labels)
    return fig


def plt_gwas(df, chr_, bp_):
    if df.empty: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["BP"], y=df["neg_log10_p"], mode="markers",
        marker=dict(size=8, color=df["neg_log10_p"],
                    colorscale="Hot", showscale=False),
        hovertemplate="%{text}<br>BP: %{x}<br>-log10(p): %{y:.2f}"
                      "<extra></extra>", text=df["SNP"]))
    fig.add_hline(y=-np.log10(0.05), line_dash="dash",
                  line_color="orange", annotation_text="p=0.05")
    fig.add_hline(y=-np.log10(0.05 / max(len(df), 1)),
                  line_dash="dash", line_color="red",
                  annotation_text="Bonferroni")
    fig.add_vline(x=bp_, line_dash="dot", line_color="blue",
                  annotation_text="SNP cible")
    fig.update_layout(title=f"GWAS local — Chr {chr_}",
                      xaxis_title="Position (bp)",
                      yaxis_title="-log10(p)", height=450)
    return fig
    # ============================================================
# ÉTAT
# ============================================================
STATE_KEYS = [
    "gt", "ind_df", "snp_df",
    "gt_filt", "ind_filt", "snp_filt", "qc_stats",
    "gt_snp", "ind_snp", "snp_snp",
    "gt_pruned", "snp_pruned",
    "pca_scores", "pca_var", "mds_coords", "king_matrix",
    "roh_df", "froh", "ld_df",
    "group_results", "group_pair", "gwas_results",
    "_ax_ped", "_ax_map", "_ax_n", "_ax_samples",
    "_genes_df",
]


def init_state():
    for k in STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if "trait_groups" not in st.session_state:
        st.session_state["trait_groups"] = dict(DEFAULT_GROUPS)


def reset_derived():
    for k in ["gt_filt", "ind_filt", "snp_filt", "qc_stats",
              "gt_snp", "ind_snp", "snp_snp",
              "gt_pruned", "snp_pruned",
              "pca_scores", "pca_var", "mds_coords", "king_matrix",
              "roh_df", "froh", "ld_df",
              "group_results", "group_pair", "gwas_results"]:
        st.session_state[k] = None


def has_data(): return st.session_state.gt is not None
def has_qc():
    return (st.session_state.gt_filt is not None
            and st.session_state.qc_stats is not None)
def has_snp(): return st.session_state.gt_snp is not None
def has_pruned():
    return (st.session_state.gt_pruned is not None
            and st.session_state.snp_pruned is not None)


# ============================================================
# MAIN
# ============================================================
def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v7.0")
    st.caption("Pipeline compact + groupes + annotation + GWAS")

    # ---------- SIDEBAR ----------
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :",
                        ["Demo", "PED/MAP", "VCF", "Convertisseur Axiom"],
                        index=0, key="sb_mode")

        if mode == "Demo":
            c1, c2 = st.columns(2)
            ni = c1.number_input("Ind.", 20, 1000, 150, 10, key="sb_ni")
            ns = c2.number_input("SNPs", 100, 10000, 800, 100, key="sb_ns")
            np_ = st.slider("Populations", 2, 14, 4, key="sb_np")
            if st.button("🎲 Générer", use_container_width=True,
                         key="sb_gen"):
                with st.spinner("Génération..."):
                    gt, ind, snp = demo_data(int(ni), int(ns), int(np_))
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind
                    st.session_state.snp_df = snp
                    reset_derived()
                st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")

        elif mode == "PED/MAP":
            pf = st.file_uploader(".ped", type=["ped", "gz", "txt"],
                                    key="sb_ped")
            mf = st.file_uploader(".map", type=["map", "gz", "txt"],
                                    key="sb_map")
            if pf and mf and st.button("📥 Charger",
                                         use_container_width=True,
                                         key="sb_load_ped"):
                try:
                    map_df = parse_map(mf.read())
                    gt, ind = parse_ped(pf.read(), len(map_df))
                    if len(map_df) != gt.shape[1]:
                        map_df = map_df.iloc[:gt.shape[1]].reset_index(drop=True)
                    if ind["FID"].nunique() <= 1:
                        ind["FID"] = ind["IID"].astype(str).apply(
                            lambda x: x.split("_")[0])
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind
                    st.session_state.snp_df = map_df
                    reset_derived()
                    st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} SNPs")
                except Exception as e:
                    st.error(f"❌ {e}")

        elif mode == "VCF":
            vf = st.file_uploader(".vcf", type=["vcf", "gz", "txt"],
                                    key="sb_vcf")
            if vf and st.button("📥 Charger", use_container_width=True,
                                 key="sb_load_vcf"):
                try:
                    gt, ind, snp = parse_vcf(vf.read())
                    if ind["FID"].nunique() <= 1:
                        ind["FID"] = ind["IID"].astype(str).apply(
                            lambda x: x.split("_")[0])
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind
                    st.session_state.snp_df = snp
                    reset_derived()
                    st.success(f"✅ {gt.shape[0]} ind × {gt.shape[1]} variants")
                except Exception as e:
                    st.error(f"❌ {e}")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("--geno", 0.0, 0.5, DEFAULTS["geno"], 0.01,
                          key="sb_geno")
        mind = st.slider("--mind", 0.0, 0.5, DEFAULTS["mind"], 0.01,
                          key="sb_mind")
        maf_t = st.slider("MAF", 0.0, 0.5, DEFAULTS["maf"], 0.01,
                           key="sb_maf")
        hwe_t = st.number_input("HWE", value=DEFAULTS["hwe"],
                                 format="%.0e", key="sb_hwe")
        het_sd = st.slider("HET ±σ", 1.0, 5.0, DEFAULTS["het_sd"], 0.1,
                            key="sb_het")

        st.divider()
        st.header("🐄 Groupes de races")
        st.caption("Laitier / Viande / Cornu / Rustique")
        if has_data():
            all_pops = sorted(st.session_state.ind_df["FID"].astype(str)
                              .unique())
            with st.expander("Définir les groupes", expanded=False):
                for gname in ["Laitier", "Viande", "Cornu", "Rustique"]:
                    sel = st.multiselect(
                        f"Races → {gname}", all_pops,
                        default=st.session_state["trait_groups"].get(gname, []),
                        key=f"grp_{gname}")
                    st.session_state["trait_groups"][gname] = sel

        st.divider()
        st.caption("v7.0 — Groupes + GTF + GWAS 10cM")

    # ---------- CONVERTISSEUR AXIOM ----------
    if mode == "Convertisseur Axiom":
        render_axiom_page()
        return

    # ---------- ACCUEIL ----------
    if not has_data():
        st.info("👉 Générez un jeu de démo, importez PED/MAP/VCF, "
                "ou utilisez le Convertisseur Axiom.")
        st.markdown("""
        ### 🧬 Fonctionnalités v7.0

        | Onglet | Contenu |
        |---|---|
        | 🏠 Aperçu | Statistiques |
        | 🧹 QC | Missingness / MAF / HWE / HET |
        | 🧬 Structure | PCA + MDS + KING |
        | 🎯 **Groupes** | **Signatures par groupe + comparaison** |
        | 🧬 **Annotation** | **GTF + gènes amont/aval** |
        | 📊 **GWAS 10 cM** | **Association locale par groupe** |
        | 📤 Export | PLINK + VCF |

        ### 🆕 Nouveautés v7.0
        - ✅ Convertisseur Axiom intégré
        - ✅ Groupes de races (Laitier / Viande / Cornu / Rustique)
        - ✅ Signatures par groupe **groupe vs reste**
        - ✅ Comparaison **Groupe A vs Groupe B**
        - ✅ Annotation GTF (gènes amont/aval)
        - ✅ GWAS 10 cM avec phénotype groupe
        - ✅ Fix `detect_roh`
        """)
        return

    gt, ind_df, snp_df = (st.session_state.gt, st.session_state.ind_df,
                           st.session_state.snp_df)

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "🧬 Structure",
                    "🎯 Groupes", "🧬 Annotation",
                    "📊 GWAS 10 cM", "📤 Export"])

    # ---------- TAB 0 : Aperçu ----------
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        st.dataframe(ind_df.head(20), use_container_width=True)

    # ---------- TAB 1 : QC ----------
    with tabs[1]:
        st.subheader("🧹 Contrôle Qualité")
        if st.button("🧬 Filtrer autosomes 1-29",
                     use_container_width=True, key="qc_auto"):
            g2, s2 = filter_autosomes(gt, snp_df)
            st.session_state.gt = g2
            st.session_state.snp_df = s2
            st.success(f"✅ {g2.shape[1]} SNPs autosomiques")

        if st.button("▶ Lancer le QC complet", type="primary",
                     use_container_width=True, key="qc_run"):
            try:
                with st.spinner("QC..."):
                    p = {"geno": geno, "mind": mind, "maf": maf_t,
                         "hwe": hwe_t, "het_sd": het_sd}
                    (gtf, indf, snpf, st_d,
                     gts, inds, snps) = qc_filters(
                        st.session_state.gt,
                        st.session_state.ind_df,
                        st.session_state.snp_df, p)
                    for k, v in [("gt_filt", gtf), ("ind_filt", indf),
                                  ("snp_filt", snpf), ("qc_stats", st_d),
                                  ("gt_snp", gts), ("ind_snp", inds),
                                  ("snp_snp", snps)]:
                        st.session_state[k] = v
                st.success(
                    f"✅ QC terminé\n"
                    f"• gt_filt (structure) : {gtf.shape[0]} × {gtf.shape[1]:,}\n"
                    f"• gt_snp (sélection) : {gts.shape[0]} × {gts.shape[1]:,}")
            except Exception as e:
                st.error(f"❌ {e}")

        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind. finaux", s["n_ind_final"])
            c2.metric("SNPs finaux", s["n_snp_final"])
            c3.metric("Exclus (ind)", s["excluded_ind"])
            c4.metric("Exclus (SNPs)", s["excluded_snp"])
            st.info(f"ℹ️ **gt_snp** : {s['n_ind_snp']} ind × "
                    f"{s['n_snp_snp']:,} SNP (sélection)")
            full = st.session_state.gt
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Missingness / ind",
                                                "Missingness / SNP"))
            fig.add_trace(go.Histogram(x=miss_ind(full), nbinsx=60,
                                        marker_color="skyblue"),
                          row=1, col=1)
            fig.add_trace(go.Histogram(x=miss_snp(full), nbinsx=60,
                                        marker_color="coral"),
                          row=1, col=2)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ---------- TAB 2 : Structure ----------
    with tabs[2]:
        st.subheader("🧬 Structure (PCA + MDS + KING)")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gtf = st.session_state.gt_filt
            if st.button("▶ PCA + MDS + KING", type="primary",
                         use_container_width=True, key="struct_run"):
                try:
                    with st.spinner("PCA..."):
                        sc, vr = pca_calc(gtf, 10)
                        st.session_state.pca_scores = sc
                        st.session_state.pca_var = vr
                    with st.spinner("MDS..."):
                        st.session_state.mds_coords = mds_calc(gtf, 5)
                    with st.spinner("KING..."):
                        st.session_state.king_matrix = king_matrix(gtf)
                    st.success("✅ Terminé")
                except Exception as e:
                    st.error(f"❌ {e}")
            labels = st.session_state.ind_filt["FID"].values
            if st.session_state.pca_scores is not None:
                st.plotly_chart(plt_pca(st.session_state.pca_scores,
                                         st.session_state.pca_var, labels),
                                use_container_width=True)
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plt_mds(st.session_state.mds_coords,
                                         labels),
                                use_container_width=True)

    # ---------- TAB 3 : Groupes ----------
    with tabs[3]:
        st.subheader("🎯 Signatures de sélection par groupe")
        st.info("ℹ️ **Sélection** utilise **gt_snp** "
                "(sans retrait des outliers HET).")

        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            groups = st.session_state["trait_groups"]
            active = {k: v for k, v in groups.items() if v}

            if not active:
                st.warning("⚠️ Définissez les groupes dans la sidebar.")
            else:
                ind_use = (st.session_state.ind_snp
                           if has_snp() else st.session_state.ind_filt)
                gt_use = (st.session_state.gt_snp
                          if has_snp() else st.session_state.gt_filt)
                snp_use = (st.session_state.snp_snp
                           if has_snp() else st.session_state.snp_filt)
                labels = ind_use["FID"].astype(str).values

                st.dataframe(pd.DataFrame([
                    {"Groupe": k, "Races": ", ".join(v),
                     "N": int(pd.Series(labels).isin(v).sum())}
                    for k, v in active.items()]),
                    use_container_width=True)

                sub1, sub2 = st.tabs(["Groupe vs reste",
                                       "Groupe A vs B"])
                with sub1:
                    c1, c2 = st.columns(2)
                    q_g = c1.slider("Quantile", 0.90, 0.9999, 0.99,
                                     0.001, format="%.4f", key="q_g")
                    fst_g = c2.slider("FST min", 0.0, 0.5, 0.15,
                                       0.01, key="mf_g")
                    if st.button("▶ Analyser tous les groupes",
                                 type="primary",
                                 use_container_width=True, key="btn_grps"):
                        try:
                            res = {}
                            for name, pops in active.items():
                                with st.spinner(f"Analyse {name}..."):
                                    res[name] = analyze_group_vs_rest(
                                        gt_use, snp_use, labels,
                                        name, pops, q_g, fst_g)
                            st.session_state.group_results = res
                            st.success(f"✅ {len(res)} groupes")
                        except Exception as e:
                            st.error(f"❌ {e}")

                    if st.session_state.group_results:
                        res = st.session_state.group_results
                        summary = [{
                            "Groupe": k,
                            "Races": ", ".join(r["pops"]),
                            "Outliers": r["n_snp_outliers"],
                            "FST moyen": round(r["fst_mean"], 4),
                            "FST max": round(r["fst_max"], 4),
                            "Seuil": round(r["threshold"], 4)}
                            for k, r in res.items()
                            if r.get("status") == "success"]
                        if summary:
                            df_s = pd.DataFrame(summary)
                            st.dataframe(df_s, use_container_width=True)
                            fig = px.bar(df_s, x="Groupe",
                                         y="Outliers",
                                         color="FST moyen",
                                         color_continuous_scale="Reds",
                                         title="SNP outliers par groupe",
                                         height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            sel_g = st.selectbox("Voir :",
                                                  df_s["Groupe"].tolist(),
                                                  key="view_g")
                            r = res[sel_g]
                            if r.get("status") == "success":
                                fig = plt_manhattan(
                                    r["all_fst_df"]["FST"].values,
                                    r["all_fst_df"]["CHR"].values,
                                    title=f"Manhattan — {sel_g} vs reste")
                                if fig:
                                    st.plotly_chart(fig,
                                                     use_container_width=True)
                                st.dataframe(r["outliers_df"].head(100),
                                              use_container_width=True)
                                csv = (r["outliers_df"]
                                       .to_csv(index=False)
                                       .encode("utf-8"))
                                st.download_button(
                                    f"⬇ SNP {sel_g}", csv,
                                    f"signature_{sel_g}.csv", "text/csv",
                                    key=f"dl_{sel_g}")

                with sub2:
                    if len(active) < 2:
                        st.warning("⚠️ Définissez au moins 2 groupes.")
                    else:
                        c1, c2 = st.columns(2)
                        gA = c1.selectbox("Groupe A", list(active.keys()),
                                            key="gA")
                        gB = c2.selectbox("Groupe B",
                                            [g for g in active if g != gA],
                                            key="gB")
                        c1, c2 = st.columns(2)
                        q_p = c1.slider("Quantile", 0.90, 0.9999, 0.99,
                                         0.001, format="%.4f", key="q_p")
                        fst_p = c2.slider("FST min", 0.0, 0.5, 0.15,
                                           0.01, key="mf_p")
                        if st.button(f"▶ {gA} vs {gB}",
                                     type="primary",
                                     use_container_width=True, key="btn_pair"):
                            try:
                                with st.spinner("Calcul..."):
                                    st.session_state.group_pair = \
                                        analyze_group_pair(
                                            gt_use, snp_use, labels,
                                            active, gA, gB, q_p, fst_p)
                                st.success("✅ Terminé")
                            except Exception as e:
                                st.error(f"❌ {e}")
                        r = st.session_state.group_pair
                        if r and r.get("status") == "success":
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("SNPs", r["n_snp_analyzed"])
                            c2.metric("Outliers", r["n_snp_outliers"])
                            c3.metric("FST moyen", f"{r['fst_mean']:.4f}")
                            c4.metric("FST max", f"{r['fst_max']:.4f}")
                            fig = plt_manhattan(
                                r["all_fst_df"]["FST"].values,
                                r["all_fst_df"]["CHR"].values,
                                title=f"Manhattan — {r['group_A']} "
                                      f"vs {r['group_B']}")
                            if fig:
                                st.plotly_chart(fig,
                                                 use_container_width=True)
                            st.dataframe(r["outliers_df"].head(100),
                                          use_container_width=True)
                            csv = (r["outliers_df"].to_csv(index=False)
                                   .encode("utf-8"))
                            st.download_button("⬇ SNP outliers", csv,
                                                f"diff_{gA}_vs_{gB}.csv",
                                                "text/csv", key="dl_pair")

    # ---------- TAB 4 : Annotation ----------
    with tabs[4]:
        st.subheader("🧬 Annotation GTF")
        st.markdown("""
        **Téléchargez le GTF bovin** :
        [Ensembl ARS-UCD1.2](https://ftp.ensembl.org/pub/release-110/gtf/bos_taurus/Bos_taurus.ARS-UCD1.2.110.gtf.gz)
        """)
        gtf_file = st.file_uploader("GTF (.gtf / .gtf.gz)",
                                      type=["gtf", "gz"], key="gtf_file")
        flank = st.slider("Fenêtre flanquante (kb)", 10, 5000, 500, 10,
                           key="flank")
        if gtf_file and st.button("📥 Charger GTF",
                                    use_container_width=True, key="gtf_btn"):
            try:
                with st.spinner("Chargement..."):
                    tmp = "/tmp/ann.gtf"
                    if gtf_file.name.endswith(".gz"):
                        tmp += ".gz"
                    with open(tmp, "wb") as f:
                        f.write(gtf_file.getbuffer())
                    gdf = load_gtf(tmp)
                    st.session_state["_genes_df"] = gdf
                st.success(f"✅ {len(gdf):,} gènes chargés")
            except Exception as e:
                st.error(f"❌ {e}")

        if st.session_state.get("_genes_df") is not None:
            gdf = st.session_state["_genes_df"]
            st.info(f"Gènes en mémoire : **{len(gdf):,}**")

            results = st.session_state.group_results
            if results:
                st.divider()
                sel = st.selectbox("Groupe à annoter :",
                                    list(results.keys()), key="ann_grp")
                r = results[sel]
                if r.get("status") == "success":
                    if st.button("📋 Annoter les SNP outliers",
                                 type="primary",
                                 use_container_width=True, key="ann_btn"):
                        try:
                            with st.spinner("Annotation..."):
                                ann = annotate_snps(
                                    r["outliers_df"], gdf,
                                    flank_kb=int(flank))
                                st.session_state.group_results[sel][
                                    "annotated_df"] = ann
                            st.success(f"✅ {len(ann)} SNPs annotés")
                        except Exception as e:
                            st.error(f"❌ {e}")
                    if "annotated_df" in r:
                        ann = r["annotated_df"]
                        with_genes = ann.dropna(subset=["gene_name"])
                        c1, c2, c3 = st.columns(3)
                        c1.metric("SNPs totaux", len(ann))
                        c2.metric("SNPs avec gène", len(with_genes))
                        c3.metric("Gènes uniques",
                                  with_genes["gene_name"].nunique()
                                  if not with_genes.empty else 0)
                        cols = ["CHR", "SNP", "BP", "FST",
                                "gene_name", "distance", "region"]
                        cols = [c for c in cols if c in ann.columns]
                        st.dataframe(ann[cols].head(200),
                                      use_container_width=True)
                        csv = ann.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇ CSV annoté", csv,
                            f"annotation_{sel}.csv", "text/csv",
                            key="dl_ann")

    # ---------- TAB 5 : GWAS 10 cM ----------
    with tabs[5]:
        st.subheader("📊 GWAS 10 cM")
        st.info("ℹ️ Utilise **gt_snp** (sans retrait HET outliers).")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            groups = st.session_state["trait_groups"]
            active = {k: v for k, v in groups.items() if v}
            if not active:
                st.warning("⚠️ Définissez les groupes dans la sidebar.")
            else:
                ind_use = (st.session_state.ind_snp
                           if has_snp() else st.session_state.ind_filt)
                gt_use = (st.session_state.gt_snp
                          if has_snp() else st.session_state.gt_filt)
                snp_use = (st.session_state.snp_snp
                           if has_snp() else st.session_state.snp_filt)

                st.markdown("### 1. Phénotype")
                source = st.radio("Source :",
                                   ["Groupe de races", "Fichier CSV"],
                                   horizontal=True, key="gwas_src")
                if source == "Groupe de races":
                    gname = st.selectbox("Groupe = cas (phénotype 1) :",
                                          list(active.keys()),
                                          key="gwas_grp")
                    ph = group_phenotype(ind_use, active, gname).rename(
                        columns={"pheno": "pheno_group"})
                    ph = ph[["IID", "pheno_group"]]
                    n_cases = int(ph["pheno_group"].sum())
                    st.success(f"✅ Phénotype '{gname}' : {n_cases} cas "
                               f"/ {len(ph)} individus")
                    pheno_col = "pheno_group"
                else:
                    pf = st.file_uploader("CSV phénotypes (IID,col)",
                                            type=["csv"], key="gwas_csv")
                    if pf is None:
                        st.stop()
                    ph = pd.read_csv(pf)
                    if "IID" not in ph.columns:
                        st.error("❌ Colonne 'IID' manquante")
                        st.stop()
                    ph["IID"] = ph["IID"].astype(str)
                    num_cols = [c for c in ph.columns if c != "IID"
                                and pd.api.types.is_numeric_dtype(ph[c])]
                    if not num_cols:
                        st.error("❌ Aucun phénotype numérique")
                        st.stop()
                    pheno_col = st.selectbox("Phénotype :", num_cols,
                                              key="gwas_col")
                    st.success(f"✅ {len(ph)} lignes chargées")

                st.markdown("### 2. SNP cible")
                # Utiliser les outliers du groupe si dispo
                grp_res = st.session_state.group_results
                if grp_res:
                    avail = [k for k, v in grp_res.items()
                             if v.get("status") == "success"]
                    if avail:
                        sel_grp = st.selectbox("Depuis groupe :", avail,
                                                key="gwas_grp_sel")
                        cands = grp_res[sel_grp]["outliers_df"]
                        if not cands.empty:
                            top_n = st.slider("Top N", 1, 20, 5,
                                               key="gwas_topn")
                            top = cands.head(top_n)
                            st.dataframe(top[["CHR", "SNP", "BP", "FST"]],
                                          use_container_width=True)
                            win = st.slider("Fenêtre (cM ≈ Mb)", 1, 20, 10,
                                            key="gwas_win")
                            if st.button("▶ Lancer GWAS",
                                         type="primary",
                                         use_container_width=True,
                                         key="gwas_run"):
                                all_res = {}
                                prog = st.progress(0)
                                for i, (_, s) in enumerate(top.iterrows()):
                                    with st.spinner(f"GWAS {s['SNP']}..."):
                                        r = gwas_local(
                                            gt_use, snp_use, ind_use, ph,
                                            s["CHR"], s["BP"], pheno_col,
                                            window_cm=win)
                                    if not r.empty:
                                        all_res[s["SNP"]] = r
                                    prog.progress((i + 1) / len(top))
                                st.session_state.gwas_results = all_res
                                st.success(f"✅ {len(all_res)} GWAS")

                gres = st.session_state.gwas_results
                if gres:
                    st.divider()
                    summary = []
                    for snp_n, df in gres.items():
                        nb = int((df["p_bonf"] < 0.05).sum())
                        ns = int((df["p_value"] < 0.05).sum())
                        summary.append({
                            "SNP cible": snp_n,
                            "N testés": len(df),
                            "p < 0.05": ns,
                            "Bonferroni": nb,
                            "p min": round(df["p_value"].min(), 6)})
                    st.dataframe(pd.DataFrame(summary),
                                  use_container_width=True)
                    sel = st.selectbox("Voir :", list(gres.keys()),
                                        key="gwas_view")
                    df = gres[sel]
                    snp_row = snp_use[snp_use["SNP"] == sel]
                    if not snp_row.empty:
                        fig = plt_gwas(df, snp_row.iloc[0]["CHR"],
                                       int(snp_row.iloc[0]["BP"]))
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(df.head(50), use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ GWAS {sel}", csv,
                                        f"gwas_{sel}.csv", "text/csv",
                                        key="dl_gwas")

    # ---------- TAB 6 : Export ----------
    with tabs[6]:
        st.subheader("📤 Export PLINK")
        if not has_qc():
            st.warning("⚠️ Lancez d'abord le QC.")
        else:
            gt_e = (st.session_state.gt_pruned if has_pruned()
                    else st.session_state.gt_filt)
            ind_e = st.session_state.ind_filt
            snp_e = (st.session_state.snp_pruned if has_pruned()
                     else st.session_state.snp_filt)
            prefix = st.text_input("Préfixe", "bovine_qc", key="exp_pref")
            if st.button("📦 Générer ZIP PLINK",
                         use_container_width=True, key="exp_gen"):
                try:
                    with st.spinner("Génération..."):
                        data = build_zip(gt_e, ind_e, snp_e, prefix)
                    st.download_button("⬇ Télécharger", data,
                                        f"{prefix}.zip",
                                        "application/zip",
                                        key="exp_dl")
                except Exception as e:
                    st.error(f"❌ {e}")
            c1, c2 = st.columns(2)
            c1.metric("Individus", gt_e.shape[0])
            c2.metric("SNPs", gt_e.shape[1])


if __name__ == "__main__":
    main()

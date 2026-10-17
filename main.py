"""
🐄 Bovine SNP Platform v6.2 — Version finale et complète
Corrections : detect_roh (json.loads), Axiom, code complet
Fonctionnalités : QC · LD · Structure · Admixture · Démographie ·
                  Sélection · Phylogénie · Binômes (5 modes) ·
                  Signatures par population · Signatures par groupes ·
                  Groupe A vs B · Annotation GTF · GWAS 10 cM · Export
"""
import gzip, json, os, warnings, zipfile
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
from scipy.spatial.distance import squareform
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import MDS as SklearnMDS

warnings.filterwarnings("ignore", category=RuntimeWarning)

st.set_page_config(page_title="🐄 Bovine SNP v6.2", page_icon="🐄",
                    layout="wide", initial_sidebar_state="expanded")

DEFAULTS = {"geno": 0.05, "mind": 0.05, "maf": 0.05,
            "hwe": 1e-6, "het_sd": 3.0, "king": 0.354, "ld_r2": 0.2}
HWE_EXACT_MAX = 20_000
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
RACE_POOL = ["AND", "EBG", "ELN", "EZP", "FGN", "LJR", "NAR",
             "NBD", "NKA", "PRS", "PSH", "PTP", "SHO", "YBS"]


# ==================== CACHE ====================
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


def to_np_str(a):
    return np.asarray(list(a), dtype=object)


def impute(gt):
    gt2 = gt.astype(np.float32, copy=True)
    cm = np.nanmean(gt2, axis=0)
    cm = np.where(np.isnan(cm), 0.0, cm)
    nm = np.isnan(gt2)
    if nm.any():
        gt2[nm] = np.take(cm, np.where(nm)[1])
    return gt2


def _chr_key(c):
    s = str(c).upper().replace("CHR", "").replace("CHROMOSOME", "")
    if s.isdigit(): return (0, int(s), "")
    sp = {"X": 100, "Y": 101, "MT": 102, "M": 102}
    if s in sp: return (1, sp[s], "")
    return (2, 0, s)


def _auto_weights():
    L = np.array([ARS_LEN[c] for c in AUTOSOMES], dtype=float)
    return L / L.sum()


def step_header(n, t, q, w, a):
    with st.expander(f"📖 Étape {n} — {t}", expanded=False):
        st.markdown(f"**🎯 {q}**\n\n**❓ {w}**\n\n**✅ {a}**")


# ==================== CONVERTISSEUR AXIOM ====================
def _clean_allele(a):
    if a is None: return "0"
    a = str(a).strip()
    return "0" if a in ("", ".", "-", "NA", "nan", "None", "0", "00") else a


def sample_to_fid_iid(name, fid_parts=2):
    n = str(name).strip()
    for e in (".CEL", ".cel", ".TXT", ".txt", ".gz", ".GZ"):
        if n.endswith(e): n = n[:-len(e)]
    n = (n.replace("(", "_").replace(")", "")
         .replace("[", "_").replace("]", "")
         .replace(" ", "_").replace("/", "_"))
    while "__" in n: n = n.replace("__", "_")
    n = n.strip("_")
    p = n.split("_")
    return ("_".join(p[:fid_parts]) if len(p) >= fid_parts else p[0]), n


def parse_axiom(raw, fn=""):
    if fn.endswith(".gz") or (len(raw) >= 2 and raw[:2] == b"\x1f\x8b"):
        try: raw = gzip.decompress(raw)
        except Exception as e: raise ValueError(f".gz corrompu : {e}")
    text = raw.decode("utf-8", errors="replace")
    lines = [l.rstrip("\r\n") for l in text.splitlines() if l.strip()]
    if not lines: raise ValueError("Fichier vide")
    skip = 1 if (lines[0].startswith("#") or lines[0].startswith("Sample")
                 or "Sample ID" in lines[0][:50]) else 0
    samples, geno = [], []
    for line in lines[skip:]:
        p = line.split("\t")
        if len(p) < 2:
            p = line.split()
            if len(p) < 2: continue
        al = []
        for cell in [c.strip() for c in p[1:]]:
            t = cell.split()
            if len(t) == 2:
                al.append((_clean_allele(t[0]), _clean_allele(t[1])))
            elif len(t) == 1:
                s = t[0]
                if len(s) == 2 and all(ch in "ACGT0" for ch in s):
                    al.append((s[0], s[1]))
                elif "/" in s:
                    a1, a2 = s.split("/", 1)
                    al.append((_clean_allele(a1), _clean_allele(a2)))
                else:
                    al.append((_clean_allele(s), _clean_allele(s)))
            else:
                al.append(("0", "0"))
        samples.append(p[0].strip())
        geno.append(al)
    if not samples: raise ValueError("Aucun échantillon")
    c = Counter([len(r) for r in geno]).most_common(1)[0][0]
    for i, r in enumerate(geno):
        if len(r) > c: geno[i] = r[:c]
        elif len(r) < c: geno[i] = r + [("0", "0")] * (c - len(r))
    return samples, geno, c


def render_axiom_page():
    st.header("🧬 Convertisseur Axiom → PED/MAP")
    st.markdown("Transforme un fichier Axiom (`A/G`, `T T`) en "
                "**PED** + **MAP** PLINK.")
    st.divider()
    f = st.file_uploader("Fichier Axiom", type=["txt", "tsv", "csv",
                                                  "ped", "gz"], key="ax_f")
    c1, c2 = st.columns(2)
    fp = c1.number_input("Parties FID", 1, 5, 2, 1, key="ax_fp")
    sep = c2.selectbox("Séparateur", ["Tabulation", "Espace"], key="ax_s")
    if f and st.button("Convertir", type="primary",
                        use_container_width=True, key="ax_btn"):
        try:
            with st.spinner("Parsing..."):
                samples, geno, n_snp = parse_axiom(f.read(), f.name)
            map_t = "\n".join(f"1\tSNP_{i+1:06d}\t0\t{i+1}"
                              for i in range(n_snp)) + "\n"
            ped_l = []
            for i, s in enumerate(samples):
                fid, iid = sample_to_fid_iid(s, int(fp))
                cols = [fid, iid, "0", "0", "0", "-9"]
                for a1, a2 in geno[i]:
                    cols.extend([a1, a2])
                ped_l.append("\t".join(cols))
            ped_t = "\n".join(ped_l) + "\n"
            if sep == "Espace":
                ped_t = ped_t.replace("\t", " ")
                map_t = map_t.replace("\t", " ")
            st.session_state["_ax_ped"] = ped_t
            st.session_state["_ax_map"] = map_t
            st.session_state["_ax_n"] = n_snp
            st.session_state["_ax_samples"] = samples
            st.success(f"✅ {len(samples)} ind × {n_snp} SNPs")
        except Exception as e:
            st.error(f"❌ {e}")
    if st.session_state.get("_ax_ped"):
        st.subheader("Téléchargement")
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇ .ped",
                            st.session_state["_ax_ped"].encode("utf-8"),
                            "converted.ped", "text/plain",
                            use_container_width=True, key="dl_ax_p")
        c2.download_button("⬇ .map",
                            st.session_state["_ax_map"].encode("utf-8"),
                            "converted.map", "text/plain",
                            use_container_width=True, key="dl_ax_m")
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("converted.ped", st.session_state["_ax_ped"])
            zf.writestr("converted.map", st.session_state["_ax_map"])
        c3.download_button("⬇ .zip", bio.getvalue(), "converted.zip",
                            "application/zip", use_container_width=True,
                            key="dl_ax_z")


# ==================== PARSERS ====================
def _enc(b):
    if len(b) >= 2 and b[:2] == b"\x1f\x8b": return "gz"
    if len(b) >= 2 and b[:2] == b"\x6c\x1b": return "bin"
    return "txt"


def parse_map(b):
    if _enc(b) == "gz":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    elif _enc(b) == "bin":
        raise ValueError("❌ .map binaire PLINK")
    else:
        text = b.decode("utf-8", errors="replace")
    rows = []
    for line in text.splitlines():
        p = line.strip().split()
        if len(p) < 4: continue
        try: bp = int(float(p[3]))
        except (ValueError, TypeError): continue
        try: cm = float(p[2]) if p[2] not in ("0", ".", "NA") else 0.0
        except ValueError: cm = 0.0
        rows.append({"CHR": str(p[0]), "SNP": str(p[1]),
                     "CM": cm, "BP": bp})
    if not rows: raise ValueError("❌ .map vide")
    return pd.DataFrame(rows)


def parse_ped(b, n_map):
    if _enc(b) == "gz":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    elif _enc(b) == "bin":
        raise ValueError("❌ .ped binaire PLINK")
    else:
        text = b.decode("utf-8", errors="replace")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: raise ValueError("❌ .ped vide")
    if len(lines) >= 2:
        n0, n1 = len(lines[0].split()), len(lines[1].split())
        if (n0 < 7 and n1 >= 7) or (n1 > 0 and n0 < n1 * 0.5):
            lines = lines[1:]
            st.info("ℹ️ Header PED ignoré")
    nc = len(lines[0].split())
    if nc < 7: raise ValueError(f"❌ {nc} colonnes PED")
    n_ped = (nc - 6) // 2
    n = min(n_map, n_ped) if n_ped > 0 else n_map
    exp = 6 + 2 * n
    fids, iids, genos = [], [], []
    for line in lines:
        p = line.split()
        if len(p) < exp: continue
        fids.append(p[0]); iids.append(p[1])
        genos.append(p[6:6 + 2 * n])
    ni = len(iids)
    if ni == 0: raise ValueError("❌ Aucun individu")
    counts = [Counter() for _ in range(n)]
    for row in genos:
        for j in range(n):
            a1, a2 = row[2*j], row[2*j+1]
            if a1 != "0" and a2 != "0":
                counts[j][a1] += 1; counts[j][a2] += 1
    minor = [min(c, key=c.get) if len(c) >= 2 else None for c in counts]
    gt = np.full((ni, n), np.nan, dtype=np.float32)
    for i, row in enumerate(genos):
        for j in range(n):
            m = minor[j]
            if m is None: continue
            a1, a2 = row[2*j], row[2*j+1]
            if a1 == "0" or a2 == "0": continue
            gt[i, j] = (a1 == m) + (a2 == m)
    return gt, pd.DataFrame({"FID": fids, "IID": iids})


def parse_vcf(b):
    if len(b) >= 2 and b[:2] == b"\x1f\x8b":
        text = gzip.decompress(b).decode("utf-8", errors="replace")
    else:
        text = b.decode("utf-8", errors="replace")
    samples, rows = [], []
    for line in text.splitlines():
        if not line or line.startswith("##"): continue
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]; continue
        if line.startswith("#"): continue
        p = line.split("\t")
        if len(p) < 10 or "," in p[4]: continue
        try: pos = int(p[1])
        except ValueError: continue
        fmt = p[8].split(":")
        if "GT" not in fmt: continue
        gi = fmt.index("GT")
        gts = np.full(len(samples), np.nan, dtype=np.float32)
        for k, s in enumerate(p[9:]):
            f = s.split(":")
            if gi >= len(f): continue
            g = f[gi]; sep_ = "|" if "|" in g else "/"
            al = g.split(sep_)
            if any(a in (".", "") for a in al): continue
            try: gts[k] = sum(int(a) for a in al)
            except ValueError: continue
        rows.append({"CHR": str(p[0]),
                     "SNP": str(p[2] or f"{p[0]}:{pos}"),
                     "CM": 0.0, "BP": pos, "A1": p[3], "A2": p[4],
                     "GT": gts})
    if not rows: raise ValueError("❌ Aucun variant")
    gt = np.zeros((len(samples), len(rows)), dtype=np.float32)
    for j, r in enumerate(rows): gt[:, j] = r["GT"]
    for j in range(len(rows)):
        col = gt[:, j]; v = col[~np.isnan(col)]
        if len(v) and v.mean() / 2.0 > 0.5:
            gt[:, j] = np.where(np.isnan(col), np.nan, 2.0 - col)
    snp_df = pd.DataFrame({
        "CHR": [r["CHR"] for r in rows],
        "SNP": [r["SNP"] for r in rows],
        "CM": 0.0, "BP": [r["BP"] for r in rows],
        "A1": [r["A1"] for r in rows], "A2": [r["A2"] for r in rows]})
    return gt, pd.DataFrame({"FID": samples, "IID": samples}), snp_df


def demo_data(ni=150, ns=800, np_=4, seed=42):
    rng = np.random.default_rng(seed)
    pops = RACE_POOL[:min(np_, len(RACE_POOL))]
    per, rem = ni // len(pops), ni % len(pops)
    cnt = [per + (1 if i < rem else 0) for i in range(len(pops))]
    p_anc = rng.beta(0.6, 0.6, ns)
    gt = np.zeros((sum(cnt), ns), dtype=np.float32)
    rows = []; k = 0
    for pop, n_i in zip(pops, cnt):
        pk = np.clip(p_anc + rng.normal(0, 0.15, ns), 0.02, 0.98)
        for i in range(n_i):
            a1 = (rng.random(ns) < pk).astype(np.float32)
            a2 = (rng.random(ns) < pk).astype(np.float32)
            gt[k] = a1 + a2
            rows.append({"FID": pop, "IID": f"{pop}_{i+1:03d}"})
            k += 1
    gt[rng.random(gt.shape) < 0.02] = np.nan
    chrs = rng.choice(AUTOSOMES, ns, p=_auto_weights())
    bps = np.array([rng.integers(1, ARS_LEN[c]) for c in chrs],
                   dtype=np.int64)
    snp_df = pd.DataFrame({"CHR": chrs,
                           "SNP": [f"rs{i:07d}" for i in range(ns)],
                           "CM": 0.0, "BP": bps})
    snp_df["_k"] = snp_df["CHR"].map(_chr_key)
    snp_df = (snp_df.sort_values(["_k", "BP"])
              .drop(columns="_k").reset_index(drop=True))
    return gt, pd.DataFrame(rows), snp_df
  # ==================== QC ====================
def miss_ind(g): return np.isnan(g).mean(axis=1)
def miss_snp(g): return np.isnan(g).mean(axis=0)
def af(g): return np.nanmean(g, axis=0) / 2.0
def maf(g): return np.minimum(af(g), 1 - af(g))
def het(g):
    with np.errstate(invalid="ignore"):
        return np.nanmean(g == 1, axis=1)


def _hwe_exact(n1, n0, n2):
    n = n1 + n0 + n2
    if n == 0: return np.nan
    if n1 == 0 and (n0 == 0 or n2 == 0): return 1.0
    rare = 2 * min(n0, n2) + n1
    mid = (rare * (2 * n - rare)) // (2 * n)
    if mid % 2 != rare % 2: mid += 1
    p = np.zeros(rare + 1); p[mid] = 1.0; s = 1.0
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch <= rare - 2:
        p[ch + 2] = p[ch] * 4 * chr_ * chc / ((ch + 2) * (ch + 1))
        s += p[ch + 2]; ch += 2; chr_ -= 1; chc -= 1
    ch, chr_, chc = mid, (rare - mid) // 2, n - mid - (rare - mid) // 2
    while ch >= 2:
        p[ch - 2] = p[ch] * ch * (ch - 1) / (4 * (chr_ + 1) * (chc + 1))
        s += p[ch - 2]; ch -= 2; chr_ += 1; chc += 1
    po = p[n1] if n1 < len(p) else 0
    return float(min(p[p <= po + 1e-7].sum() / s, 1.0))


@cached
def hwe_pv(g):
    _, m = g.shape
    pv = np.full(m, np.nan)
    nm = np.isnan(g)
    n0 = ((g == 0) & ~nm).sum(0)
    n1 = ((g == 1) & ~nm).sum(0)
    n2 = ((g == 2) & ~nm).sum(0)
    nv = n0 + n1 + n2
    if m <= HWE_EXACT_MAX:
        for j in range(m):
            if nv[j] < 5: continue
            pv[j] = _hwe_exact(int(n1[j]), int(n0[j]), int(n2[j]))
    else:
        p = (n1 + 2*n2) / (2.0 * np.where(nv > 0, nv, np.nan))
        with np.errstate(invalid="ignore", divide="ignore"):
            eh = 2*p*(1-p)*nv
            v = (eh >= 5) & np.isfinite(eh)
            chi2 = (np.abs(n1-eh) - 0.5)**2 / eh
            pv[v] = 1 - stats.chi2.cdf(chi2[v], df=1)
    return pv


def _align(g, i, s):
    n = min(g.shape[0], len(i)); m = min(g.shape[1], len(s))
    return (g[:n, :m], i.iloc[:n].reset_index(drop=True),
            s.iloc[:m].reset_index(drop=True))


def qc_filters(g, i, s, p):
    g, i, s = _align(g, i, s)
    n0, m0 = g.shape
    tr = {}
    k = miss_snp(g) <= p["geno"]
    g, s = g[:, k], s[k].reset_index(drop=True)
    tr["geno"] = int(m0 - g.shape[1])
    k = miss_ind(g) <= p["mind"]
    g, i = g[k], i[k].reset_index(drop=True)
    tr["mind"] = int(n0 - g.shape[0])
    if g.shape[1] == 0 or g.shape[0] == 0:
        raise ValueError("Tout exclu (missingness)")
    mb = g.shape[1]; m = maf(g)
    k = np.isfinite(m) & (m >= p["maf"])
    g, s = g[:, k], s[k].reset_index(drop=True)
    tr["maf"] = int(mb - g.shape[1])
    if g.shape[1] == 0: raise ValueError("Tout exclu MAF")
    mb = g.shape[1]
    pv = hwe_pv(g); k = np.isnan(pv) | (pv >= 1e-6)
    g, s = g[:, k], s[k].reset_index(drop=True)
    if g.shape[1] > 0:
        pv2 = hwe_pv(g); k = np.isnan(pv2) | (pv2 >= p["hwe"])
        g, s = g[:, k], s[k].reset_index(drop=True)
    tr["hwe"] = int(mb - g.shape[1])
    if g.shape[1] == 0: raise ValueError("Tout exclu HWE")
    gt_snp, ind_snp, snp_snp = g.copy(), i.copy(), s.copy()
    h = het(g); keep = np.ones(len(h), dtype=bool)
    for fid in i["FID"].unique():
        mk = (i["FID"] == fid).values
        sub = h[mk]
        if len(sub) < 5 or np.nanstd(sub) < 1e-9: continue
        z = (sub - np.nanmean(sub)) / np.nanstd(sub)
        idx = np.where(mk)[0]
        keep[idx[np.abs(z) > p["het_sd"]]] = False
    nb = g.shape[0]
    g, i = g[keep], i[keep].reset_index(drop=True)
    tr["het"] = int(nb - g.shape[0])
    if g.shape[0] == 0: raise ValueError("Tout exclu HET")
    return (g, i, s,
            {"n_ind_init": n0, "n_snp_init": m0,
             "n_ind_final": g.shape[0], "n_snp_final": g.shape[1],
             "excluded_ind": n0 - g.shape[0],
             "excluded_snp": m0 - g.shape[1],
             "n_ind_snp": gt_snp.shape[0], "n_snp_snp": gt_snp.shape[1],
             "trace": tr},
            gt_snp, ind_snp, snp_snp)


def filter_auto(g, s):
    cs = s["CHR"].astype(str).str.upper().str.replace("CHR", "")
    k = cs.isin([str(i) for i in range(1, 30)])
    return g[:, k.values], s[k].reset_index(drop=True)


@cached
def ld_pruning(g, w=50, st=5, r2t=0.2):
    n = g.shape[1]
    if n < 3: return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    X = impute(g); X = X - X.mean(0)
    i = 0
    while i < n:
        we = min(i + st, n)
        while we < n and (we - i) < w:
            j = we
            if keep[j] and keep[i]:
                a, b = X[:, i], X[:, j]
                if a.std() > 1e-8 and b.std() > 1e-8:
                    r = np.corrcoef(a, b)[0, 1]
                    if r * r > r2t: keep[j] = False
            we += 1
        i += st
    return keep


# ==================== STRUCTURE ====================
@cached
def pca_calc(g, nc=10):
    X = impute(g); X = X - X.mean(0)
    nc = min(nc, X.shape[0] - 1, X.shape[1])
    p = PCA(nc).fit(X)
    return p.transform(X), p.explained_variance_ratio_ * 100


@cached
def mds_calc(g, nc=5):
    X = impute(g); n = X.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        D[i] = np.abs(X - X[i]).sum(1) / X.shape[1]
    return SklearnMDS(nc, dissimilarity="precomputed", random_state=42,
                      n_init=1, max_iter=300).fit_transform(D)


@cached
def king_matrix(g):
    n = g.shape[0]
    h = np.nansum(g == 1, axis=1).astype(float)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            bh = np.nansum((g[i] == 1) & (g[j] == 1))
            d = h[i] + h[j]
            K[i, j] = K[j, i] = (d - 2*bh) / d if d > 0 else 0
    np.fill_diagonal(K, 0.5)
    return K


@cached
def admix_nmf(g, K=3, seed=42, mi=500):
    X = np.clip(impute(g), 0, 2)
    m = NMF(n_components=K, init="nndsvda", random_state=seed, max_iter=mi)
    W = m.fit_transform(X)
    s = W.sum(1, keepdims=True); s[s == 0] = 1
    return W / s, m.components_


@cached
def admix_cv(g, Kr=(2, 6), n_rep=3, ho=0.05, seed=42):
    rng = np.random.default_rng(seed)
    X = np.clip(impute(g), 0, 2)
    n, m = X.shape
    mask = rng.random((n, m)) < ho
    Xt = X.copy(); Xt[mask] = 0
    res = {}
    for K in range(Kr[0], Kr[1]):
        errs = []
        for rep in range(n_rep):
            mo = NMF(n_components=K, init="nndsvda",
                     random_state=seed + rep, max_iter=300)
            W = mo.fit_transform(Xt); H = mo.components_
            errs.append(np.mean((X[mask] - (W @ H)[mask]) ** 2))
        res[K] = {"mean": float(np.mean(errs)), "std": float(np.std(errs))}
    return res


# ==================== LD DECAY / ROH / Ne ====================
@cached
def ld_decay(g, bp, mk=1000, ms=1500, seed=42):
    n = g.shape[1]
    if n < 2: return pd.DataFrame(columns=["dist_kb", "r2"])
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n, ms, replace=False))
           if n > ms else np.arange(n))
    gs = impute(g[:, idx])
    bp_s = np.asarray(bp)[idx].astype(np.float64)
    X = gs - gs.mean(0)
    sd = gs.std(0); sd[sd < 1e-8] = np.nan
    X = X / sd
    C = (X.T @ X) / X.shape[0]; R2 = C ** 2
    iu, ju = np.triu_indices(len(idx), k=1)
    dkb = (bp_s[ju] - bp_s[iu]) / 1000
    m = (dkb > 0) & (dkb <= mk)
    return pd.DataFrame({"dist_kb": dkb[m], "r2": R2[iu[m], ju[m]]})


@cached
def detect_roh(g, snp_json, min_s=30, min_kb=500):
    """FIX : json.loads au lieu de pd.read_json"""
    s = pd.DataFrame(json.loads(snp_json))
    chr_a = s["CHR"].astype(str).values
    bp = s["BP"].astype(np.int64).values
    n = g.shape[0]; rohs, fbp = [], np.zeros(n)
    tot = 0.0
    for ch in pd.unique(chr_a):
        ix = np.where(chr_a == ch)[0]
        if len(ix) < min_s: continue
        ix = ix[np.argsort(bp[ix])]
        span = bp[ix[-1]] - bp[ix[0]]
        if span < 1: continue
        tot += span
        for i in range(n):
            col = g[i, ix]; sk = None
            for k in range(len(ix)):
                h = (col[k] == 0) or (col[k] == 2)
                if h:
                    if sk is None: sk = k
                else:
                    if sk is not None:
                        nr = k - sk
                        lkb = (bp[ix[k-1]] - bp[ix[sk]]) / 1000
                        if nr >= min_s and lkb >= min_kb:
                            rohs.append({"IID_idx": i, "CHR": str(ch),
                                         "start_bp": int(bp[ix[sk]]),
                                         "end_bp": int(bp[ix[k-1]]),
                                         "n_snp": int(nr),
                                         "length_kb": float(lkb)})
                            fbp[i] += bp[ix[k-1]] - bp[ix[sk]]
                        sk = None
            if sk is not None:
                nr = len(ix) - sk
                lkb = (bp[ix[-1]] - bp[ix[sk]]) / 1000
                if nr >= min_s and lkb >= min_kb:
                    rohs.append({"IID_idx": i, "CHR": str(ch),
                                 "start_bp": int(bp[ix[sk]]),
                                 "end_bp": int(bp[ix[-1]]),
                                 "n_snp": int(nr),
                                 "length_kb": float(lkb)})
                    fbp[i] += bp[ix[-1]] - bp[ix[sk]]
    froh = fbp / tot if tot > 0 else fbp
    roh = (pd.DataFrame(rohs) if rohs
           else pd.DataFrame(columns=["IID_idx", "CHR", "start_bp",
                                       "end_bp", "n_snp", "length_kb"]))
    return roh, froh


@cached
def estimate_ne(g, s, ms=2000):
    bp = s["BP"].astype(np.float64).values
    ch = s["CHR"].astype(str).values
    n = g.shape[1]
    if n > ms:
        idx = np.linspace(0, n - 1, ms).astype(int)
        g, bp, ch = g[:, idx], bp[idx], ch[idx]
    X = impute(g); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    r2, d = [], []
    m = X.shape[1]
    for i in range(m - 1):
        for j in range(i + 1, min(i + 50, m)):
            if ch[i] != ch[j]: continue
            dd = abs(bp[j] - bp[i]) / 1e6
            if dd < 0.001 or dd > 50: continue
            rr = np.corrcoef(X[:, i], X[:, j])[0, 1]
            r2.append(rr * rr); d.append(dd)
    if not d: return pd.DataFrame({"GenAgo": [], "Ne": []})
    r2, d = np.array(r2), np.array(d)
    bins = np.logspace(np.log10(d.min() + 1e-6), np.log10(d.max()), 20)
    bc = (bins[:-1] + bins[1:]) / 2
    rb = np.array([r2[(d >= bins[k]) & (d < bins[k+1])].mean()
                   if ((d >= bins[k]) & (d < bins[k+1])).sum() > 0
                   else np.nan for k in range(len(bins)-1)])
    m_ = np.isfinite(rb)
    if m_.sum() < 3: return pd.DataFrame({"GenAgo": [], "Ne": []})
    c = bc[m_] * 0.01; r = rb[m_]
    ne = np.where(r > 1e-4, 1 / (4*c*r), np.nan)
    df = pd.DataFrame({"GenAgo": bc[m_] * 100, "Ne": ne}).dropna()
    df = df[df["Ne"] > 0].sort_values("GenAgo")
    if len(df) > 3:
        df["Ne"] = df["Ne"].rolling(3, center=True, min_periods=1).mean()
    return df


# ==================== FST ====================
@cached
def fst_per_snp(g, labels):
    labels = np.asarray(labels, dtype=object)
    pops = np.unique(labels)
    if len(pops) < 2: return np.full(g.shape[1], np.nan)
    out = np.full(g.shape[1], np.nan)
    mk = {p: labels == p for p in pops}
    for j in range(g.shape[1]):
        ps, ns = [], []
        for p in pops:
            v = g[mk[p], j]; v = v[~np.isnan(v)]
            if len(v) < 3: continue
            ps.append(v.mean()/2); ns.append(len(v))
        if len(ps) < 2: continue
        pa, na = np.array(ps), np.array(ns)
        pb = np.average(pa, weights=na)
        hs = np.average(2*pa*(1-pa), weights=na)
        ht = 2*pb*(1-pb)
        if ht > 1e-9: out[j] = (ht - hs) / ht
    return out


@cached
def fst_pair(g, labels):
    labels = np.asarray(labels, dtype=object)
    pops = sorted(np.unique(labels))
    K = len(pops); M = np.full((K, K), np.nan); np.fill_diagonal(M, 0)
    mk = {p: labels == p for p in pops}
    for i in range(K):
        for j in range(i + 1, K):
            g1, g2 = g[mk[pops[i]]], g[mk[pops[j]]]
            p1, p2 = np.nanmean(g1, 0)/2, np.nanmean(g2, 0)/2
            n1, n2 = (~np.isnan(g1)).sum(0), (~np.isnan(g2)).sum(0)
            dn = np.where(n1+n2 > 0, n1+n2, np.nan)
            pb = (p1*n1 + p2*n2)/dn
            hs = (2*p1*(1-p1)*n1 + 2*p2*(1-p2)*n2)/dn
            ht = 2*pb*(1-pb)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = (ht-hs)/ht
            M[i, j] = M[j, i] = float(np.nanmean(f))
    return M, pops


@cached
def fst_grp_vs_rest(g, labels, gp):
    labels = np.asarray(labels, dtype=object)
    mg = np.isin(labels, list(gp)); mr = ~mg
    if mg.sum() < 5 or mr.sum() < 5: return pd.DataFrame()
    p_g, p_r = np.nanmean(g[mg], 0)/2, np.nanmean(g[mr], 0)/2
    n_g, n_r = (~np.isnan(g[mg])).sum(0), (~np.isnan(g[mr])).sum(0)
    dn = np.where(n_g+n_r > 0, n_g+n_r, np.nan)
    pb = (p_g*n_g + p_r*n_r)/dn
    hs = (2*p_g*(1-p_g)*n_g + 2*p_r*(1-p_r)*n_r)/dn
    ht = 2*pb*(1-pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ht-hs)/ht
    f = np.clip(np.where(np.isfinite(f), f, np.nan), 0, 1)
    return pd.DataFrame({"FST": f, "P_group": p_g, "P_rest": p_r,
                         "n_group": n_g, "n_rest": n_r})


@cached
def fst_grp_vs_grp(g, labels, pa, pb_):
    labels = np.asarray(labels, dtype=object)
    ma, mb = np.isin(labels, list(pa)), np.isin(labels, list(pb_))
    if ma.sum() < 5 or mb.sum() < 5: return pd.DataFrame()
    p_A, p_B = np.nanmean(g[ma], 0)/2, np.nanmean(g[mb], 0)/2
    n_A, n_B = (~np.isnan(g[ma])).sum(0), (~np.isnan(g[mb])).sum(0)
    dn = np.where(n_A+n_B > 0, n_A+n_B, np.nan)
    p_bar = (p_A*n_A + p_B*n_B)/dn
    hs = (2*p_A*(1-p_A)*n_A + 2*p_B*(1-p_B)*n_B)/dn
    ht = 2*p_bar*(1-p_bar)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ht-hs)/ht
    f = np.clip(np.where(np.isfinite(f), f, np.nan), 0, 1)
    return pd.DataFrame({"FST": f, "P_A": p_A, "P_B": p_B,
                         "n_A": n_A, "n_B": n_B})


def _extract_out(fst_df, snp_df, q, mf):
    if fst_df.empty: return pd.DataFrame(), {}
    df = pd.concat([snp_df.reset_index(drop=True),
                    fst_df.reset_index(drop=True)], axis=1
                   ).dropna(subset=["FST"])
    if df.empty: return pd.DataFrame(), {}
    thr = max(float(np.nanquantile(df["FST"], q)), mf)
    out = (df[df["FST"] >= thr]
           .sort_values("FST", ascending=False)
           .reset_index(drop=True))
    return out, {"n_snp_analyzed": len(df), "n_snp_outliers": len(out),
                 "fst_mean": float(df["FST"].mean()),
                 "fst_median": float(df["FST"].median()),
                 "fst_max": float(df["FST"].max()),
                 "threshold": float(thr)}


def analyze_pop_vs_rest(g, s, labels, q=0.99, mf=0.15):
    res = {}
    for pop in sorted(np.unique(labels)):
        if (labels == pop).sum() < 3: continue
        f = fst_grp_vs_rest(g, labels, [pop])
        if f.empty: continue
        out, st_ = _extract_out(f, s, q, mf)
        res[pop] = {"pop": pop, "status": "success", "pops": [pop],
                    "all_fst_df": pd.concat(
                        [s.reset_index(drop=True),
                         f.reset_index(drop=True)], axis=1),
                    "outliers_df": out, **st_}
    return res


def analyze_grp_vs_rest(g, s, labels, groups, q=0.99, mf=0.15):
    res = {}
    for name, pops in groups.items():
        if not pops: continue
        f = fst_grp_vs_rest(g, labels, pops)
        if f.empty:
            res[name] = {"status": "insufficient", "group": name}
            continue
        out, st_ = _extract_out(f, s, q, mf)
        res[name] = {"group": name, "pops": list(pops),
                     "status": "success",
                     "all_fst_df": pd.concat(
                         [s.reset_index(drop=True),
                          f.reset_index(drop=True)], axis=1),
                     "outliers_df": out, **st_}
    return res


def analyze_grp_pair(g, s, labels, groups, gA, gB, q=0.99, mf=0.15):
    f = fst_grp_vs_grp(g, labels, groups.get(gA, []),
                        groups.get(gB, []))
    if f.empty: return {"status": "insufficient"}
    out, st_ = _extract_out(f, s, q, mf)
    return {"status": "success", "group_A": gA, "group_B": gB,
            "pops_A": groups.get(gA, []), "pops_B": groups.get(gB, []),
            "all_fst_df": pd.concat(
                [s.reset_index(drop=True),
                 f.reset_index(drop=True)], axis=1),
            "outliers_df": out, **st_}


@cached
def analyze_pair(g, s, labels, a, b, q=0.99, mf=0.15):
    labels = np.asarray(labels, dtype=object)
    mA, mB = labels == a, labels == b
    if mA.sum() < 3 or mB.sum() < 3:
        return {"status": "insufficient", "pop_A": a, "pop_B": b}
    gA, gB = g[mA], g[mB]
    pA = np.nanmean(gA, 0)/2; pB = np.nanmean(gB, 0)/2
    nA = (~np.isnan(gA)).sum(0); nB = (~np.isnan(gB)).sum(0)
    dn = np.where(nA+nB > 0, nA+nB, np.nan)
    pb = (pA*nA + pB*nB)/dn
    hs = (2*pA*(1-pA) + 2*pB*(1-pB))/2
    ht = 2*pb*(1-pb)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (ht-hs)/ht
    f = np.clip(np.where(np.isfinite(f), f, np.nan), 0, 1)
    fst_df = pd.DataFrame({"FST": f, "P_A": pA, "P_B": pB,
                           "P_bar": pb, "n_A": nA, "n_B": nB})
    out, st_ = _extract_out(fst_df, s, q, mf)
    return {"status": "success", "pop_A": a, "pop_B": b,
            "n_ind_A": int(mA.sum()), "n_ind_B": int(mB.sum()),
            "outliers_df": out,
            "all_fst_df": pd.concat(
                [s.reset_index(drop=True),
                 fst_df.reset_index(drop=True)], axis=1), **st_}


def grp_phenotype(ind_df, groups, gname):
    df = ind_df.copy()
    df["pheno"] = 0.0
    df.loc[df["FID"].astype(str).isin(groups.get(gname, [])),
           "pheno"] = 1.0
    return df


def auto_pairs_mds(mds, labels, qlo=0.15, qhi=0.85):
    import itertools
    mds = np.asarray(mds); labels = np.asarray(labels, dtype=object)
    pops = sorted(np.unique(labels))
    cents, cnts = {}, {}
    for p in pops:
        m = labels == p
        if m.sum() < 3: continue
        cents[p] = mds[m, :2].mean(0); cnts[p] = int(m.sum())
    if len(cents) < 2: return pd.DataFrame()
    pairs = [{"pop_A": a, "pop_B": b,
              "distance_MDS": float(np.linalg.norm(cents[a]-cents[b])),
              "n_ind_A": cnts[a], "n_ind_B": cnts[b]}
             for a, b in itertools.combinations(cents.keys(), 2)]
    df = pd.DataFrame(pairs).sort_values("distance_MDS").reset_index(drop=True)
    lo = df["distance_MDS"].quantile(qlo)
    hi = df["distance_MDS"].quantile(qhi)
    def cat(d):
        return "🟢 Proche" if d <= lo else ("🔴 Éloignée" if d >= hi else "🟡 Moyenne")
    df["catégorie"] = df["distance_MDS"].apply(cat)
    return df


# ==================== GTF + ANNOTATION ====================
def load_gtf(path):
    genes = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.strip().split("\t")
            if len(p) < 9 or p[2] != "gene": continue
            ch = p[0].replace("chr", "")
            gn = None
            for attr in p[8].split(";"):
                a = attr.strip()
                if a.startswith("gene_name"):
                    gn = a.split('"')[1]
            genes.append({"CHR": str(ch), "start": int(p[3]),
                          "end": int(p[4]), "gene_name": gn})
    return pd.DataFrame(genes)


def annotate_snps(out, gdf, fl=500):
    if out.empty or gdf is None or gdf.empty: return out.copy()
    flb = fl * 1000
    rows = []
    for _, snp in out.iterrows():
        ch = str(snp["CHR"]).replace("CHR", "").strip()
        bp = int(snp["BP"])
        gc = gdf[gdf["CHR"] == ch]
        if gc.empty:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None}); continue
        cd = []
        for _, gn in gc.iterrows():
            if gn["start"] <= bp <= gn["end"]:
                cd.append((gn["gene_name"], 0, "intragenic"))
            else:
                d = min(abs(bp-gn["start"]), abs(bp-gn["end"]))
                if d <= flb:
                    r = "upstream" if bp < gn["start"] else "downstream"
                    cd.append((gn["gene_name"], d, r))
        if cd:
            b = min(cd, key=lambda x: x[1])
            rows.append({**snp.to_dict(), "gene_name": b[0],
                         "distance": b[1], "region": b[2]})
        else:
            rows.append({**snp.to_dict(), "gene_name": None,
                         "distance": None, "region": None})
    return pd.DataFrame(rows)


# ==================== GWAS 10 cM ====================
def gwas_local(g, s, i, ph, ch, bp_, pn, wc=10):
    wbp = wc * 1e6
    mask = ((s["CHR"].astype(str) == str(ch))
            & (s["BP"] >= bp_-wbp) & (s["BP"] <= bp_+wbp))
    idx = np.where(mask)[0]
    if len(idx) == 0: return pd.DataFrame()
    i = i.copy(); i["IID"] = i["IID"].astype(str)
    ph = ph.set_index("IID")
    com = [x for x in i["IID"] if x in ph.index]
    if len(com) < 10: return pd.DataFrame()
    im = i["IID"].isin(com).values
    gs = g[im][:, idx]
    ins = i[im].reset_index(drop=True)
    pv = []
    for iid in ins["IID"]:
        try:
            v = ph.loc[iid, pn]
            if isinstance(v, pd.Series): v = v.iloc[0]
            pv.append(float(v))
        except Exception: pv.append(np.nan)
    pv = np.array(pv)
    pops = ins["FID"].astype(str).values
    up = sorted(set(pops))
    dm = np.zeros((len(pops), max(0, len(up)-1)))
    for k, p in enumerate(pops):
        for j, u in enumerate(up[1:]):
            if p == u: dm[k, j] = 1
    res = []
    sw = s.iloc[idx]
    for j in range(len(idx)):
        gg = gs[:, j]
        v = ~np.isnan(gg) & ~np.isnan(pv)
        if v.sum() < 10: continue
        gv, pvv = gg[v], pv[v]
        X = (np.column_stack([np.ones(v.sum()), gv, dm[v]])
             if dm.shape[1] > 0
             else np.column_stack([np.ones(v.sum()), gv]))
        try:
            beta, _, _, _ = np.linalg.lstsq(X, pvv, rcond=None)
            r = pvv - X @ beta
            ssr = np.sum(r**2); sst = np.sum((pvv-pvv.mean())**2)
            r2 = 1 - ssr/sst if sst > 0 else 0
            dr = v.sum() - X.shape[1]
            if dr > 0:
                mse = ssr/dr
                try:
                    cv = np.linalg.inv(X.T @ X)
                    se = np.sqrt(mse*cv[1, 1])
                    tv = beta[1]/se
                    pval = 2*(1-stats.t.cdf(abs(tv), dr))
                except np.linalg.LinAlgError:
                    pval, tv = np.nan, np.nan
            else: pval, tv = np.nan, np.nan
            res.append({"SNP": sw.iloc[j]["SNP"],
                        "CHR": sw.iloc[j]["CHR"],
                        "BP": int(sw.iloc[j]["BP"]),
                        "beta": float(beta[1]),
                        "t_stat": float(tv) if np.isfinite(tv) else np.nan,
                        "p_value": float(pval) if np.isfinite(pval) else np.nan,
                        "r2": float(r2), "n_obs": int(v.sum())})
        except Exception: continue
    df = pd.DataFrame(res)
    if not df.empty:
        df["p_bonf"] = np.minimum(1, df["p_value"]*len(df))
        df["neg_log10_p"] = -np.log10(df["p_value"].clip(lower=1e-300))
        df = df.sort_values("p_value")
    return df


# ==================== EXPORT ====================
def _dos_bits(c):
    b = np.zeros(len(c), dtype=np.uint8)
    b[np.isnan(c)] = 0b01
    b[c == 0] = 0b11; b[c == 1] = 0b10; b[c == 2] = 0b00
    return b


def build_bed(g):
    n, m = g.shape; nb = (n+3)//4
    buf = bytearray([0x6C, 0x1B, 0x01])
    for j in range(m):
        bits = _dos_bits(g[:, j])
        p = np.zeros(nb, dtype=np.uint8)
        for k in range(n):
            p[k//4] |= (bits[k] & 3) << (2*(k % 4))
        buf.extend(p.tobytes())
    return bytes(buf)


def build_zip(g, ind, snp, pre="bovine"):
    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{pre}.bed", build_bed(g))
        bim = []
        for _, r in snp.iterrows():
            a1 = str(r.get("A1", "A")) if "A1" in r else "A"
            a2 = str(r.get("A2", "G")) if "A2" in r else "G"
            if a1 in ("nan", ""): a1 = "A"
            if a2 in ("nan", "", a1): a2 = "G" if a1 == "A" else "A"
            bim.append(f"{r['CHR']}\t{r['SNP']}\t{r['CM']}\t"
                       f"{int(r['BP'])}\t{a1}\t{a2}")
        zf.writestr(f"{pre}.bim", "\n".join(bim) + "\n")
        fam = "\n".join(f"{r['FID']}\t{r['IID']}\t0\t0\t0\t-9"
                        for _, r in ind.iterrows()) + "\n"
        zf.writestr(f"{pre}.fam", fam)
    return bio.getvalue()


# ==================== VISUALISATION ====================
def plt_hist(v, t, xl, c="#3498db"):
    fig = go.Figure(go.Histogram(x=v, nbinsx=80, marker_color=c))
    fig.update_layout(title=t, xaxis_title=xl, yaxis_title="Fréquence",
                      height=380)
    return fig


def plt_pca(sc, vp, lb):
    df = pd.DataFrame({"PC1": sc[:, 0], "PC2": sc[:, 1], "Pop": lb})
    fig = px.scatter(df, x="PC1", y="PC2", color="Pop",
                     title=f"PCA ({vp[0]:.1f}% / {vp[1]:.1f}%)", height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig


def plt_mds(c, lb):
    df = pd.DataFrame({"MDS1": c[:, 0], "MDS2": c[:, 1], "Pop": lb})
    fig = px.scatter(df, x="MDS1", y="MDS2", color="Pop", title="MDS",
                     height=550)
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    return fig


def plt_man(fst, ch, t="Manhattan — FST/SNP"):
    df = pd.DataFrame({"FST": np.asarray(fst),
                       "CHR": np.asarray(ch)}).dropna()
    if df.empty: return None
    o = df["CHR"].map(_chr_key)
    df["_0"] = o.map(lambda x: x[0]); df["_1"] = o.map(lambda x: x[1])
    df = df.sort_values(["_0", "_1"]).reset_index(drop=True)
    cum, ticks, lb = 0, [], []
    pos = np.zeros(len(df), dtype=int)
    for c in df["CHR"].unique():
        ix = np.where(df["CHR"] == c)[0]
        pos[ix] = np.arange(cum, cum + len(ix))
        ticks.append((cum + cum + len(ix) - 1) / 2)
        lb.append(str(c)); cum += len(ix)
    df["x"] = pos
    fig = go.Figure()
    for i, c in enumerate(df["CHR"].unique()):
        sub = df[df["CHR"] == c]
        color = "#2c3e50" if i % 2 == 0 else "#7f8c8d"
        fig.add_trace(go.Scatter(x=sub["x"], y=sub["FST"], mode="markers",
                                  marker=dict(size=5, color=color),
                                  showlegend=False, hoverinfo="skip"))
    q = float(np.nanquantile(fst, 0.99))
    fig.add_hline(y=q, line_dash="dash", line_color="red",
                  annotation_text=f"Top 1% = {q:.4f}")
    fig.update_layout(title=t, xaxis_title="Chromosome", yaxis_title="FST",
                      height=500)
    fig.update_xaxes(tickvals=ticks, ticktext=lb)
    return fig


def plt_gwas(df, ch, bp):
    if df.empty: return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["BP"], y=df["neg_log10_p"], mode="markers",
        marker=dict(size=8, color=df["neg_log10_p"], colorscale="Hot",
                    showscale=False),
        hovertemplate="%{text}<br>BP: %{x}<br>-log10(p): %{y:.2f}"
                      "<extra></extra>", text=df["SNP"]))
    fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="orange",
                  annotation_text="p=0.05")
    fig.add_hline(y=-np.log10(0.05/max(len(df), 1)), line_dash="dash",
                  line_color="red", annotation_text="Bonferroni")
    fig.add_vline(x=bp, line_dash="dot", line_color="blue",
                  annotation_text="SNP cible")
    fig.update_layout(title=f"GWAS local — Chr {ch}",
                      xaxis_title="Position (bp)", yaxis_title="-log10(p)",
                      height=450)
    return fig
  # ==================== ÉTAT ====================
STATE_KEYS = ["gt", "ind_df", "snp_df", "gt_filt", "ind_filt", "snp_filt",
              "qc_stats", "gt_snp", "ind_snp", "snp_snp",
              "gt_pruned", "snp_pruned", "pca_scores", "pca_var",
              "mds_coords", "king_matrix", "ld_df", "roh_df", "froh",
              "admixture_Q", "admixture_K", "cv_results",
              "reynolds_D", "reynolds_pops", "fst_global",
              "binome_result", "all_binomes", "group_results",
              "group_pair", "gwas_results", "genes_df",
              "_ax_ped", "_ax_map", "_ax_n", "_ax_samples"]


def init_state():
    for k in STATE_KEYS:
        if k not in st.session_state:
            st.session_state[k] = None
    if "trait_groups" not in st.session_state:
        st.session_state["trait_groups"] = dict(DEFAULT_GROUPS)


def reset_derived():
    for k in ["gt_filt", "ind_filt", "snp_filt", "qc_stats",
              "gt_snp", "ind_snp", "snp_snp", "gt_pruned", "snp_pruned",
              "pca_scores", "pca_var", "mds_coords", "king_matrix",
              "ld_df", "roh_df", "froh", "admixture_Q", "admixture_K",
              "cv_results", "reynolds_D", "reynolds_pops",
              "fst_global", "binome_result", "all_binomes",
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


# ==================== MAIN ====================
def main():
    init_state()
    st.title("🐄 Bovine SNP Platform v6.2")
    st.caption("Pipeline complet · Binômes · Signatures population · "
               "Groupes de races · Annotation GTF · GWAS 10 cM")

    # ----- SIDEBAR -----
    with st.sidebar:
        st.header("📁 Données")
        mode = st.radio("Source :", ["Demo", "PED/MAP", "VCF",
                                     "Convertisseur Axiom"],
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
                st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
        elif mode == "PED/MAP":
            pf = st.file_uploader(".ped", type=["ped", "gz", "txt"],
                                    key="sb_p")
            mf = st.file_uploader(".map", type=["map", "gz", "txt"],
                                    key="sb_m")
            if pf and mf and st.button("📥 Charger",
                                         use_container_width=True,
                                         key="sb_lp"):
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
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")
        elif mode == "VCF":
            vf = st.file_uploader(".vcf", type=["vcf", "gz", "txt"],
                                    key="sb_v")
            if vf and st.button("📥 Charger", use_container_width=True,
                                 key="sb_lv"):
                try:
                    gt, ind, snp = parse_vcf(vf.read())
                    if ind["FID"].nunique() <= 1:
                        ind["FID"] = ind["IID"].astype(str).apply(
                            lambda x: x.split("_")[0])
                    st.session_state.gt = gt
                    st.session_state.ind_df = ind
                    st.session_state.snp_df = snp
                    reset_derived()
                    st.success(f"✅ {gt.shape[0]} × {gt.shape[1]}")
                except Exception as e:
                    st.error(f"❌ {e}")

        st.divider()
        st.header("⚙️ Seuils QC")
        geno = st.slider("--geno", 0.0, 0.5, DEFAULTS["geno"], 0.01,
                          key="sb_g")
        mind = st.slider("--mind", 0.0, 0.5, DEFAULTS["mind"], 0.01,
                          key="sb_mi")
        maf_t = st.slider("MAF", 0.0, 0.5, DEFAULTS["maf"], 0.01,
                           key="sb_m")
        hwe_t = st.number_input("HWE", value=DEFAULTS["hwe"],
                                 format="%.0e", key="sb_h")
        het_sd = st.slider("HET ±σ", 1.0, 5.0, DEFAULTS["het_sd"], 0.1,
                            key="sb_hs")
        ld_r2 = st.slider("LD r²", 0.05, 0.5, DEFAULTS["ld_r2"], 0.05,
                           key="sb_ld")

        st.divider()
        st.header("🐄 Groupes de races")
        if has_data():
            all_pops = sorted(st.session_state.ind_df["FID"].astype(str)
                              .unique())
            with st.expander("Définir", expanded=False):
                for gn in ["Laitier", "Viande", "Cornu", "Rustique"]:
                    sel = st.multiselect(
                        f"→ {gn}", all_pops,
                        default=st.session_state["trait_groups"].get(gn, []),
                        key=f"grp_{gn}")
                    st.session_state["trait_groups"][gn] = sel

        st.divider()
        if st.button("🚀 Pipeline complet", type="primary",
                     use_container_width=True, key="sb_pipe"):
            st.session_state["_run_pipe"] = True
        st.caption("v6.2 — Complete")

    # ----- CONVERTISSEUR AXIOM -----
    if mode == "Convertisseur Axiom":
        render_axiom_page()
        return

    # ----- ACCUEIL -----
    if not has_data():
        st.info("👉 Générez un jeu de démo, importez PED/MAP/VCF, "
                "ou utilisez le Convertisseur Axiom.")
        st.markdown("""
        ### 🧬 v6.2 — 12 onglets

        | Onglet | Contenu |
        |---|---|
        | 🏠 Aperçu | Statistiques |
        | 🧹 QC | Missingness / MAF / HWE / HET |
        | ✂️ LD Pruning | Indep-pairwise |
        | 🧬 Structure | PCA + MDS + KING |
        | 🎨 Admixture | NMF + CV |
        | 📈 Démographie | LD decay + ROH + Ne |
        | 🔍 Sélection | FST Manhattan |
        | 🌳 Phylogénie | Reynolds + NJ |
        | 📤 Export | PLINK |
        | 🧬 **Binômes** | 5 modes + **signature/pop** + **groupes** |
        | 🧪 **SNP & Gènes** | **GWAS 10 cM + GTF** |
        | 📄 Rapport | HTML |

        ### 🆕 v6.2
        - ✅ Fix `detect_roh`
        - ✅ Convertisseur Axiom
        - ✅ Signatures par **population** (vs reste)
        - ✅ Signatures par **groupes** (Laitier / Viande / Cornu / Rustique)
        - ✅ Comparaison **Groupe A vs B**
        - ✅ GWAS 10 cM avec **phénotype groupe**
        - ✅ `gt_snp` strict (sans retrait HET) pour sélection
        """)
        return

    gt, ind_df, snp_df = (st.session_state.gt, st.session_state.ind_df,
                           st.session_state.snp_df)

    tabs = st.tabs(["🏠 Aperçu", "🧹 QC", "✂️ LD Pruning", "🧬 Structure",
                    "🎨 Admixture", "📈 Démographie", "🔍 Sélection",
                    "🌳 Phylogénie", "📤 Export", "🧬 Binômes",
                    "🧪 SNP & Gènes", "📄 Rapport"])

    # ----- TAB 0 -----
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.metric("Individus", gt.shape[0])
        c2.metric("SNPs", gt.shape[1])
        c3.metric("Populations", ind_df["FID"].nunique())
        st.dataframe(ind_df.head(20), use_container_width=True)

    # ----- TAB 1 : QC -----
    with tabs[1]:
        st.subheader("🧹 QC")
        if st.button("🧬 Filtrer autosomes 1-29",
                     use_container_width=True, key="qc_a"):
            g2, s2 = filter_auto(gt, snp_df)
            st.session_state.gt = g2
            st.session_state.snp_df = s2
            st.success(f"✅ {g2.shape[1]} SNPs")
        if st.button("▶ QC complet", type="primary",
                     use_container_width=True, key="qc_r"):
            try:
                with st.spinner("QC..."):
                    p = {"geno": geno, "mind": mind, "maf": maf_t,
                         "hwe": hwe_t, "het_sd": het_sd}
                    r = qc_filters(st.session_state.gt,
                                    st.session_state.ind_df,
                                    st.session_state.snp_df, p)
                    for k, v in zip(["gt_filt", "ind_filt", "snp_filt",
                                      "qc_stats", "gt_snp", "ind_snp",
                                      "snp_snp"], r):
                        st.session_state[k] = v
                st.success(f"✅ gt_filt : {r[0].shape[0]}×{r[0].shape[1]} · "
                           f"gt_snp : {r[4].shape[0]}×{r[4].shape[1]}")
            except Exception as e:
                st.error(f"❌ {e}")
        if has_qc():
            s = st.session_state.qc_stats
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ind.", s["n_ind_final"])
            c2.metric("SNPs", s["n_snp_final"])
            c3.metric("Exclus ind.", s["excluded_ind"])
            c4.metric("Exclus SNPs", s["excluded_snp"])
            st.info(f"gt_snp : {s['n_ind_snp']} ind × {s['n_snp_snp']:,} SNP")
            full = st.session_state.gt
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Miss / ind", "Miss / SNP"))
            fig.add_trace(go.Histogram(x=miss_ind(full), nbinsx=60,
                                        marker_color="skyblue"), 1, 1)
            fig.add_trace(go.Histogram(x=miss_snp(full), nbinsx=60,
                                        marker_color="coral"), 1, 2)
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ----- TAB 2 : LD Pruning -----
    with tabs[2]:
        st.subheader("✂️ LD Pruning")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            if st.button("▶ Pruning", type="primary",
                         use_container_width=True, key="ld_btn"):
                with st.spinner("Pruning..."):
                    k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                    st.session_state.gt_pruned = st.session_state.gt_filt[:, k]
                    st.session_state.snp_pruned = (
                        st.session_state.snp_filt[k].reset_index(drop=True))
                st.success(f"✅ {st.session_state.gt_pruned.shape[1]} SNPs")

    # ----- TAB 3 : Structure -----
    with tabs[3]:
        st.subheader("🧬 Structure")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            if st.button("▶ PCA + MDS + KING", type="primary",
                         use_container_width=True, key="str_btn"):
                with st.spinner("PCA..."):
                    sc, vp = pca_calc(gu, 10)
                    st.session_state.pca_scores = sc
                    st.session_state.pca_var = vp
                with st.spinner("MDS..."):
                    st.session_state.mds_coords = mds_calc(gu, 5)
                with st.spinner("KING..."):
                    st.session_state.king_matrix = king_matrix(gu)
                st.success("✅")
            lb = st.session_state.ind_filt["FID"].values
            if st.session_state.pca_scores is not None:
                st.plotly_chart(plt_pca(st.session_state.pca_scores,
                                         st.session_state.pca_var, lb),
                                use_container_width=True)
            if st.session_state.mds_coords is not None:
                st.plotly_chart(plt_mds(st.session_state.mds_coords, lb),
                                use_container_width=True)

    # ----- TAB 4 : Admixture -----
    with tabs[4]:
        st.subheader("🎨 Admixture")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            c1, c2 = st.columns(2)
            Km = c1.slider("K max", 3, 10, 6, key="ad_km")
            nr = c2.slider("CV reps", 1, 5, 3, key="ad_nr")
            if st.button("▶ CV", use_container_width=True, key="ad_cv"):
                with st.spinner("CV..."):
                    st.session_state.cv_results = admix_cv(
                        gu, (2, Km+1), int(nr))
                st.success("✅")
            if st.session_state.cv_results is not None:
                ks = sorted(st.session_state.cv_results.keys())
                means = [st.session_state.cv_results[k]["mean"] for k in ks]
                fig = go.Figure(go.Scatter(x=ks, y=means,
                                            mode="lines+markers",
                                            line=dict(color="crimson",
                                                      width=3),
                                            marker=dict(size=12)))
                ko = ks[int(np.argmin(means))]
                fig.add_vline(x=ko, line_dash="dash", line_color="green",
                              annotation_text=f"K={ko}")
                fig.update_layout(title="CV Error", height=400)
                st.plotly_chart(fig, use_container_width=True)
            K = st.slider("K", 2, 10, 4, key="ad_K")
            if st.button("▶ Admixture", use_container_width=True,
                         key="ad_r"):
                with st.spinner("NMF..."):
                    Q, _ = admix_nmf(gu, int(K))
                    st.session_state.admixture_Q = Q
                    st.session_state.admixture_K = K
                st.success(f"✅ Q {Q.shape}")
            if st.session_state.admixture_Q is not None:
                Q = st.session_state.admixture_Q
                K = st.session_state.admixture_K
                df = pd.DataFrame(Q, columns=[f"K{k+1}" for k in range(K)])
                df["Pop"] = st.session_state.ind_filt["FID"].values
                df["IID"] = st.session_state.ind_filt["IID"].values
                fig = go.Figure()
                for k in range(K):
                    fig.add_trace(go.Bar(x=np.arange(len(df)),
                                          y=df[f"K{k+1}"], name=f"K{k+1}"))
                fig.update_layout(barmode="stack", height=450,
                                  title=f"Admixture K={K}")
                st.plotly_chart(fig, use_container_width=True)

    # ----- TAB 5 : Démographie -----
    with tabs[5]:
        st.subheader("📈 Démographie")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            s1, s2 = st.tabs(["LD decay", "ROH"])
            with s1:
                if st.button("▶ LD decay", key="ld_d"):
                    with st.spinner("LD..."):
                        st.session_state.ld_df = ld_decay(
                            gu, st.session_state.snp_filt["BP"].values)
                    st.success("✅")
                if (st.session_state.ld_df is not None
                        and not st.session_state.ld_df.empty):
                    d = st.session_state.ld_df.copy()
                    d["bin"] = (d["dist_kb"] // 20) * 20
                    agg = d.groupby("bin")["r2"].mean().reset_index()
                    fig = px.line(agg, x="bin", y="r2",
                                  labels={"bin": "Distance (kb)",
                                          "r2": "r²"})
                    st.plotly_chart(fig, use_container_width=True)
            with s2:
                if st.button("▶ ROH", key="roh_b"):
                    try:
                        with st.spinner("ROH..."):
                            sj = st.session_state.snp_filt[["CHR", "BP"]].to_json()
                            roh, froh = detect_roh(gu, sj)
                            st.session_state.roh_df = roh
                            st.session_state.froh = froh
                        st.success(f"✅ {len(roh)} ROH")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.froh is not None:
                    df = pd.DataFrame({"FROH": st.session_state.froh,
                                       "Pop": st.session_state.ind_filt["FID"].values})
                    st.plotly_chart(px.histogram(df, x="FROH",
                                                  color="Pop", nbins=50),
                                    use_container_width=True)

    # ----- TAB 6 : Sélection globale -----
    with tabs[6]:
        st.subheader("🔍 Sélection (FST par SNP)")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)
            if st.button("▶ FST", use_container_width=True, key="fst_g"):
                with st.spinner("FST..."):
                    st.session_state.fst_global = fst_per_snp(
                        gu, iu["FID"].values)
                st.success("✅")
            if st.session_state.fst_global is not None:
                fig = plt_man(st.session_state.fst_global,
                              su["CHR"].values)
                if fig: st.plotly_chart(fig, use_container_width=True)

    # ----- TAB 7 : Phylogénie -----
    with tabs[7]:
        st.subheader("🌳 Phylogénie (Reynolds + NJ)")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            iu = st.session_state.ind_filt
            if st.button("▶ Reynolds", use_container_width=True,
                         key="rey"):
                with st.spinner("..."):
                    lb = iu["FID"].astype(str).values
                    pops = sorted(np.unique(lb))
                    ph = {p: np.nanmean(gu[lb == p], 0) / 2 for p in pops}
                    K = len(pops)
                    D = np.zeros((K, K))
                    for i in range(K):
                        for j in range(i+1, K):
                            pi, pj = ph[pops[i]], ph[pops[j]]
                            num = np.nansum((pi-pj)**2)
                            den = np.nansum(pi*(1-pj) + pj*(1-pi))
                            if den > 0: D[i, j] = D[j, i] = num/den
                    st.session_state.reynolds_D = D
                    st.session_state.reynolds_pops = pops
                st.success("✅")
            if st.session_state.reynolds_D is not None:
                D = st.session_state.reynolds_D
                pops = st.session_state.reynolds_pops
                fig = go.Figure(go.Heatmap(z=D, x=pops, y=pops,
                                            colorscale="Blues",
                                            text=np.round(D, 3),
                                            texttemplate="%{text}"))
                fig.update_layout(title="Distance de Reynolds", height=600)
                st.plotly_chart(fig, use_container_width=True)
                try:
                    from scipy.cluster.hierarchy import linkage, to_tree
                    Z = linkage(squareform(D, checks=False),
                                method="average")
                    tree = to_tree(Z)
                    def rec(n):
                        if n.is_leaf():
                            return f"{pops[n.id]}:{n.dist:.4f}"
                        return (f"({rec(n.get_left())},"
                                f"{rec(n.get_right())}):{n.dist:.4f}")
                    st.code(rec(tree), language="text")
                except Exception as e:
                    st.warning(f"NJ : {e}")

    # ----- TAB 8 : Export -----
    with tabs[8]:
        st.subheader("📤 Export PLINK")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            ge = (st.session_state.gt_pruned if has_pruned()
                  else st.session_state.gt_filt)
            ie = st.session_state.ind_filt
            se = (st.session_state.snp_pruned if has_pruned()
                  else st.session_state.snp_filt)
            pre = st.text_input("Préfixe", "bovine_qc", key="ex_p")
            if st.button("📦 Générer", use_container_width=True,
                         key="ex_g"):
                try:
                    with st.spinner("..."):
                        data = build_zip(ge, ie, se, pre)
                    st.download_button("⬇ Télécharger", data, f"{pre}.zip",
                                        "application/zip", key="ex_dl")
                except Exception as e:
                    st.error(f"❌ {e}")

    # ----- TAB 9 : BINÔMES + GROUPES -----
    with tabs[9]:
        st.subheader("🧬 Binômes & Signatures")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)
            lb = iu["FID"].astype(str).values
            avail = sorted(np.unique(lb))
            if has_snp():
                st.info(f"ℹ️ **gt_snp** : {gu.shape[0]} ind × "
                        f"{gu.shape[1]:,} SNP (sans outliers HET)")

            mode_b = st.radio(
                "Mode :",
                ["🧠 Auto-MDS", "🖱️ Manuelle", "🌐 Toutes paires",
                 "🎯 Signature par population",
                 "🐄 Signature par groupe", "⚔️ Groupe A vs B"],
                key="bin_mode")

            c1, c2 = st.columns(2)
            q_b = c1.slider("Quantile", 0.90, 0.9999, 0.99, 0.001,
                             format="%.4f", key="bin_q")
            fst_b = c2.slider("FST min", 0.0, 0.5, 0.15, 0.01,
                               key="bin_mf")

            # --- Auto-MDS ---
            if mode_b == "🧠 Auto-MDS":
                if st.session_state.mds_coords is None:
                    st.warning("⚠️ Lancez Structure d'abord.")
                else:
                    pairs = auto_pairs_mds(st.session_state.mds_coords, lb)
                    if not pairs.empty:
                        st.dataframe(pairs.head(20),
                                      use_container_width=True)
                        opts = [f"{r['pop_A']} ↔ {r['pop_B']} "
                                f"[{r['catégorie']}]"
                                for _, r in pairs.iterrows()]
                        sel = st.selectbox("Binôme :", opts, key="auto_sel")
                        a = sel.split(" ↔ ")[0].strip()
                        b = sel.split(" ↔ ")[1].split()[0].strip()
                        if st.button(f"▶ {a} vs {b}", type="primary",
                                     key="auto_run"):
                            with st.spinner("..."):
                                st.session_state.binome_result = analyze_pair(
                                    gu, su, lb, a, b, q_b, fst_b)
                        r = st.session_state.binome_result
                        if r and r.get("status") == "success":
                            st.metric("SNP outliers", r["n_snp_outliers"])
                            fig = plt_man(r["all_fst_df"]["FST"].values,
                                          r["all_fst_df"]["CHR"].values,
                                          title=f"{a} vs {b}")
                            if fig: st.plotly_chart(fig,
                                                     use_container_width=True)
                            st.dataframe(r["outliers_df"].head(100),
                                          use_container_width=True)

            # --- Manuelle ---
            elif mode_b == "🖱️ Manuelle":
                c1, c2 = st.columns(2)
                a = c1.selectbox("Pop A", avail, key="man_a")
                b = c2.selectbox("Pop B", [x for x in avail if x != a],
                                 key="man_b")
                if st.button(f"▶ {a} vs {b}", type="primary", key="man_run"):
                    with st.spinner("..."):
                        st.session_state.binome_result = analyze_pair(
                            gu, su, lb, a, b, q_b, fst_b)
                r = st.session_state.binome_result
                if r and r.get("status") == "success":
                    st.metric("SNP outliers", r["n_snp_outliers"])
                    fig = plt_man(r["all_fst_df"]["FST"].values,
                                  r["all_fst_df"]["CHR"].values,
                                  title=f"{a} vs {b}")
                    if fig: st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(r["outliers_df"].head(100),
                                  use_container_width=True)
                    csv = r["outliers_df"].to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ {a}vs{b}", csv,
                                        f"outliers_{a}_{b}.csv", "text/csv",
                                        key="man_dl")

            # --- Toutes paires ---
            elif mode_b == "🌐 Toutes paires":
                import itertools
                pairs = list(itertools.combinations(avail, 2))
                st.info(f"{len(pairs)} paires")
                if st.button(f"▶ Analyser {len(pairs)}",
                             type="primary",
                             use_container_width=True, key="all_run"):
                    res = {}
                    prog = st.progress(0)
                    for i, (a, b) in enumerate(pairs):
                        res[f"{a}_vs_{b}"] = analyze_pair(
                            gu, su, lb, a, b, q_b, fst_b)
                        prog.progress((i+1)/len(pairs))
                    st.session_state.all_binomes = res
                    st.success(f"✅ {len(res)}")
                if st.session_state.all_binomes:
                    summ = [{"Paire": k.replace("_vs_", " ↔ "),
                             "Outliers": v.get("n_snp_outliers", 0),
                             "FST moyen": round(v.get("fst_mean", 0), 4)}
                            for k, v in st.session_state.all_binomes.items()
                            if v.get("status") == "success"]
                    if summ:
                        df_s = (pd.DataFrame(summ)
                                .sort_values("FST moyen", ascending=False))
                        st.dataframe(df_s, use_container_width=True)

            # --- Signature par population ---
            elif mode_b == "🎯 Signature par population":
                st.info("FST de chaque race vs toutes les autres "
                        "regroupées.")
                if st.button("▶ Toutes les populations", type="primary",
                             use_container_width=True, key="pop_rest"):
                    res = analyze_pop_vs_rest(gu, su, lb, q_b, fst_b)
                    st.session_state.group_results = res
                    st.success(f"✅ {len(res)} populations")
                if st.session_state.group_results:
                    summ = [{"Population": k,
                             "Outliers": v["n_snp_outliers"],
                             "FST moyen": round(v["fst_mean"], 4),
                             "FST max": round(v["fst_max"], 4)}
                            for k, v in st.session_state.group_results.items()
                            if v.get("status") == "success"]
                    if summ:
                        st.dataframe(pd.DataFrame(summ).sort_values(
                            "FST moyen", ascending=False),
                            use_container_width=True)
                        sel = st.selectbox("Détail :",
                                            [s["Population"] for s in summ],
                                            key="pop_det")
                        v = st.session_state.group_results[sel]
                        if v.get("status") == "success":
                            fig = plt_man(v["all_fst_df"]["FST"].values,
                                          v["all_fst_df"]["CHR"].values,
                                          title=f"{sel} vs reste")
                            if fig: st.plotly_chart(fig,
                                                     use_container_width=True)
                            st.dataframe(v["outliers_df"].head(100),
                                          use_container_width=True)
                            csv = v["outliers_df"].to_csv(
                                index=False).encode("utf-8")
                            st.download_button(f"⬇ {sel}", csv,
                                                f"sig_{sel}.csv",
                                                "text/csv", key="pop_dl")

            # --- Signature par groupe ---
            elif mode_b == "🐄 Signature par groupe":
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                if not groups:
                    st.warning("⚠️ Définissez les groupes dans la sidebar.")
                else:
                    st.dataframe(pd.DataFrame([
                        {"Groupe": k, "Races": ", ".join(v),
                         "N": int(pd.Series(lb).isin(v).sum())}
                        for k, v in groups.items()]),
                        use_container_width=True)
                    if st.button("▶ Analyser tous les groupes",
                                 type="primary",
                                 use_container_width=True, key="grp_all"):
                        res = analyze_grp_vs_rest(gu, su, lb, groups,
                                                   q_b, fst_b)
                        st.session_state.group_results = res
                        st.success(f"✅ {len(res)}")
                    if st.session_state.group_results:
                        summ = [{"Groupe": k,
                                 "Outliers": v["n_snp_outliers"],
                                 "FST moyen": round(v["fst_mean"], 4)}
                                for k, v in
                                st.session_state.group_results.items()
                                if v.get("status") == "success"]
                        if summ:
                            st.dataframe(pd.DataFrame(summ).sort_values(
                                "FST moyen", ascending=False),
                                use_container_width=True)
                            sel = st.selectbox("Détail :",
                                                [s["Groupe"] for s in summ],
                                                key="grp_det")
                            v = st.session_state.group_results[sel]
                            if v.get("status") == "success":
                                fig = plt_man(
                                    v["all_fst_df"]["FST"].values,
                                    v["all_fst_df"]["CHR"].values,
                                    title=f"{sel} vs reste")
                                if fig: st.plotly_chart(
                                    fig, use_container_width=True)
                                st.dataframe(
                                    v["outliers_df"].head(100),
                                    use_container_width=True)
                                csv = v["outliers_df"].to_csv(
                                    index=False).encode("utf-8")
                                st.download_button(f"⬇ {sel}", csv,
                                                    f"grp_{sel}.csv",
                                                    "text/csv",
                                                    key="grp_dl")

            # --- Groupe A vs B ---
            else:
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                if len(groups) < 2:
                    st.warning("⚠️ Définissez au moins 2 groupes.")
                else:
                    c1, c2 = st.columns(2)
                    gA = c1.selectbox("Groupe A", list(groups.keys()),
                                       key="gA")
                    gB = c2.selectbox("Groupe B",
                                       [g for g in groups if g != gA],
                                       key="gB")
                    if st.button(f"▶ {gA} vs {gB}", type="primary",
                                 key="pair_run"):
                        with st.spinner("..."):
                            st.session_state.group_pair = analyze_grp_pair(
                                gu, su, lb, groups, gA, gB, q_b, fst_b)
                        st.success("✅")
                    r = st.session_state.group_pair
                    if r and r.get("status") == "success":
                        c1, c2, c3 = st.columns(3)
                        c1.metric("SNP", r["n_snp_analyzed"])
                        c2.metric("Outliers", r["n_snp_outliers"])
                        c3.metric("FST", f"{r['fst_mean']:.4f}")
                        fig = plt_man(r["all_fst_df"]["FST"].values,
                                      r["all_fst_df"]["CHR"].values,
                                      title=f"{gA} vs {gB}")
                        if fig: st.plotly_chart(fig,
                                                 use_container_width=True)
                        st.dataframe(r["outliers_df"].head(100),
                                      use_container_width=True)
                        csv = r["outliers_df"].to_csv(
                            index=False).encode("utf-8")
                        st.download_button(f"⬇ {gA}vs{gB}", csv,
                                            f"diff_{gA}_{gB}.csv",
                                            "text/csv", key="pair_dl")

    # ----- TAB 10 : SNP & Gènes -----
    with tabs[10]:
        st.subheader("🧪 SNP & Gènes")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            gu = (st.session_state.gt_snp if has_snp()
                  else st.session_state.gt_filt)
            su = (st.session_state.snp_snp if has_snp()
                  else st.session_state.snp_filt)
            iu = (st.session_state.ind_snp if has_snp()
                  else st.session_state.ind_filt)

            s1, s2 = st.tabs(["🧬 Annotation", "📊 GWAS 10 cM"])

            with s1:
                st.markdown(
                    "**GTF bovin :** [Ensembl ARS-UCD1.2](https://ftp."
                    "ensembl.org/pub/release-110/gtf/bos_taurus/"
                    "Bos_taurus.ARS-UCD1.2.110.gtf.gz)")
                gf = st.file_uploader("GTF", type=["gtf", "gz"],
                                        key="gtf_f")
                flank = st.slider("Fenêtre (kb)", 10, 5000, 500, 10,
                                    key="gtf_fl")
                if gf and st.button("📥 Charger GTF",
                                     use_container_width=True, key="gtf_l"):
                    try:
                        with st.spinner("..."):
                            tmp = "/tmp/ann.gtf"
                            if gf.name.endswith(".gz"): tmp += ".gz"
                            with open(tmp, "wb") as f:
                                f.write(gf.getbuffer())
                            st.session_state.genes_df = load_gtf(tmp)
                        st.success(
                            f"✅ {len(st.session_state.genes_df):,} gènes")
                    except Exception as e:
                        st.error(f"❌ {e}")
                if st.session_state.genes_df is not None:
                    st.info(f"Gènes : {len(st.session_state.genes_df):,}")
                    res = st.session_state.group_results
                    if res:
                        sel = st.selectbox("Groupe à annoter :",
                                            list(res.keys()), key="ann_g")
                        v = res[sel]
                        if v.get("status") == "success":
                            if st.button("📋 Annoter", type="primary",
                                         use_container_width=True,
                                         key="ann_btn"):
                                try:
                                    with st.spinner("..."):
                                        ann = annotate_snps(
                                            v["outliers_df"],
                                            st.session_state.genes_df,
                                            int(flank))
                                        st.session_state.group_results[
                                            sel]["annotated_df"] = ann
                                    st.success(f"✅ {len(ann)} SNPs")
                                except Exception as e:
                                    st.error(f"❌ {e}")
                            if "annotated_df" in v:
                                ann = v["annotated_df"]
                                wg = ann.dropna(subset=["gene_name"])
                                c1, c2, c3 = st.columns(3)
                                c1.metric("SNPs", len(ann))
                                c2.metric("Avec gène", len(wg))
                                c3.metric("Gènes uniques",
                                          wg["gene_name"].nunique()
                                          if not wg.empty else 0)
                                cols = ["CHR", "SNP", "BP", "FST",
                                        "gene_name", "distance", "region"]
                                cols = [c for c in cols if c in ann.columns]
                                st.dataframe(ann[cols].head(200),
                                              use_container_width=True)
                                csv = ann.to_csv(index=False).encode("utf-8")
                                st.download_button("⬇ CSV", csv,
                                                    f"ann_{sel}.csv",
                                                    "text/csv", key="ann_dl")

            with s2:
                groups = {k: v for k, v in
                          st.session_state["trait_groups"].items() if v}
                st.markdown("### 1. Phénotype")
                src = st.radio("Source :", ["Groupe de races",
                                             "Fichier CSV"],
                                horizontal=True, key="gw_src")
                ph_col, ph = None, None
                if src == "Groupe de races":
                    if not groups:
                        st.warning("⚠️ Définissez les groupes.")
                        st.stop()
                    gn = st.selectbox("Groupe = cas :",
                                       list(groups.keys()), key="gw_g")
                    ph = (grp_phenotype(iu, groups, gn)
                          .rename(columns={"pheno": "pheno_group"})
                          [["IID", "pheno_group"]])
                    ph_col = "pheno_group"
                    st.success(f"✅ {int(ph['pheno_group'].sum())} cas")
                else:
                    pf = st.file_uploader("CSV (IID,col)", type=["csv"],
                                            key="gw_csv")
                    if pf is None: st.stop()
                    ph = pd.read_csv(pf)
                    if "IID" not in ph.columns:
                        st.error("❌ IID manquant"); st.stop()
                    ph["IID"] = ph["IID"].astype(str)
                    nc = [c for c in ph.columns if c != "IID"
                          and pd.api.types.is_numeric_dtype(ph[c])]
                    if not nc:
                        st.error("❌ Aucun numérique"); st.stop()
                    ph_col = st.selectbox("Phénotype :", nc, key="gw_c")
                    st.success(f"✅ {len(ph)} lignes")

                st.markdown("### 2. SNP cible")
                res = st.session_state.group_results
                if res:
                    avail = [k for k, v in res.items()
                             if v.get("status") == "success"]
                    if avail:
                        sel_g = st.selectbox("Depuis :", avail, key="gw_sg")
                        cands = res[sel_g]["outliers_df"]
                        if not cands.empty:
                            top_n = st.slider("Top N", 1, 20, 5,
                                               key="gw_tn")
                            top = cands.head(top_n)
                            st.dataframe(
                                top[["CHR", "SNP", "BP", "FST"]],
                                use_container_width=True)
                            win = st.slider("Fenêtre (cM)", 1, 20, 10,
                                            key="gw_w")
                            if st.button("▶ GWAS", type="primary",
                                         use_container_width=True,
                                         key="gw_run"):
                                all_g = {}
                                prog = st.progress(0)
                                for i, (_, s) in enumerate(top.iterrows()):
                                    with st.spinner(
                                            f"GWAS {s['SNP']}..."):
                                        r = gwas_local(
                                            gu, su, iu, ph, s["CHR"],
                                            s["BP"], ph_col, win)
                                    if not r.empty:
                                        all_g[s["SNP"]] = r
                                    prog.progress((i+1)/len(top))
                                st.session_state.gwas_results = all_g
                                st.success(f"✅ {len(all_g)} GWAS")

                gres = st.session_state.gwas_results
                if gres:
                    summ = []
                    for snp_n, df in gres.items():
                        nb = int((df["p_bonf"] < 0.05).sum())
                        ns = int((df["p_value"] < 0.05).sum())
                        summ.append({"SNP cible": snp_n,
                                     "N": len(df), "p<0.05": ns,
                                     "Bonferroni": nb,
                                     "p min": round(
                                         df["p_value"].min(), 6)})
                    st.dataframe(pd.DataFrame(summ),
                                  use_container_width=True)
                    sel = st.selectbox("Voir :", list(gres.keys()),
                                        key="gw_v")
                    df = gres[sel]
                    snp_row = su[su["SNP"] == sel]
                    if not snp_row.empty:
                        fig = plt_gwas(df, snp_row.iloc[0]["CHR"],
                                        int(snp_row.iloc[0]["BP"]))
                        if fig: st.plotly_chart(fig,
                                                 use_container_width=True)
                    st.dataframe(df.head(50), use_container_width=True)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(f"⬇ GWAS {sel}", csv,
                                        f"gwas_{sel}.csv", "text/csv",
                                        key="gw_dl")

    # ----- TAB 11 : Rapport -----
    with tabs[11]:
        st.subheader("📄 Rapport HTML")
        if not has_qc():
            st.warning("⚠️ Lancez QC.")
        else:
            proj = st.text_input("Projet", "Cattle_Project", key="rep_p")
            if st.button("📄 Générer", use_container_width=True,
                         key="rep_g"):
                s = st.session_state.qc_stats
                figs = ""
                if st.session_state.pca_scores is not None:
                    f = plt_pca(st.session_state.pca_scores,
                                st.session_state.pca_var,
                                st.session_state.ind_filt["FID"].values)
                    figs += f.to_html(full_html=False,
                                       include_plotlyjs="cdn")
                html = f"""<!DOCTYPE html><html><head>
<meta charset="UTF-8"><title>Rapport v6.2</title>
<style>body{{font-family:Arial;margin:30px}}
h1{{color:#1a5276;border-bottom:2px solid #1a5276}}
.summary{{background:#f4f6f7;padding:16px;border-radius:8px}}</style>
</head><body><h1>🐄 Rapport Bovine SNP Platform v6.2</h1>
<p><b>{proj}</b> — {datetime.now():%Y-%m-%d %H:%M}</p>
<div class="summary"><h2>Résumé</h2><ul>
<li>Individus : {s['n_ind_final']} / {s['n_ind_init']}</li>
<li>SNPs : {s['n_snp_final']} / {s['n_snp_init']}</li>
</ul></div>{figs}</body></html>"""
                st.download_button(
                    "⬇ Rapport", html.encode("utf-8"),
                    f"rapport_{datetime.now():%Y%m%d}.html",
                    "text/html", key="rep_dl")

    # ----- PIPELINE COMPLET -----
    if st.session_state.get("_run_pipe"):
        st.session_state["_run_pipe"] = False
        try:
            with st.spinner("🚀 Pipeline..."):
                g2, s2 = filter_auto(st.session_state.gt,
                                      st.session_state.snp_df)
                st.session_state.gt = g2
                st.session_state.snp_df = s2
                p = {"geno": geno, "mind": mind, "maf": maf_t,
                     "hwe": hwe_t, "het_sd": het_sd}
                r = qc_filters(g2, st.session_state.ind_df, s2, p)
                for k, v in zip(["gt_filt", "ind_filt", "snp_filt",
                                  "qc_stats", "gt_snp", "ind_snp",
                                  "snp_snp"], r):
                    st.session_state[k] = v
                k = ld_pruning(st.session_state.gt_filt, 50, 5, ld_r2)
                st.session_state.gt_pruned = (st.session_state.gt_filt[:, k])
                st.session_state.snp_pruned = (
                    st.session_state.snp_filt[k].reset_index(drop=True))
                gu = st.session_state.gt_pruned
                sc, vp = pca_calc(gu, 10)
                st.session_state.pca_scores = sc
                st.session_state.pca_var = vp
                st.session_state.mds_coords = mds_calc(gu, 5)
            st.success("✅ Pipeline terminé !")
        except Exception as e:
            st.error(f"❌ {e}")


if __name__ == "__main__":
    main()

"""
agent1_header.py - Binary Header-only Naive Bayes agent (v8.0.0-binary)

Inputs : sender, receiver, date, subject (raw header strings). Never the body.
Outputs: P(safe), P(not_safe) in fixed order (0=safe, 1=not_safe).
         nb_score = P(not_safe) in [0, 1].
"""
import re, datetime
from email.header import decode_header, make_header
from email.utils import getaddresses, parsedate_tz

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar

__version__ = "8.0.0-binary"

CLASS_NAMES = ("safe", "not_safe")
N_CLASSES = 2
PROB_FLOOR = 1e-6
LOGIT_CLIP = 10.0
MAX_RAW = 4096
MAX_HDR = 512

_PREFIX_RE = re.compile(r"^\s*(?:re|fwd?|fw|antw|wg|tr|aw)\s*[:\-]\s*", re.I)
_ADDR_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_IPV4_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_TZ_TAIL_RE = re.compile(r"\b([A-Z]{1,4})\b\s*$")
_TZ_NUM_RE = re.compile(r"\d:\d{2}(?::\d{2})?\s*([+-]\d{4})\b")

FREE_MAIL = frozenset({
    "gmail.com", "yahoo.com", "hotmail.com", "aol.com", "outlook.com",
    "mail.com", "msn.com", "live.com", "ymail.com", "comcast.net",
    "sbcglobal.net", "verizon.net", "att.net", "bellsouth.net",
    "rediffmail.com", "cox.net", "charter.net", "shaw.ca", "earthlink.net"})

BRAND_TOKENS = frozenset({
    "paypal", "ebay", "amazon", "apple", "microsoft", "google",
    "chase", "wellsfargo", "bankofamerica", "citi", "citigroup",
    "barclays", "hsbc", "netflix", "irs", "usps", "fedex", "ups",
    "dhl", "facebook", "instagram", "whatsapp", "linkedin", "twitter"})
FUZZY_BRANDS = frozenset(b for b in BRAND_TOKENS if len(b) >= 5)
COMMON_WORDS = frozenset({"apply", "ample", "maple", "grape", "phase", "chose", "chasm", "apples"})

SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "loan",
    "click", "fit", "surf", "buzz", "racing", "date", "faith"})

KNOWN_TWO_LEVEL_TLDS = frozenset({
    "co.uk", "org.uk", "me.uk", "ltd.uk", "plc.uk", "net.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.nz", "net.nz", "org.nz", "ac.nz", "govt.nz",
    "co.za", "org.za", "net.za", "gov.za",
    "co.jp", "ne.jp", "or.jp", "go.jp", "ac.jp",
    "com.br", "net.br", "org.br", "gov.br",
    "com.mx", "org.mx", "net.mx", "edu.mx", "gob.mx",
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "co.in", "net.in", "org.in", "gen.in", "firm.in",
    "gc.ca", "fed.us"})

RFC_TZ_OFFSETS = {
    "UT": 0, "UTC": 0, "GMT": 0, "Z": 0,
    "EDT": -4, "EST": -5, "CDT": -5, "CST": -6,
    "MDT": -6, "MST": -7, "PDT": -7, "PST": -8,
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6,
    "G": 7, "H": 8, "I": 9, "K": 10, "L": 11, "M": 12,
    "N": -1, "O": -2, "P": -3, "Q": -4, "R": -5, "S": -6,
    "T": -7, "U": -8, "V": -9, "W": -10, "X": -11, "Y": -12}


def _decode(v, limit=MAX_HDR):
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if not isinstance(v, str):
        return ""
    raw = v[:MAX_RAW]
    try:
        dec = str(make_header(decode_header(raw)))
    except Exception:
        dec = raw
    return dec[:limit].strip()


def _is_ip(host):
    return bool(_IPV4_RE.match(str(host).strip("[] ")))


def registrable_domain(domain):
    if not domain or not isinstance(domain, str):
        return ""
    d = domain.strip().lower().strip(". >)(<[]")
    if _IPV4_RE.match(d):
        return d
    parts = d.split(".")
    if len(parts) <= 2:
        return d
    if ".".join(parts[-2:]) in KNOWN_TWO_LEVEL_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _normalize_subject_text(dec):
    txt = dec.lower()
    for _ in range(5):
        new = _PREFIX_RE.sub("", txt).strip()
        if new == txt:
            break
        txt = new
    txt = re.sub(r"\d+", "#", txt)
    txt = re.sub(r"[^\w\s#]", " ", txt)
    return " ".join(txt.split())


def normalize_subject(s):
    return _normalize_subject_text(_decode(s))


def _parse_addrs(text):
    out = []
    if not text or not isinstance(text, str):
        return out
    try:
        try:
            items = getaddresses([text], strict=False)
        except TypeError:
            items = getaddresses([text])
        out = [(n.strip(), a.strip().lower()) for n, a in items if a and "@" in a]
    except Exception:
        out = []
    if not out:
        out = [("", a.lower()) for a in _ADDR_RE.findall(text)]
    return out


def _within_one_edit(a, b):
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(c1 != c2 for c1, c2 in zip(a, b)) == 1
    short, long_ = (a, b) if la < lb else (b, a)
    return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


def canonical_sender_identity(dom, addr):
    if not addr or "@" not in str(addr):
        return None
    d = str(dom or "").strip().lower()
    if not d or d in FREE_MAIL or _is_ip(d):
        return f"addr:{str(addr).strip().lower()}"
    return f"dom:{d}"


def _ensure_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, dict):
        return pd.DataFrame([x])
    if isinstance(x, (list, tuple)):
        return pd.DataFrame(list(x))
    raise TypeError(type(x))


_SENDER_EMPTY = dict(
    s_missing=1, s_addr="", s_dom="", s_tld="", s_dom_levels=0, s_has_num=0, s_has_hyphen=0,
    s_is_freemail=0, s_suspicious_tld=0, s_brand_in_domain=0, s_brand_spoof_host=0, s_typo_brand=0,
    s_ip_domain=0, s_punycode=0, s_disp_len=0, s_disp_has_excl=0, s_disp_has_quest=0,
    s_disp_has_at=0, s_disp_has_dom=0)


def _sender_info(v):
    row = dict(_SENDER_EMPTY)
    addrs = _parse_addrs(_decode(v))
    if not addrs:
        return row
    disp, a = addrs[0]
    host = a.split("@", 1)[1].strip(".[]> <)(") if "@" in a else ""
    if not host:
        return row
    row["s_missing"] = 0
    row["s_addr"] = a
    labels = host.split(".")
    row["s_dom_levels"] = len(labels)
    row["s_disp_len"] = len(disp)
    row["s_disp_has_excl"] = int("!" in disp)
    row["s_disp_has_quest"] = int("?" in disp)
    row["s_disp_has_at"] = int("@" in disp)
    row["s_disp_has_dom"] = int(bool(re.search(r"\b[a-z0-9-]+\.[a-z]{2,}\b", disp, re.I)))
    row["s_punycode"] = int(any(l.startswith("xn--") for l in labels))
    if _is_ip(host):
        row["s_dom"] = host.strip("[]")
        row["s_ip_domain"] = 1
        return row
    reg = registrable_domain(host)
    tld = reg.split(".")[-1] if "." in reg else ""
    row["s_dom"], row["s_tld"] = reg, tld
    row["s_has_num"] = int(bool(re.search(r"\d", reg)))
    row["s_has_hyphen"] = int("-" in reg)
    row["s_is_freemail"] = int(reg in FREE_MAIL)
    row["s_suspicious_tld"] = int(tld in SUSPICIOUS_TLDS)
    reg_tokens = {t for t in re.split(r"[^a-z0-9]+", reg) if t}
    sub_part = host[: len(host) - len(reg)] if host.endswith(reg) else ""
    sub_tokens = {t for t in re.split(r"[^a-z0-9]+", sub_part) if t}
    in_reg = reg_tokens & BRAND_TOKENS
    row["s_brand_in_domain"] = int(bool(in_reg))
    row["s_brand_spoof_host"] = int(bool((sub_tokens & BRAND_TOKENS) - in_reg))
    row["s_typo_brand"] = int(any(
        len(t) >= 5 and t not in COMMON_WORDS and t not in BRAND_TOKENS and _within_one_edit(t, b)
        for t in reg_tokens for b in FUZZY_BRANDS))
    return row


def _receiver_info(v, sdom):
    dec = _decode(v)
    addrs = _parse_addrs(dec)
    undis = int("undisclosed" in dec.lower())
    row = dict(r_missing=0, r_state=0, r_count=0, r_disp_len=0, r_disp_has_at=0, r_has_undisclosed=undis,
               r_same_domain_as_sender=0, r_is_freemail=0)
    if not addrs:
        row["r_missing"] = 0 if undis else 1
        row["r_state"] = 1 if undis else 0
        return row
    disp, a = addrs[0]
    host = a.split("@", 1)[1] if "@" in a else ""
    reg = registrable_domain(host)
    row.update(r_state=1 if undis else (2 if len(addrs) == 1 else 3),
               r_count=len(addrs), r_disp_len=len(disp), r_disp_has_at=int("@" in disp),
               r_same_domain_as_sender=int(reg == sdom and bool(sdom)), r_is_freemail=int(reg in FREE_MAIL))
    return row


def _date_info(v):
    dec = _decode(v)
    if not dec:
        return 0, -1, -1, 0
    try:
        pt = parsedate_tz(dec)
    except Exception:
        pt = None
    if pt is None:
        return 3, -1, -1, 0
    y, mo, d, hr = pt[0], pt[1], pt[2], pt[3]
    if not (0 <= hr <= 23):
        return 3, -1, -1, 0
    try:
        wd = datetime.date(y, mo, d).weekday()
    except Exception:
        return 3, -1, -1, 0
    body = re.sub(r"\([^)]*\)\s*$", "", dec).strip()
    if _TZ_NUM_RE.search(body):
        return 2, hr, wd, int(round((pt[9] or 0) / 3600.0))
    m = _TZ_TAIL_RE.search(body)
    if m and m.group(1) in RFC_TZ_OFFSETS:
        return 2, hr, wd, RFC_TZ_OFFSETS[m.group(1)]
    return 1, hr, wd, 0


def _subject_info(v):
    dec = _decode(v)
    if not dec:
        return dict(u_missing=1, u_len=0, u_num_words=0, u_caps_ratio=0.0, u_num_punct=0,
                    u_num_digits=0, u_is_fwd=0, u_is_reply=0, u_norm="")
    letters = [c for c in dec if c.isalpha()]
    return dict(
        u_missing=0, u_len=len(dec), u_num_words=len(dec.split()),
        u_caps_ratio=(sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0,
        u_num_punct=sum(c in "!?;:#$%&'\"" for c in dec),
        u_num_digits=sum(c.isdigit() for c in dec),
        u_is_fwd=int(bool(re.search(r"^\s*(?:fwd?|fw)\s*[:\-]", dec, re.I))),
        u_is_reply=int(bool(re.search(r"^\s*(?:re|antw|wg|tr|aw)\s*[:\-]", dec, re.I))),
        u_norm=_normalize_subject_text(dec))


def extract_features(df):
    df = _ensure_df(df)
    N = len(df)
    col = lambda name: df[name].tolist() if name in df else [None] * N
    S = pd.DataFrame([_sender_info(v) for v in col("sender")], index=df.index)
    R = pd.DataFrame([_receiver_info(v, d) for v, d in zip(col("receiver"), S["s_dom"].tolist())], index=df.index)
    D = pd.DataFrame([_date_info(v) for v in col("date")], index=df.index,
                     columns=["d_date_status", "d_hour", "d_weekday", "d_tz_h"])
    U = pd.DataFrame([_subject_info(v) for v in col("subject")], index=df.index)
    return pd.concat([S, R, D, U], axis=1)


SPEC = {
    "s_missing": ("bin",), "s_dom_levels": ("cap", 6), "s_has_num": ("bin",), "s_has_hyphen": ("bin",),
    "s_is_freemail": ("bin",), "s_suspicious_tld": ("bin",), "s_brand_in_domain": ("bin",),
    "s_brand_spoof_host": ("bin",), "s_typo_brand": ("bin",), "s_ip_domain": ("bin",), "s_punycode": ("bin",),
    "s_disp_len": ("quant",), "s_disp_has_excl": ("bin",), "s_disp_has_quest": ("bin",),
    "s_disp_has_at": ("bin",), "s_disp_has_dom": ("bin",), "s_tld": ("cat",),
    "r_missing": ("bin",), "r_state": ("cap", 3), "r_count": ("cap", 5), "r_disp_len": ("quant",), "r_disp_has_at": ("bin",),
    "r_has_undisclosed": ("bin",), "r_same_domain_as_sender": ("bin",), "r_is_freemail": ("bin",),
    "d_date_status": ("cap", 3), "d_hour": ("edges", list(range(0, 24, 2))),
    "d_weekday": ("cap", 6), "d_tz_h": ("edges", list(range(-12, 14, 2))),
    "u_missing": ("bin",), "u_len": ("quant",), "u_num_words": ("quant",), "u_caps_ratio": ("quant",),
    "u_num_punct": ("quant",), "u_num_digits": ("quant",), "u_is_fwd": ("bin",), "u_is_reply": ("bin",)}

FEATURE_GROUPS = {g: [f for f in SPEC if f.startswith(p)]
                  for g, p in (("sender", "s_"), ("receiver", "r_"), ("date", "d_"), ("subject", "u_"))}
ALL_FEATURES = [f for g in ("sender", "receiver", "date", "subject") for f in FEATURE_GROUPS[g] if f != "r_state"]
COMPACT_RECEIVER = ["r_state", "r_same_domain_as_sender", "r_is_freemail"]
ALL_FEATURES_COMPACT = [f for f in ALL_FEATURES if not f.startswith("r_")] + COMPACT_RECEIVER
MISSINGNESS_FEATURES = ["s_missing", "r_missing", "r_state", "u_missing", "d_date_status"]
CLOCK_FEATURES = ["d_hour", "d_weekday", "d_tz_h"]


class HeaderPreprocessor:
    def __init__(self, features, n_bins=24, top_tld=20):
        self.features = list(features)
        self.n_bins, self.top_tld = n_bins, top_tld

    def fit(self, Xf):
        self.edges_, self.tlds_ = {}, []
        M = self._masks(Xf)
        for f in self.features:
            kind = SPEC[f][0]
            if kind == "quant":
                v = pd.to_numeric(Xf[f], errors="coerce").to_numpy(dtype=float)
                am = self.abstain_mask(f, M)
                if am is not None:
                    v = v[~am]
                v = v[~np.isnan(v)]
                self.edges_[f] = (np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1])).tolist()
                                  if len(v) else [])
            elif kind == "edges":
                self.edges_[f] = list(SPEC[f][1])
        if "s_tld" in self.features and self.top_tld > 0:
            vc = Xf["s_tld"].astype(str).replace("nan", "").value_counts()
            self.tlds_ = [t for t in vc.index if t][: self.top_tld]
        self.n_categories_ = [self._k(f) for f in self.features]
        return self

    def _k(self, f):
        kind = SPEC[f][0]
        if kind == "bin":
            return 3
        if kind == "cap":
            return SPEC[f][1] + 2
        if kind in ("quant", "edges"):
            return len(self.edges_[f]) + 3
        if kind == "cat":
            return len(self.tlds_) + 3
        raise ValueError(kind)

    def missing_category(self, f):
        if f in ("s_missing", "r_missing", "u_missing"):
            return 1
        if f in ("d_date_status", "r_state"):
            return 0
        if SPEC[f][0] in ("quant", "edges", "cat"):
            return self._k(f) - 2
        return None

    @staticmethod
    def _masks(Xf):
        n = len(Xf)
        z = np.zeros(n, bool)
        g = lambda c: Xf[c].to_numpy() if c in Xf.columns else None
        s_m = g("s_missing"); s_m = s_m.astype(bool) if s_m is not None else z
        r_m = g("r_missing"); r_m = r_m.astype(bool) if r_m is not None else z
        u_m = g("u_missing"); u_m = u_m.astype(bool) if u_m is not None else z
        rc = g("r_count"); r_noaddr = (rc == 0) if rc is not None else z
        st = g("d_date_status")
        d_bad = np.isin(st, (0, 3)) if st is not None else z
        d_notz = (st != 2) if st is not None else z
        return dict(s=s_m, r=r_m, u=u_m, r_noaddr=np.asarray(r_noaddr, bool),
                    d_bad=np.asarray(d_bad, bool), d_notz=np.asarray(d_notz, bool))

    @staticmethod
    def abstain_mask(f, M):
        if f in MISSINGNESS_FEATURES:
            return None
        if f.startswith("s_"):
            return M["s"]
        if f.startswith("r_"):
            if f == "r_same_domain_as_sender":
                return M["r"] | M["r_noaddr"] | M["s"]
            return M["r"] if f in ("r_count", "r_has_undisclosed") else (M["r"] | M["r_noaddr"])
        if f.startswith("u_"):
            return M["u"]
        if f.startswith("d_"):
            return M["d_notz"] if f == "d_tz_h" else M["d_bad"]
        return None

    def transform(self, Xf):
        M = self._masks(Xf)
        cols = []
        for f in self.features:
            K = self._k(f)
            kind = SPEC[f][0]
            if kind == "bin":
                c = np.clip(pd.to_numeric(Xf[f], errors="coerce").fillna(0).to_numpy(dtype=int), 0, 1)
            elif kind == "cap":
                c = np.clip(pd.to_numeric(Xf[f], errors="coerce").fillna(0).to_numpy(dtype=int), 0, SPEC[f][1])
            elif kind in ("quant", "edges"):
                ed = self.edges_[f]
                v = pd.to_numeric(Xf[f], errors="coerce").to_numpy(dtype=float)
                nan = np.isnan(v)
                c = np.searchsorted(ed, np.nan_to_num(v, nan=0.0), side="right")
                c[nan] = len(ed) + 1
            else:
                m = {t: i for i, t in enumerate(self.tlds_)}
                s = Xf[f].astype(str).str.lower()
                c = np.array([len(self.tlds_) + 1 if (not x or x == "nan") else m.get(x, len(self.tlds_))
                              for x in s], dtype=int)
            am = self.abstain_mask(f, M)
            if am is not None:
                c = np.where(am, K - 1, c)
            cols.append(c)
        return np.column_stack(cols).astype(np.int32)


class SmoothedCategoricalNB:
    def __init__(self, alpha=1.0, lik_scale=1.0, max_missing_log_odds=1.5,
                 feat_bound=2.5, smoothing="marginal"):
        self.alpha, self.lik_scale = float(alpha), float(lik_scale)
        self.max_missing_log_odds = float(max_missing_log_odds)
        self.feat_bound = float(feat_bound) if feat_bound is not None else 0.0
        self.smoothing = smoothing

    def fit(self, X, y, n_categories, preprocessor=None, feature_weights=None):
        y = np.asarray(y, dtype=int)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError(f"Binary labels {{0, 1}} required, got {np.unique(y).tolist()}")
        C = N_CLASSES
        self.n_internal_ = C
        cc = np.bincount(y, minlength=C).astype(float)
        if (cc == 0).any():
            raise ValueError(f"Training data must contain both classes {{0, 1}}; counts={cc.tolist()}")
        self.log_prior_ = np.log(cc / cc.sum())
        self.prior_probs_ = cc / cc.sum()
        self.weights_ = np.ones(len(n_categories)) if feature_weights is None else np.asarray(feature_weights, float)
        self.tables_ = []
        for f, K in enumerate(n_categories):
            col = X[:, f]
            real = col != (K - 1)
            cnt = np.vstack([np.bincount(col[(y == c) & real], minlength=K)[: K - 1] for c in range(C)]).astype(float)
            n_real = cnt.sum(1)
            tot = cnt.sum(0)
            if self.smoothing == "laplace":
                p = (cnt + self.alpha) / (n_real[:, None] + self.alpha * (K - 1))
            else:
                marg = (tot + 1.0) / (tot.sum() + (K - 1))
                p = (cnt + self.alpha * marg) / (n_real[:, None] + self.alpha)
            ll = np.zeros((K, C))
            ll[: K - 1] = np.log(p).T
            ll[: K - 1][tot == 0] = 0.0
            if self.feat_bound > 0.0:
                for k in range(K - 1):
                    ll[k] = np.clip(ll[k] - np.mean(ll[k]), -self.feat_bound, self.feat_bound)
            if self.max_missing_log_odds > 0.0 and preprocessor is not None:
                miss = preprocessor.missing_category(preprocessor.features[f])
                if miss is not None and miss < K - 1:
                    ll[miss] = np.clip(ll[miss] - np.mean(ll[miss]), -self.max_missing_log_odds, self.max_missing_log_odds)
            self.tables_.append(ll)
        return self

    def log_joint(self, X):
        acc = np.zeros((X.shape[0], self.n_internal_))
        for f, ll in enumerate(self.tables_):
            acc += self.weights_[f] * ll[X[:, f]]
        return self.log_prior_ + self.lik_scale * acc

    def predict_proba(self, X):
        z = self.log_joint(X)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)


class SubjectTokenNB:
    def __init__(self, min_df=10, max_vocab=3000, alpha=0.5, clip=2.0):
        self.min_df, self.max_vocab, self.alpha, self.clip = int(min_df), int(max_vocab), float(alpha), float(clip)

    @staticmethod
    def _docs(texts):
        return [t.split() if isinstance(t, str) else [] for t in texts]

    def _matrix(self, docs):
        rows, cols = [], []
        for i, toks in enumerate(docs):
            for t in set(toks):
                j = self.vocab_.get(t)
                if j is not None:
                    rows.append(i); cols.append(j)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(self.vocab_)))

    def fit(self, texts, y, n_classes=N_CLASSES):
        docs = self._docs(texts)
        df_count = {}
        for d in docs:
            for t in set(d):
                df_count[t] = df_count.get(t, 0) + 1
        items = sorted(((t, c) for t, c in df_count.items() if c >= self.min_df), key=lambda x: (-x[1], x[0]))
        items = items[: self.max_vocab]
        self.vocab_ = {t: i for i, (t, _) in enumerate(items)}
        self.n_classes_ = int(n_classes)
        V = len(self.vocab_)
        if V == 0:
            self.L_ = np.zeros((0, self.n_classes_))
            return self
        X = self._matrix(docs)
        y = np.asarray(y, dtype=int)
        Y = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(self.n_classes_, len(y)))
        cnt = np.asarray((Y @ X).todense())
        tot = cnt.sum(1, keepdims=True)
        L = np.log((cnt + self.alpha) / (tot + self.alpha * V)).T
        L = L - L.mean(1, keepdims=True)
        self.L_ = np.clip(L, -self.clip, self.clip)
        return self

    def contribution(self, texts):
        n_docs = len(texts)
        if len(self.vocab_) == 0:
            return np.zeros((n_docs, self.n_classes_))
        X = self._matrix(self._docs(texts))
        n = np.asarray(X.sum(1)).ravel()
        return np.asarray(X @ self.L_) / np.sqrt(np.maximum(n, 1.0))[:, None]


def _finalize(p):
    p = np.clip(p, PROB_FLOOR, 1.0 - PROB_FLOOR)
    return p / p.sum(1, keepdims=True)


def prob_to_logit(p1):
    p1 = np.clip(np.asarray(p1, dtype=float), PROB_FLOOR, 1.0 - PROB_FLOOR)
    return np.log(p1 / (1.0 - p1))


def clipped_logit(p1, clip=LOGIT_CLIP):
    return np.clip(prob_to_logit(p1), -float(clip), float(clip))


def apply_temperature(P, T):
    logit = np.clip(prob_to_logit(P[:, 1]) / float(T), -50.0, 50.0)
    p1 = 1.0 / (1.0 + np.exp(-logit))
    return _finalize(np.column_stack([1.0 - p1, p1]))


def fit_temperature(P, y):
    from sklearn.metrics import log_loss
    res = minimize_scalar(lambda t: log_loss(y, apply_temperature(P, t), labels=[0, 1]),
                          bounds=(0.2, 10.0), method="bounded")
    return float(res.x)


def binary_stats_at_preds(y, pred):
    y = np.asarray(y, dtype=int); pred = np.asarray(pred, dtype=int)
    tp = float(((y == 1) & (pred == 1)).sum())
    tn = float(((y == 0) & (pred == 0)).sum())
    fp = float(((y == 0) & (pred == 1)).sum())
    fn = float(((y == 1) & (pred == 0)).sum())
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1 = (2 * prec1 * rec1 / (prec1 + rec1)) if (prec1 + rec1) > 0 else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2 * prec0 * rec0 / (prec0 + rec0)) if (prec0 + rec0) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return dict(f1_1=f1_1, f1_0=f1_0, macro_f1=0.5 * (f1_0 + f1_1), prec=prec1, rec=rec1, fpr=fpr)


def threshold_table(P, y, grid=np.linspace(-2.5, 2.5, 101)):
    logit = prob_to_logit(P[:, 1])
    y = np.asarray(y, dtype=int)
    f1s, mf1s, prs, rcs, fprs = [], [], [], [], []
    for t in grid:
        st = binary_stats_at_preds(y, (logit >= t).astype(int))
        f1s.append(st["f1_1"]); mf1s.append(st["macro_f1"])
        prs.append(st["prec"]); rcs.append(st["rec"]); fprs.append(st["fpr"])
    return dict(thresholds=np.asarray(grid, float), f1=np.asarray(f1s), macro_f1=np.asarray(mf1s),
                precision=np.asarray(prs), recall=np.asarray(rcs), fpr=np.asarray(fprs))


def pick_threshold(tab, mode="f1_opt", min_precision=0.70):
    ts = tab["thresholds"]
    mag = np.abs(ts)
    if mode == "argmax":
        return 0.0, dict(mode_used="argmax", feasible=True, threshold_prob=0.5,
                         f1=float("nan"), macro_f1=float("nan"),
                         precision=float("nan"), recall=float("nan"), fpr=float("nan"))
    n = len(ts)
    i_f1 = max(range(n), key=lambda i: (round(tab["f1"][i], 9), -mag[i]))
    i_mf1 = max(range(n), key=lambda i: (round(tab["macro_f1"][i], 9), -mag[i]))
    chosen, used, feas = (i_mf1 if mode == "macro_f1" else i_f1), ("macro_f1" if mode == "macro_f1" else "f1_opt"), True
    if mode == "recall_at_precision":
        ok = [i for i in range(n) if tab["precision"][i] >= min_precision]
        if ok:
            chosen = max(ok, key=lambda i: (round(tab["recall"][i], 9), round(tab["f1"][i], 9), -mag[i]))
            used = "recall_at_precision"
        else:
            feas = False
    t_star = float(ts[chosen])
    p_star = float(1.0 / (1.0 + np.exp(-t_star)))
    return t_star, dict(mode_used=used, feasible=feas, threshold_prob=p_star,
                        f1=float(tab["f1"][chosen]), macro_f1=float(tab["macro_f1"][chosen]),
                        precision=float(tab["precision"][chosen]), recall=float(tab["recall"][chosen]),
                        fpr=float(tab["fpr"][chosen]))


def tune_threshold(P, y, grid=np.linspace(-2.5, 2.5, 101), mode="f1_opt", min_precision=0.70):
    t, info = pick_threshold(threshold_table(P, y, grid), mode, min_precision)
    return t, info["f1"]


class HeaderAgent:
    def __init__(self, features=None, n_bins=24, alpha=1.0, lik_scale=1.0,
                 max_missing_log_odds=1.5, feat_bound=2.5, use_missingness_features=True,
                 group_weighting="flat", subject_tokens=False, token_weight=1.0, smoothing="marginal"):
        raw = list(features) if features is not None else ALL_FEATURES
        self.use_missingness_features = bool(use_missingness_features)
        self.features = raw if self.use_missingness_features else [f for f in raw if f not in MISSINGNESS_FEATURES]
        self.n_bins, self.alpha, self.lik_scale = int(n_bins), float(alpha), float(lik_scale)
        self.max_missing_log_odds, self.feat_bound = float(max_missing_log_odds), float(feat_bound)
        self.group_weighting, self.subject_tokens = group_weighting, bool(subject_tokens)
        self.token_weight, self.smoothing = float(token_weight), smoothing
        self.temperature = 1.0
        self.threshold_logit = 0.0
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = False

    def _weights(self):
        if self.group_weighting != "inv_group":
            return np.ones(len(self.features))
        sizes = {g: sum(f in self.features for f in fs) for g, fs in FEATURE_GROUPS.items()}
        mean_n = np.mean([n for n in sizes.values() if n > 0])
        w = []
        for f in self.features:
            g = next(g for g, fs in FEATURE_GROUPS.items() if f in fs)
            w.append(np.sqrt(mean_n / sizes[g]))
        return np.array(w)

    def fit_features(self, Xf, y):
        y = np.asarray(y, dtype=int)
        self.preprocessor_ = HeaderPreprocessor(self.features, self.n_bins).fit(Xf)
        Z = self.preprocessor_.transform(Xf)
        raw = np.bincount(y, minlength=N_CLASSES).astype(float)
        self.training_prior_ = raw[:N_CLASSES] / raw.sum()
        self.model_ = SmoothedCategoricalNB(
            alpha=self.alpha, lik_scale=self.lik_scale,
            max_missing_log_odds=self.max_missing_log_odds, feat_bound=self.feat_bound,
            smoothing=self.smoothing).fit(Z, y, self.preprocessor_.n_categories_, self.preprocessor_, self._weights())
        self.token_nb_ = None
        if self.subject_tokens:
            self.token_nb_ = SubjectTokenNB().fit(Xf["u_norm"].tolist(), y, N_CLASSES)
        self.temperature, self.threshold_logit = 1.0, 0.0
        self.is_fitted_ = True
        return self

    def set_temperature(self, T):
        self.temperature = float(T)

    def set_threshold_logit(self, t):
        self.threshold_logit = float(t)

    def proba_features(self, Xf, calibrated=True):
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted.")
        lj = self.model_.log_joint(self.preprocessor_.transform(Xf))
        if self.token_nb_ is not None:
            lj = lj + self.token_weight * self.token_nb_.contribution(Xf["u_norm"].tolist())
        lj = lj - lj.max(1, keepdims=True)
        p = np.exp(lj)
        p = _finalize(p / p.sum(1, keepdims=True))
        return apply_temperature(p, self.temperature) if calibrated else p

    def predict_label_features(self, Xf):
        P = self.proba_features(Xf, calibrated=True)
        return (prob_to_logit(P[:, 1]) >= self.threshold_logit).astype(int)

    @staticmethod
    def completeness(Xf):
        s = Xf["s_missing"].to_numpy().astype(int)
        r = Xf["r_missing"].to_numpy().astype(int)
        d = (Xf["d_date_status"] == 0).to_numpy().astype(int)
        u = Xf["u_missing"].to_numpy().astype(int)
        return ((1 - s) + (1 - r) + (1 - d) + (1 - u)) / 4.0

    def detailed_features(self, Xf, index=None):
        P = self.proba_features(Xf, calibrated=True)
        comp = self.completeness(Xf)
        nb_score = P[:, 1]
        return pd.DataFrame({
            "p_safe": P[:, 0],
            "nb_score": nb_score,
            "nb_logit": clipped_logit(nb_score),
            "pred_class": (prob_to_logit(nb_score) >= self.threshold_logit).astype(int),
            "header_completeness": comp,
            "insufficient_evidence": (comp <= 0.5).astype(int)},
            index=Xf.index if index is None else index)

    def fit(self, X_emails, y):
        return self.fit_features(extract_features(_ensure_df(X_emails)), y)

    def predict_proba(self, X_emails, calibrated=True):
        df = _ensure_df(X_emails)
        return pd.DataFrame(self.proba_features(extract_features(df), calibrated), index=df.index, columns=list(CLASS_NAMES))

    def predict_detailed(self, X_emails):
        df = _ensure_df(X_emails)
        return self.detailed_features(extract_features(df), df.index)

    def save(self, path):
        import joblib
        joblib.dump({"agent": self, "version": __version__}, path)

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)["agent"]


def score_emails(agent, emails):
    p = agent.predict_proba(emails, calibrated=True).to_numpy(dtype=float)
    return np.clip(p[:, 1], 0.0, 1.0)


def score_email(agent, email):
    return float(score_emails(agent, [email])[0])

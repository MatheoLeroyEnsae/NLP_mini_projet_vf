"""
analysis_dataset.py
===================

The `analysis_dataset` module is a utility module that gathers all the functions for **exploratory data analysis**
The notebook ``00_analysis.ipynb`` is the main driver of the EDA.

Table of Contents
--------
1. Mapping party -> family + normalisation
2. Graphical configuration (palettes, rcParams)
3. Section 2.1 — Overview of the corpus
4. Section 2.2 — Length distributions
5. Section 2.3 — Global vocabulary (top unigrams / bigrams)
6. Section 2.4 — Lexical density
7. Section 2.5 — Wordclouds by year
8. Section 2.6 — TF-IDF: distinctive terms by year
9. Section 2.7 — Descriptive analyses on metadata

Author : Mathéo LEROY
"""
from __future__ import annotations

import time
import re
from networkx import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langdetect import detect, DetectorFactory, LangDetectException
from collections import Counter
from typing import List, Optional, Tuple, Dict
from matplotlib.colors import LinearSegmentedColormap

DetectorFactory.seed = 42

# ============================================================================
# 0.  Mapping party -> family + normalisation   
# ============================================================================
MAPPING_FILE_PATH = "/Users/matheoleroy/Downloads/data_nlp/mapping_partis_familles_1.txt"
party_to_family: dict[str, str] = {}

YEAR_PALETTE = {
    "1981": "#c1432b",   
    "1988": "#d9a441",   
    "1993": "#2c5f7c",   
}

PARTY_PALETTE: Dict[str, str] = {
    # Extreme left-wing
    "EXTREME_GAUCHE":   "#6b1a1a",  
    "COMMUNISTE":       "#a01818",  

    # Left-wing parties
    "SOCIALISTE":       "#e63946",  
    "COALITION_GAUCHE": "#c97064",  
    # Green parties
    "ECOLOGISTE":       "#3a8a4a",  

    # Center parties
    "CENTRE":           "#f4c430",  
    "CENTRE_DROIT":     "#5da9d6",  

    # Right-wing parties
    "DROITE":           "#1f4e79",  
    "EXTREME_DROITE":   "#3b3b3b",  

    # Other / unknown
    "DIVERS":           "#9a9a9a",  
    "UNKNOWN":          "#cccccc",  
}

GERMAN_STOPWORDS = {
    "die", "der", "das", "den", "dem", "des",
    "und", "oder", "aber", "denn", "weil", "dass",
    "ist", "sind", "war", "waren", "sein", "wird", "werden", "wurde",
    "haben", "hat", "hatte", "habe",
    "von", "für", "mit", "auf", "aus", "bei", "nach", "über", "unter",
    "zu", "zur", "zum", "im", "ins", "vom", "beim",
    "ich", "du", "er", "sie", "es", "wir", "ihr",
    "mein", "dein", "sein", "ihr", "unser", "euer",
    "ein", "eine", "einen", "einem", "einer", "eines",
    "nicht", "kein", "keine", "auch", "noch", "schon", "nur",
    "wenn", "als", "wie", "was", "wer", "wo", "warum",
    "diese", "dieser", "dieses", "jene", "jeder", "alle",
    "sich", "man", "uns", "euch", "ihnen",
}

WORD_RE = re.compile(r"\b[a-zäöüßA-ZÄÖÜ]+\b")

PATH = "/Users/matheoleroy/Downloads/data_nlp/figures"


def party_color(party: Optional[str], fallback: str = "#9a9a9a") -> str:
    """Return the color associated with a political party, or a fallback if the party is unknown."""
    if party is None:
        return fallback
    key = str(party).upper().strip()
    return PARTY_PALETTE.get(key, fallback)


def load_party_mapping(path: str = MAPPING_FILE_PATH) -> dict[str, str]:
    """
    Load the party -> family mapping from a TXT file.
    
    The result is cached in a global variable of the module.
    """
    global party_to_family
    
    mapping: dict[str, str] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '\t' in line:
                label_brut, famille = line.split('\t', 1)
                mapping[label_brut.strip().lower()] = famille.strip()
    
    party_to_family = mapping
    return party_to_family


def normalize_party(soutien)-> str:
    if soutien is None:
        return "UNKNOWN"
    
    s = str(soutien).strip().lower()
    
    if not s or s in {"non mentionné", "nan", "none"}:
        return "UNKNOWN"
    
    if s in party_to_family:
        return party_to_family[s]
    

    return "UNKNOWN"


# ============================================================================
# 1. Graphical configuration (palettes, rcParams) + dataset loading
# ============================================================================

def setup_matplotlib():
    """Set up a consistent graphical style for all the plots in the EDA."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfaf7",
        "axes.edgecolor": "#2b2b2b",
        "axes.labelcolor": "#2b2b2b",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e1d6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": "#2b2b2b",
        "ytick.color": "#2b2b2b",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })
    sns.set_palette(list(YEAR_PALETTE.values()))
    print("Matplotlib configured with custom style and palettes.")


def setup_nltk(download: bool = True):
    """Download the necessary NLTK resources for tokenization and stopwords."""
    import ssl
    import nltk
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    if download:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("stopwords", quiet=True)


def load_dataset(parquet_path: str) -> pd.DataFrame:
    """Load the parquet file, normalize the `date` column and automatically construct
    the `party` column from `titulaire-soutien` by applying the normalization rules defined in the module.
    """
    df = pd.read_parquet(parquet_path, engine="fastparquet")
    df["date"] = df["date"].astype(str).str.strip()

    load_party_mapping()

    df["party"] = df["titulaire-soutien"].apply(normalize_party).astype(str).str.upper().str.strip()

    print(f"Dataset loaded : {df.shape[0]} documents, {df.shape[1]} columns.")
    print("\nParty distribution (after normalization) :")
    print(df["party"].value_counts())
    return df

def german_stopword_ratio(text: str) -> float:
    """Proportion de tokens qui sont des stopwords allemands."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = [t.lower() for t in WORD_RE.findall(text)]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in GERMAN_STOPWORDS)
    return hits / len(tokens)


def safe_detect(text: str) -> str:
    """Détection de langue robuste (renvoie 'unknown' en cas d'échec)."""
    if not isinstance(text, str) or len(text.strip()) < 20:
        return "unknown"
    try:
        # On limite à 2000 caractères : largement suffisant et plus rapide
        return detect(text[:2000])
    except LangDetectException:
        return "unknown"


def flag_german_documents(df: pd.DataFrame,
                          text_col: str = "text",
                          stopword_threshold: float = 0.05) -> pd.DataFrame:
    """
    Add columns to the DataFrame to flag documents that are likely in German.
        - lang          : language detected by langdetect
        - german_score  : proportion of German stopwords in the text
        - is_german     : True if lang == 'de' OR german_score >= threshold
    The double criterion avoids both types of errors:
        - langdetect can be wrong on short or bilingual texts
          → the stopword ratio catches these cases
        - a French text that cites some German words will have a low ratio
            → we don't mark it as German by mistake

    """
    print(f"Number of documents: {len(df)}")

    df = df.copy()
    df["lang"] = df[text_col].apply(safe_detect)
    df["german_score"] = df[text_col].apply(german_stopword_ratio)
    df["is_german"] = (df["lang"] == "de") | (df["german_score"] >= stopword_threshold)

    return df

def _german_ratio(text: str) -> float:
    tokens = [t.lower() for t in WORD_RE.findall(text)]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in GERMAN_STOPWORDS) / len(tokens)
 
 
def _is_french_segment(segment: str,
                       min_chars: int = 30,
                       german_threshold: float = 0.05) -> bool:
    """
    Decide if a text segment is in French or not, based on its length, the output of langdetect and the ratio of German stopwords.
    Strategy:
        - Very short segments (< min_chars) : we keep by default
            (acronyms, slogans, proper names: not enough signal to decide)
        - Otherwise, double criterion:
            * langdetect must say 'fr' (or anything other than 'de')
            * ratio of German stopwords < threshold
    """
    s = segment.strip()
    if len(s) < min_chars:
        return True  
 
    # Critère 1 : ratio de stopwords allemands
    if _german_ratio(s) >= german_threshold:
        return False
 
    # Critère 2 : langdetect
    try:
        lang = detect(s)
    except LangDetectException:
        return True  # incertain → on garde
    return lang != "de"
 
 
def filter_french_only(text: str,
                       min_chars: int = 30,
                       german_threshold: float = 0.05) -> str:
    """
    Remove German segments from the text, keeping only the French parts.
    2 steps segmentation:
        1. By paragraphs (blocks separated by empty lines)
        2. If a whole paragraph is rejected, we stop there.
           Otherwise we look sentence by sentence inside it.
    """
    if not isinstance(text, str) or not text.strip():
        return text
 
    kept_paragraphs = []
    paragraphs = re.split(r"\n\s*\n", text)
 
    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue
 
        if _is_french_segment(para_clean, min_chars, german_threshold):
            kept_paragraphs.append(para_clean)
            continue
 
        
        sentences = re.split(r"(?<=[.!?])\s+", para_clean)
        kept_sentences = [
            s for s in sentences
            if _is_french_segment(s, min_chars, german_threshold)
        ]
        if kept_sentences:
            kept_paragraphs.append(" ".join(kept_sentences))
 
    return "\n\n".join(kept_paragraphs)
 
 
def add_french_only_column(df: pd.DataFrame,
                           text_col: str = "text",
                           only_flagged: bool = True,
                           flag_col: str = "is_german") -> pd.DataFrame:
    """
    Add a column `text_fr` to the DataFrame, where German segments have been removed from the original `text` column.
        Also add a column `removed_ratio` indicating the proportion of text that was removed.
        
        If `only_flagged` is True, we only apply the cleaning to documents that were flagged as German by `flag_german_documents`.
        This allows to save time by not processing documents that are very likely in French.
        If False, we apply the cleaning to all documents (which can be useful for a thorough analysis but takes more time).
    """
    df = df.copy()
 
    if only_flagged and flag_col in df.columns:
        mask = df[flag_col].fillna(False).astype(bool)
        print(f"Filtrage de l'allemand sur {mask.sum()} documents marqués...")
        df["text_fr"] = df[text_col]
        df.loc[mask, "text_fr"] = df.loc[mask, text_col].apply(filter_french_only)
    else:
        print(f"Filtrage de l'allemand sur {len(df)} documents...")
        df["text_fr"] = df[text_col].apply(filter_french_only)
 
    orig_len = df[text_col].fillna("").str.len()
    new_len = df["text_fr"].fillna("").str.len()
    df["removed_ratio"] = 1 - (new_len / orig_len.where(orig_len > 0, 1))
 
    return df

# ============================================================================
# 2.  Section 2.1 — Overview of the corpus
# ============================================================================

def plot_top_titulaire_party_pareto(df: pd.DataFrame, top_n: int = 5):
    """Figure — Top `top_n` modalities of `titulaire-soutien` (left) and `party`
    (right), as bar plots, with a cumulative curve (% of the global corpus) on
    a secondary Y axis.

    The Pareto curve relates cumulative counts to the TOTAL number of non-null
    documents in the column, not just to the cumulative sum of the top N — it
    therefore indicates the share of the corpus covered by the top modalities.
    """
    cols = ["titulaire-soutien", "party"]

    fig, axes = plt.subplots(1, 2, figsize=(19, 5.2))

    titles = [
        f"Top {top_n} — titulaire-soutien (raw)",
        f"Top {top_n} — party (normalized)",
    ]

    for ax, col, title in zip(axes, cols, titles):
        # Cleaning: drop empty values / "non mentionné" / "UNKNOWN"
        s = (df[col].astype(str).str.strip()) # .replace({"": np.nan, "nan": np.nan, "None": np.nan, "non mentionné": np.nan, "UNKNOWN": np.nan}))        
        s = s.dropna()

        total = len(s)
        vc = s.value_counts().head(top_n)
        cum_pct = vc.cumsum() / total * 100

        # Colors: party palette for the `party` column, neutral tone otherwise
        if col == "party":
            colors = [party_color(p) for p in vc.index]
        else:
            colors = ["#2c5f7c"] * len(vc)

        # Bar plot (left axis)
        x = np.arange(len(vc))
        bars = ax.bar(x, vc.values, color=colors, edgecolor="white", width=0.7)
        for rect, v in zip(bars, vc.values):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    v + max(vc.values) * 0.01,
                    f"{v}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(vc.index, rotation=90, ha="right")
        ax.set_ylabel("Number of documents")
        ax.set_ylim(0, max(vc.values) * 1.18)
        ax.set_title(title, loc="left", fontsize=12, pad=10)

        # Cumulative curve (right axis, in %)
        ax2 = ax.twinx()
        ax2.plot(x, cum_pct.values, "o-", color="#c1432b",
                 lw=1.8, markersize=6, label="Cumulative % of corpus")
        for xi, pct in zip(x, cum_pct.values):
            ax2.text(xi, pct + 2, f"{pct:.1f}%",
                     ha="center", va="bottom", fontsize=8.5,
                     color="#c1432b", fontweight="bold")
        ax2.set_ylabel("Cumulative share of corpus (%)", color="#c1432b")
        ax2.tick_params(axis="y", colors="#c1432b")
        ax2.set_ylim(0, 105)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)


    fig.suptitle(f"Top {top_n} modalities and cumulative coverage of the corpus",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PATH}/pareto.pdf", bbox_inches="tight", dpi=300)
    plt.show()

def overview(df: pd.DataFrame, n_top_parties: int = 12):
    """Show an overview of the dataset: shape, missing values, distribution of documents by year and by party."""
    from IPython.display import display as ipy_display

    print("=" * 70)
    print(f"  Documents      : {len(df):>6}")
    print(f"  Columns       : {list(df.columns)}")
    print("=" * 70)
    print("\nMissing values :")
    ipy_display(df.isnull().sum().to_frame("n_missing").T)

    n_per_year = df["date"].value_counts().sort_index()
    print("\nDocuments per year :")
    for y, n in n_per_year.items():
        print(f"  {y} : {n:>5}  {'█' * int(n/n_per_year.max()*40)}")

    

# ============================================================================
# 3.  Section 2.2 — Length distributions
# ============================================================================

def compute_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Add (in place) the columns ``text_length`` and ``word_count``.

    Returns also a small `describe` for inspection."""
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    summary = df[["text_length", "word_count"]].describe().round(1)
    print(summary)
    return summary


def plot_length_distribution(df: pd.DataFrame):
    """ Figure 1 : global histogram + density plots by year (excluding P99 outliers)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

    # (a) Global histogram with median / 95th percentile annotations
    ax = axes[0]
    med = df["word_count"].median()
    p95 = df["word_count"].quantile(0.95)
    ax.hist(df["word_count"], bins=60, color="#2c5f7c", alpha=0.85,
            edgecolor="white")
    ax.axvline(med, color="#c1432b", lw=1.6, ls="--",
               label=f"MMedian = {med:.0f}")
    ax.axvline(p95, color="#d9a441", lw=1.6, ls="--",
               label=f"P95 = {p95:.0f}")
    ax.set_xlabel("Number of words per document")
    ax.set_ylabel("Number of documents")
    ax.set_title("Distribution of lengths (complete corpus)", loc="left")
    ax.legend()

    # (b) Densities by year (excluding extreme outliers for readability)
    ax = axes[1]
    mask = df["word_count"] < df["word_count"].quantile(0.99)
    for year in sorted(df["date"].unique()):
        sub = df.loc[mask & (df["date"] == year), "word_count"]
        if len(sub) == 0:
            continue
        sns.kdeplot(sub, ax=ax, fill=True, alpha=0.35,
                    color=YEAR_PALETTE.get(year, "#888"),
                    label=year, linewidth=1.8, common_norm=False)
    ax.set_xlabel("Number of words per document")
    ax.set_ylabel("Density")
    ax.set_title("Distribution by year (excluding P99)", loc="left")
    ax.legend(title="Year")

    fig.suptitle("Length of Speeches", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PATH}/length_speeches.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_length_by_party_year(df: pd.DataFrame, top_n_parties: int = 8):
    """Figure 2 : median length by party and by year (heatmap)."""
    if "party" not in df.columns:
        print("Column 'party' absent — graph skipped.")
        return

    top_parties = df["party"].value_counts().head(top_n_parties).index.tolist()
    sub = df[df["party"].isin(top_parties)]

    pivot = (sub.groupby(["party", "date"])["word_count"]
                .median()
                .unstack("date")
                .reindex(top_parties))

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(top_parties) + 1.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlBu_r",
                cbar_kws={"label": "Median length (words)"},
                linewidths=0.6, linecolor="white", ax=ax)
    ax.set_title("Median length of speeches (words) — party × year",
                 loc="left", pad=12)
    ax.set_xlabel("Year"); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PATH}/median_length.pdf", bbox_inches="tight", dpi=300)
    plt.show()

def plot_doc_share_by_party_year(df: pd.DataFrame, top_n_parties: int = 8):
    """Figure — temporal distribution of documents per party (% of each
    party's own total). Each row sums to 100%, showing in which year a
    party was most active relative to its own total output."""
    if "party" not in df.columns:
        print("Column 'party' absent — graph skipped.")
        return

    top_parties = df["party"].value_counts().head(top_n_parties).index.tolist()
    sub = df[df["party"].isin(top_parties)]

    counts = (sub.groupby(["party", "date"])
                 .size()
                 .unstack("date", fill_value=0)
                 .reindex(top_parties))

    # Row-wise normalization: each party sums to 100% across years
    pivot_pct = counts.div(counts.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(top_parties) + 1.5))
    sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="YlOrBr",
                cbar_kws={"label": "% of party's total documents"},
                linewidths=0.6, linecolor="white", ax=ax,
                vmin=0, vmax=100)
    ax.set_title("Temporal distribution of documents per party ",
                 loc="left", pad=12)
    ax.set_xlabel("Year"); ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig("/Users/matheoleroy/Downloads/data_nlp/figures/temporal_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    # Optional: print raw counts alongside, useful for context
    print("\nRaw document counts (party × year):")
    print(counts.to_string())


# ============================================================================
# 4.  Section 2.3 — Global vocabulary
# ============================================================================

def quick_tokenize_corpus(df: pd.DataFrame) -> Tuple[Counter, set]:
    """Quick tokenization via NLTK : lowercase, alphabetical,
    removal of French stopwords, tokens ≥ 3 characters.

    Adds a ``tokens`` column to the DataFrame.

    Returns
    -------
    word_freq : Counter
        Global word counter.
    fr_stopwords : set
        Set of French stopwords used by NLTK (reused for wordclouds and distinct TF-IDF).
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    fr_stopwords = set(stopwords.words("french"))

    def quick_tokenize(text):
        toks = word_tokenize(str(text).lower(), language="french")
        return [w for w in toks if w.isalpha()
                and w not in fr_stopwords and len(w) > 2]

    t0 = time.time()
    df["tokens"] = df["text"].apply(quick_tokenize)
    print(f"Tokenisation NLTK : {time.time()-t0:.1f}s")

    all_words = [w for toks in df["tokens"] for w in toks]
    word_freq = Counter(all_words)
    print(f"Global vocabulary : {len(word_freq):,} unique words | "
          f"{len(all_words):,} tokens.")
    return word_freq, fr_stopwords


def plot_top_words_and_bigrams(df: pd.DataFrame, word_freq: Counter,
                               fr_stopwords: set, n: int = 20):
    """Figure 3 : top n unigrams and bigrams side by side."""
    from sklearn.feature_extraction.text import CountVectorizer

    top_words = word_freq.most_common(n)
    words, counts = zip(*top_words)

    bg_vec = CountVectorizer(ngram_range=(2, 2),
                             stop_words=list(fr_stopwords),
                             min_df=5, max_features=2000)
    X_bg = bg_vec.fit_transform(df["text"])
    bg_sums = X_bg.sum(axis=0).A1
    bg_pairs = sorted(zip(bg_vec.get_feature_names_out(), bg_sums),
                      key=lambda x: -x[1])[:n]
    bg_words, bg_counts = zip(*bg_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    ax = axes[0]
    bars = ax.barh(range(len(words)), counts, color="#2c5f7c", edgecolor="white")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Number of occurrences")
    ax.set_title(f"Top {n} unigrams", loc="left")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{c:,}", va="center", fontsize=9, color="#2b2b2b")

    ax = axes[1]
    bars = ax.barh(range(len(bg_words)), bg_counts,
                   color="#c1432b", edgecolor="white")
    ax.set_yticks(range(len(bg_words)))
    ax.set_yticklabels(bg_words)
    ax.invert_yaxis()
    ax.set_xlabel("Number of occurrences")
    ax.set_title(f"Top {n} bigrams", loc="left")
    for bar, c in zip(bars, bg_counts):
        ax.text(bar.get_width() + max(bg_counts) * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{c:,}", va="center", fontsize=9, color="#2b2b2b")

    fig.suptitle("Global vocabulary of the corpus", fontsize=15,
                 fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 5.  Section 2.4 — Lexical density
# ============================================================================

def compute_lexical_density(df: pd.DataFrame):
    """Lexical density = unique vocabulary / total number of words.
    The higher it is, the less repetitive the text is.

    Modifies ``df`` in place (columns ``unique_words``, ``lexical_density``).
    """
    df["unique_words"] = df["tokens"].apply(lambda x: len(set(x)))
    df["lexical_density"] = np.where(df["word_count"] > 0,
                                     df["unique_words"] / df["word_count"],
                                     np.nan)
    print(df["lexical_density"].describe().round(3))


def plot_lexical_density(df: pd.DataFrame, top_n_parties: int = 8):
    """Figure 4 : violins (by year) + boxplots (by party)."""
    has_party = "party" in df.columns
    n_panels = 2 if has_party else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # (a) Violins by year
    ax = axes[0]
    order = sorted(df["date"].unique())
    parts = ax.violinplot(
        [df.loc[df["date"] == y, "lexical_density"].dropna() for y in order],
        showmeans=False, showmedians=True, widths=0.8,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(YEAR_PALETTE.get(order[i], "#888"))
        pc.set_alpha(0.7); pc.set_edgecolor("#2b2b2b")
    for k in ("cmins", "cmaxes", "cbars", "cmedians"):
        if k in parts:
            parts[k].set_color("#2b2b2b")
    ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
    ax.set_xlabel("Year"); ax.set_ylabel("Lexical density")
    ax.set_title("Lexical density by year", loc="left")

    # (b) Boxplots by party
    if has_party:
        ax = axes[1]
        top_parties = df["party"].value_counts().head(top_n_parties).index.tolist()
        sub = df[df["party"].isin(top_parties)]
        box_data = [sub.loc[sub["party"] == p, "lexical_density"].dropna()
                    for p in top_parties]
        bp = ax.boxplot(box_data, labels=top_parties, patch_artist=True,
                        showfliers=False, widths=0.6,
                        medianprops=dict(color="white", linewidth=1.6))
        for patch, p in zip(bp["boxes"], top_parties):
            patch.set_facecolor(party_color(p)); patch.set_alpha(0.85)
        ax.set_xlabel("Party"); ax.set_ylabel("Lexical density")
        ax.set_title(f"Lexical density by party (top {top_n_parties})", loc="left")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Richness of the vocabulary", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PATH}/lexical_density.pdf", bbox_inches="tight", dpi=300)
    plt.show()


# ============================================================================
# 6.  Section 2.5 — Wordclouds per year
# ============================================================================

def plot_wordclouds_by_year(df: pd.DataFrame, fr_stopwords: set,
                            max_words: int = 140, extra_stopwords: Optional[set] = None):
    """Figure 5 : a wordcloud per year (3 panels)."""
    from wordcloud import WordCloud
    default_extra = {
        "sciences", "po", "sciencespo",
        "fonds", "fond",
        "cevipof",
    }
    if extra_stopwords:
        default_extra |= {w.lower() for w in extra_stopwords}

    # Merge without mutating the caller's set
    stopwords_full = set(fr_stopwords) | default_extra


    years = sorted(df["date"].unique())
    fig, axes = plt.subplots(1, len(years), figsize=(5.5 * len(years), 4.8))
    if len(years) == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        text = " ".join(df.loc[df["date"] == year, "text"]
                          .astype(str).tolist())
        base = YEAR_PALETTE.get(year, "#2c5f7c")
        cmap = LinearSegmentedColormap.from_list(
            f"cmap_{year}", ["#1a1a1a", base, "#f4d35e"]
        )
        wc = WordCloud(width=900, height=620, background_color="#0d0d0d",
                       stopwords=stopwords_full, colormap=cmap,
                       max_words=max_words, prefer_horizontal=0.95,
                       relative_scaling=0.45).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"  {year}", loc="left", fontsize=18,
                     color=base, fontweight="bold", pad=10)

    fig.suptitle("Lexical Field: Dominant Lexical Champ per Scrutiny", fontsize=15,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PATH}/wordclouds_by_year.pdf", bbox_inches="tight", dpi=300)
    plt.show()


# ============================================================================
# 7.  Section 2.6 — Distinct TF-IDF per year
# ============================================================================

def plot_tfidf_distinctive_words(df: pd.DataFrame, fr_stopwords: set,
                                 n: int = 15, max_features: int = 3000):
    """Figure 6 : Terms most distinctive by year (TF-IDF surplus).

    For each year, we calculate the average TF-IDF of the terms in that
    year, subtract the average TF-IDF of the terms in the rest of the
    corpus, and retain the ``n`` largest surpluses.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_eda = TfidfVectorizer(max_features=max_features,
                                stop_words=list(fr_stopwords),
                                min_df=5, max_df=0.95)
    X_tfidf = tfidf_eda.fit_transform(df["text"])
    terms = np.array(tfidf_eda.get_feature_names_out())

    def top_distinctive(year, n_top=n):
        mask = (df["date"] == year).values
        if mask.sum() == 0:
            return []
        mean_year = X_tfidf[mask].mean(axis=0).A1
        mean_other = X_tfidf[~mask].mean(axis=0).A1
        delta = mean_year - mean_other
        idx = np.argsort(-delta)[:n_top]
        return list(zip(terms[idx], mean_year[idx], delta[idx]))

    years = sorted(df["date"].unique())
    fig, axes = plt.subplots(1, len(years),
                             figsize=(5.5 * len(years), 5.5), sharex=False)
    if len(years) == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        rows = top_distinctive(year, n_top=n)
        if not rows:
            ax.axis("off"); continue
        ws, _, deltas = zip(*rows)
        color = YEAR_PALETTE.get(year, "#2c5f7c")
        ax.barh(range(len(ws)), deltas, color=color,
                edgecolor="white", alpha=0.9)
        ax.set_yticks(range(len(ws))); ax.set_yticklabels(ws)
        ax.invert_yaxis()
        ax.set_xlabel("TF-IDF surplus vs rest of corpus")
        ax.set_title(year, loc="left", color=color, fontsize=14)

    fig.suptitle("Terms most distinctive by year (TF-IDF — relative surplus)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PATH}/tfidf_distinctive.pdf", bbox_inches="tight", dpi=300)
    plt.show()


# ============================================================================
# 8.  Section 2.7 — Metadata analysis : gender, age, profession
# ============================================================================

def plot_women_share_by_party_year(df: pd.DataFrame, top_n_parties: int = 8):
    """Figure 5a — Part of women among the titular candidates
    (heatmap party × year)."""
    sexe_col = "titulaire-sexe"
    if sexe_col not in df.columns:
        print(f"Column '{sexe_col}' is missing.")
        return

    sub = df.copy()
    sub["is_femme"] = (
        sub[sexe_col].astype(str).str.lower().str.strip()
        .isin(["féminin", "feminin", "f", "femme"])
        .astype(int)
    )
    mask_known = sub[sexe_col].astype(str).str.lower().str.strip().isin(
        ["féminin", "feminin", "f", "femme",
         "masculin", "m", "homme"]
    )
    sub = sub[mask_known]

    top_parties = sub["party"].value_counts().head(top_n_parties).index.tolist()
    sub = sub[sub["party"].isin(top_parties)]

    pivot = (sub.groupby(["party", "date"])["is_femme"]
                .mean()
                .unstack("date")
                .reindex(top_parties)) * 100

    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(top_parties) + 1.8))
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".1f",
                cmap="rocket_r", cbar_kws={"label": "% femmes"},
                linewidths=0.6, linecolor="white")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title("Part of women among the titular candidates (%) — party × year",
                 loc="left", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"{PATH}/women_share_by_party_year.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    tot = (sub.groupby("date")["is_femme"].mean() * 100).round(1)
    print("\nGlobal part of women among the titular candidates (%) :")
    print(tot.to_string())


def plot_age_distribution(df: pd.DataFrame, year_focus: str = "1981",
                          top_n_parties: int = 8):
    """Figure 5b — Distribution of ages by year + median age by party
    for the target year."""
    age_col = "titulaire-age-calcule"
    if age_col not in df.columns:
        print(f"Column '{age_col}' is missing.")
        return

    sub = df.copy()
    sub["age_num"] = pd.to_numeric(sub[age_col], errors="coerce")
    sub = sub.dropna(subset=["age_num"])
    sub = sub[(sub["age_num"] >= 18) & (sub["age_num"] <= 90)]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for year in sorted(sub["date"].unique()):
        s = sub.loc[sub["date"] == year, "age_num"]
        ax.hist(s, bins=np.arange(20, 86, 3),
                alpha=0.55,
                label=f"{year} (n={len(s)}, médiane {s.median():.0f})",
                color=YEAR_PALETTE.get(year, "#666"),
                edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Age of titular candidate")
    ax.set_ylabel("Number of professions")
    ax.set_title("Distribution of ages of titular candidates",
                 loc="left", fontsize=13, pad=12)
    ax.legend(title="Election", loc="upper right")
    plt.tight_layout()
    plt.show()

    # Median age by party for the target year
    sub_y = sub[sub["date"] == year_focus]
    top_parties = sub_y["party"].value_counts().head(top_n_parties).index.tolist()
    if len(top_parties) >= 2:
        med = (sub_y[sub_y["party"].isin(top_parties)]
                  .groupby("party")["age_num"].median()
                  .sort_values())
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(med) + 1.5))
        colors = [party_color(p) for p in med.index]
        ax.barh(med.index, med.values, color=colors, edgecolor="white")
        for i, v in enumerate(med.values):
            ax.text(v + 0.3, i, f"{v:.0f} ans", va="center", fontsize=10)
        ax.set_xlabel("Median age (years)")
        ax.set_title(f"Median age of candidates by party — {year_focus}",
                     loc="left", fontsize=13, pad=10)
        plt.tight_layout()
        plt.show()


def plot_top_professions(df: pd.DataFrame, top_n_global: int = 15,
                         top_n_parties: int = 8, top_n_prof: int = 10):
    """Figure 5c — Professions of candidates : top global + heatmap
    party × profession."""
    prof_col = "titulaire-profession"
    if prof_col not in df.columns:
        print(f"Column '{prof_col}' is missing.")
        return

    sub = df.copy()
    sub["prof_clean"] = (
        sub[prof_col].astype(str).str.lower().str.strip()
        .replace({"non mentionné": np.nan, "nan": np.nan, "": np.nan})
    )
    sub = sub.dropna(subset=["prof_clean"])

    # Top professions sur l'ensemble du corpus
    top_prof = sub["prof_clean"].value_counts().head(top_n_global)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(top_prof.index[::-1], top_prof.values[::-1],
            color="#2c5f7c", edgecolor="white")
    for i, v in enumerate(top_prof.values[::-1]):
        ax.text(v + max(top_prof) * 0.01, i, str(v),
                va="center", fontsize=9)
    ax.set_xlabel("Number of titular candidates")
    ax.set_title(f"Most frequent professions — top {top_n_global} (complete corpus)",
                 loc="left", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()

    # Heatmap party × profession
    top_parties_p = sub["party"].value_counts().head(top_n_parties).index.tolist()
    top_prof_idx = top_prof.head(top_n_prof).index.tolist()
    sub_pp = sub[sub["party"].isin(top_parties_p)
                 & sub["prof_clean"].isin(top_prof_idx)]
    pivot = (sub_pp.groupby(["party", "prof_clean"]).size()
                  .unstack(fill_value=0)
                  .reindex(index=top_parties_p, columns=top_prof_idx,
                           fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1).replace(0, 1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(top_parties_p) + 2))
    sns.heatmap(pivot_pct, ax=ax, annot=True, fmt=".0f",
                cmap="YlOrBr", cbar_kws={"label": "% of candidates of the party"},
                linewidths=0.6, linecolor="white")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"Profession of candidates by party (top {top_n_prof} professions)",
                 loc="left", fontsize=13, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    plt.show()


def plot_political_capital(df: pd.DataFrame, top_n_parties: int = 8):
    """Figure 5d — Part of candidates with a mandate (current / past /
    outgoing) by party."""
    def has_mandate(value):
        s = str(value).strip().lower()
        return s not in {"", "nan", "none", "non mentionné"}

    mc_col, mp_col = "titulaire-mandat-en-cours", "titulaire-mandat-passe"
    if mc_col not in df.columns or mp_col not in df.columns:
        print("Columns of mandate absent.")
        return

    sub = df.copy()
    sub["has_current"] = sub[mc_col].apply(has_mandate).astype(int)
    sub["has_past"] = sub[mp_col].apply(has_mandate).astype(int)
    sub["is_sortant"] = sub[mp_col].astype(str).str.lower().str.contains(
        "député", na=False
    ).astype(int)

    top_parties = sub["party"].value_counts().head(top_n_parties).index.tolist()
    sub_p = sub[sub["party"].isin(top_parties)]

    agg = (sub_p.groupby("party")[["has_current", "has_past", "is_sortant"]]
                .mean() * 100).reindex(top_parties).round(1)
    agg.columns = ["Mandat en cours", "Mandat passé", "Député sortant"]

    fig, ax = plt.subplots(figsize=(9, 0.55 * len(agg) + 2))
    x = np.arange(len(agg))
    width = 0.27
    ax.bar(x - width, agg["Mandat en cours"], width,
           label="Mandat en cours", color="#1f4e79", edgecolor="white")
    ax.bar(x, agg["Mandat passé"], width,
           label="Mandat passé", color="#5da9d6", edgecolor="white")
    ax.bar(x + width, agg["Député sortant"], width,
           label="Député sortant", color="#c1432b", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(agg.index, rotation=25, ha="right")
    ax.set_ylabel("% of candidates of the party")
    ax.set_title("PPolitical capital of candidates by party",
                 loc="left", fontsize=13, pad=12)
    ax.legend(loc="upper right", ncol=1)
    plt.tight_layout()
    plt.show()

    print("\nDétail (in %) :")
    print(agg.to_string())


def plot_geographic_coverage(df: pd.DataFrame, top_n_dept: int = 15):
    """Figure 5e — Geographic coverage of the corpus (top departments)."""
    geo_col = "departement-nom" if "departement-nom" in df.columns else "departement"
    if geo_col not in df.columns:
        print("Geographic column is missing.")
        return

    sub = df.copy()
    sub["geo"] = (
        sub[geo_col].astype(str).str.strip()
        .replace({"non mentionné": np.nan, "nan": np.nan, "": np.nan})
    )
    sub = sub.dropna(subset=["geo"])

    n_dept = sub["geo"].nunique()
    print(f"The corpus covers {n_dept} distinct departments.")

    top_geo = sub["geo"].value_counts().head(top_n_dept)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_geo.index[::-1], top_geo.values[::-1],
            color="#3a8a4a", edgecolor="white")
    for i, v in enumerate(top_geo.values[::-1]):
        ax.text(v + max(top_geo) * 0.01, i, str(v),
                va="center", fontsize=9)
    ax.set_xlabel("Number of professions of faith")
    ax.set_title(f"Top {top_n_dept} departments by volume — complete corpus "
                 f"({n_dept} departments)",
                 loc="left", fontsize=13, pad=12)
    plt.tight_layout()
    plt.show()


def plot_ticket_gender_composition(df: pd.DataFrame):
    """Figure 5f — Party × profession heatmap."""
    ts, ss = "titulaire-sexe", "suppleant-sexe"
    if ts not in df.columns or ss not in df.columns:
        print("Columns of sex titulaire/suppléant absent.")
        return

    def norm_sex(v):
        s = str(v).lower().strip()
        if s in {"féminin", "feminin", "f", "femme"}:
            return "F"
        if s in {"masculin", "m", "homme"}:
            return "H"
        return None

    sub = df.copy()
    sub["t_sex"] = sub[ts].apply(norm_sex)
    sub["s_sex"] = sub[ss].apply(norm_sex)
    sub = sub.dropna(subset=["t_sex", "s_sex"])
    sub["binome"] = sub["t_sex"] + "/" + sub["s_sex"]

    pivot = (sub.groupby(["date", "binome"]).size()
                .unstack(fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    order = [c for c in ["H/H", "H/F", "F/H", "F/F"] if c in pivot_pct.columns]
    pivot_pct = pivot_pct[order]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    binome_colors = {"H/H": "#1f4e79", "H/F": "#5da9d6",
                     "F/H": "#e63946", "F/F": "#a01818"}
    bottom = np.zeros(len(pivot_pct))
    for col in order:
        ax.bar(pivot_pct.index, pivot_pct[col], bottom=bottom,
               label=col, color=binome_colors.get(col, "#777"),
               edgecolor="white", width=0.55)
        bottom += pivot_pct[col].values
    ax.set_ylabel("% of couples")
    ax.set_title("Gender composition of the titular / substitute couple",
                 loc="left", fontsize=13, pad=12)
    ax.legend(title="Couple", loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(f"{PATH}/gender_composition.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    print("\nRepartition (%) of couples per election :")
    print(pivot_pct.round(1).to_string())

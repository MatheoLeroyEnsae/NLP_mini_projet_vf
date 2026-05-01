"""
topic_modeling.py
=================

Module python that gathers all the functions for the topic modeling pipeline 
on the election manifestos of the 1981-1988-1993 French legislative elections.

The notebook ``02_topic_modeling_pipeline.ipynb`` simply calls these functions; all the reusable logic lives here.

Summary of contents
--------
1. Pre-processing text (lemmatization, stop-words)
2. Coherence infrastructure (gensim Dictionary + 4 metrics)
3. Plot helpers (top-words grids, sweep curves)
4. LDA : sweep of k + direct training with k* + pyLDAvis visualization
5. NMF : sweep of k + direct training with k*
6. BERTopic : training with k* search + topics_over_time
7. Political analyses (5.1 propensity, 5.2 specialization, 5.3 mapping)
8. Metadata × topics (5.4)

Author : Mathéo LEROY
"""

from __future__ import annotations

import time
import textwrap
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# 0.  Global constants and utilities
# ============================================================================

COHERENCE_METRICS: List[str] = ["c_v", "c_npmi"] # c_uci removed "u_mass"
N_TOP_WORDS_COH: int = 10

PARTY_PALETTE: Dict[str, str] = {
    "EXTREME_GAUCHE":   "#6b1a1a",  
    "COMMUNISTE":       "#a01818", 

    "SOCIALISTE":       "#e63946", 
    "COALITION_GAUCHE": "#c97064",  

    "ECOLOGISTE":       "#3a8a4a",  

    "CENTRE":           "#f4c430",  
    "CENTRE_DROIT":     "#5da9d6",  

    "DROITE":           "#1f4e79",  
    "EXTREME_DROITE":   "#3b3b3b", 

    "DIVERS":           "#9a9a9a",  
    "UNKNOWN":          "#cccccc",  
}


def party_color(party: Optional[str], fallback: str = "#9a9a9a") -> str:
    """Return a stable color for a party, with a gray fallback."""
    if party is None:
        return fallback
    key = str(party).upper().strip()
    return PARTY_PALETTE.get(key, fallback)


# ============================================================================
# 1.  Pre-processing text (lemmatization, stop-words)
# ============================================================================

def lemmatize_corpus(texts: Iterable[str], spacy_model: str = "fr_core_news_sm",
                     batch_size: int = 64, min_token_len: int = 3) -> List[str]:
    """Lemmatize corpus via a corpus of texts using spaCy.

    Parameters
    ----------
    texts : iterable of str
        The corpus to lemmatize.
    spacy_model : str
        Model spaCy to load (default : `fr_core_news_sm`).
    batch_size : int
        Batch size for `nlp.pipe`.
    min_token_len : int
        Tokens with lemmas shorter than this length are removed.

    Returns
    -------
    list of str
        A list of strings, where each string is the lemmatized version
        (with stopwords spaCy removed) of the original document.
    """
    import spacy

    nlp = spacy.load(spacy_model, disable=["parser", "ner"])

    t0 = time.time()
    out = []
    for doc in nlp.pipe([str(t) for t in texts], batch_size=batch_size):
        out.append(" ".join(
            tok.lemma_.lower() for tok in doc
            if tok.is_alpha and not tok.is_stop
            and len(tok.lemma_) > min_token_len - 1
        ))
    print(f"Lemmatisation : {time.time() - t0:.1f}s")
    return out


def build_stopwords(extra_path: Optional[str] = None, extra_metier: Optional[set] = None) -> List[str]:
    """Build the final list of stop-words.

    Combine NLTK French + an optional project list (one word per line)
    + an internal list of métier extras (very frequent verbs and politeness
    formulas specific to religious professions).
    """
    from nltk.corpus import stopwords as nltk_sw

    if extra_metier is None:
        extra_metier = set()
    else:
        extra_metier = set(extra_metier)

    extra_metier = extra_metier | {
    # Verbes auxiliaires et modaux (vrai bruit)
    "être", "avoir", "faire", "aller", "voir", "savoir",
    "pouvoir", "vouloir", "devoir", "falloir", 
    "pouvons", "voulez", "doivent", "faut", "devons", "pouvez",
    
    # Quantifieurs et adverbes ultra-fréquents
    "tout", "tous", "toute", "toutes", "très", "plus",
    "moins", "aussi", "ainsi", "alors",
    
    # Formules d'adresse
    "cher", "chère", "madame", "monsieur", "mesdames", "messieurs",
    
    # Métadiscours électoral procédural
    "circonscription", "candidat", "candidate",
    "électeur", "électrice", "électeurs", "élection",
    "suppléant", "suppléante", "suppleant",
    "colistier", "colistière",
    "bulletin", "urnes", "dimanche", "votez",
    "législative", "législatives", "legislatives", "législatif", "election", "liste", "tour",
    "vote", "voter", "député", "députée", "maire", "conseiller", "conseillère",
    "français", "française", "françaises", "france", "mandat",
    "national", "science", "politique", "nationale", "nationales",
    "majorité", "président", "présidente", "présidents", "présidentes", "gouvernement",
    "parti", "partis", "partie", "parties", "programme", "promesse", "promesses",
    "pays", "citoyen", "citoyenne", "citoyens", "citoyennes", "gauche", "droite", "centre",
    "communisme", "socialisme", "écologie", "démocratie", "république", "liberté", "égalité", "fraternité",
    "loi", "lois", "constitution", "institutions", "parlement", "assemblée", "sénat",
    "union", "rassemblement", "coalition", "majorité",
    
    # Dates (déjà en métadonnée)
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    "1981", "1988", "1993", "21", "28", "14",
    
    # Artefacts d'archivage CEVIPOF
    "sciences", "po", "cevipof", "fonds", "nom",
    
    # Granularité temporelle peu informative
    "ans", "an", "année",
}

# gouvernement, gouverner, majorité, président, politique,


    fr_sw = set(nltk_sw.words("french"))

    project_sw: set = set()
    if extra_path is not None:
        try:
            with open(extra_path, "r", encoding="utf-8") as f:
                project_sw = {x.strip() for x in f.readlines() if x.strip()}
        except FileNotFoundError:
            print(f"Stop-words projet introuvables ({extra_path}) — on continue sans.")

    out = sorted(fr_sw | project_sw | extra_metier)
    print(f"Stop-words : {len(out)} entrées.")
    return out


# ============================================================================
# 2.  Coherence infrastructure (gensim Dictionary + 4 metrics)
# ============================================================================

def build_coherence_infra(lemmatized_texts: Iterable[str],
                          no_below: int = 5, no_above: float = 0.95
                          ) -> Tuple[List[List[str]], "Dictionary"]:
    """Build the coherence infrastructure (gensim Dictionary + 4 metrics).

    Returns
    -------
    tokenized_texts : list[list[str]]
        The re-tokenized corpus (split on spaces) ; ``CoherenceModel``
        s'attend à ce format.
    gensim_dictionary : gensim.corpora.Dictionary
        The filtered dictionary (same thresholds as sklearn vectorizers
        for comparable comparisons).
    """
    from gensim.corpora import Dictionary

    tokenized_texts = [str(doc).split() for doc in lemmatized_texts]
    gensim_dictionary = Dictionary(tokenized_texts)
    gensim_dictionary.filter_extremes(no_below=no_below, no_above=no_above)

    print(f"Corpus tokenisé   : {len(tokenized_texts)} documents")
    print(f"Dictionnaire utile: {len(gensim_dictionary)} tokens "
          f"(no_below={no_below}, no_above={no_above})")
    return tokenized_texts, gensim_dictionary


def get_topics_from_components(components: np.ndarray,
                               feature_names: np.ndarray,
                               n_top_words: int = N_TOP_WORDS_COH
                               ) -> List[List[str]]:
    """For a sklearn-like model (LDA / NMF), return the list of
    top-words for each topic. Compatible with ``CoherenceModel``."""
    topics = []
    for topic in components:
        top_idx = np.argsort(-topic)[:n_top_words]
        topics.append([feature_names[i] for i in top_idx])
    return topics


def compute_coherence_scores(topics: List[List[str]],
                             tokenized_texts: List[List[str]],
                             gensim_dictionary: "Dictionary",
                             metrics: List[str] = COHERENCE_METRICS
                             ) -> Dict[str, float]:
    """Calculate the 4 coherence scores for a list of topics."""
    from gensim.models.coherencemodel import CoherenceModel

    out = {}
    for met in metrics:
        cm = CoherenceModel(
            topics=topics,
            texts=tokenized_texts,
            dictionary=gensim_dictionary,
            coherence=met,
        )
        out[met] = cm.get_coherence()
    return out


# ============================================================================
# 3.  Plot helpers
# ============================================================================

def plot_top_words_grid(model, vectorizer, n_top_words: int = 10,
                        title: str = "", n_cols: int = 5,
                        cmap=plt.cm.viridis):
    """bar plot grid : top-words per topic.

    Compatible with any sklearn model exposing ``components_``.
    """
    feature_names = vectorizer.get_feature_names_out()
    n_topics = model.components_.shape[0]
    n_rows = int(np.ceil(n_topics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 2.6 * n_rows),
                             sharex=False)
    axes = np.array(axes).flatten()

    for idx, topic in enumerate(model.components_):
        top_idx = topic.argsort()[-n_top_words:]
        words = feature_names[top_idx]
        weights = topic[top_idx]
        colors = cmap(np.linspace(0.2, 0.85, len(words)))
        ax = axes[idx]
        ax.barh(words, weights, color=colors, edgecolor="white")
        ax.set_title(f"Topic {idx+1}", fontsize=11, loc="left",
                     color="#1a1a1a", fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    for j in range(n_topics, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/top_words_grid_{model}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_coherence_sweep(results: Dict[str, Dict[int, float]],
                         model_name: str,
                         ax=None, mark_best: bool = True
                         ) -> Tuple[Optional[int], Dict[str, int]]:
    """Plot the 4 normalized metrics on the k-axis.

    `results` is a dict {metric: {k: score}}.

    Returns
    -------
    k_best : int or None
        The k that maximizes the average of the 4 normalized metrics.
    best_per_metric : dict
        For each metric, the k that maximizes it.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    palette = {
        "u_mass": "#c1432b", "c_v":    "#2c5f7c",
        "c_uci":  "#d9a441", "c_npmi": "#3a8a4a",
    }

    best_per_metric: Dict[str, int] = {}
    for met in COHERENCE_METRICS:
        ks = sorted(results[met].keys())
        if not ks:
            continue
        scores = np.array([results[met][k] for k in ks])
        rng = scores.max() - scores.min()
        norm = (scores - scores.min()) / rng if rng > 0 else np.zeros_like(scores)
        ax.plot(ks, norm, marker="o", label=met,
                color=palette[met], linewidth=2)
        best_per_metric[met] = ks[int(np.argmax(scores))]

    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel("NNormalized coherence (min-max)")
    ax.set_title(f"{model_name} — fine-tuning of the number of topics", loc="left")
    ax.grid(True, alpha=0.4)

    if mark_best:
        ks_all = sorted(results[COHERENCE_METRICS[0]].keys())
        if not ks_all:
            ax.legend(loc="best", ncol=2)
            return None, best_per_metric
        mean_norm = np.zeros(len(ks_all))
        for met in COHERENCE_METRICS:
            scores = np.array([results[met][k] for k in ks_all])
            rng = scores.max() - scores.min()
            if rng > 0:
                mean_norm += (scores - scores.min()) / rng
        mean_norm /= len(COHERENCE_METRICS)
        k_best = ks_all[int(np.argmax(mean_norm))]
        ax.axvline(k_best, color="black", linestyle="--", alpha=0.6,
                   label=f"k* = {k_best}")
        ax.legend(loc="best", ncol=2)
        return k_best, best_per_metric

    ax.legend(loc="best", ncol=2)
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/coherence_sweep_{model_name}.pdf", bbox_inches="tight", dpi=300)
    plt.show()

    return None, best_per_metric


# ============================================================================
# 4.  LDA — sweep + training with optimal k + pyLDAvis
# ============================================================================

def fit_lda_with_optimal_k(lemmatized_texts: pd.Series,
                           tokenized_texts: List[List[str]],
                           gensim_dictionary,
                           stopwords: List[str],
                           topic_range: List[int] = (5, 8, 10, 12, 15, 20, 25),
                           n_features: int = 1500,
                           random_state: int = 2026,
                           verbose: bool = True
                           ) -> Dict:
    """Complete LDA pipeline: sweeps `k`, selects the optimal k and
    trains a single final model with this k*.

    Returns
    -------
    dict containing :
        - 'cv', 'X_count' : vectorizer and matrix
        - 'results' : coherence scores per metric and per k
        - 'k_best' : optimal k (average of the 4 normalized metrics)
        - 'best_per_metric' : optimal k per metric
        - 'lda' : the trained model directly with k*
    """
    cv = CountVectorizer(max_df=0.95, min_df=5, max_features=n_features,
                         stop_words=stopwords, ngram_range=(1, 2))
    X_count = cv.fit_transform(lemmatized_texts)
    feature_names = cv.get_feature_names_out()

    results = {met: {} for met in COHERENCE_METRICS}

    t0 = time.time()
    for k in topic_range:
        lda_k = LatentDirichletAllocation(
            n_components=k, max_iter=20,
            learning_method="online", learning_offset=50.0,
            random_state=random_state, n_jobs=-1,
        )
        lda_k.fit(X_count)
        topics_k = get_topics_from_components(lda_k.components_, feature_names)
        scores = compute_coherence_scores(topics_k, tokenized_texts,
                                          gensim_dictionary)
        for met, val in scores.items():
            results[met][k] = val
        if verbose:
            print(f"  k={k:>2}  |  " +
                  "  ".join(f"{m}={results[m][k]:+.4f}" for m in COHERENCE_METRICS))

    if verbose:
        print(f"\nTemps total LDA sweep : {time.time()-t0:.1f}s")

    # Choix de k*
    ks_all = sorted(topic_range)
    mean_norm = np.zeros(len(ks_all))
    for met in COHERENCE_METRICS:
        scores = np.array([results[met][k] for k in ks_all])
        rng = scores.max() - scores.min()
        if rng > 0:
            mean_norm += (scores - scores.min()) / rng
    mean_norm /= len(COHERENCE_METRICS)
    k_best = ks_all[int(np.argmax(mean_norm))]
    best_per_metric = {met: max(results[met], key=results[met].get)
                       for met in COHERENCE_METRICS}

    # Entraînement FINAL direct avec k*
    lda = LatentDirichletAllocation(
        n_components=k_best, max_iter=20,
        learning_method="online", learning_offset=50.0,
        random_state=random_state, n_jobs=-1,
    )
    lda.fit(X_count)
    if verbose:
        print(f"\nLDA trained directly with k* = {k_best} topics.")

    return {
        "cv": cv, "X_count": X_count,
        "results": results, "k_best": k_best,
        "best_per_metric": best_per_metric,
        "lda": lda,
    }


def plot_pyldavis(lda_model, X_count, cv_vectorizer):
    """Construct the pyLDAvis visualization for an sklearn LDA model.

    Returns the `PreparedData` object that can be displayed in the notebook
    via `pyLDAvis.display(prepared_data)`. The sub-module to use depends
    on the version of pyLDAvis :

    - pyLDAvis ≥ 3.4 : ``pyLDAvis.lda_model``
    - pyLDAvis < 3.4 : ``pyLDAvis.sklearn``
    """
    import pyLDAvis
    pyLDAvis.enable_notebook()

    # Tentative pyLDAvis ≥ 3.4
    try:
        import pyLDAvis.lda_model as pyldavis_lda
    except ImportError:
        # Fallback pour les versions antérieures
        import pyLDAvis.sklearn as pyldavis_lda

    prepared = pyldavis_lda.prepare(lda_model, X_count, cv_vectorizer,
                                    mds="tsne")
    return prepared


# ============================================================================
# 5.  NMF — sweep + training with optimal k
# ============================================================================

def fit_nmf_with_optimal_k(lemmatized_texts: pd.Series,
                           tokenized_texts: List[List[str]],
                           gensim_dictionary,
                           stopwords: List[str],
                           topic_range: List[int] = (5, 8, 10, 12, 15, 20, 25, 30),
                           n_features: int = 1500,
                           ngram_range: Tuple[int, int] = (1, 2),
                           random_state: int = 2026,
                           verbose: bool = True
                           ) -> Dict:
    """Pipeline NMF : balaye `k`, choosen le k* et train a single final model.

    Returns
    -------
    dict containing :
        - 'tfidf_vec', 'X_tfidf' : TF-IDF vectorizer and matrix
        - 'results', 'k_best', 'best_per_metric'
        - 'nmf' : the trained model with k*
        - 'W' : document × topic matrix
        - 'H' : topic × word matrix
        - 'W_norm' : W normalized by row (probabilities)
        - 'topic_label' : dict {topic_id -> "T0", "T1", ...} by default
    """
    tfidf_vec = TfidfVectorizer(max_df=0.95, min_df=5,
                                max_features=n_features,
                                stop_words=stopwords,
                                ngram_range=ngram_range)
    X_tfidf = tfidf_vec.fit_transform(lemmatized_texts)
    feature_names = tfidf_vec.get_feature_names_out()

    results = {met: {} for met in COHERENCE_METRICS}

    t0 = time.time()
    for k in topic_range:
        nmf_k = NMF(n_components=k, init="nndsvd",
                    random_state=random_state, max_iter=400)
        nmf_k.fit(X_tfidf)
        topics_k = get_topics_from_components(nmf_k.components_, feature_names)
        scores = compute_coherence_scores(topics_k, tokenized_texts,
                                          gensim_dictionary)
        for met, val in scores.items():
            results[met][k] = val
        if verbose:
            print(f"  k={k:>2}  |  " +
                  "  ".join(f"{m}={results[m][k]:+.4f}" for m in COHERENCE_METRICS))

    if verbose:
        print(f"\nTotal amount of time for NMF sweep : {time.time()-t0:.1f}s")

    ks_all = sorted(topic_range)
    mean_norm = np.zeros(len(ks_all))
    for met in COHERENCE_METRICS:
        scores = np.array([results[met][k] for k in ks_all])
        rng = scores.max() - scores.min()
        if rng > 0:
            mean_norm += (scores - scores.min()) / rng
    mean_norm /= len(COHERENCE_METRICS)
    k_best = ks_all[int(np.argmax(mean_norm))]
    best_per_metric = {met: max(results[met], key=results[met].get)
                       for met in COHERENCE_METRICS}

    # Entraînement FINAL avec k*
    nmf = NMF(n_components=k_best, init="nndsvd",
              random_state=random_state, max_iter=400)
    W = nmf.fit_transform(X_tfidf)
    H = nmf.components_
    W_norm = W / W.sum(axis=1, keepdims=True).clip(min=1e-12)

    topic_label = {i: f"T{i}" for i in range(k_best)}

    if verbose:
        print(f"\nNMF trained directly with k* = {k_best} topics. "
              f"W={W.shape}, H={H.shape}")

    return {
        "tfidf_vec": tfidf_vec, "X_tfidf": X_tfidf,
        "results": results, "k_best": k_best,
        "best_per_metric": best_per_metric,
        "nmf": nmf, "W": W, "H": H, "W_norm": W_norm,
        "topic_label": topic_label,
    }


def topic_top_words(H: np.ndarray, vectorizer, n: int = 12) -> Dict[int, List[str]]:
    """For each topic, returns the n words with the highest weights."""
    feats = vectorizer.get_feature_names_out()
    return {i: feats[topic.argsort()[-n:][::-1]].tolist()
            for i, topic in enumerate(H)}


def plot_dominant_topic_distribution(W_norm: np.ndarray,
                                     topic_label: Dict[int, str]):
    """Bar-plot of the distribution of documents by dominant topic."""
    n_topics = W_norm.shape[1]
    dominant = np.argmax(W_norm, axis=1)
    topic_counts = pd.Series(dominant).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_topics))
    bars = ax.bar([f"T{i}\n{topic_label[i]}" for i in topic_counts.index],
                  topic_counts.values, color=colors, edgecolor="white")
    for bar, c in zip(bars, topic_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{c}", ha="center", fontsize=9, color="#2b2b2b")
    ax.set_ylabel("Number of documents")
    ax.set_title("Dominant Topic — Distribution across the corpus", loc="left")
    plt.tight_layout()
    plt.savefig("/Users/matheoleroy/Downloads/data_nlp/figures_topic/dominant_topic_distribution.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def inspect_random_documents(df: pd.DataFrame, W_norm: np.ndarray,
                             topic_label: Dict[int, str],
                             n: int = 3, random_state: int = 2026, length: int = 1200):
    """  Inspect qualitatively n documents randomly selected at random with their
    thematic distribution."""
    n_topics = W_norm.shape[1]
    sample = df.sample(n=n, random_state=random_state)

    for row in sample.itertuples():
        doc_id = row.Index
        dist = W_norm[doc_id]
        top_t = int(np.argmax(dist))
        print(f"\n{'─'*70}")
        print(f"Doc {doc_id} | year {row.date}", end="")
        if "party" in df.columns:
            print(f" | party {row.party}", end="")
        print(f" | dominant = T{top_t} ({topic_label[top_t]}, score {dist[top_t]:.2f})")
        print(textwrap.fill(str(row.text)[:length], width=95))

        fig, ax = plt.subplots(figsize=(11, 2.8))
        bar_colors = ["#d6d2c4"] * n_topics
        bar_colors[top_t] = "#c1432b"
        ax.bar([f"T{i}" for i in range(n_topics)], dist,
               color=bar_colors, edgecolor="white")
        ax.set_ylabel("Proportion")
        ax.set_title(f"Topic Distribution — doc {doc_id}",
                     loc="left", fontsize=11)
        ax.set_ylim(0, max(dist) * 1.15)
        plt.tight_layout()
        plt.savefig("/Users/matheoleroy/Downloads/data_nlp/figures_topic/inspect_random_documents.pdf", bbox_inches="tight", dpi=300)

        plt.show()


# ============================================================================
# 6.  BERTopic — training with optimal k search + topics_over_time
# ============================================================================

def _bertopic_topics_as_words(model, n_top_words: int = N_TOP_WORDS_COH
                              ) -> List[List[str]]:
    """Extract, for each topic ≠ -1, the n_top_words c-TF-IDF words."""
    topic_info = model.get_topic_info()
    real_ids = [tid for tid in topic_info["Topic"].tolist() if tid != -1]
    topics = []
    for tid in real_ids:
        ws = model.get_topic(tid) or []
        topics.append([w for w, _ in ws[:n_top_words] if w])
    return [t for t in topics if len(t) >= 2]


def fit_bertopic_with_optimal_k(docs: List[str],
                                tokenized_texts: List[List[str]],
                                gensim_dictionary,
                                stopwords: List[str],
                                topic_range: List[int] = (5, 8, 10, 12, 15, 20),
                                random_state: int = 2026,
                                min_cluster_size: int = 50,
                                embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                                verbose: bool = True,
                                cluster_selection_method: str = "eom",
                                n_neighbors: int = 15,
                                n_components: int = 5,
                                min_samples: int = 10,
                                ) -> Dict:
    """Complete Pipeline BERTopic  :
    1. Encode the documents ONCE (the most costly operation).
    2. Sweep `k` via `reduce_topics` to find the optimal k*.
    3. Train the final model ONCE and reduce it directly to k*.

    Returns
    -------
    dict containing :
        - 'embedding_model', 'embeddings'
        - 'topic_model' : BERTopic model reduced to k* topics
        - 'bert_topics', 'bert_probs' : assignments + probabilities
        - 'results', 'k_best', 'best_per_metric'
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    embedding_model = SentenceTransformer(embedding_model_name)
    if verbose:
        print(f"Encodage des {len(docs)} documents…")
    t0 = time.time()
    embeddings = embedding_model.encode(docs, show_progress_bar=verbose, batch_size=32)
    if verbose:
        print(f"  shape={embeddings.shape}  |  {time.time()-t0:.1f}s")

    # 2) Vectorizer for the c-TF-IDF (labeling)
    vectorizer_model = CountVectorizer(
        stop_words=stopwords, min_df=2, ngram_range=(1, 2), max_df=0.95,
    )

    # 3) Sweep — for each k we train BERTopic then reduce
    results = {met: {} for met in COHERENCE_METRICS}
    if verbose:
        print(f"\n--- Sweep BERTopic sur k ∈ {list(topic_range)} ---")

    t0 = time.time()
    for k in topic_range:
        bt_k = BERTopic(
            embedding_model=embedding_model,
            umap_model=UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0,
                            metric="cosine", random_state=random_state),
            hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size,
                                  metric="euclidean",
                                  cluster_selection_method=cluster_selection_method,
                                  min_samples=min_samples,
                                  prediction_data=True),
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,  
            verbose=False,
        )
        bt_k.fit(docs, embeddings)

        n_found = len([t for t in bt_k.get_topic_info()["Topic"] if t != -1])
        if n_found > k:
            bt_k.reduce_topics(docs, nr_topics=k)

        topics_k = _bertopic_topics_as_words(bt_k)
        if len(topics_k) < 2:
            if verbose:
                print(f"  k={k:>2}  | <2 topics utiles, skip.")
            continue

        scores = compute_coherence_scores(topics_k, tokenized_texts,
                                          gensim_dictionary)
        for met, val in scores.items():
            results[met][k] = val
        if verbose:
            print(f"  k={k:>2}  |  " +
                  "  ".join(f"{m}={results[m][k]:+.4f}" for m in COHERENCE_METRICS))

    if verbose:
        print(f"\nTemps total BERTopic sweep : {time.time()-t0:.1f}s")

    # 4) Choice of k* (mean of the 4 normalized metrics)
    ks_all = sorted(k for k in topic_range if k in results[COHERENCE_METRICS[0]])
    if not ks_all:
        raise RuntimeError("Aucun k n'a produit suffisamment de topics.")
    mean_norm = np.zeros(len(ks_all))
    for met in COHERENCE_METRICS:
        scores = np.array([results[met][k] for k in ks_all])
        rng = scores.max() - scores.min()
        if rng > 0:
            mean_norm += (scores - scores.min()) / rng
    mean_norm /= len(COHERENCE_METRICS)
    k_best = ks_all[int(np.argmax(mean_norm))]
    best_per_metric = {met: max(results[met], key=results[met].get)
                       for met in COHERENCE_METRICS}

    # 5) Final training directly with k*
    if verbose:
        print(f"\nEntraînement final BERTopic avec k* = {k_best}…")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                        metric="cosine", random_state=random_state),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster_size,
                              metric="euclidean",
                              cluster_selection_method="eom",
                              prediction_data=True),
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,  # nécessaire pour W_norm
        verbose=False,
    )
    bert_topics, bert_probs = topic_model.fit_transform(docs, embeddings)
    n_found = len([t for t in topic_model.get_topic_info()["Topic"] if t != -1])
    if n_found > k_best:
        topic_model.reduce_topics(docs, nr_topics=k_best)
        
        bert_topics = topic_model.topics_
        bert_probs = topic_model.probabilities_
    if verbose:
        n_topics_final = len([t for t in topic_model.get_topic_info()["Topic"]
                              if t != -1])
        n_outliers = (np.array(bert_topics) == -1).sum()
        print(f"BERTopic ready — {n_topics_final} topics, "
              f"{n_outliers} non-classified documents (-1).")

    return {
        "embedding_model": embedding_model,
        "embeddings": embeddings,
        "topic_model": topic_model,
        "bert_topics": bert_topics,
        "bert_probs": bert_probs,
        "results": results,
        "k_best": k_best,
        "best_per_metric": best_per_metric,
    }


def plot_bertopic_top_words(topic_model, n_top_words: int = 10, n_show: int = 10):
    """Grid of the top words for the main BERTopic topics."""
    topic_info = topic_model.get_topic_info()
    real_topic_ids = [tid for tid in topic_info["Topic"].tolist() if tid != -1]
    n_show = min(n_show, len(real_topic_ids))
    topics_to_plot = real_topic_ids[:n_show]

    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 2.6 * n_rows),
                             sharex=False)
    axes = np.array(axes).flatten()

    for k, tid in enumerate(topics_to_plot):
        words_scores = topic_model.get_topic(tid)
        if not words_scores:
            axes[k].axis("off"); continue
        words = [w for w, _ in words_scores][:n_top_words][::-1]
        scores = [s for _, s in words_scores][:n_top_words][::-1]
        colors = plt.cm.plasma(np.linspace(0.2, 0.85, len(words)))
        ax = axes[k]
        ax.barh(words, scores, color=colors, edgecolor="white")
        size = topic_info.loc[topic_info["Topic"] == tid, "Count"].iloc[0]
        ax.set_title(f"BERTopic {tid}  (n={size})",
                     fontsize=11, loc="left",
                     color="#1a1a1a", fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"BERTopic — Top words of the {n_show} main topics",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/top_words_grid_bert.pdf", bbox_inches="tight", dpi=300)

    plt.show()


def compute_topics_over_time(topic_model, docs: List[str],
                             timestamps: List, nr_bins: Optional[int] = None,
                             ) -> pd.DataFrame:
    """Wrapper around `topic_model.topics_over_time`.

    Parameters
    ----------
    topic_model : BERTopic
    docs : list of str
    timestamps : list
        Timestamps (ex : column `date` of DataFrame).
    nr_bins : int or None
        Number of temporal bins (by default, we use the number of distinct years
        in `timestamps`).
    datetime_format : str or None
        Format of the timestamps if they are strings (ex : `"%Y"`).
    """
    if nr_bins is None:
        nr_bins = len(set(timestamps))
    print(f"Calculate topic distribution over time with nr_bins={nr_bins}…")
    kwargs = {"nr_bins": nr_bins}
    
    topics_over_time = topic_model.topics_over_time(docs, timestamps, **kwargs)
    return topics_over_time


def visualize_topics_over_time_plot(topic_model, topics_over_time: pd.DataFrame,
                                    top_n_topics: int = 10):
    """Wrapper around `topic_model.visualize_topics_over_time` (Plotly)."""
    fig = topic_model.visualize_topics_over_time(topics_over_time,
                                                 top_n_topics=top_n_topics)
    return fig


# ============================================================================
# 7.  Temporal Evolution (Section 4 — version NMF)
# ============================================================================

def topic_distribution_over_years(df: pd.DataFrame, W_norm: np.ndarray,
                                  topic_label: Dict[int, str]) -> pd.DataFrame:
    """DataFrame [topic × year] : mean part of each topic, by year."""
    years = sorted(df["date"].unique())
    n_topics = W_norm.shape[1]
    return pd.DataFrame(
        {y: W_norm[(df["date"] == y).values].mean(axis=0) for y in years},
        index=[f"T{i}-{topic_label[i]}" for i in range(n_topics)]
    )


def plot_stacked_area_by_year(topic_by_year: pd.DataFrame):
    """Stacked area plot — thematic composition by year."""
    n_topics = len(topic_by_year)
    years = list(topic_by_year.columns)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.cm.tab10(np.linspace(0, 1, n_topics))
    bottom = np.zeros(len(years))

    for i, label in enumerate(topic_by_year.index):
        vals = topic_by_year.loc[label].values
        ax.bar(years, vals, bottom=bottom, color=cmap[i],
               edgecolor="white", linewidth=1.2, label=label, width=0.55)
        for j, v in enumerate(vals):
            if v > 0.05:
                ax.text(j, bottom[j] + v / 2, f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if i % 2 == 0 else "#1a1a1a",
                        fontweight="bold")
        bottom += vals

    ax.set_ylabel("Mean part of the topic")
    ax.set_ylim(0, 1.001)
    ax.set_title("Thematic composition of the corpus by year",
                 loc="left", fontsize=14, pad=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=9, title="Topics")
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/stacked_area.pdf", bbox_inches="tight", dpi=300)

    plt.show()


def plot_topic_trajectories(topic_by_year: pd.DataFrame,
                            topic_label: Dict[int, str]):
    """Small multiples : trajectory of each topic across the elections."""
    n_topics = len(topic_by_year)
    years = list(topic_by_year.columns)

    n_cols = 5
    n_rows = int(np.ceil(n_topics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows),
                             sharey=True)
    axes = np.array(axes).flatten()

    cmap = plt.cm.tab10(np.linspace(0, 1, n_topics))
    ymax = topic_by_year.values.max() * 1.15

    for i in range(n_topics):
        ax = axes[i]
        vals = topic_by_year.iloc[i].values
        color = cmap[i]
        ax.fill_between(range(len(years)), vals, alpha=0.35, color=color)
        ax.plot(range(len(years)), vals, "o-", color=color, lw=2.2, markersize=7)
        for j, v in enumerate(vals):
            ax.text(j, v + ymax * 0.04, f"{v*100:.1f}%", ha="center",
                    fontsize=8.5, color="#1a1a1a")
        ax.set_xticks(range(len(years))); ax.set_xticklabels(years)
        ax.set_ylim(0, ymax)
        ax.set_title(f"T{i} — {topic_label[i]}", fontsize=10,
                     loc="left", color=color, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n_topics, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Trajectory of each topic across the three elections",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/topic_trajectories.pdf", bbox_inches="tight", dpi=300)

    plt.show()


# ============================================================================
# 8.  Analyses politiques (Section 5)
# ============================================================================

# ----- 5.1 Thematic propensity ----------------------------------------------

def topic_distribution_by_party(df: pd.DataFrame, W_norm: np.ndarray,
                                topic_label: Dict[int, str],
                                year, min_docs: int = 10) -> pd.DataFrame:
    """Average topic distribution by party for a given year."""
    if "party" not in df.columns:
        raise ValueError("No 'party' column in df.")

    mask_year = (df["date"] == year).values
    sub = df[mask_year]
    W_sub = W_norm[mask_year]

    out = {}
    for party, idx in sub.groupby("party").indices.items():
        if len(idx) < min_docs:
            continue
        out[party] = W_sub[idx].mean(axis=0)

    n_topics = W_norm.shape[1]
    return pd.DataFrame(out,
                        index=[f"T{i}-{topic_label[i]}"
                               for i in range(n_topics)]).T


def plot_propensity_heatmap(prop: pd.DataFrame, year):
    """Heatmap parti × topic + barre latérale d'entropie (diversité)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(prop) + 2.5),
                             gridspec_kw={"width_ratios": [4, 1]})

    ax = axes[0]
    sns.heatmap(prop, ax=ax, cmap="rocket_r", vmin=0,
                annot=True, fmt=".2f", annot_kws={"fontsize": 9},
                cbar_kws={"label": "Average topic distribution", "pad": 0.02},
                linewidths=0.8, linecolor="white")
    ax.set_title(f"Thematic distribution by party — election {year}",
                 loc="left", fontsize=14, pad=12)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    def entropy(p):
        p = np.asarray(p) + 1e-12
        return -(p * np.log(p)).sum() / np.log(len(p))

    ent = prop.apply(entropy, axis=1).sort_values(ascending=True)
    colors = [party_color(p) for p in ent.index]
    ax = axes[1]
    ax.barh(ent.index, ent.values, color=colors, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Normalised Entropy\n(0 = unithematic, 1 = balanced)")
    ax.set_title("Thematic Diversity", loc="left", fontsize=12)
    for i, (party, v) in enumerate(ent.items()):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center",
                fontsize=9, color="#1a1a1a")

    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/propensity_heatmap_{year}.pdf", bbox_inches="tight", dpi=300)

    plt.show()


def plot_radar_top_parties(prop: pd.DataFrame, df: pd.DataFrame, year,
                            n_top: int = 4):
    """Radar chart to compare the thematic profiles of the n_top parties
    most represented this year."""
    top = (df[df["date"] == year]["party"]
              .value_counts().head(n_top).index.tolist())
    selected = {p: prop.loc[p].values for p in top if p in prop.index}

    if len(selected) < 2:
        print("Not enough parties to compare in radar chart.")
        return

    n = len(next(iter(selected.values())))
    labels = [f"T{i}" for i in range(n)]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    vmax = max(arr.max() for arr in selected.values()) * 1.1
    for party, vals in selected.items():
        v = np.concatenate([vals, [vals[0]]])
        c = party_color(party)
        ax.plot(angles, v, "o-", lw=2, color=c, label=party, markersize=5)
        ax.fill(angles, v, alpha=0.15, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, vmax)
    ax.set_yticks(np.linspace(0, vmax, 4))
    ax.set_yticklabels([f"{x:.2f}" for x in np.linspace(0, vmax, 4)],
                       fontsize=8, color="#666")
    ax.set_title(f"Thematic Profiles — Top Parties {year}",
                 fontsize=13, pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=10)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/radar_chart_{year}.pdf", bbox_inches="tight", dpi=300)

    plt.show()


# ----- 5.2 Specialization ---------------------------------------------------

def specialization_for_year(df: pd.DataFrame, W_norm: np.ndarray,
                            topic_label: Dict[int, str],
                            year, min_docs: int = 10
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns 2 DataFrames [party × topic] : delta vs national average, and ratio."""
    mask_year = (df["date"] == year).values
    W_sub = W_norm[mask_year]
    sub = df[mask_year]

    national_mean = W_sub.mean(axis=0)
    deltas, ratios = {}, {}
    for party, idx in sub.groupby("party").indices.items():
        if len(idx) < min_docs:
            continue
        party_mean = W_sub[idx].mean(axis=0)
        deltas[party] = party_mean - national_mean
        ratios[party] = party_mean / np.maximum(national_mean, 1e-9)

    n_topics = W_norm.shape[1]
    cols = [f"T{i}-{topic_label[i]}" for i in range(n_topics)]
    return (pd.DataFrame(deltas, index=cols).T.sort_index(),
            pd.DataFrame(ratios, index=cols).T.sort_index())


def plot_specialization(deltas: pd.DataFrame, ratios: pd.DataFrame, year):
    """Heatmap divergente Δ + heatmap log₂-ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.55 * len(deltas) + 2.5))

    vmax = max(abs(deltas.values.min()), abs(deltas.values.max()))
    sns.heatmap(deltas, ax=axes[0], cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                annot=True, fmt="+.2f", annot_kws={"fontsize": 8.5},
                cbar_kws={"label": "Deviation from National Average",
                          "pad": 0.02},
                linewidths=0.8, linecolor="white")
    axes[0].set_title(f"Specialization — Δ vs National Average ({year})",
                      loc="left", fontsize=13, pad=10)
    plt.setp(axes[0].get_xticklabels(), rotation=35, ha="right")
    axes[0].set_ylabel(""); axes[0].set_xlabel("")

    log_ratios = np.log2(ratios.replace(0, np.nan))
    vmax2 = np.nanpercentile(np.abs(log_ratios.values), 95)
    sns.heatmap(log_ratios, ax=axes[1], cmap="PiYG", center=0,
                vmin=-vmax2, vmax=vmax2,
                annot=True, fmt="+.1f", annot_kws={"fontsize": 8.5},
                cbar_kws={"label": "log₂(ratio) — 0=national average", "pad": 0.02},
                linewidths=0.8, linecolor="white")
    axes[1].set_title(f"Sur/sous-représentation log₂-ratio — {year}",
                      loc="left", fontsize=13, pad=10)
    plt.setp(axes[1].get_xticklabels(), rotation=35, ha="right")
    axes[1].set_ylabel(""); axes[1].set_xlabel("")

    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/specialization_{year}.pdf", bbox_inches="tight", dpi=300)

    plt.show()


def plot_party_signatures(deltas: pd.DataFrame, year, n: int = 3):
    """For each party, top n and bottom n topics (Δ vs national average)."""
    rows = []
    for party in deltas.index:
        s = deltas.loc[party].sort_values(ascending=False)
        for rank, (t, v) in enumerate(s.head(n).items(), 1):
            rows.append({"party": party, "topic": t, "delta": v,
                         "kind": "sur", "rank": rank})
        for rank, (t, v) in enumerate(s.tail(n).iloc[::-1].items(), 1):
            rows.append({"party": party, "topic": t, "delta": v,
                         "kind": "sous", "rank": rank})
    sig = pd.DataFrame(rows)
    parties_in_sig = sorted(sig["party"].unique())
    n_parties = len(parties_in_sig)

    fig, axes = plt.subplots(n_parties, 1,
                             figsize=(11, 1.6 * n_parties + 0.8),
                             sharex=True)
    if n_parties == 1:
        axes = [axes]

    for ax, party in zip(axes, parties_in_sig):
        rows_p = sig[sig["party"] == party].copy().sort_values("delta")
        colors = ["#c1432b" if v < 0 else "#2c5f7c" for v in rows_p["delta"]]
        bars = ax.barh(rows_p["topic"], rows_p["delta"], color=colors,
                       edgecolor="white", height=0.7)
        ax.axvline(0, color="#1a1a1a", lw=0.8)
        ax.set_title(party, loc="left", fontsize=12,
                     color=party_color(party), fontweight="bold")
        ax.set_yticks(range(len(rows_p)))
        ax.set_yticklabels(rows_p["topic"], fontsize=9)
        for bar, v in zip(bars, rows_p["delta"]):
            offset = 0.002 if v >= 0 else -0.002
            ha = "left" if v >= 0 else "right"
            ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                    f"{v:+.2f}", va="center", ha=ha, fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Deviation from National Average (Δ)")
    fig.suptitle(f"Thematic Signatures of Parties — {year}",
                 fontsize=14, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/thematic_signatures_{year}.pdf", bbox_inches="tight", dpi=300)

    plt.show()


def plot_specialization_evolution(df: pd.DataFrame, W_norm: np.ndarray,
                                  topic_label: Dict[int, str],
                                  min_docs: int = 10, max_parties: int = 6):
    """Evolution of Δ by topic across the elections, party by party."""
    n_topics = W_norm.shape[1]
    all_deltas = {}
    for y in sorted(df["date"].unique()):
        d, _ = specialization_for_year(df, W_norm, topic_label, y, min_docs=min_docs)
        all_deltas[y] = d

    common_parties = sorted(set.intersection(
        *[set(d.index) for d in all_deltas.values()]
    ))
    print(f"Parties present at all 3 elections : {common_parties}")

    if len(common_parties) < 2:
        return

    n_show = min(len(common_parties), max_parties)
    fig, axes = plt.subplots(n_show, 1, figsize=(11, 1.5 * n_show + 0.8),
                             sharex=True)
    if n_show == 1:
        axes = [axes]
    years_sorted = sorted(all_deltas.keys())
    cmap = plt.cm.tab10(np.linspace(0, 1, n_topics))

    for ax, party in zip(axes, common_parties[:n_show]):
        for i in range(n_topics):
            vals = [all_deltas[y].loc[party].iloc[i] for y in years_sorted]
            ax.plot(years_sorted, vals, "o-", color=cmap[i],
                    lw=1.7, markersize=5,
                    label=f"T{i}" if ax is axes[0] else None)
        ax.axhline(0, color="#1a1a1a", lw=0.6, ls="--")
        ax.set_title(party, loc="left", fontsize=11,
                     color=party_color(party), fontweight="bold")
        ax.set_ylabel("Δ vs national average")

    axes[0].legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=9, ncol=1, title="Topics")
    axes[-1].set_xlabel("Year")
    fig.suptitle("Temporal Evolution of Thematic Specializations",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/specialization_evolution.pdf", bbox_inches="tight", dpi=300)

    plt.show()


# ----- 5.3 Semantic mapping --------------------------------------------------

def party_year_vectors(df: pd.DataFrame, W_norm: np.ndarray,
                       min_docs: int = 10) -> Tuple[np.ndarray, List[Tuple]]:
    """Mean thematic vector for each (party, year)."""
    rows = []
    labels = []
    for (party, year), idx in df.groupby(["party", "date"]).indices.items():
        if len(idx) < min_docs:
            continue
        rows.append(W_norm[idx].mean(axis=0))
        labels.append((party, year, len(idx)))
    return np.array(rows), labels


def plot_pca_semantic_map(vecs: np.ndarray, labels: List[Tuple],
                          random_state: int = 2026):
    """PCA of thematic profiles + trajectories of the same party over time."""
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(vecs)
    explained = pca.explained_variance_ratio_
    loadings = pca.components_

    def axis_top_topics(loading, n=3):
        idx_pos = np.argsort(-loading)[:n]
        idx_neg = np.argsort(loading)[:n]
        return f"+ ({', '.join(f'T{i}' for i in idx_pos)})    vs    − ({', '.join(f'T{i}' for i in idx_neg)})"

    labels_df = pd.DataFrame(labels, columns=["party", "year", "n_docs"])

    fig, ax = plt.subplots(figsize=(11, 7.5))
    for i, (party, year, n_docs) in enumerate(labels):
        color = party_color(party)
        marker = {"1981": "o", "1988": "s", "1993": "^"}.get(str(year), "D")
        size = 80 + 8 * np.sqrt(n_docs)
        ax.scatter(coords[i, 0], coords[i, 1], c=color, marker=marker,
                   s=size, edgecolors="white", linewidth=1.4, alpha=0.9, zorder=3)
        ax.annotate(f"{party}\n{year}", (coords[i, 0], coords[i, 1]),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=9, color="#1a1a1a",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="#cccccc", lw=0.7, alpha=0.85))

    for party in labels_df["party"].unique():
        sub_idx = [i for i, (p, _, _) in enumerate(labels) if p == party]
        if len(sub_idx) < 2:
            continue
        sub_idx_sorted = sorted(sub_idx, key=lambda k: labels[k][1])
        pts = coords[sub_idx_sorted]
        ax.plot(pts[:, 0], pts[:, 1], "-", color=party_color(party),
                lw=1.4, alpha=0.55, zorder=1)

    ax.axhline(0, color="#888", lw=0.5, ls=":")
    ax.axvline(0, color="#888", lw=0.5, ls=":")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% de variance)\n{axis_top_topics(loadings[0])}",
                  fontsize=10)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% de variance)\n{axis_top_topics(loadings[1])}",
                  fontsize=10)
    ax.set_title("Semantic Map of Parties (PCA on Thematic Profiles)",
                 loc="left", fontsize=14, pad=12)

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                   markersize=10, label="1981"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
                   markersize=10, label="1988"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#888",
                   markersize=10, label="1993"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", title="Année",
              frameon=True, edgecolor="#cccccc")

    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/semantic_map_{year}.pdf", bbox_inches="tight", dpi=300)

    plt.show()
    print(f"\nExplained variance by the 2 first components : "
          f"{explained.sum()*100:.1f}%")


def plot_tsne_documents(df: pd.DataFrame, W_norm: np.ndarray, year,
                        min_docs: int = 10, random_state: int = 2026):
    """t-SNE on individual documents for a given year + centroids by party."""
    mask_year = (df["date"] == year).values
    W_year = W_norm[mask_year]
    parties_year = df.loc[mask_year, "party"].values

    counts = pd.Series(parties_year).value_counts()
    keep = counts[counts >= min_docs].index.tolist()
    sel = np.isin(parties_year, keep)
    W_sel = W_year[sel]
    parties_sel = parties_year[sel]

    n_samples = len(W_sel)
    if n_samples < 4:
        print("Pas assez de documents pour t-SNE.")
        return

    perp = max(5, min(30, n_samples // 4))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perp,
                init="pca", learning_rate="auto")
    coords_tsne = tsne.fit_transform(W_sel)

    fig, ax = plt.subplots(figsize=(11, 8))
    for party in keep:
        pts = coords_tsne[parties_sel == party]
        ax.scatter(pts[:, 0], pts[:, 1], s=22, alpha=0.45,
                   color=party_color(party), edgecolors="none",
                   label=f"{party} (n={len(pts)})")

    for party in keep:
        pts = coords_tsne[parties_sel == party]
        cx, cy = pts.mean(axis=0)
        ax.scatter(cx, cy, s=380, color=party_color(party),
                   edgecolors="grey", linewidth=2.2, zorder=5,
                   marker="X")
        ax.annotate(party, (cx, cy), xytext=(0, 0),
                    textcoords="offset points",
                    fontsize=11, fontweight="bold",
                    color="black", ha="center", va="center", zorder=6)

    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Semantic Map of Documents ({year}) — t-SNE",
                 loc="left", fontsize=14, pad=12)
    ax.legend(loc="best", title="Party", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/semantic_map_documents_{year}.pdf", bbox_inches="tight", dpi=300)

    plt.show()


def plot_cosine_similarity_parties(vecs: np.ndarray, labels: List[Tuple], year):
    """Heatmap of cosine similarity between parties for a given year."""
    year_str = str(year)
    year_vecs_idx = [i for i, (_, y, _) in enumerate(labels) if str(y) == year_str]
    if len(year_vecs_idx) < 2:
        print(f"Not enough parties to calculate a similarity matrix in {year}.")
        return

    V = vecs[year_vecs_idx]
    parties_v = [labels[i][0] for i in year_vecs_idx]
    sim = cosine_similarity(V)
    sim_df = pd.DataFrame(sim, index=parties_v, columns=parties_v)

    from scipy.cluster.hierarchy import linkage, leaves_list
    try:
        Z = linkage(1 - sim_df.values + np.eye(len(sim_df)) * 1e-6,
                    method="average")
        order = leaves_list(Z)
        sim_df = sim_df.iloc[order, order]
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(sim_df, ax=ax, cmap="rocket_r", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"fontsize": 9},
                cbar_kws={"label": "Cosine Similarity"},
                linewidths=0.6, linecolor="white", square=True)
    ax.set_title(f"Thematic Proximity between Parties — {year}",
                 loc="left", fontsize=13, pad=10)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/cosine_similarity_{year}.pdf", bbox_inches="tight", dpi=300)

    plt.show()


# ============================================================================
# 9.  Métadonnées × topics (Section 5.4)
# ============================================================================

def plot_metadata_gender(df: pd.DataFrame, W_norm: np.ndarray,
                         topic_label: Dict[int, str], year):
    """Topics surinvestis par les candidates femmes vs hommes."""
    mask_year = (df["date"] == year).values
    W_y = W_norm[mask_year]
    sub = df[mask_year].copy()

    sub["sex_norm"] = (
        sub["titulaire-sexe"].astype(str).str.lower().str.strip()
        .map(lambda s: "F" if s in {"féminin", "feminin", "f", "femme"}
                       else ("H" if s in {"masculin", "m", "homme"} else None))
    )
    known = sub["sex_norm"].notna().values
    W_known = W_y[known]
    sex_known = sub.loc[known, "sex_norm"].values

    if (sex_known == "F").sum() < 10 or (sex_known == "H").sum() < 10:
        print(f"Pas assez de candidates en {year} pour la comparaison.")
        return

    profile_F = W_known[sex_known == "F"].mean(axis=0)
    profile_H = W_known[sex_known == "H"].mean(axis=0)
    delta = profile_F - profile_H

    n_topics = W_norm.shape[1]
    cols = [f"T{i}-{topic_label[i]}" for i in range(n_topics)]
    df_delta = pd.DataFrame({"Δ (F − H)": delta},
                            index=cols).sort_values("Δ (F − H)")

    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(cols) + 1.5))
    colors = ["#a01818" if v > 0 else "#1f4e79" for v in df_delta["Δ (F − H)"]]
    ax.barh(df_delta.index, df_delta["Δ (F − H)"],
            color=colors, edgecolor="white")
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Δ mean part of the topic (women − men)")
    ax.set_title(f"Topics surinvested by women candidates — {year}\n"
                 f"(n_F={int((sex_known=='F').sum())}, "
                 f"n_H={int((sex_known=='H').sum())})",
                 loc="left", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/gender_comparison_{year}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_metadata_age(df: pd.DataFrame, W_norm: np.ndarray,
                      topic_label: Dict[int, str], year,
                      bins=(17, 35, 50, 65, 100),
                      bin_labels=("≤ 35 years", "36–50 years", "51–65 years", "> 65 years")):
    """Thematic profile by age group."""
    mask_year = (df["date"] == year).values
    W_y = W_norm[mask_year]
    sub = df[mask_year].copy()

    age = pd.to_numeric(sub["titulaire-age-calcule"], errors="coerce")
    sub["age_bin"] = pd.cut(age, bins=list(bins), labels=list(bin_labels), right=True)

    mask_age = sub["age_bin"].notna().values
    W_age = W_y[mask_age]
    ages = sub.loc[mask_age, "age_bin"].values

    n_topics = W_norm.shape[1]
    cols = [f"T{i}-{topic_label[i]}" for i in range(n_topics)]
    profiles = {}
    for lab in bin_labels:
        idx = (ages == lab)
        if idx.sum() >= 10:
            profiles[f"{lab} (n={int(idx.sum())})"] = W_age[idx].mean(axis=0)

    if not profiles:
        print(f"Not enough age entries in {year}.")
        return

    df_age = pd.DataFrame(profiles, index=cols).T
    national = W_y.mean(axis=0)
    df_age_delta = df_age - national

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(df_age_delta) + 1.8))
    sns.heatmap(df_age_delta, ax=ax, center=0, cmap="RdBu_r",
                annot=True, fmt=".02f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Δ vs national average"},
                linewidths=0.6, linecolor="white")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"Thematic Specialization by Generation — {year}",
                 loc="left", fontsize=13, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/age_comparison_{year}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_metadata_sortants(df: pd.DataFrame, W_norm: np.ndarray,
                           topic_label: Dict[int, str], year):
    """Comparison of exiting members vs new entrants."""
    mask_year = (df["date"] == year).values
    W_y = W_norm[mask_year]
    sub = df[mask_year].copy()

    sub["is_sortant"] = (
        sub["titulaire-mandat-passe"].astype(str).str.lower()
        .str.contains("député", na=False)
    )

    def has_any_mandate(row):
        for c in ["titulaire-mandat-passe", "titulaire-mandat-en-cours"]:
            s = str(row.get(c, "")).strip().lower()
            if s and s not in {"non mentionné", "nan", "none"}:
                return True
        return False

    sub["is_nouveau"] = ~sub.apply(has_any_mandate, axis=1)

    n_sort = int(sub["is_sortant"].sum())
    n_neuf = int(sub["is_nouveau"].sum())
    print(f"En {year} — sortants : {n_sort}, nouveaux entrants : {n_neuf}")

    if n_sort < 10 or n_neuf < 10:
        print("Effectifs insuffisants pour la comparaison.")
        return

    p_sort = W_y[sub["is_sortant"].values].mean(axis=0)
    p_neuf = W_y[sub["is_nouveau"].values].mean(axis=0)
    delta = p_sort - p_neuf

    n_topics = W_norm.shape[1]
    cols = [f"T{i}-{topic_label[i]}" for i in range(n_topics)]
    df_delta = pd.DataFrame({"Δ (sortants − nouveaux)": delta},
                            index=cols).sort_values("Δ (sortants − nouveaux)")

    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(cols) + 1.5))
    colors = ["#1f4e79" if v > 0 else "#c1432b"
              for v in df_delta["Δ (sortants − nouveaux)"]]
    ax.barh(df_delta.index, df_delta["Δ (sortants − nouveaux)"],
            color=colors, edgecolor="white")
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Δ part moyenne du topic (sortants − nouveaux entrants)")
    ax.set_title(f"Topics distinctifs des sortants vs nouveaux entrants — {year}",
                 loc="left", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/sortants_comparison_{year}.pdf", bbox_inches="tight", dpi=300)
    plt.show()


# Règles de catégorisation des professions
PROF_RULES = [
    ("enseignement",       ["instituteur", "institutrice", "professeur",
                            "enseignant", "maître", "directeur d'école",
                            "universitaire"]),
    ("santé",              ["médecin", "infirmier", "infirmière",
                            "pharmacien", "chirurgien", "dentiste",
                            "kinésithérapeute"]),
    ("droit",              ["avocat", "notaire", "magistrat", "juriste",
                            "huissier"]),
    ("agriculture",        ["agriculteur", "agricultrice", "exploitant agricole",
                            "viticulteur", "éleveur", "paysan"]),
    ("ouvrier / employé",  ["ouvrier", "ouvrière", "employé", "employée",
                            "salarié", "technicien"]),
    ("cadre / ingénieur",  ["ingénieur", "cadre", "directeur", "directrice",
                            "chef d'entreprise", "consultant",
                            "conseiller entreprise"]),
    ("commerce / artisan", ["commerçant", "artisan", "restaurateur"]),
    ("fonction publique",  ["fonctionnaire", "haut fonctionnaire",
                            "préfet", "sous-préfet", "administrateur"]),
    ("élu permanent",      ["député", "sénateur", "maire", "conseiller",
                            "permanent", "homme politique"]),
    ("journalisme / com",  ["journaliste", "écrivain", "communic"]),
]


def categorize_prof(value) -> Optional[str]:
    """Catégorise un libellé de profession en grande famille socioprofessionnelle."""
    s = str(value).strip().lower()
    if not s or s in {"non mentionné", "nan", "none"}:
        return None
    for cat, kws in PROF_RULES:
        if any(k in s for k in kws):
            return cat
    return None


def plot_metadata_profession(df: pd.DataFrame, W_norm: np.ndarray,
                             topic_label: Dict[int, str], year):
    """Profil thématique par grande famille professionnelle."""
    mask_year = (df["date"] == year).values
    W_y = W_norm[mask_year]
    sub = df[mask_year].copy()
    sub["prof_cat"] = sub["titulaire-profession"].apply(categorize_prof)

    mask_p = sub["prof_cat"].notna().values
    W_p = W_y[mask_p]
    cats = sub.loc[mask_p, "prof_cat"].values

    n_topics = W_norm.shape[1]
    cols = [f"T{i}-{topic_label[i]}" for i in range(n_topics)]
    profiles = {}
    for cat in pd.Series(cats).unique():
        idx = (cats == cat)
        if idx.sum() >= 10:
            profiles[f"{cat} (n={int(idx.sum())})"] = W_p[idx].mean(axis=0)

    if len(profiles) < 2:
        print("Trop peu de catégories professionnelles renseignées.")
        return

    df_prof = pd.DataFrame(profiles, index=cols).T
    from scipy.cluster.hierarchy import linkage, leaves_list
    try:
        Z = linkage(df_prof.values, method="average")
        order = leaves_list(Z)
        df_prof = df_prof.iloc[order]
    except Exception:
        pass

    national = W_y.mean(axis=0)
    df_prof_delta = df_prof - national

    fig, ax = plt.subplots(figsize=(11, 0.55 * len(df_prof_delta) + 2))
    sns.heatmap(df_prof_delta, ax=ax, center=0, cmap="RdBu_r",
                annot=True, fmt=".02f", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Δ vs moyenne nationale"},
                linewidths=0.6, linecolor="white")
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"Spécialisation thématique par catégorie professionnelle — {year}",
                 loc="left", fontsize=13, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(f"/Users/matheoleroy/Downloads/data_nlp/figures_topic/profession_comparison_{year}.pdf", bbox_inches="tight", dpi=300)
    plt.show()

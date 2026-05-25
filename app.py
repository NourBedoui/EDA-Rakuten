import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from PIL import Image
import os
import re
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

st.set_page_config(page_title="Rakuten EDA", page_icon="🛒", layout="wide")

st.markdown("""
<style>
input[type="radio"] { accent-color: #BF0000; }
div[role="radiogroup"] label svg { display: none; }
</style>
""", unsafe_allow_html=True)

DATA_PATH  = r"C:\Users\nour.bedoui\Desktop\Rakuten\data"
LOGO_PATH  = r"C:\Users\nour.bedoui\Desktop\Rakuten\data\Rakuten_Global_Brand_Logo.svg.webp"
TRAIN_PATH = r"C:\Users\nour.bedoui\Desktop\Rakuten\data\images\images\image_train"

CAT_NAMES = {
    10: "Livres occasion", 40: "Jeux vidéo consoles", 50: "Accessoires gaming",
    60: "Consoles jeux", 1140: "Collection vintage", 1160: "Cartes de jeux",
    1180: "Jouets divers", 1280: "Jouets enfants", 1281: "Jeux société",
    1300: "Modélisme", 1301: "Vêtements bébé", 1302: "Jeux plein air",
    1320: "Puériculture", 1560: "Mobilier/Déco", 1920: "Linge maison",
    1940: "Alimentation", 2060: "Déco intérieure", 2220: "Animalerie",
    2280: "Journaux/Revues", 2403: "Livres neufs", 2462: "Jeux vidéo occasion",
    2522: "Fournitures bureau", 2582: "Mobilier jardin", 2583: "Piscine/Spa",
    2585: "Bricolage", 2705: "DVD/Films", 2905: "Jeux vidéo PC"
}

@st.cache_data
def load_data():
    X_train = pd.read_csv(f"{DATA_PATH}/X_train_update.csv", index_col=0)
    Y_train = pd.read_csv(f"{DATA_PATH}/Y_train_CVw08PX.csv", index_col=0)
    df = pd.concat([X_train, Y_train], axis=1)
    df["label"] = df["prdtypecode"].map(CAT_NAMES)

    stop_fr     = set(stopwords.words("french"))
    stop_en     = set(stopwords.words("english"))
    stop_custom = {
        "les","des","est","plus","par","sur","pour","avec","sans",
        "non","lot","set","new","cm","ml","mg","kg","mm","xl","xs","générique",
        "der","die","das","und","fur","mit","von","auf","ein","eine",
        "den","dem","bei","aus","oder","wie","zum","zur","ist"
    }
    stop_all = stop_fr | stop_en | stop_custom

    def nettoyer(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zàâçéèêëîïôûùüÿæœ\s]", " ", text)
        return " ".join([m for m in text.split() if m not in stop_all and len(m) > 2])

    df["designation_clean"] = df["designation"].apply(nettoyer)
    df["nb_mots"]           = df["designation"].apply(lambda x: len(str(x).split()))
    df["has_desc"]          = df["description"].notna() & (df["description"].str.strip() != "")
    return df

@st.cache_data
def load_quality():
    try:
        return pd.read_csv(f"{DATA_PATH}/quality_df.csv")
    except:
        return None

df         = load_data()
quality_df = load_quality()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.image(LOGO_PATH, use_container_width=True)
st.sidebar.markdown("---")

section = st.sidebar.radio("Navigation", [
    "Vue d'ensemble",
    "Variable cible",
    "Analyse textuelle",
    "Contenu par catégorie",
    "Description",
    "Qualité des images",
])

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres")

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Tout sélect."):
    st.session_state["cats"] = [f"{c} — {CAT_NAMES.get(c,'')}" for c in sorted(df["prdtypecode"].unique())]
if col_btn2.button("Tout désélect."):
    st.session_state["cats"] = []

cats_dispo    = sorted(df["prdtypecode"].unique())
cats_label    = [f"{c} — {CAT_NAMES.get(c,'')}" for c in cats_dispo]
selection     = st.sidebar.multiselect(
    "Catégories",
    options=cats_label,
    default=st.session_state.get("cats", cats_label),
    key="cats"
)
cats_selected = [int(s.split(" — ")[0]) for s in selection]
df_filtered   = df[df["prdtypecode"].isin(cats_selected)] if cats_selected else df

# ── SECTION 1 — VUE D'ENSEMBLE ────────────────────────────────────────────────
if section == "Vue d'ensemble":
    st.title("Vue d'ensemble du dataset")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Produits total",        f"{len(df):,}")
    col2.metric("Catégories",            f"{df['prdtypecode'].nunique()}")
    col3.metric("Description manquante", f"{df['description'].isna().mean()*100:.1f}%")
    col4.metric("Nb mots moyen",         f"{df['nb_mots'].mean():.1f}")

    st.markdown("---")
    st.subheader("Valeurs manquantes")

    cols_brutes = ["designation", "description", "productid", "imageid", "prdtypecode"]
    t1 = pd.DataFrame({
        "Colonne"             : cols_brutes,
        "Type"                : [str(df[c].dtype) for c in cols_brutes],
        "Valeurs renseignées" : [df[c].notnull().sum() for c in cols_brutes],
        "Valeurs manquantes"  : [df[c].isnull().sum() for c in cols_brutes],
        "% manquant"          : [(df[c].isnull().mean() * 100).round(1) for c in cols_brutes],
        "Statut"              : ["Incomplet" if df[c].isnull().any() else "Complet" for c in cols_brutes]
    }).sort_values("% manquant", ascending=False).reset_index(drop=True)

    def color_statut(v):
        if v == "Incomplet": return "color: red; font-weight: bold"
        return "color: green; font-weight: bold"

    st.dataframe(t1.style
        .format({"% manquant": "{:.1f}%", "Valeurs renseignées": "{:,}", "Valeurs manquantes": "{:,}"})
        .applymap(color_statut, subset=["Statut"]),
        use_container_width=True)

    st.markdown("---")
    st.subheader("Doublons")

    n_total = len(df)
    t2 = pd.DataFrame({
        "Vérification" : ["Lignes dupliquées", "Désignations dupliquées", "ProductID dupliqués", "ImageID dupliqués"],
        "Nb doublons"  : [df.duplicated().sum(), df["designation"].duplicated().sum(),
                          df["productid"].duplicated().sum(), df["imageid"].duplicated().sum()],
        "% sur total"  : [round(df.duplicated().sum()/n_total*100, 2),
                          round(df["designation"].duplicated().sum()/n_total*100, 2),
                          round(df["productid"].duplicated().sum()/n_total*100, 2),
                          round(df["imageid"].duplicated().sum()/n_total*100, 2)]
    })
    st.dataframe(t2.style.format({"Nb doublons": "{:,}", "% sur total": "{:.2f}%"}),
                 use_container_width=True)

# ── SECTION 2 — VARIABLE CIBLE ────────────────────────────────────────────────
elif section == "Variable cible":
    st.title("Distribution de la variable cible")

    counts = df_filtered["prdtypecode"].value_counts()
    colors = cm.turbo(np.linspace(0.1, 0.9, len(counts)))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(counts)), counts.values, color=colors, width=0.7)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index.astype(str), rotation=45, fontsize=9)
    ax.set_xlabel("prdtypecode")
    ax.set_ylabel("Nombre de produits")
    ax.set_title("Nombre de produits par catégorie", fontweight="bold")
    total = counts.sum()
    for i, val in enumerate(counts.values):
        ax.annotate(f"{100*val/total:.1f}%", (i, val), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "Le dataset n'est pas équilibré : la catégorie 2583 (Piscine/Spa) représente à elle seule 12% des données, "
        "tandis que les 5 dernières catégories sont chacune sous les 1%. "
        "Ce déséquilibre est modéré — pas catastrophique — mais il faudra en tenir compte lors de l'évaluation "
        "en utilisant le macro F1-score plutôt que l'accuracy, et potentiellement ajuster les poids des classes."
    )

# ── SECTION 3 — ANALYSE TEXTUELLE ─────────────────────────────────────────────
elif section == "Analyse textuelle":
    st.title("Analyse textuelle")

    moy_cat = (df_filtered.groupby("prdtypecode")["nb_mots"]
                           .mean()
                           .sort_values(ascending=False))
    colors = cm.turbo(np.linspace(0.1, 0.9, len(moy_cat)))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(moy_cat)), moy_cat.values, color=colors, width=0.7)
    ax.axhline(df["nb_mots"].mean(), color="red", linestyle="--",
               linewidth=1.5, label=f"Moyenne : {df['nb_mots'].mean():.1f} mots")
    ax.set_xticks(range(len(moy_cat)))
    ax.set_xticklabels(moy_cat.index.astype(str), rotation=45, fontsize=9)
    ax.set_xlabel("prdtypecode")
    ax.set_ylabel("Nb mots moyen")
    ax.set_title("Nb de mots moyen par catégorie", fontweight="bold")
    for i, val in enumerate(moy_cat.values):
        ax.annotate(f"{val:.1f}", (i, val), ha="center", va="bottom", fontsize=7)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "Les désignations sont globalement courtes — environ 11 mots en moyenne. "
        "Les Journaux/Revues (2280) ont les titres les plus longs car ils incluent souvent la date et le numéro. "
        "Les DVD/Films (2705) ont les plus courts — le titre du film suffit. "
        "Dans tous les cas, on est loin de la limite de 512 tokens de CamemBERT, donc pas de troncature à craindre."
    )

# ── SECTION 4 — CONTENU PAR CATÉGORIE ─────────────────────────────────────────
elif section == "Contenu par catégorie":
    st.title("Contenu textuel par catégorie")

    cats_for_select = cats_selected if cats_selected else sorted(df["prdtypecode"].unique())
    cat_choice = st.selectbox(
        "Choisir une catégorie",
        options=[f"{c} — {CAT_NAMES.get(c,'')}" for c in cats_for_select]
    )
    cat_code = int(cat_choice.split(" — ")[0])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top 10 mots — {cat_choice}")
        textes    = " ".join(df[df["prdtypecode"] == cat_code]["designation_clean"])
        freq_dict = dict(Counter(textes.split()).most_common(10))
        top10     = pd.DataFrame(freq_dict.items(), columns=["Mot", "Fréquence"])
        colors    = cm.turbo(np.linspace(0.1, 0.9, len(top10)))
        fig, ax   = plt.subplots(figsize=(7, 5))
        ax.barh(top10["Mot"], top10["Fréquence"], color=colors)
        ax.invert_yaxis()
        ax.set_title(f"Top 10 mots — {cat_code}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Nuage de mots")
        wc = WordCloud(
            width=500, height=300, background_color="white",
            colormap="turbo", max_words=10, collocations=False
        ).generate_from_frequencies(freq_dict)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        plt.tight_layout()
        st.pyplot(fig2)

    st.info(
        "Les mots les plus fréquents reflètent bien le contenu de chaque catégorie — "
        "ce qui est rassurant pour la classification. "
        "Certaines catégories proches (ex: Jouets enfants / Jeux société) partagent des mots comme "
        "'jeu', 'enfants', 'jouet' — ce sont les cas où le modèle risque de se tromper le plus souvent."
    )

# ── SECTION 5 — DESCRIPTION ───────────────────────────────────────────────────
elif section == "Description":
    st.title("Analyse de la description")

    st.subheader("Taux de remplissage par catégorie")
    fill_rate = (df_filtered.groupby("prdtypecode")["has_desc"]
                             .mean()
                             .mul(100)
                             .sort_values(ascending=False))
    colors = cm.turbo(np.linspace(0.1, 0.9, len(fill_rate)))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(fill_rate)), fill_rate.values, color=colors, width=0.7)
    ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="50%")
    ax.axhline(fill_rate.mean(), color="gray", linestyle="--",
               linewidth=1.5, label=f"Moyenne : {fill_rate.mean():.1f}%")
    ax.set_xticks(range(len(fill_rate)))
    ax.set_xticklabels(fill_rate.index.astype(str), rotation=45, fontsize=9)
    ax.set_xlabel("prdtypecode")
    ax.set_ylabel("% produits avec description")
    ax.set_title("Taux de remplissage description par catégorie", fontweight="bold")
    for i, val in enumerate(fill_rate.values):
        ax.annotate(f"{val:.0f}%", (i, val), ha="center", va="bottom", fontsize=7)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "Le taux de remplissage varie énormément selon les catégories. "
        "Les produits physiques (mobilier, piscine, bricolage) ont presque toujours une description — "
        "les vendeurs la renseignent naturellement. "
        "Pour les livres et médias, c'est rarement le cas : le titre identifie déjà le produit. "
        "Pour ces catégories, le modèle devra se baser presque exclusivement sur la désignation."
    )

    st.markdown("---")
    st.subheader("Mean Description Coverage")

    @st.cache_data
    def compute_coverage(data):
        def desc_coverage(row):
            if pd.isna(row["description"]) or str(row["description"]).strip() == "":
                return 0.0
            mots_titre = set(str(row["designation"]).lower().split())
            mots_desc  = set(str(row["description"]).lower().split())
            if not mots_titre: return 0.0
            return len(mots_titre & mots_desc) / len(mots_titre)
        return data.apply(desc_coverage, axis=1)

    with st.spinner("Calcul de la coverage..."):
        df_cov             = df_filtered.copy()
        df_cov["coverage"] = compute_coverage(df_cov)

    moy_cov = (df_cov.groupby("prdtypecode")["coverage"]
                      .mean()
                      .sort_values(ascending=False))
    colors = cm.turbo(np.linspace(0.1, 0.9, len(moy_cov)))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(len(moy_cov)), moy_cov.values * 100, color=colors, width=0.7)
    ax.axhline(moy_cov.mean() * 100, color="gray", linestyle="--",
               linewidth=1.5, label=f"Moyenne : {moy_cov.mean()*100:.1f}%")
    ax.set_xticks(range(len(moy_cov)))
    ax.set_xticklabels(moy_cov.index.astype(str), rotation=45, fontsize=9)
    ax.set_xlabel("prdtypecode")
    ax.set_ylabel("Coverage moyenne (%)")
    ax.set_title("Mean Description Coverage par catégorie", fontweight="bold")
    for i, val in enumerate(moy_cov.values):
        ax.annotate(f"{val*100:.0f}%", (i, val*100), ha="center", va="bottom", fontsize=7)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "Une coverage élevée (ex: Linge maison 67%) signifie que la description reprend largement "
        "les mêmes mots que le titre — elle n'apporte pas grand chose de nouveau. "
        "Une coverage faible (ex: Journaux 1%) signifie que quand une description existe, "
        "elle parle vraiment d'autre chose que le titre — c'est là qu'elle est utile. "
        "À noter : les catégories avec peu de descriptions ET une coverage faible sont les plus intéressantes "
        "à exploiter si on arrive à récupérer ces descriptions."
    )

# ── SECTION 6 — QUALITÉ DES IMAGES ───────────────────────────────────────────
elif section == "Qualité des images":
    st.title("Qualité des images")

    if quality_df is None:
        st.warning("Le fichier quality_df.csv n'est pas encore disponible.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Images totales",     f"{len(quality_df):,}")
        col2.metric("Images vides",       f"{quality_df['est_vide'].sum():,}")
        col3.metric("Produit trop petit", f"{quality_df['est_petit'].sum():,}")
        col4.metric("Images dupliquées",  f"{quality_df['est_dup'].sum():,}")

        st.markdown("---")
        st.subheader("Répartition par catégorie")

        probleme = st.selectbox(
            "Type de problème",
            options=["est_petit", "est_dup", "est_vide"],
            format_func=lambda x: {
                "est_vide" : "Images vides",
                "est_petit": "Produit trop petit",
                "est_dup"  : "Images dupliquées"
            }[x]
        )

        pb_cat = (quality_df[quality_df[probleme]]
                  .groupby("prdtypecode")
                  .size()
                  .sort_values(ascending=False)
                  .reset_index())
        pb_cat.columns = ["prdtypecode", "Nb images"]
        pb_cat["label"] = pb_cat["prdtypecode"].map(CAT_NAMES)

        colors = cm.turbo(np.linspace(0.1, 0.9, len(pb_cat)))
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(range(len(pb_cat)), pb_cat["Nb images"], color=colors, width=0.7)
        ax.set_xticks(range(len(pb_cat)))
        ax.set_xticklabels(pb_cat["prdtypecode"].astype(str), rotation=45, fontsize=9)
        ax.set_title("Nb images problématiques par catégorie", fontweight="bold")
        for i, val in enumerate(pb_cat["Nb images"]):
            ax.annotate(str(val), (i, val), ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)

        st.info(
            "Les images vides et produits trop petits sont marginaux (< 1%) — pas de nettoyage urgent. "
            "En revanche, les 8 957 images dupliquées (10.5%) méritent attention : "
            "plusieurs produits différents partagent la même image. "
            "Ce n'est pas forcément une erreur — un même visuel peut illustrer plusieurs références — "
            "mais ça peut compliquer l'apprentissage du modèle image qui ne pourra pas distinguer ces produits visuellement."
        )

        st.markdown("---")
        st.subheader("Aperçu des images")

        n_afficher = st.slider("Nombre d'images à afficher", 5, 30, 10)
        subset_imgs = quality_df[quality_df[probleme]].head(n_afficher)

        n_cols = 5
        rows   = [subset_imgs.iloc[i:i+n_cols] for i in range(0, len(subset_imgs), n_cols)]

        for row in rows:
            cols = st.columns(n_cols)
            for col, (_, r) in zip(cols, row.iterrows()):
                img_path = os.path.join(TRAIN_PATH, r["filename"])
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    col.image(img, caption=f"{CAT_NAMES.get(r['prdtypecode'], '')}", use_container_width=True)
                else:
                    col.warning("Image non trouvée")
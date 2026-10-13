#import "@preview/typslides:1.3.3": *
#import "@preview/diagraph:0.3.0": raw-render
#import "@preview/algo:0.3.6": algo, comment, d, i
#import "schemas.typ": *

#show: typslides.with(
  ratio: "16-9",
  theme: "purply",
  font: "Fira Sans",
  font-size: 20pt,
  link-style: "color",
)

#front-slide(
  title: "Deep Dive : Mistral 7B",
  subtitle: [Mistral.AI - 2023],
  authors: [
    Loïc Desmarès, Fayssal Zakaria, Yassine Chaouch\
    #link("https://github.com/LugolBis/mistral-7b-deep-dive")
  ],
)

#table-of-contents(title: "Sommaire")

#slide(title: "Introduction", outlined: true)[
  *Qu'est ce qu'un LLM ?*\
  \
  Un LLM (Large Language Model) est un modèle d'intelligence artificielle entraîné sur de grandes quantités de données afin d'apprendre les régularités du langage et de représenter les relations entre les mots.\
  Il repose généralement sur une architecture de réseau de neurones appelée _Transformer_ et génère du texte en prédisant successivement le prochain *token* (un mot) à partir du contexte précédent.\
  Cette solution est non déterministe et la qualité des poids (weights) d'un modèle dépend de celle des données.
]

#slide(title: "Introduction", outlined: true)[
  #align(center)[
    #NN
  ]
]

#slide(title: "Introduction", outlined: true)[
  *Pipeline d'Inférence* :\
  \
  #InferencePipeline
]

#slide(title: "Pre-fill", outlined: true)[
  *Qu'est ce que le _Pre-fill_* ?\
  Le _Pre-fill_ est la phase d'encodage des tokens (mots) extraits du prompt, en vecteurs de représentations.
  Cet encodage se fait à partir des couches du modèles et donc des poids appris par celui-ci.\
  \
  On obtient donc pour chaque token, 3 vecteurs :
  - $Q$ : Les "queries" utilisées pour rechercher les informations contextuelles pertinentes.
  - $K$ : Les "keys" sont comparées avec chaque "query" pour calculer un score d'attention entre les tokens.
  - $V$ : Les "values" contiennent les informations qui sont agrégées selon la pondération du score d'attention.

]

#slide(title: "Pre-fill", outlined: true)[
  *SWA* :\
  \
  La formule de l'attention : $A = "softmax"((Q K^T) / sqrt(d_k))V$\
  $->$ On a donc $O = A V$ qui correspond a $V$ aggrégé selon l'attention.\
  \
  Calculer le score d'attention agrégé pour tous les tokens est couteux en temps, il augmente de manière quadratique selon _N_.\
  Le _*Sliding Window Attention*_ (*SWA*) résout ce problème en réduisant la fenêtre des clés $K$ comparées à une requête _Q_ pour chaque couche du modèle lors du calcul du score d'attention _A_.
]

#slide(title: "Pre-fill", outlined: true)[
  Dans l'exemple suivant $N = 6$ et $W = 2$ les tokens $t_i$ sont lus par les $Q_j$ avec $j in [j; j + W]$
  #align(center)[#SWA]
]

#slide(title: "Pre-fill", outlined: true)[
  *Chunking* :\
  TODO
]

#slide(title: "Cache KV", outlined: true)[
  *Qu'est ce que le cache KV* ?\
  TODO
]

#slide(title: "Cache KV", outlined: true)[
  *GQA* :\
  TODO
]

#slide(title: "Cache KV", outlined: true)[
  *Rolling Buffer* :\
  TODO
]

#slide(title: "Décodage", outlined: true)[
  *Comment fonctionne la phase de _Décodage_* ?\
  TODO
]

#slide(title: "Décodage", outlined: true)[
  *SWA* :\
  TODO
]

#slide(title: "Paramètres de Mistral 7B", outlined: true)[
  *Paramètres de _Mistral 7B_*@jiang2023mistral7b :\

  #table(
    columns: (auto, auto, 1fr),
    align: (left, center, left),
    [*Paramètre*], [*Valeur*], [*Description*],
    [dim], [4096], [Dimension du modèle (dimension du vecteur représentant un token)],
    [n_layers], [32 couches], [Nombre de couches cachées],
    [n_heads], [32 têtes], [Nombre de têtes d'attention],
    [head_dim], [128], [Dimension par tête d'attention],
    [n_kv_heads], [8 têtes], [Nombre de têtes clé/valeur (KV)],
    [window_size (#emph[W])], [4096 tokens], [Taille de la fenêtre glissante d'attention],
    [context_len], [8192 tokens], [Longueur de contexte],
    [vocab_size], [32000 tokens], [Taille du vocabulaire (nombre de tokens distinct)],
    [buffer_cache], [4096 tokens], [Taille du cache tampon circulaire],
    [n_parameters], [7 milliards], [Nombre de paramètres du modèle],
  )
]

#slide(title: "Conclusion", outlined: true)[
  *_TODO_*
]

// Bibliography
#bibliography-slide(bibliography("references.bib"))

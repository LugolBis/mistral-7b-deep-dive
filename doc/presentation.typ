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
  On obtient donc pour chaque token, 3 ensembles de vecteurs :
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
  Exemple : $N = 6$ et $W = 2$, les tokens $t_i$ sont lus par les $Q_j$ avec $j in [i; i + W]$
  #align(center)[#SWA]
]

#slide(title: "Pre-fill", outlined: true)[
  *Chunking* :\
  On découpe le prompt en chunks de _C_ tokens. Les calculs des scores d'attention des tokens d'un chunk peuvent alors être parallélisés, tout en respectant le masque causal et la *SWA*.\
  \
  _Mistral 7B_ fait le choix de poser : $C = W$. Cela permet de profiter du cache mémoire lors du calcul du score d'attention et plus particulièrement lorsque l'on passe d'un chunk à un autre.\
  #PrefillCache
]

#slide(title: "Cache KV", outlined: true)[
  *Qu'est ce que le cache KV* ?\
  Le *KV* cache permet de mettre en cache les représentations vectorielles (*K* et *V*) des tokens du prompt pour chaque couche du modèle.\
  \
  Le cache *KV* est rempli durant la phase de _Pre-fill_ et peut être mis à jour lors de la phase d'encodage ou avant (voir ci-après).\
  \
  _Mistral 7B_ dispose de 8 têtes *KV* par couche du modèle, qui sont chacune utilisées par 4 têtes de requêtes (*Q*) $->$ c'est le *GQA*.
]

#slide(title: "Cache KV", outlined: true)[
  *GQA* :\
  Le _Grouped Query Attention_ consiste à regrouper des requêtes (*Q*) distinctes pour qu'elles partagent les mêmes vecteurs (*K*, *V*).\
  #GQA
]

#slide(title: "Cache KV", outlined: true)[
  *Rolling Buffer* :\
  Un cache de taille fixe de longueur $W$. On écrit donc le token $t_i$ à la position $i mod W$ du buffer cache.\
  \
  Étant donné que pour calculer les représentations (*Q*, *K*, *V*) d'un token $t_i$ nous n'avons besoin que des $W$ derniers tokens, cela signifie que nous n'avons pas besoin de garder les tokens précédents, qui ont déjà impacté les représentations (*Q*, *K*, *V*) des $W$ derniers tokens du *KV* cache.\
  \
  #underline()[Il y a propagation de l'information !]\
  L'influence des tokens passés peut atteindre $1 + k (W-1)$ après $k$ couches.\
  De plus la taille de la mémoire est linéaire en $O(min(N, W))$.
]


#slide(title: "Décodage", outlined: true)[
  *Comment fonctionne la phase de décodage ?*

  Le *KV* cache construit pendant le _Pre-fill_ contient, pour chaque couche, les clés $K$ et les valeurs $V$ des tokens déjà traités.

  \

  *Initialisation : générer le premier token*

  À la fin du _Pre-fill_, les logits $l_0$ de la dernière position
  du prompt permettent de calculer la distribution du prochain token :
  $p_"next_token" = "softmax"(l_0)$.\
  On sélectionne alors le premier token $y_1$ selon la stratégie
  de décodage choisie.
]

#slide(title: "Décodage", outlined: true)[
  *Boucle de décodage : prédire chaque token suivant*

  + On transmet le token courant $y_t$ au modèle. À chaque couche, le modèle calcule ses ($Q$, $K$, $V$) et met à jour le *KV* cache.

  + Le modèle calcule $O$ selon la *SWA*.

  + Le vecteur caché final $h_t$ est projeté sur le vocabulaire pour
    produire les logits du token suivant :
    $l_t = W_"vocab" h_t$.

  + On calcule la distribution de probabilité des tokens candidats : $p_"next_token" = "softmax"(l)$. Puis on sélectionne le token $y_(t+1)$ selon la stratégie de décodage (glouton, échantillonnage, etc.).

  + Si un token de fin est généré ou si une limite de génération
    est atteinte, on arrête. Sinon, on transmet $y_(t+1)$ au modèle
    et on répète la boucle.
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
    [chunk_size (#emph[C])], [4096 tokens], [Taille d'un chunk lors du Pre-fill],
    [context_len], [8192 tokens], [Longueur de contexte],
    [vocab_size], [32000 tokens], [Taille du vocabulaire (nombre de tokens distinct)],
    [buffer_cache], [4096 tokens], [Taille du cache tampon circulaire],
    [n_parameters], [7 milliards], [Nombre de paramètres du modèle],
  )
]

#slide(title: "Conclusion", outlined: true)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 0.6cm,
    [
      #figtitle[Cache d'une séquence selon sa longueur]
      #OptiGQA

      #cap[Linéaire en N : GQA divise par 4.\ À 32 K tokens : 16 → 4 → 0,5 Gio.]
    ],
    [
      #figtitle[Scores d'attention calculés (par head et par layer)]
      #OptiSWA

      #cap[Classique : O(N²).\ SWA : O(N × W), donc linéaire en N au-delà de W.]
    ],
  )
]

#slide(title: "Conclusion", outlined: true)[
  #grid(
    columns: (1fr, 1fr),
    column-gutter: 0.6cm,
    [
      #figtitle[Entrées stockées selon la longueur]
      #OptiRolling

      #cap[Le rolling buffer plafonne à W = 4096 :\ le stockage devient constant, ÷ 8 à 32 K tokens.]
    ],
    [
      #figtitle[Taille logique de la matrice de scores (par head)]
      #OptiPrefill

      #cap[Le chunking remplace N × N par C × (W + C) = 34 M,\ constant dès que N > W.]
    ],
  )
]

// Bibliography
#bibliography-slide(bibliography("references.bib"))

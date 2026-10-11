#import "@preview/fletcher:0.5.8": diagram, edge, node

#set page(width: auto, height: auto, margin: 8mm)
#set text(size: 9pt)

#let n = 6          // neurones par couche (input + 2 hidden)
#let r = 4.5mm      // rayon des neurones

#let NN = [#diagram(
    spacing: (48mm, 13mm),
    node-stroke: 0.7pt,
    edge-stroke: 0.4pt + luma(80),
    {
      // ---------- Noms des couches ----------
      node((0, -1.2), text(size: 1 * r)[*Input layer*], stroke: none, fill: rgb("#959ce7"))
      node((1, -1.2), text(size: 1 * r)[*Layer 1*], stroke: none, fill: rgb(221, 194, 82, 255))
      node((2, -1.2), text(size: 1 * r)[*Layer 2*], stroke: none, fill: rgb(246, 219, 115, 255))
      node((3, -1.2), text(size: 1 * r)[*Output layer*], stroke: none, fill: rgb(82, 221, 170, 255))

      // ---------- Neurones ----------
      // Input layer
      for i in range(n) {
        node((0, i), text(size: 1 * r)[$x_#(i + 1)$], shape: circle, radius: r, fill: rgb("#959ce7"))
      }
      // Hidden layer 1 et 2
      for i in range(n) {
        node((1, i), text(size: 1 * r)[$h^((1))_#(i + 1)$], shape: circle, radius: r, fill: rgb(221, 194, 82, 255))
        node((2, i), text(size: 1 * r)[$h^((2))_#(i + 1)$], shape: circle, radius: r, fill: rgb(246, 219, 115, 255))
      }
      // Output layer (un seul neurone, centré verticalement)
      node((3, (n - 1) / 2), text(size: 1 * r)[$hat(y)$], shape: circle, radius: r, fill: rgb(82, 221, 170, 255))

      // ---------- Arêtes pondérées (w0, w1, ...) ----------
      let k = 0

      // Input -> Layer 1
      for i in range(n) {
        for j in range(n) {
          edge(
            (0, i),
            (1, j),
            "->",
            label: text(size: 5pt)[$w_(#k)$],
            label-pos: 0.72,
            label-fill: white,
            label-sep: 1pt,
          )
          k += 1
        }
      }

      // Layer 1 -> Layer 2
      for i in range(n) {
        for j in range(n) {
          edge(
            (1, i),
            (2, j),
            "->",
            label: text(size: 5pt)[$w_(#k)$],
            label-pos: 0.72,
            label-fill: white,
            label-sep: 1pt,
          )
          k += 1
        }
      }

      // Layer 2 -> Output layer
      for i in range(n) {
        edge(
          (2, i),
          (3, (n - 1) / 2),
          "->",
          label: text(size: 5pt)[$w_(#k)$],
          label-pos: 0.6,
          label-fill: white,
          label-sep: 1pt,
        )
        k += 1
      }
    },
  )
]

#let InferencePipeline = [#diagram(
  spacing: (48mm, 13mm),
  node-inset: 10pt,
  node-stroke: 0.7pt,
  edge-stroke: 1.5pt + luma(80),
  node-corner-radius: 7pt,
  {
    node((0, 0), [*Prompt*\ de $N$ tokens], shape: rect, fill: rgb("#959ce7"))
    node((0.75, 0), [*Pre-fill*\ les tokens\ du prompt], shape: rect, fill: rgb("#eea189"))
    node((1.5, 0), [*KV cache*\ ($K$, $V$)\ par Token\ par Layer], shape: rect, fill: rgb("#5cbefb"))
    node((2.25, 0), [*Décodage*\ 1 Token/pas], shape: rect, fill: rgb(246, 219, 115, 255))
    node((3, 0), [*Token*\ généré], shape: rect, fill: rgb(82, 221, 170, 255))

    edge((0, 0), (0.75, 0), "->")
    edge((0.75, 0), (1.5, 0), "->")
    edge((1.5, 0), (2.25, 0), "->")
    edge((2.25, 0), (3, 0), "->")
    edge(
      (3, 0),
      (1.5, 0),
      "->",
      bend: 50deg,
      label: [Si la limite de contexte n'est pas atteinte\ on met à jour le *KV* cache et on poursuit\ la génération.],
    )
  },
)]

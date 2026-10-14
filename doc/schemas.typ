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

#let SWA = [
  #let build-diagram() = {
    let tokens = ("The", "cat", "sat", "on", "the", "mat")
    let layer-names = ("Input", "Layer 1", "Layer 2", "Output")
    let W = 2
    let n-tok = 6
    let n-lay = 4
    let focus = 5
    let xs = 2
    let ys = 0.6

    let reaches = range(n-lay).map(L => {
      let steps = n-lay - 1 - L
      let start = calc.max(0, focus - W * steps)
      range(start, focus + 1)
    })
    let in-cone(L, i) = i in reaches.at(L)

    let red-fill = rgb("#E8401C")
    let red-stroke = rgb("#A82712")
    let pale-fill = rgb("#FBE5DC")
    let pale-stroke = rgb("#D97742")
    let gray-edge = rgb("#C4C4C4")
    let blue-head = rgb("#3A6EA5")
    let blue-dark = rgb("#21416E")
    let elems = ()

    for L in range(n-lay) {
      for i in range(n-tok) {
        let on = in-cone(L, i)
        elems.push(node(
          (L * xs, i * ys),
          text(size: 8.5pt, weight: "bold", fill: if on { white } else { rgb("#222") })[t#i],
          stroke: (if on { red-stroke } else { pale-stroke }) + 1.3pt,
          fill: if on { red-fill } else { pale-fill },
          inset: 7pt,
          corner-radius: 50%,
        ))
      }
    }

    for L in range(n-lay) {
      elems.push(node(
        (L * xs, -0.5),
        box(
          fill: blue-head,
          stroke: blue-dark + 0.8pt,
          inset: (x: 10pt, y: 4pt),
          radius: 4pt,
          text(size: 9.5pt, weight: "bold", fill: white)[#layer-names.at(L)],
        ),

        inset: 0pt,
      ))
    }

    for i in range(n-tok) {
      elems.push(node(
        (-0.5, i * ys),
        text(size: 9pt, style: "italic", fill: rgb("#333"))[#tokens.at(i)],

        inset: 0pt,
      ))
    }

    for L in range(n-lay - 1) {
      for i in range(n-tok) {
        for j in range(calc.max(0, i - W), i + 1) {
          let on = in-cone(L + 1, i) and in-cone(L, j)
          elems.push(edge(
            (L * xs, j * ys),
            ((L + 1) * xs, i * ys),
            marks: "->",
            stroke: (if on { red-fill } else { gray-edge }) + (if on { 1.5pt } else { 0.5pt }),
          ))
        }
      }
    }

    diagram(
      node-stroke: none,
      node-inset: 0pt,
      axes: (ltr, ttb),
      ..elems,
    )
  }


  #build-diagram()
]

#let PrefillCache = [
  #diagram(
    node-stroke: 0.6pt,
    node-fill: white,
    spacing: (1.5em, 2.8em),

    node(
      (0, 0),
      [
        #grid(
          column-gutter: 5pt,
          columns: 2,
          [*Chunk 1*],
          table(
            inset: 7pt,
            columns: 4,
            [$t_1$], [$t_2$], [$t_3$], [$t_4$],
          ),
        )
      ],
      fill: rgb("#D7EBDD"),
      name: <c1>,
      inset: 10pt,
    ),

    node(
      (1, 0),
      [
        #grid(
          column-gutter: 5pt,
          columns: 2,
          [*Chunk 2*],
          table(
            inset: 7pt,
            columns: 4,
            [$t_5$], [$t_6$], [$t_7$], [$t_8$],
          ),
        )
      ],
      fill: rgb("#D7EBDD"),
      name: <c2>,
      inset: 10pt,
    ),

    node(
      (2, 0),
      [
        #grid(
          column-gutter: 5pt,
          columns: 2,
          [*Chunk 3*],
          table(
            inset: 7pt,
            columns: 4,
            [$t_9$], [$t_10$], [$t_11$], [$t_12$],
          ),
        )
      ],
      fill: rgb("#D7EBDD"),
      name: <c3>,
      inset: 10pt,
    ),

    node(
      (0, 1),
      [
        #grid(
          column-gutter: 5pt,
          columns: 2,
          [*KV cache*],
          table(
            inset: 7pt,
            columns: 4,
            [$t_1$], [$t_2$], [$t_3$], [$t_4$],
          ),
        )
      ],
      fill: rgb("#5cbefb"),
      name: <kv1>,
      inset: 10pt,
    ),

    node(
      (1, 1),
      [
        #grid(
          column-gutter: 5pt,
          columns: 2,
          [*KV cache*],
          table(
            inset: 7pt,
            columns: 4,
            [$t_1$], [...], [$t_7$], [$t_8$],
          ),
        )
      ],
      fill: rgb("#5cbefb"),
      name: <kv2>,
      inset: 10pt,
    ),

    node(
      (2, 1),
      [
        #grid(
          column-gutter: 5pt,
          columns: 2,
          [*KV cache*],
          table(
            inset: 7pt,
            columns: 4,
            [$t_1$], [...], [$t_11$], [$t_12$],
          ),
        )
      ],
      fill: rgb("#5cbefb"),
      name: <kv3>,
      inset: 10pt,
    ),

    edge(<c1>, <kv1>, "->", label: [SWA], label-side: left),
    edge(<c2>, <kv2>, "->", label: [SWA], label-side: left),
    edge(<c3>, <kv3>, "->", label: [SWA], label-side: left),

    edge(<kv1>, <c2>, "->", bend: -10deg, stroke: rgb("#356AC3"), label-side: right),

    edge(<kv2>, <c3>, "->", bend: -10deg, stroke: rgb("#356AC3"), label-side: right),
  )
]

#let GQA = [
  #diagram(
    node-stroke: 0.4pt,

    for idx_left in range(8) {
      let label_query = idx_left + 1
      let node_q = (0.45 * idx_left, 0)
      let node_kv = if calc.odd(idx_left) {
        (0.45 * idx_left, 2.5)
      } else {
        (0.45 * idx_left, 2)
      }

      node(
        node_q,
        [$Q_(#label_query)$],
        fill: rgb("#f4d2f9"),
      )

      node(
        node_kv,
        [$K_(#label_query)$,$V_(#label_query)$],
        fill: rgb("#5cbefb"),
      )

      edge(
        node_q,
        node_kv,
        "->",
      )
    },

    edge(
      (3.55, -0.5),
      (3.55, 3),
      "-",
    ),

    for idx_kv_right in range(2) {
      let label_kv = idx_kv_right + 1
      let node_kv = (4.5 + 1.8 * idx_kv_right, 2.5)

      node(
        node_kv,
        [$K_(#label_kv)$,$V_(#label_kv)$],
        fill: rgb("#5cbefb"),
      )

      for idx_q_right in range(4) {
        let label_query = 4 * idx_kv_right + idx_q_right + 1
        let node_q = (3.8 + 1.8 * idx_kv_right + 0.45 * idx_q_right, 1)

        node(
          node_q,
          [$Q_(#label_query)$],
          fill: rgb("#f4d2f9"),
        )

        edge(
          node_q,
          node_kv,
          "->",
        )
      }
    },
  )
]

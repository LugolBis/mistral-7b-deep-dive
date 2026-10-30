#import "@preview/fletcher:0.5.8": diagram, edge, node

#set page(width: auto, height: auto, margin: 8mm)
#set text(size: 9pt)

// 1. Neural networks
#let neurons-per-layer = 6
#let neuron-radius = 4.5mm

#let layer-specs = (
  (name: "Input layer", fill: rgb("#959ce7")),
  (name: "Layer 1", fill: rgb(221, 194, 82)),
  (name: "Layer 2", fill: rgb(246, 219, 115)),
  (name: "Output layer", fill: rgb(82, 221, 170)),
)

#let NN = [#diagram(
  spacing: (48mm, 13mm),
  node-stroke: 0.7pt,
  edge-stroke: 0.4pt + luma(80),
  {
    for (layer-index, spec) in layer-specs.enumerate() {
      node(
        (layer-index, -1.2),
        text(size: neuron-radius)[*#spec.name*],
        stroke: none,
        fill: spec.fill,
      )
    }

    for neuron-index in range(neurons-per-layer) {
      node(
        (0, neuron-index),
        text(size: neuron-radius)[$x_#(neuron-index + 1)$],
        shape: circle,
        radius: neuron-radius,
        fill: layer-specs.at(0).fill,
      )
    }

    for neuron-index in range(neurons-per-layer) {
      for hidden-layer in (1, 2) {
        node(
          (hidden-layer, neuron-index),
          text(size: neuron-radius)[$h^((#hidden-layer))_#(neuron-index + 1)$],
          shape: circle,
          radius: neuron-radius,
          fill: layer-specs.at(hidden-layer).fill,
        )
      }
    }

    let output-y = (neurons-per-layer - 1) / 2
    node(
      (3, output-y),
      text(size: neuron-radius)[$hat(y)$],
      shape: circle,
      radius: neuron-radius,
      fill: layer-specs.at(3).fill,
    )

    let weight-index = 0

    for source-layer in (0, 1) {
      for source-neuron in range(neurons-per-layer) {
        for target-neuron in range(neurons-per-layer) {
          edge(
            (source-layer, source-neuron),
            (source-layer + 1, target-neuron),
            "->",
            label: text(size: 5pt)[$w_(#weight-index)$],
            label-pos: 0.72,
            label-fill: white,
            label-sep: 1pt,
          )
          weight-index += 1
        }
      }
    }

    for source-neuron in range(neurons-per-layer) {
      edge(
        (2, source-neuron),
        (3, output-y),
        "->",
        label: text(size: 5pt)[$w_(#weight-index)$],
        label-pos: 0.6,
        label-fill: white,
        label-sep: 1pt,
      )
      weight-index += 1
    }
  },
)]

// 2. Inference pipeline
#let pipeline-steps = (
  (label: [*Prompt*\ de $N$ tokens], fill: rgb("#959ce7")),
  (label: [*Pre-fill*\ les tokens\ du prompt], fill: rgb("#eea189")),
  (label: [*KV cache*\ ($K$, $V$)\ par Token\ par Layer], fill: rgb("#5cbefb")),
  (label: [*Décodage*\ 1 Token/pas], fill: rgb(246, 219, 115)),
  (label: [*Token*\ généré], fill: rgb(82, 221, 170)),
)

#let InferencePipeline = [#diagram(
  spacing: (48mm, 13mm),
  node-inset: 10pt,
  node-stroke: 0.7pt,
  edge-stroke: 1.5pt + luma(80),
  node-corner-radius: 7pt,
  {
    for (step-index, step) in pipeline-steps.enumerate() {
      node(
        (0.75 * step-index, 0),
        step.label,
        shape: rect,
        fill: step.fill,
      )
    }

    for step-index in range(pipeline-steps.len() - 1) {
      edge(
        (0.75 * step-index, 0),
        (0.75 * (step-index + 1), 0),
        "->",
      )
    }

    edge(
      (3, 0),
      (1.5, 0),
      "->",
      bend: 50deg,
      label: [Si la limite de contexte n'est pas atteinte\ on met à jour le *KV* cache et on poursuit\ la génération.],
    )
  },
)]

// 3. Sliding Window Attention (SWA)
#let SWA = [
  #let build-diagram() = {
    let tokens = ("The", "cat", "sat", "on", "the", "mat")
    let layer-names = ("Input", "Layer 1", "Layer 2", "Output")
    let attention-window = 2
    let token-count = 6
    let layer-count = 4
    let focus-token-index = 5
    let layer-dx = 2
    let token-dy = 0.6

    let red-fill = rgb("#E8401C")
    let red-stroke = rgb("#A82712")
    let pale-fill = rgb("#FBE5DC")
    let pale-stroke = rgb("#D97742")
    let gray-edge = rgb("#C4C4C4")
    let header-fill = rgb("#3A6EA5")
    let header-stroke = rgb("#21416E")

    let attention-reach(layer-index) = {
      let steps-back = layer-count - 1 - layer-index
      let start-token = calc.max(0, focus-token-index - attention-window * steps-back)
      range(start-token, focus-token-index + 1)
    }
    let in-attention-cone(layer-index, token-index) = token-index in attention-reach(layer-index)

    let elements = ()

    for layer-index in range(layer-count) {
      for token-index in range(token-count) {
        let highlighted = in-attention-cone(layer-index, token-index)
        elements.push(node(
          (layer-index * layer-dx, token-index * token-dy),
          text(
            size: 8.5pt,
            weight: "bold",
            fill: if highlighted { white } else { rgb("#222") },
          )[t#token-index],
          stroke: (if highlighted { red-stroke } else { pale-stroke }) + 1.3pt,
          fill: if highlighted { red-fill } else { pale-fill },
          inset: 7pt,
          corner-radius: 50%,
        ))
      }
    }

    for layer-index in range(layer-count) {
      elements.push(node(
        (layer-index * layer-dx, -0.5),
        box(
          fill: header-fill,
          stroke: header-stroke + 0.8pt,
          inset: (x: 10pt, y: 4pt),
          radius: 4pt,
          text(size: 9.5pt, weight: "bold", fill: white)[#layer-names.at(layer-index)],
        ),
        inset: 0pt,
      ))
    }

    for token-index in range(token-count) {
      elements.push(node(
        (-0.5, token-index * token-dy),
        text(size: 9pt, style: "italic", fill: rgb("#333"))[#tokens.at(token-index)],
        inset: 0pt,
      ))
    }

    for layer-index in range(layer-count - 1) {
      for target-token in range(token-count) {
        for source-token in range(
          calc.max(0, target-token - attention-window),
          target-token + 1,
        ) {
          let highlighted = (
            in-attention-cone(layer-index + 1, target-token) and in-attention-cone(layer-index, source-token)
          )
          elements.push(edge(
            (layer-index * layer-dx, source-token * token-dy),
            ((layer-index + 1) * layer-dx, target-token * token-dy),
            marks: "->",
            stroke: (
              (if highlighted { red-fill } else { gray-edge }) + (if highlighted { 1.5pt } else { 0.5pt })
            ),
          ))
        }
      }
    }

    diagram(
      node-stroke: none,
      node-inset: 0pt,
      axes: (ltr, ttb),
      ..elements,
    )
  }

  #build-diagram()
]

// 4. Prefill avec KV cache
#let chunks = (
  ("Chunk 1", ($t_1$, $t_2$, $t_3$, $t_4$)),
  ("Chunk 2", ($t_5$, $t_6$, $t_7$, $t_8$)),
  ("Chunk 3", ($t_9$, $t_10$, $t_11$, $t_12$)),
)
#let chunk-fill = rgb("#D7EBDD")
#let kv-fill = rgb("#5cbefb")
#let link-stroke = rgb("#356AC3")

#let make-chunk-box(label, tokens) = [
  #grid(
    column-gutter: 5pt,
    columns: 2,
    [*#label*], table(inset: 7pt, columns: 4, ..tokens),
  )
]

#let PrefillCache = [
  #diagram(
    node-stroke: 0.6pt,
    node-fill: white,
    spacing: (1.5em, 2.8em),
    {
      for (chunk-index, (chunk-label, chunk-tokens)) in chunks.enumerate() {
        node(
          (chunk-index, 0),
          make-chunk-box(chunk-label, chunk-tokens),
          fill: chunk-fill,
          inset: 10pt,
        )
        node(
          (chunk-index, 1),
          make-chunk-box("KV cache", chunk-tokens),
          fill: kv-fill,
          inset: 10pt,
        )
        edge(
          (chunk-index, 0),
          (chunk-index, 1),
          "->",
          label: [SWA],
          label-side: left,
        )
      }

      for chunk-index in range(1, chunks.len()) {
        edge(
          (chunk-index - 1, 1),
          (chunk-index, 0),
          "->",
          bend: -10deg,
          stroke: link-stroke,
          label-side: right,
        )
      }
    },
  )
]


// 5. Grouped Query Attention (GQA)

#let GQA = [
  #diagram(
    node-stroke: 0.4pt,
    {
      let query-fill = rgb("#f4d2f9")
      let kv-fill = rgb("#5cbefb")

      for head-index in range(8) {
        let query-label = head-index + 1
        let query-position = (0.45 * head-index, 0)
        let kv-position = if calc.odd(head-index) {
          (0.45 * head-index, 2.5)
        } else {
          (0.45 * head-index, 2)
        }

        node(query-position, [$Q_(#query-label)$], fill: query-fill)
        node(kv-position, [$K_(#query-label)$,$V_(#query-label)$], fill: kv-fill)
        edge(query-position, kv-position, "->")
      }

      edge((3.55, -0.5), (3.55, 3), "-")

      node((20cm, 9cm), [Avec *GQA* on a donc :], stroke: none)

      let queries-per-kv = 4
      for kv-index in range(2) {
        let kv-label = kv-index + 1
        let kv-position = (4.5 + 1.8 * kv-index, 2.25)
        node(kv-position, [$K_(#kv-label)$,$V_(#kv-label)$], fill: kv-fill)

        for query-offset in range(queries-per-kv) {
          let query-label = queries-per-kv * kv-index + query-offset + 1
          let query-position = (
            3.8 + 1.8 * kv-index + 0.45 * query-offset,
            0.5,
          )
          node(query-position, [$Q_(#query-label)$], fill: query-fill)
          edge(query-position, kv-position, "->")
        }
      }
    },
  )
]

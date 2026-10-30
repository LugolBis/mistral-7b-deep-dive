
#let c-teal = rgb("#9FE1CB")
#let c-tealD = rgb("#085041")
#let c-tealM = rgb("#1D9E75")
#let c-coral = rgb("#F5C4B3")
#let c-coralD = rgb("#712B13")
#let c-coralM = rgb("#D85A30")
#let c-amber = rgb("#EF9F27")
#let c-amberL = rgb("#FAEEDA")
#let c-amberD = rgb("#633806")
#let c-purple = rgb("#CECBF6")
#let c-purpleM = rgb("#7F77DD")
#let c-purpleD = rgb("#3C3489")
#let c-blue = rgb("#B5D4F4")
#let c-blueM = rgb("#378ADD")
#let c-blueD = rgb("#0C447C")
#let c-gray = rgb("#D3D1C7")
#let c-grayL = rgb("#F1EFE8")
#let c-ink = rgb("#2C2C2A")
#let c-mute = rgb("#5F5E5A")


#let cap(body) = text(size: 12pt, fill: c-mute, body)

#let figtitle(body) = text(size: 15pt, weight: "bold", fill: c-ink, body)

#let plot(
  w: 9cm,
  h: 4.6cm,
  xmin: 0,
  xmax: 1,
  ymin: 0,
  ymax: 1,
  xticks: (),
  yticks: (),
  xfmt: v => str(v),
  yfmt: v => str(v),
  xlab: none,
  ylab: none,
  series: (),
  notes: (),
  legend: true,
) = {
  let px(x) = (x - xmin) / (xmax - xmin) * w
  let py(y) = h - (y - ymin) / (ymax - ymin) * h
  box(inset: (left: 1.2cm, bottom: 1.0cm, top: 0.55cm, right: 0.5cm), box(
    width: w,
    height: h,
    stroke: (left: 0.7pt + c-mute, bottom: 0.7pt + c-mute),
    {
      for t in yticks {
        place(top + left, line(start: (0pt, py(t)), end: (w, py(t)), stroke: 0.3pt + c-gray))
        place(top + left, dx: -1.15cm, dy: py(t) - 0.15cm, box(width: 1.05cm, align(right, text(
          size: 9pt,
          fill: c-mute,
          yfmt(t),
        ))))
      }
      for t in xticks {
        place(top + left, line(start: (px(t), h), end: (px(t), h + 0.09cm), stroke: 0.6pt + c-mute))
        place(top + left, dx: px(t) - 0.9cm, dy: h + 0.12cm, box(width: 1.8cm, align(center, text(
          size: 9pt,
          fill: c-mute,
          xfmt(t),
        ))))
      }
      for s in series {
        let pts = s.pts
        for i in range(pts.len() - 1) {
          place(top + left, line(
            start: (px(pts.at(i).at(0)), py(pts.at(i).at(1))),
            end: (px(pts.at(i + 1).at(0)), py(pts.at(i + 1).at(1))),
            stroke: (paint: s.color, thickness: 1.8pt, cap: "round", dash: s.at("dash", default: none)),
          ))
        }
      }
      for n in notes {
        if n.at("dot", default: false) {
          place(top + left, dx: px(n.x) - 2.5pt, dy: py(n.y) - 2.5pt, circle(radius: 2.5pt, fill: n.at(
            "color",
            default: c-ink,
          )))
        }
        place(top + left, dx: px(n.x) + n.at("dx", default: 0.15cm), dy: py(n.y) + n.at("dy", default: -0.15cm), text(
          size: 9pt,
          weight: "bold",
          fill: n.at("color", default: c-ink),
          n.text,
        ))
      }
      if xlab != none {
        place(top + left, dx: 0cm, dy: h + 0.5cm, box(width: w, align(center, text(size: 9pt, fill: c-ink, xlab))))
      }
      if ylab != none { place(top + left, dx: -1.2cm, dy: -0.5cm, text(size: 9pt, fill: c-ink, ylab)) }
      if legend {
        place(top + left, dx: 0.25cm, dy: 0.1cm, grid(
          columns: (0.6cm, auto), row-gutter: 3.5pt, column-gutter: 5pt, align: horizon,
          ..for s in series {
            (
              line(length: 0.5cm, stroke: (
                paint: s.color,
                thickness: 1.8pt,
                cap: "round",
                dash: s.at("dash", default: none),
              )),
              text(size: 8pt, fill: c-ink, s.label),
            )
          }
        ))
      }
    },
  ))
}

#import "@preview/cetz:0.4.2": canvas, draw, decorations
#import draw: content, line

#set page(width: auto, height: auto, margin: 8pt)

#let neuron(pos, fill: white, text: none) = {
  draw.content(pos, text, frame: "circle", fill: fill, stroke: 0.5pt, padding: 1pt, text: (size: 8pt))
}

#let connect-layers(start-pos, start-count, start-offset, end-pos, end-count, end-offset) = {
  for ii in range(start-count) {
    for jj in range(end-count) {
      let start = (start-pos, start-offset - ii * 0.8)
      let end = (end-pos, end-offset - jj * 0.8)
      draw.line(start, end, stroke: rgb("#aaa") + .25pt)
    }
  }
}

#canvas({
  // Configuration: (x-pos, count, fill, prefix, superscript)
  // Architecture: 16 -> 8 -> 4 -> 2 -> 4 -> 8 -> 16
  let raw-layers = (
    (0, 16, rgb("#f6db71"), "x", none),     // Input (16)
    (3, 8, rgb("#eee"), "h", "1"),          // Hidden 1 (8)
    (5.5, 4, rgb("#eee"), "h", "2"),        // Hidden 2 (4)
    (7.5, 2, rgb("#ff9999"), "z", none),    // Latent Space (2)
    (9.5, 4, rgb("#eee"), "h", "3"),        // Hidden 3 (4)
    (12, 8, rgb("#eee"), "h", "4"),         // Hidden 4 (8)
    (15, 16, rgb("#cecef9"), "hat(x)", none) // Output (16)
  )

  // 1. Draw Connections
  for idx in range(raw-layers.len() - 1) {
    let (x1, n1, ..) = raw-layers.at(idx)
    let (x2, n2, ..) = raw-layers.at(idx + 1)
    let y-off1 = (n1 - 1) * 0.4
    let y-off2 = (n2 - 1) * 0.4
    connect-layers(x1, n1, y-off1, x2, n2, y-off2)
  }

  // 2. Draw Highlights
  let brace-y = -6.8 
  // Moved label-y down slightly to account for larger font height
  let label-y = -8.2 

  // Encoder Brace
  decorations.brace((0, brace-y), (5.5, brace-y), flip: true)
  content(((0+5.5)/2, label-y), text(size: 2.5em)[*Encoder*])

  // Latent Label
  // Moved down to -2.2 to prevent overlap with neurons
  content((7.5, -2.2), align(center, text(size: 2.5em)[*Latent\ Space*]))

  // Decoder Brace
  decorations.brace((9.5, brace-y), (15, brace-y), flip: true)
  content(((9.5+15)/2, label-y), text(size: 2.5em)[*Decoder*])

  // 3. Draw Neurons
  for (x, count, fill, prefix, sup) in raw-layers {
    let y-offset = (count - 1) * 0.4 
    for idx in range(count) {
      let y-pos = y-offset - idx * 0.8
      let text-content = if count < 10 or idx < 2 or idx >= count - 2 {
         if sup != none { $prefix^sup_idx$ } 
         else if prefix == "hat(x)" { $hat(x)_idx$ } 
         else { $prefix_idx$ }
      } else if idx == 2 {
        $dots.v$
      } else {
        none
      }
      neuron((x, y-pos), fill: fill, text: text-content)
    }
  }
})

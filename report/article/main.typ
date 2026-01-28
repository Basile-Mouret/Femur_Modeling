#import "@preview/ssrn-scribe:0.9.1": *

// if you do not want to use the integrated packages, you can comment out the following lines
#import "extra.typ": *

#show: great-theorems-init

#show: paper.with(
  font: "PT Serif",                 // core body font family
  fontsize: 12pt,                   // core body font size
  maketitle: true,                  // true → dedicate a cover page; false → inline title
  title: [3D Reconstruction of Femurs],               // document title
  subtitle: "Modeling Project",   // optional subtitle line

  // Cover-page–only spacing and typography (ignored when maketitle=false)
  cover-text-width: 90%,            // width of the abstract/keywords block
  cover-line-leading: 1.32em,       // line height for cover/front matter paragraphs
  cover-paragraph-spacing: 0.7em,   // paragraph spacing on the cover/front matter

  // Author grid controls (shared across both modes)
  author-columns: 4,                // override the auto-detected column count
  author-alignment: center,         // column alignment for author details
  authors: (
    (
      name: "Boyer Timothé",
      affiliation: "Grenoble INP - Ensimag",
      //email: "boyer.timothé@grenoble-inp.org",
    ),
    (
      name: "Hacini Malik",
      affiliation: "Grenoble INP - Ensimag",
    ),
    (
      name: "Lainé Martin",
      affiliation: "Grenoble INP - Ensimag",
    ),
    (
      name: "Mouret Basile",
      affiliation: "Grenoble INP - Ensimag",
    ),
  ),
  date: "January 2026",                // version/date string (shown in both modes)
  abstract: include("chapters/abstract.typ"),             // optional abstract (rendered front matter)
//   keywords: [
//     Imputation,
//     Multiple Imputation,
//     Bayesian,],                    // keyword list
//  // JEL: [G11, G12],                  // optional JEL codes
  //frontmatter-gap: 12pt,            // spacing between abstract/keywords/JEL entries

  // Body typography (applies to both modes)
  body-line-leading: 1.32em,        // main-text line height
  body-paragraph-spacing: 0.7em,    // spacing between main-text paragraphs
  body-text-spacing: 106%,          // glyph tracking for the body text

  bibliography: bibliography("../bibliography.bib", title: "References", style: "apa"), // attach your references
)

= Introduction

#include("chapters/introduction.typ")

= Motivation

#include("chapters/motivation.typ")

= Methods

#include("chapters/lin_pca.typ")
#include("chapters/methods.typ")

= Results

#include("chapters/results.typ")

= Discussion

#include("chapters/discussion.typ")

= Conclusion and future works

#include("chapters/conclusion.typ")

#v(1em)

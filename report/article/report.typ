#import "@preview/hei-synd-report:0.1.1": *
#import "metadata.typ": *
#import "tail/bibliography.typ": *
#import "tail/glossary.typ": *
#import "extra.typ": *
#show:make-glossary
#register-glossary(entry-list)

//-------------------------------------
// Template config
//
#show: report.with(
  option: option,
  doc: doc,
  date: date,
  tableof: tableof,
)

//-------------------------------------
// Content
//

#nonumber[= Abstract]
#include "chapters/abstract.typ"
#nonumber[= Introduction]
#include "chapters/introduction.typ"

#pagebreak()

= Motivation
#include "chapters/motivation.typ"
= PCA
#include "chapters/pca.typ"
= Autoencoder 
#include "chapters/autoencoder.typ"
= LDDMM
#include "chapters/lddmm.typ"
= Discussion
#include "chapters/discussion.typ"
= Conclusion and future works
#include "chapters/conclusion.typ"

#heading(numbering:none, outlined: false)[] <sec:end>

//-------------------------------------
// Glossary
//
#make_glossary(gloss:gloss, title:i18n("gloss-title"))

//-------------------------------------
// Bibliography
//
#make_bibliography(bib:bib, title:i18n("bib-title"))

//-------------------------------------
// Appendix
//
#if appendix == true {[
  #pagebreak()
  #counter(heading).update(0)
  #set heading(numbering:"A")
  = #i18n("appendix-title") <sec:appendix>
  //#include "tail/a-appendix.typ"
]}

#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *
#import "/tail/bibliography.typ": *
#import "/tail/glossary.typ": *
#import "/extra.typ": *
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
//#include "chapters/abstract.typ"
//#include "chapters/introduction.typ"
// #include "chapters/motivation.typ"
// #include "chapters/methods.typ"
// #include "chapters/results.typ"
// #include "chapters/discussion.typ"

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

//-------------------------------------
// Document options
//
#let option = (
  type : "final",
  //type : "draft",
  lang : "en",
  //lang : "de",
  //lang : "fr",
)
//-------------------------------------
// Optional generate titlepage image
//
#import "@preview/fractusist:0.1.1":*  // only for the generated images

#let titlepage_logo= dragon-curve(
  12,
  step-size: 10,
  stroke-style: stroke(
    //paint: gradient.linear(..color.map.rocket, angle: 135deg),
    paint: gradient.radial(..color.map.rocket),
    thickness: 3pt, join: "round"),
  height: 10cm,
)

//-------------------------------------
// Metadata of the document
//
#let doc= (
 
 title    : [*Statistical and Neural approaches in 3D Shape Modeling* ],
  abbr     : "SNSM",
  url      : "https://github.com/Basile-Mouret/Femur_Modeling/",
  logos: (
    tp_topleft  : image("resources/img/uga.svg", height: 1.2cm),
    tp_topright : image("resources/img/ensimag.svg", height: 1.5cm),
    tp_main     : image("resources/img/femur_3D.png",),
    header      : image("resources/img/femur_3D.png", width: 2.5cm),
  ),
  authors: (
    (
      name        : "Martin Lainé",
      abbr        : "MaL",
      email       : "",
      url         : "",
    ),
    (
      name        : "Basile Mouret",
      abbr        : "BaM",
      email       : "",
      url         : "",
    ),
    (
      name        : "Timothé Boyer",
      abbr        : "TiB",
      email       : "",
      url         : "",
    ),
    (
      name        : "Malik Hacini",
      abbr        : "MaH",
      email       : "",
      url         : "",
    ),
  ),
  school: (
    name        : "Grenoble INP : ENSIMAG",
    major       : "Master's in Applied Mathematics",
  ),
  course: (
    name     : "Modeling Project in C++",
    prof     : "Marek Bucki",
    semester : "Spring 2026",
  ),
  keywords : ("Typst", "Femur", "Report", "ENSIMAG", "Autoencoder", "LDDMM"),
  version  : "v0.1.0",
)

#let date= datetime.today()

//-------------------------------------
// Settings
//
#let tableof = (
  toc: true,
  tof: false,
  tot: false,
  tol: false,
  toe: false,
  maxdepth: 3,
)

#let gloss    = true
#let appendix = false
#let bib = (
  display : true,
  path  : "/tail/bibliography.bib",
  style : "ieee", //"apa", "chicago-author-date", "chicago-notes", "mla"
)

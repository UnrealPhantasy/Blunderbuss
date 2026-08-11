//! Cœur du moteur d'échecs.
//!
//! Ce crate contient **toute la logique** : l'interface `Position`, la
//! recherche, l'ordonnancement des coups et l'évaluation. Les crates `uci/` et
//! `bindings/` (à venir) ne seront que de fines enveloppes autour de lui —
//! voir ADR-014 dans `kb/fiches/02-decisions.html`.
//!
//! # Où on en est
//!
//! Rien n'est encore implémenté. La première pierre à poser est le module
//! `position`, qui définit la frontière entre le moteur et la couche règles
//! empruntée (`cozy-chess`). Cette frontière est le sujet d'ADR-005, et sa
//! *forme* — copie-joue ou joue/annule — est la question ouverte d'ADR-015,
//! à trancher avant d'écrire la moindre ligne d'interface.
//!
//! Marche à suivre détaillée : `kb/fiches/09-etat-et-reprise.html`.
//!
//! # Plan des modules
//!
//! ```text
//! position    l'interface Position + son implémentation sur cozy-chess
//! evaluation  matériel, tables de position (PST), puis le reste
//! ordre       ordonnancement des coups — MVV-LVA, killer moves, history
//! recherche   alpha-beta, quiescence, iterative deepening, transposition
//! ```
//!
//! Ces modules n'existent pas encore ; les déclarations sont ajoutées au fur et
//! à mesure. En Rust, un module n'est compilé que s'il est déclaré ici par
//! `pub mod <nom>;` — un fichier `src/position.rs` orphelin serait ignoré
//! silencieusement, ce qui est une source classique de confusion.

// pub mod position;
// pub mod evaluation;
// pub mod ordre;
// pub mod recherche;

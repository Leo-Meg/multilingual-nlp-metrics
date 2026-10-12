"""
Tests de la version 1 — corpus parallèle et modèles de langue n-grammes.

    python -m tests.test_ngrammes
"""

from __future__ import annotations

import math

from src.corpus import (
    CORPUS_PARALLELE,
    INCONNU,
    decouper_train_test,
    segmenter_caracteres,
    segmenter_mots,
    statistiques,
    type_token_ratio,
)
from src.ngrammes import (
    ModeleNGramme,
    comparer_langues,
    correlation,
    demontrer_incomparabilite,
)


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def test_corpus_strictement_parallele() -> None:
    """Sans cela, toute comparaison entre langues est biaisée par le contenu."""
    tailles = {langue: len(v) for langue, v in CORPUS_PARALLELE.items()}
    assert len(set(tailles.values())) == 1, tailles


def test_segmentation_isole_la_ponctuation() -> None:
    assert segmenter_mots("Bonjour, le monde !") == ["bonjour", ",", "le", "monde", "!"]


def test_segmentation_caracteres_normalise_unicode() -> None:
    """« é » composé et « é » précomposé doivent donner le même résultat."""
    precompose = "café"
    compose = "café"  # e + accent aigu combinant
    assert segmenter_caracteres(precompose) == segmenter_caracteres(compose)


def test_decoupage_reproductible_et_complet() -> None:
    phrases = CORPUS_PARALLELE["francais"]
    a1, a2 = decouper_train_test(phrases, graine=7)
    b1, b2 = decouper_train_test(phrases, graine=7)
    assert a1 == b1 and a2 == b2
    assert sorted(a1 + a2) == sorted(phrases)
    assert set(a1).isdisjoint(a2), "train et test ne doivent pas se recouvrir"


def test_ttr_borne_et_decroissant_avec_la_taille() -> None:
    assert type_token_ratio(["a b a b"]) == 0.5
    assert type_token_ratio(["a b c"]) == 1.0
    court = type_token_ratio(CORPUS_PARALLELE["francais"][:3])
    long = type_token_ratio(CORPUS_PARALLELE["francais"])
    assert court > long, (court, long)


def test_le_turc_est_le_plus_riche_morphologiquement() -> None:
    """Vérifie que le corpus reflète bien la typologie annoncée."""
    profils = {langue: statistiques(textes)
               for langue, textes in CORPUS_PARALLELE.items()}
    assert max(profils, key=lambda l: profils[l]["ttr"]) == "turc"
    assert max(profils, key=lambda l: profils[l]["caracteres_par_token"]) == "turc"
    assert min(profils, key=lambda l: profils[l]["caracteres_par_token"]) == "anglais"


# --------------------------------------------------------------------------- #
# Modèle n-gramme
# --------------------------------------------------------------------------- #


def _modele(lissage: str = "kneser_ney", ordre: int = 3) -> ModeleNGramme:
    train, _ = decouper_train_test(CORPUS_PARALLELE["francais"], graine=0)
    m = ModeleNGramme(ordre=ordre, lissage=lissage)
    m.entrainer(train)
    return m


def test_les_probabilites_sont_dans_zero_un() -> None:
    for lissage in ("add_k", "backoff", "kneser_ney"):
        m = _modele(lissage)
        for contexte in [(), ("le",), ("le", "chat")]:
            for mot in ("chat", "dort", INCONNU, "motjamaisvu"):
                p = m.probabilite(contexte, mot)
                assert 0.0 < p <= 1.0, (lissage, contexte, mot, p)


def test_add_k_est_une_vraie_distribution() -> None:
    """La somme sur tout le vocabulaire doit valoir 1 (aux arrondis près).

    C'est vrai pour le lissage additif, et volontairement FAUX pour le repli
    « stupide » — ses auteurs l'assument dans le titre de leur article.
    """
    m = _modele("add_k", ordre=2)
    total = sum(m.probabilite(("le",), mot) for mot in m.vocabulaire)
    assert abs(total - 1.0) < 1e-6, total


def test_aucun_n_gramme_ne_recoit_une_probabilite_nulle() -> None:
    """Sans lissage, une seule séquence inconnue rendrait la perplexité infinie."""
    for lissage in ("add_k", "backoff", "kneser_ney"):
        m = _modele(lissage)
        p = m.probabilite(("séquence", "totalement"), "improbable")
        assert p > 0.0, lissage


def test_perplexite_finie_et_superieure_a_un() -> None:
    train, test = decouper_train_test(CORPUS_PARALLELE["francais"], graine=0)
    for lissage in ("add_k", "backoff", "kneser_ney"):
        m = ModeleNGramme(ordre=3, lissage=lissage)
        m.entrainer(train)
        ppl = m.perplexite(test)
        assert math.isfinite(ppl), lissage
        assert ppl > 1.0, (lissage, ppl)


def test_le_modele_prefere_son_train_a_du_bruit() -> None:
    """Contrôle de santé : le modèle doit reconnaître ce qu'il a appris."""
    train, _ = decouper_train_test(CORPUS_PARALLELE["francais"], graine=0)
    m = ModeleNGramme(ordre=3)
    m.entrainer(train)
    bruit = ["zzz qqq xxx www", "aaa bbb ccc ddd"]
    assert m.perplexite(train) < m.perplexite(bruit)


def test_traitement_du_hors_vocabulaire() -> None:
    """Les mots rares du train deviennent <unk>, et les inconnus du test aussi."""
    m = ModeleNGramme(ordre=2, seuil_unk=1)
    m.entrainer(["le chat le chien le chat", "le chat dort"])
    assert INCONNU in m.vocabulaire
    assert "chien" not in m.vocabulaire, "un hapax doit être remplacé par <unk>"
    assert "chat" in m.vocabulaire
    # Un mot totalement inconnu reçoit la probabilité de <unk>, pas zéro.
    assert m.probabilite(("le",), "hippopotame") > 0.0


def test_symbole_de_fin_present() -> None:
    """Sans lui, le modèle ne peut pas représenter la fin d'une phrase."""
    from src.corpus import FIN

    m = _modele()
    assert FIN in m.vocabulaire
    assert m.probabilite(("le", "chat"), FIN) > 0.0


def test_kneser_ney_prefere_les_mots_polyvalents() -> None:
    """Le cœur de Kneser-Ney : compter les CONTEXTES, pas les occurrences.

    Je construis le cas canonique : « francisco » est aussi fréquent que
    « chat », mais n'apparaît que dans un seul contexte (« san francisco »).
    Sa probabilité de continuation doit être plus faible.
    """
    phrases = (["san francisco est grand"] * 5
               + ["le chat dort", "un chat mange", "ce chat court",
                  "mon chat saute", "ton chat joue"])
    m = ModeleNGramme(ordre=2, lissage="kneser_ney", seuil_unk=0)
    m.entrainer(phrases)

    assert len(m.contextes_du_mot["francisco"]) < len(m.contextes_du_mot["chat"])
    # Après un contexte inédit, « chat » doit être préféré à « francisco ».
    p_chat = m.probabilite(("bonjour",), "chat")
    p_francisco = m.probabilite(("bonjour",), "francisco")
    assert p_chat > p_francisco, (p_chat, p_francisco)


def test_un_ordre_superieur_aide_sur_le_train() -> None:
    """Contrôle : un trigramme mémorise mieux son propre corpus qu'un unigramme."""
    train, _ = decouper_train_test(CORPUS_PARALLELE["francais"], graine=0)
    perplexites = []
    for ordre in (1, 2, 3):
        m = ModeleNGramme(ordre=ordre, lissage="backoff")
        m.entrainer(train)
        perplexites.append(m.perplexite(train))
    assert perplexites[0] > perplexites[-1], perplexites


# --------------------------------------------------------------------------- #
# Les résultats méthodologiques
# --------------------------------------------------------------------------- #


def test_perplexite_correle_negativement_au_hors_vocabulaire() -> None:
    """Le résultat central : la métrique AVANTAGE les langues difficiles.

    Plus une langue a de tokens hors-vocabulaire, plus ils sont remplacés par
    <unk>, qui est très facile à prédire — donc plus la perplexité baisse. La
    perplexité récompense donc le refus de prédire.
    """
    resultats = comparer_langues(CORPUS_PARALLELE, ordre=3)
    r = correlation([x["perplexite"] for x in resultats],
                    [x["hors_vocabulaire"] for x in resultats])
    assert r < -0.5, f"corrélation attendue nettement négative, obtenue {r:+.3f}"

    # Concrètement : le turc, le plus dur à modéliser, a la meilleure perplexité.
    meilleur = min(resultats, key=lambda x: x["perplexite"])
    plus_de_hv = max(resultats, key=lambda x: x["hors_vocabulaire"])
    assert meilleur["langue"] == plus_de_hv["langue"] == "turc"


def test_perplexite_depend_de_la_segmentation() -> None:
    """Même texte, deux segmentations, deux perplexités incomparables."""
    d = demontrer_incomparabilite(CORPUS_PARALLELE)
    assert d["tokens_caracteres"] > 4 * d["tokens_mots"]
    assert d["perplexite_mots"] != d["perplexite_caracteres"]
    # Le modèle de caractères doit tout prédire : plus de bits au total.
    bits_mots = d["entropie_mots"] * d["tokens_mots"]
    bits_car = d["entropie_caracteres"] * d["tokens_caracteres"]
    assert bits_car > bits_mots


def test_correlation_cas_limites() -> None:
    assert abs(correlation([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
    assert abs(correlation([1, 2, 3], [6, 4, 2]) + 1.0) < 1e-12
    assert correlation([1, 1, 1], [1, 2, 3]) == 0.0
    assert correlation([1.0], [2.0]) == 0.0


def executer_tous_les_tests() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    echecs = 0
    for test in tests:
        try:
            test()
            print(f"  [OK]     {test.__name__}")
        except AssertionError as e:
            echecs += 1
            print(f"  [ÉCHEC]  {test.__name__} : {e}")
    print(f"\n{len(tests) - echecs}/{len(tests)} tests passés.")
    if echecs:
        raise SystemExit(1)


if __name__ == "__main__":
    print("=== Tests — version 1 : modèles n-grammes ===\n")
    executer_tous_les_tests()

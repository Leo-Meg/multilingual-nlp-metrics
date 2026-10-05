"""
Tests de la version 2 — BLEU, chrF, et la démonstration de fragilité.

    python -m tests.test_bleu
"""

from __future__ import annotations

import math

from src.bleu import (
    bleu,
    bleu_corpus,
    chrf,
    compter_ngrammes,
    compter_sur_un_corpus,
    nombre_de_permutations,
    penalite_brievete,
    permutations_a_score_bleu_egal,
    precision_modifiee,
    segmenter_caracteres_espaces,
    segmenter_sous_mots_naif,
)
from src.corpus import segmenter_mots


def test_comptage_de_ngrammes() -> None:
    assert compter_ngrammes(["a", "b", "c"], 2) == {("a", "b"): 1, ("b", "c"): 1}
    assert compter_ngrammes(["a"], 2) == {}


def test_ecretage_de_la_precision() -> None:
    """L'astuce centrale de BLEU : « le le le le » ne doit pas scorer 1,0."""
    hypothese = ["le"] * 4
    reference = [["le", "chat", "dort"]]
    num, den = precision_modifiee(hypothese, reference, 1)
    assert den == 4
    assert num == 1, "le compte doit être plafonné au maximum dans la référence"


def test_precision_parfaite_sur_copie_exacte() -> None:
    tokens = ["le", "chat", "noir", "dort", "bien"]
    for n in range(1, 5):
        num, den = precision_modifiee(tokens, [tokens], n)
        assert num == den > 0


def test_penalite_brievete() -> None:
    assert penalite_brievete(10, [10]) == 1.0        # même longueur
    assert penalite_brievete(15, [10]) == 1.0        # plus longue : pas de pénalité
    assert 0 < penalite_brievete(5, [10]) < 1.0      # plus courte : pénalisée
    assert penalite_brievete(0, [10]) == 0.0
    assert math.isclose(penalite_brievete(5, [10]), math.exp(1 - 2.0))


def test_bleu_parfait_sur_la_reference() -> None:
    references = ["le chat noir dort sur le tapis du salon tranquillement"]
    assert math.isclose(bleu(references, references), 100.0)


def test_bleu_nul_sans_aucun_mot_commun() -> None:
    assert bleu(["aaa bbb ccc ddd eee"], ["fff ggg hhh iii jjj"]) == 0.0


def test_bleu_penalise_une_hypothese_trop_courte() -> None:
    reference = ["le chat noir dort sur le tapis du salon tranquillement"]
    complete = ["le chat noir dort sur le tapis du salon tranquillement"]
    tronquee = ["le chat noir dort"]
    assert bleu(tronquee, reference) < bleu(complete, reference)


def test_bleu_corpus_renvoie_ses_composantes() -> None:
    resultat = bleu_corpus(
        [segmenter_mots("le chat noir dort bien")],
        [[segmenter_mots("le chat noir dort bien")]],
    )
    assert set(resultat) == {"bleu", "penalite_brievete", "precisions",
                             "longueur_hyp", "longueur_ref"}
    assert len(resultat["precisions"]) == 4
    assert all(p == 1.0 for p in resultat["precisions"])


def test_bleu_est_agrege_au_niveau_du_corpus() -> None:
    """BLEU-corpus n'est PAS la moyenne des BLEU par phrase.

    C'est une erreur très répandue (et que j'ai commencé par faire en M2).
    Ce test montre concrètement que les deux quantités diffèrent.
    """
    hyps = ["le chat dort bien ici", "un chien court vite dehors"]
    refs = ["le chat dort bien ici", "le chat mange une pomme rouge"]

    corpus = bleu(hyps, refs)
    moyenne = sum(bleu([h], [r]) for h, r in zip(hyps, refs)) / 2
    assert not math.isclose(corpus, moyenne), (corpus, moyenne)


# --------------------------------------------------------------------------- #
# La démonstration de Callison-Burch et al. (2006)
# --------------------------------------------------------------------------- #


def test_les_permutations_preservent_presque_le_score() -> None:
    """Questions 3 et 5 du TP, effectivement démontrées.

    En déplaçant des blocs de 5 mots, seuls les n-grammes qui chevauchent les
    frontières changent. Le score reste donc très élevé alors que la phrase est
    devenue du charabia.
    """
    tokens = segmenter_mots(
        "le chat noir dort tranquillement sur le tapis du salon "
        "pendant que les enfants jouent dans la cour de l' école "
        "et que la pluie tombe doucement sur les toits"
    )
    variantes = permutations_a_score_bleu_egal(tokens)
    assert len(variantes) > 100

    scores = [bleu_corpus([v], [[tokens]])["bleu"] for v in variantes]
    assert min(scores) > 50.0, f"score minimal trop bas : {min(scores):.2f}"
    assert sum(scores) / len(scores) > 70.0

    # Toutes les variantes contiennent exactement les mêmes mots.
    for variante in variantes[:20]:
        assert sorted(variante) == sorted(tokens)


def test_les_permutations_sont_bien_distinctes() -> None:
    tokens = segmenter_mots("un deux trois quatre cinq six sept huit neuf dix "
                            "onze douze treize quatorze quinze")
    variantes = permutations_a_score_bleu_egal(tokens)
    uniques = {tuple(v) for v in variantes}
    assert len(uniques) == len(variantes)


def test_denombrement_des_permutations() -> None:
    """Question 4 du TP."""
    assert nombre_de_permutations(10) == math.factorial(2)
    assert nombre_de_permutations(30) == math.factorial(6)
    assert nombre_de_permutations(50) == math.factorial(10)
    # Sur un corpus, les possibilités se multiplient.
    total = compter_sur_un_corpus(["a b c d e f g h i j"] * 3)
    assert total == 2 ** 3


# --------------------------------------------------------------------------- #
# L'effet de la tokenisation (question 7)
# --------------------------------------------------------------------------- #


def test_la_tokenisation_change_le_score() -> None:
    """Le même système, trois tokenisations, trois scores. D'où sacreBLEU."""
    references = ["le chat noir dort tranquillement sur le tapis du salon"]
    hypotheses = ["le chat noir dort paisiblement sur le tapis du salon"]

    scores = {
        "mots": bleu(hypotheses, references, segmenteur=segmenter_mots),
        "sous_mots": bleu(hypotheses, references, segmenteur=segmenter_sous_mots_naif),
        "caracteres": bleu(hypotheses, references,
                           segmenteur=segmenter_caracteres_espaces),
    }
    assert len(set(round(v, 2) for v in scores.values())) > 1, scores
    # Plus les unités sont fines, plus le score monte.
    assert scores["caracteres"] > scores["mots"], scores


# --------------------------------------------------------------------------- #
# chrF
# --------------------------------------------------------------------------- #


def test_chrf_parfait_et_nul() -> None:
    ref = ["le chat noir dort"]
    assert math.isclose(chrf(ref, ref), 100.0)
    assert chrf(["zzzzzz"], ["ppppp"]) == 0.0


def test_chrf_credite_les_variantes_morphologiques() -> None:
    """Le point où chrF est structurellement supérieur à BLEU.

    Pour BLEU, « mangeaient » et « mangent » sont deux mots sans rapport —
    exactement comme « mangeaient » et « dormait ». chrF voit qu'ils partagent
    presque tous leurs n-grammes de caractères.
    """
    reference = ["les enfants mangeaient des pommes rouges"]
    variante_morpho = ["les enfants mangent des pommes rouges"]
    sans_rapport = ["le chien dormait sous la table verte"]

    chrf_morpho = chrf(variante_morpho, reference)
    chrf_autre = chrf(sans_rapport, reference)
    assert chrf_morpho > chrf_autre + 30, (chrf_morpho, chrf_autre)


def test_chrf_ne_depend_pas_des_espaces() -> None:
    """chrF travaille sur les caractères : la segmentation en mots ne compte pas."""
    a = chrf(["le chat dort"], ["le chat dort"])
    b = chrf(["lechatdort"], ["le chat dort"])
    assert math.isclose(a, b)


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
    print("=== Tests — version 2 : BLEU et chrF ===\n")
    executer_tous_les_tests()

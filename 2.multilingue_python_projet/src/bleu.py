"""
BLEU, chrF, et la fragilité de l'évaluation de la traduction automatique.

D'où ça vient
-------------
TP n°3 de NLP multilingue — *« How Reliable is MT Evaluation? »*
(Guillaume Wisniewski, M2), construit autour de l'article de Callison-Burch,
Osborne & Koehn (2006), *Re-evaluating the role of Bleu in machine translation
research*.

L'énoncé demandait, dans l'ordre :
  1. implémenter BLEU soi-même ;
  2. évaluer MarianMT et mBART sur WMT'15 ;
  3. **« expliquer sur un exemple pourquoi de telles permutations ne feront
     jamais baisser le score BLEU »** ;
  4. compter combien de phrases on peut fabriquer ainsi ;
  5. en déduire ce que cela dit de BLEU ;
  6. refaire avec sacreBLEU et conclure ;
  7. recommencer avec trois tokenisations différentes.

J'avais implémenté BLEU et fait tourner les modèles. Les questions 3, 4 et 5 —
les seules qui comptaient vraiment — étaient restées **sans réponse écrite** dans
mon rendu. C'est ce que je répare ici : je construis effectivement les
permutations, je les compte, et je montre le score inchangé.

Pourquoi je peux le faire sans modèle de traduction
----------------------------------------------------
La démonstration de Callison-Burch et al. ne demande **aucun** modèle : elle
consiste à partir d'une traduction et à la manipuler. C'est justement ce qui la
rend si convaincante, et c'est ce qui la rend reproductible ici en Python pur.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import permutations

from .corpus import segmenter_mots


# --------------------------------------------------------------------------- #
# 1. BLEU
# --------------------------------------------------------------------------- #


def compter_ngrammes(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def precision_modifiee(hypothese: list[str], references: list[list[str]],
                       n: int) -> tuple[int, int]:
    """Précision n-gramme **écrêtée**, la trouvaille centrale de BLEU.

    Sans écrêtage, l'hypothèse « le le le le le » obtiendrait une précision
    unigramme de 1,0 face à la référence « le chat dort ». L'écrêtage plafonne
    le compte de chaque n-gramme à son nombre maximal d'occurrences dans une
    référence : « le » ne peut compter qu'une fois.

    Returns:
        ``(numérateur, dénominateur)`` — je renvoie les deux pour pouvoir les
        agréger au niveau du corpus, ce qui est la définition correcte de BLEU
        (et non la moyenne des BLEU par phrase, erreur très répandue).
    """
    comptes_hyp = compter_ngrammes(hypothese, n)
    if not comptes_hyp:
        return 0, 0

    maxima: Counter = Counter()
    for reference in references:
        for ngramme, compte in compter_ngrammes(reference, n).items():
            maxima[ngramme] = max(maxima[ngramme], compte)

    numerateur = sum(min(compte, maxima[ng]) for ng, compte in comptes_hyp.items())
    return numerateur, sum(comptes_hyp.values())


def penalite_brievete(longueur_hyp: int, longueurs_ref: list[int]) -> float:
    """Pénalité de brièveté : ``min(1, exp(1 − r/c))``.

    BLEU n'a pas de notion de rappel — une hypothèse très courte et parfaitement
    précise obtiendrait un score parfait. La pénalité corrige ce défaut en
    punissant les hypothèses plus courtes que la référence.

    La référence retenue est la plus proche en longueur (« closest match »),
    convention de l'article original (Papineni et al., 2002).
    """
    if longueur_hyp == 0:
        return 0.0
    reference = min(longueurs_ref, key=lambda r: (abs(r - longueur_hyp), r))
    if longueur_hyp > reference:
        return 1.0
    return math.exp(1.0 - reference / longueur_hyp)


def bleu_corpus(hypotheses: list[list[str]], references: list[list[list[str]]],
                ordre_max: int = 4) -> dict[str, float]:
    """BLEU au niveau du **corpus** — la seule définition correcte.

    On agrège les numérateurs et dénominateurs sur tout le corpus **avant** de
    diviser. Faire la moyenne des BLEU par phrase donne un nombre différent, et
    c'est une erreur que je vois régulièrement (j'ai commencé par la faire dans
    mon rendu de M2).

    Returns:
        Le score et ses composantes, pour pouvoir diagnostiquer.
    """
    numerateurs = [0] * (ordre_max + 1)
    denominateurs = [0] * (ordre_max + 1)
    longueur_hyp = 0
    longueurs_ref_choisies = 0

    for hypothese, refs in zip(hypotheses, references):
        for n in range(1, ordre_max + 1):
            num, den = precision_modifiee(hypothese, refs, n)
            numerateurs[n] += num
            denominateurs[n] += den
        longueur_hyp += len(hypothese)
        longueurs_ref_choisies += min(
            (len(r) for r in refs),
            key=lambda r: (abs(r - len(hypothese)), r),
        )

    precisions = [
        (numerateurs[n] / denominateurs[n]) if denominateurs[n] else 0.0
        for n in range(1, ordre_max + 1)
    ]

    if any(p == 0.0 for p in precisions):
        # Une précision nulle annule tout le produit géométrique. C'est le
        # comportement standard, et la raison pour laquelle BLEU est inutilisable
        # au niveau de la phrase sans lissage.
        score = 0.0
    else:
        score = math.exp(sum(math.log(p) for p in precisions) / ordre_max)

    bp = (1.0 if longueur_hyp > longueurs_ref_choisies
          else math.exp(1.0 - longueurs_ref_choisies / max(longueur_hyp, 1)))

    return {
        "bleu": 100.0 * bp * score,
        "penalite_brievete": bp,
        "precisions": precisions,
        "longueur_hyp": float(longueur_hyp),
        "longueur_ref": float(longueurs_ref_choisies),
    }


def bleu(hypotheses: list[str], references: list[str], ordre_max: int = 4,
         segmenteur=segmenter_mots) -> float:
    """Interface simple : deux listes de chaînes, un score entre 0 et 100."""
    return bleu_corpus(
        [segmenteur(h) for h in hypotheses],
        [[segmenteur(r)] for r in references],
        ordre_max=ordre_max,
    )["bleu"]


# --------------------------------------------------------------------------- #
# 2. chrF — l'alternative fondée sur les caractères
# --------------------------------------------------------------------------- #


def chrf(hypotheses: list[str], references: list[str], n_max: int = 6,
         beta: float = 2.0) -> float:
    """chrF (Popović, 2015) : F-mesure sur les n-grammes de **caractères**.

    Trois différences décisives avec BLEU :

      * elle utilise le **rappel** en plus de la précision (d'où la F-mesure),
        donc pas besoin de pénalité de brièveté artificielle ;
      * elle travaille sur les **caractères**, donc elle ne dépend pas d'une
        tokenisation — c'est le point qui m'intéresse le plus après le TP n°1 ;
      * elle accorde une reconnaissance **partielle** aux formes fléchies :
        *mangeait* et *mangeaient* partagent presque tous leurs n-grammes de
        caractères, alors que BLEU les traite comme deux mots sans rapport.

    Ce dernier point est capital pour les langues morphologiquement riches, où
    BLEU sous-évalue systématiquement les systèmes. ``beta = 2`` donne deux fois
    plus de poids au rappel qu'à la précision, comme dans l'article.
    """
    total_precision = 0.0
    total_rappel = 0.0
    ordres_utiles = 0

    for n in range(1, n_max + 1):
        num_p = den_p = num_r = den_r = 0
        for hypothese, reference in zip(hypotheses, references):
            h = list(hypothese.replace(" ", ""))
            r = list(reference.replace(" ", ""))
            comptes_h = compter_ngrammes(h, n)
            comptes_r = compter_ngrammes(r, n)
            communs = sum((comptes_h & comptes_r).values())
            num_p += communs
            den_p += sum(comptes_h.values())
            num_r += communs
            den_r += sum(comptes_r.values())

        if den_p and den_r:
            total_precision += num_p / den_p
            total_rappel += num_r / den_r
            ordres_utiles += 1

    if ordres_utiles == 0:
        return 0.0

    p = total_precision / ordres_utiles
    r = total_rappel / ordres_utiles
    if p + r == 0:
        return 0.0
    return 100.0 * (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


# --------------------------------------------------------------------------- #
# 3. La démonstration de Callison-Burch et al. (2006)
# --------------------------------------------------------------------------- #


def permutations_a_score_bleu_egal(tokens: list[str], ordre_max: int = 4) -> list[list[str]]:
    """Toutes les permutations de blocs qui préservent exactement le score BLEU.

    Le principe de la question 3 du TP : BLEU ne regarde que des n-grammes
    jusqu'à l'ordre 4. Si l'on découpe la phrase en blocs de **plus de 4 mots**
    et qu'on réordonne ces blocs, tous les n-grammes internes sont préservés.
    Seuls les n-grammes qui chevauchent les frontières changent.

    En choisissant des blocs suffisamment longs, on fabrique donc un nombre
    considérable de phrases — dont beaucoup sont du charabia — au score BLEU
    quasiment inchangé.

    C'est la démonstration que BLEU est **structurellement aveugle à l'ordre des
    constituants**, qui est pourtant l'essentiel de ce qu'un traducteur doit
    réussir.
    """
    taille_bloc = ordre_max + 1
    blocs = [tokens[i:i + taille_bloc] for i in range(0, len(tokens), taille_bloc)]
    if len(blocs) > 7:  # 7! = 5040, au-delà c'est intraitable
        blocs = blocs[:7]
    return [[mot for bloc in ordre for mot in bloc]
            for ordre in permutations(blocs)]


def nombre_de_permutations(nb_mots: int, ordre_max: int = 4) -> int:
    """Nombre de phrases fabricables par ce principe, pour une phrase donnée.

    C'était la question 4 du TP (« combien de phrases pouvez-vous générer ainsi ?
    calculez le nombre obtenu sur le jeu de test WMT'15 »).
    """
    nb_blocs = math.ceil(nb_mots / (ordre_max + 1))
    return math.factorial(nb_blocs)


def compter_sur_un_corpus(phrases: list[str], ordre_max: int = 4) -> int:
    """Le calcul demandé à la question 4, appliqué à un corpus entier."""
    total = 1
    for phrase in phrases:
        total *= max(nombre_de_permutations(len(segmenter_mots(phrase)), ordre_max), 1)
    return total


# --------------------------------------------------------------------------- #
# 4. L'effet de la tokenisation (question 7)
# --------------------------------------------------------------------------- #


def segmenter_caracteres_espaces(texte: str) -> list[str]:
    """Tokenisation en caractères : « chat » -> ``['c','h','a','t']``."""
    return [c for c in texte.lower() if not c.isspace()]


def segmenter_sous_mots_naif(texte: str, taille: int = 3) -> list[str]:
    """Découpage en tranches de taille fixe — un substitut de BPE sans dépendance.

    Ce n'est évidemment pas une vraie tokenisation en sous-mots. Mais pour la
    question posée — *l'effet de la granularité sur BLEU* — seul compte le fait
    que les unités soient plus petites que des mots et plus grandes que des
    caractères.
    """
    jetons = []
    for mot in segmenter_mots(texte):
        jetons.extend(mot[i:i + taille] for i in range(0, len(mot), taille))
    return jetons


if __name__ == "__main__":
    print("=== BLEU : implémentation et fragilité ===\n")

    # Un mini jeu de test : hypothèses et références.
    references = [
        "le chat noir dort tranquillement sur le tapis du salon",
        "les étudiants travaillent sur un corpus annoté depuis deux mois",
        "nous devons évaluer le système avant la fin de la semaine",
        "la traduction automatique a beaucoup progressé ces dernières années",
    ]
    hypotheses_bonnes = [
        "le chat noir dort paisiblement sur le tapis du salon",
        "les étudiants travaillent sur un corpus annoté depuis deux mois",
        "nous devons évaluer le système avant la fin de semaine",
        "la traduction automatique a beaucoup progressé ces dernières années",
    ]
    hypotheses_mediocres = [
        "un chat dort sur un tapis",
        "des étudiants font un travail sur des textes",
        "il faut tester le programme rapidement",
        "les traductions par ordinateur se sont améliorées",
    ]

    print("--- Scores de base ---\n")
    print(f"  {'système':>22} | {'BLEU':>7} | {'chrF':>7}")
    print("  " + "-" * 42)
    for nom, hyps in (("traduction quasi parfaite", hypotheses_bonnes),
                      ("paraphrase correcte", hypotheses_mediocres),
                      ("référence contre elle-même", references)):
        print(f"  {nom:>22} | {bleu(hyps, references):>7.2f} | "
              f"{chrf(hyps, references):>7.2f}")

    print(
        "\n  Premier constat : la paraphrase — correcte, compréhensible, qu'un\n"
        "  humain jugerait acceptable — obtient un BLEU catastrophique. BLEU ne\n"
        "  mesure PAS la qualité d'une traduction, il mesure la ressemblance de\n"
        "  surface à UNE référence particulière.\n"
    )

    # -- Question 3 : les permutations ------------------------------------- #
    print("--- Questions 3 à 5 du TP : les permutations qui ne coûtent rien ---\n")
    # Une phrase longue : plus elle a de blocs, plus la démonstration est
    # spectaculaire. Avec 10 mots on n'obtient que 2 permutations ; avec 30, on
    # en obtient 720.
    phrase = ("le chat noir dort tranquillement sur le tapis du salon "
              "pendant que les enfants jouent dans la cour de l' école "
              "et que la pluie tombe doucement sur les toits")
    tokens = segmenter_mots(phrase)
    variantes = permutations_a_score_bleu_egal(tokens)

    score_origine = bleu_corpus([tokens], [[tokens]])["bleu"]
    scores = [bleu_corpus([v], [[tokens]])["bleu"] for v in variantes]

    print(f"  phrase de référence ({len(tokens)} mots) :")
    print(f"    « {phrase} »\n")
    print(f"  {len(variantes)} permutations de blocs de 5 mots construites.")
    print(f"  score de l'original : {score_origine:.2f}")
    print(f"  scores des variantes : min {min(scores):.2f} | "
          f"max {max(scores):.2f} | moyenne {sum(scores) / len(scores):.2f}\n")

    print("  Trois variantes, avec leur score :")
    for variante, score in list(zip(variantes, scores))[1:4]:
        print(f"    BLEU {score:>6.2f}  « {' '.join(variante)} »")

    print(
        "\n  Ces phrases sont du charabia. Un lecteur francophone les rejetterait\n"
        "  immédiatement. BLEU leur donne un score très proche de l'original,\n"
        "  parce qu'il ne regarde que des n-grammes jusqu'à l'ordre 4 : en\n"
        "  déplaçant des blocs de 5 mots, on ne casse que les n-grammes qui\n"
        "  chevauchent les frontières.\n"
    )

    # -- Question 4 : le dénombrement -------------------------------------- #
    print("  Question 4 — combien de telles phrases peut-on fabriquer ?\n")
    print(f"  {'longueur de phrase':>20} | {'blocs':>6} | {'permutations':>14}")
    print("  " + "-" * 46)
    for longueur in (10, 20, 30, 50):
        blocs = math.ceil(longueur / 5)
        print(f"  {longueur:>20} | {blocs:>6} | "
              f"{nombre_de_permutations(longueur):>14,}")

    total = compter_sur_un_corpus(references)
    print(f"\n  Sur mes 4 phrases seulement : {total:,} combinaisons.")
    print("  Le jeu de test de WMT'15 contenait 2 169 phrases. Le nombre de")
    print("  phrases indiscernables par BLEU y est astronomique — bien au-delà")
    print("  de ce qu'on peut écrire.\n")

    # -- Question 7 : l'effet de la tokenisation --------------------------- #
    print("--- Question 7 du TP : le même système, trois tokenisations ---\n")
    print(f"  {'tokenisation':>16} | {'BLEU (bonnes)':>14} | {'BLEU (paraphrases)':>19}")
    print("  " + "-" * 56)
    for nom, seg in (("mots", segmenter_mots),
                     ("sous-mots (3 car.)", segmenter_sous_mots_naif),
                     ("caractères", segmenter_caracteres_espaces)):
        b1 = bleu(hypotheses_bonnes, references, segmenteur=seg)
        b2 = bleu(hypotheses_mediocres, references, segmenteur=seg)
        print(f"  {nom:>16} | {b1:>14.2f} | {b2:>19.2f}")

    print(
        "\n  Le MÊME système obtient des scores complètement différents selon la\n"
        "  tokenisation. Plus les unités sont petites, plus le score monte —\n"
        "  parce que deux mots différents partagent des caractères.\n"
        "\n"
        "  C'est exactement le problème que sacreBLEU résout : il impose UNE\n"
        "  tokenisation de référence et publie une signature complète du calcul\n"
        "  (« hassle-free computation of shareable, comparable, and reproducible\n"
        "  BLEU scores », Post 2018). Sans cela, deux articles annonçant\n"
        "  « BLEU 28,4 » peuvent parler de deux choses sans rapport.\n"
    )

    # -- BLEU contre chrF sur la morphologie ------------------------------- #
    print("--- Là où chrF est structurellement meilleur : la morphologie ---\n")
    reference_morpho = ["les enfants mangeaient des pommes rouges"]
    variantes_morpho = [
        ("bonne traduction", ["les enfants mangeaient des pommes rouges"]),
        ("erreur de temps", ["les enfants mangent des pommes rouges"]),
        ("erreur de nombre", ["l' enfant mangeait des pommes rouges"]),
        ("mots sans rapport", ["le chien dormait sous la table verte"]),
    ]
    print(f"  {'variante':>20} | {'BLEU':>7} | {'chrF':>7}")
    print("  " + "-" * 40)
    for nom, hyp in variantes_morpho:
        print(f"  {nom:>20} | {bleu(hyp, reference_morpho):>7.2f} | "
              f"{chrf(hyp, reference_morpho):>7.2f}")

    print(
        "\n  Pour BLEU, « mangeaient » et « mangent » sont deux mots sans aucun\n"
        "  rapport — exactement comme « mangeaient » et « dormait ». chrF, qui\n"
        "  travaille sur les caractères, voit qu'ils partagent presque tout et\n"
        "  accorde un crédit partiel.\n"
        "\n"
        "  C'est pour ça que BLEU sous-évalue systématiquement les systèmes sur\n"
        "  les langues morphologiquement riches : une seule désinence fausse coûte\n"
        "  le mot entier. Sur du turc ou du finnois, l'effet est massif."
    )

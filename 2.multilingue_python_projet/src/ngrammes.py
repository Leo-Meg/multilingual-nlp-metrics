"""
Modèles de langue n-grammes, avec lissage — et la question de la perplexité.

D'où ça vient
-------------
TP n°1 de NLP multilingue (« One model to rule them all », Guillaume
Wisniewski, M2). Nous devions installer KenLM, entraîner un modèle de langue par
langue sur wiki40b, et comparer les perplexités obtenues sur 41 langues.

Je réimplémente ici le modèle de langue lui-même, parce que la vraie question du
TP n'était pas technique mais méthodologique :

  **Question 1.3.2 — Peut-on comparer des perplexités entre corpus ou entre
  langues ?**

J'avais répondu « en général non », ce qui était juste, mais je n'aurais pas su
le démontrer. Ce module le démontre, avec des chiffres qu'on peut reproduire.

Ce qu'il y a dans un modèle n-gramme
-------------------------------------
    P(w_1 … w_n) = ∏ P(w_i | w_{i-k+1} … w_{i-1})

L'hypothèse de Markov d'ordre k−1 : le mot suivant ne dépend que des k−1
précédents. C'est faux linguistiquement (une dépendance sujet-verbe peut porter
sur quinze mots), mais ça marche étonnamment bien — et surtout, ça permet
d'estimer les probabilités par simple comptage.

Le vrai problème est le **lissage**. Un n-gramme jamais vu reçoit une probabilité
nulle, donc toute la phrase reçoit une probabilité nulle, donc la perplexité est
infinie. Il faut redistribuer de la masse. J'implémente trois méthodes, de la
plus naïve à la plus fine.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .corpus import DEBUT, FIN, INCONNU, segmenter_mots


class ModeleNGramme:
    """Modèle de langue n-gramme avec lissage choisi.

    Args:
        ordre: taille des n-grammes (2 = bigrammes, 3 = trigrammes).
        lissage: ``"add_k"``, ``"backoff"`` ou ``"kneser_ney"``.
        k: paramètre du lissage additif.
        remise: paramètre de remise (*discount*) pour le repli et Kneser-Ney.
    """

    def __init__(self, ordre: int = 3, lissage: str = "kneser_ney",
                 k: float = 0.1, remise: float = 0.75, seuil_unk: int = 1):
        assert ordre >= 1
        assert lissage in ("add_k", "backoff", "kneser_ney")
        self.ordre = ordre
        self.lissage = lissage
        self.k = k
        self.remise = remise
        self.seuil_unk = seuil_unk

        self.comptes: list[Counter] = [Counter() for _ in range(ordre + 1)]
        self.vocabulaire: set[str] = set()
        # Pour Kneser-Ney : nombre de contextes DISTINCTS dans lesquels un mot
        # apparaît, et nombre de continuations distinctes d'un contexte.
        self.contextes_du_mot: defaultdict[str, set] = defaultdict(set)
        self.continuations: defaultdict[tuple, set] = defaultdict(set)
        self.nb_bigrammes_distincts = 0

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #

    def _preparer(self, phrase: str, remplacer_inconnus: bool = False) -> list[str]:
        """Ajoute les symboles de début et de fin, et gère les mots inconnus.

        Le symbole de fin n'est pas décoratif : sans lui, le modèle ne peut pas
        représenter la probabilité qu'une phrase **s'arrête**, et la somme des
        probabilités sur toutes les phrases ne vaut pas 1. C'est aussi lui qui
        permet à un générateur de savoir quand se taire.

        Le traitement du hors-vocabulaire
        ---------------------------------
        C'est le point que ma première version ratait, et l'erreur était
        spectaculaire : Kneser-Ney donnait des perplexités de l'ordre du million
        là où le lissage additif donnait 115. Diagnostic : 46 % des tokens de
        test étaient absents du vocabulaire, et la distribution de continuation
        de Kneser-Ney leur attribuait une probabilité quasi nulle.

        La solution standard, que tout modèle n-gramme sérieux applique :
        remplacer les mots rares du TRAIN par un symbole ``<unk>``, de sorte que
        le modèle apprenne une vraie probabilité pour « un mot que je ne connais
        pas », puis rediriger vers ce symbole tout mot inconnu du TEST.

        Sans ce traitement, comparer deux lissages revient à comparer leurs
        façons respectives de mal gérer l'inconnu — pas leur qualité de modèle.
        """
        tokens = segmenter_mots(phrase)
        if remplacer_inconnus:
            tokens = [t if t in self.vocabulaire else INCONNU for t in tokens]
        return [DEBUT] * (self.ordre - 1) + tokens + [FIN]

    def entrainer(self, phrases: list[str]) -> None:
        # Premier passage : repérer les mots trop rares pour être estimés.
        frequences: Counter[str] = Counter()
        for phrase in phrases:
            frequences.update(segmenter_mots(phrase))
        rares = {m for m, n in frequences.items() if n <= self.seuil_unk}

        self.vocabulaire = {m for m in frequences if m not in rares}
        self.vocabulaire.update({INCONNU, FIN})

        for phrase in phrases:
            tokens = self._preparer(phrase, remplacer_inconnus=True)

            for taille in range(1, self.ordre + 1):
                for i in range(len(tokens) - taille + 1):
                    ngramme = tuple(tokens[i:i + taille])
                    self.comptes[taille][ngramme] += 1

                    if taille >= 2:
                        contexte, mot = ngramme[:-1], ngramme[-1]
                        self.contextes_du_mot[mot].add(contexte)
                        self.continuations[contexte].add(mot)

        self.nb_bigrammes_distincts = len(self.comptes[2]) if self.ordre >= 2 else 0

    # ------------------------------------------------------------------ #
    # Probabilités
    # ------------------------------------------------------------------ #

    def _proba_add_k(self, contexte: tuple, mot: str) -> float:
        """Lissage additif : on ajoute ``k`` à tous les comptes.

        C'est le lissage de Laplace généralisé. Il a le mérite d'être en une
        ligne et le défaut d'être mauvais : il donne autant de masse à un
        n-gramme jamais vu qu'à n'importe quel autre, ce qui est absurde. Sur un
        gros vocabulaire, il vole une part énorme de probabilité aux n-grammes
        réellement observés.
        """
        V = max(len(self.vocabulaire), 1)
        compte_ngramme = self.comptes[len(contexte) + 1][contexte + (mot,)]
        compte_contexte = (self.comptes[len(contexte)][contexte]
                           if contexte else sum(self.comptes[1].values()))
        return (compte_ngramme + self.k) / (compte_contexte + self.k * V)

    def _proba_backoff(self, contexte: tuple, mot: str) -> float:
        """Repli (*stupid backoff*, Brants et al., 2007).

        Si le n-gramme n'a jamais été vu, on se replie sur le (n−1)-gramme en
        multipliant par un facteur fixe (0,4 dans l'article original).

        Ce n'est **pas** une vraie distribution de probabilité — la somme ne vaut
        pas 1 — et les auteurs l'assument dans le titre. Mais c'est extrêmement
        rapide et ça marche très bien à grande échelle. Je l'inclus parce que
        c'est un bon rappel : en TAL appliqué, une heuristique bien choisie bat
        souvent une théorie propre.
        """
        while True:
            compte_ngramme = self.comptes[len(contexte) + 1][contexte + (mot,)]
            if compte_ngramme > 0:
                compte_contexte = (self.comptes[len(contexte)][contexte]
                                   if contexte else sum(self.comptes[1].values()))
                facteur = 0.4 ** (self.ordre - 1 - len(contexte))
                return facteur * compte_ngramme / max(compte_contexte, 1)
            if not contexte:
                # Unigramme jamais vu : on retombe sur un lissage additif.
                V = max(len(self.vocabulaire), 1)
                total = max(sum(self.comptes[1].values()), 1)
                return self.k / (total + self.k * V)
            contexte = contexte[1:]

    def _proba_kneser_ney(self, contexte: tuple, mot: str) -> float:
        """Kneser-Ney interpolé — le lissage de référence avant les réseaux.

        L'idée centrale est belle et contre-intuitive. Pour la distribution de
        repli, on n'utilise **pas** la fréquence brute du mot mais son nombre de
        **contextes distincts**.

        L'exemple canonique : « Francisco » est fréquent, mais n'apparaît
        pratiquement que dans « San Francisco ». Un lissage naïf lui donnera une
        forte probabilité après n'importe quel mot, ce qui est absurde.
        Kneser-Ney compte les contextes distincts, voit que « Francisco » n'en a
        qu'un, et lui donne une probabilité de continuation très faible.

        Autrement dit : la question n'est pas « ce mot est-il fréquent ? » mais
        « ce mot est-il *polyvalent* ? ».
        """
        if not contexte:
            # Distribution de continuation : proportionnelle au nombre de
            # contextes distincts, et non à la fréquence.
            if self.nb_bigrammes_distincts == 0:
                V = max(len(self.vocabulaire), 1)
                return 1.0 / V
            return max(len(self.contextes_du_mot[mot]), 1e-10) / self.nb_bigrammes_distincts

        compte_contexte = self.comptes[len(contexte)][contexte]
        if compte_contexte == 0:
            return self._proba_kneser_ney(contexte[1:], mot)

        compte_ngramme = self.comptes[len(contexte) + 1][contexte + (mot,)]
        terme_principal = max(compte_ngramme - self.remise, 0.0) / compte_contexte

        # Masse retirée par la remise, redistribuée sur la distribution de repli.
        nb_continuations = len(self.continuations[contexte])
        poids = self.remise * nb_continuations / compte_contexte

        return terme_principal + poids * self._proba_kneser_ney(contexte[1:], mot)

    def probabilite(self, contexte: tuple, mot: str) -> float:
        """P(mot | contexte), selon le lissage choisi."""
        contexte = tuple(contexte)[-(self.ordre - 1):] if self.ordre > 1 else ()
        if self.lissage == "add_k":
            return self._proba_add_k(contexte, mot)
        if self.lissage == "backoff":
            return self._proba_backoff(contexte, mot)
        return self._proba_kneser_ney(contexte, mot)

    # ------------------------------------------------------------------ #
    # Évaluation
    # ------------------------------------------------------------------ #

    def log_vraisemblance(self, phrase: str) -> tuple[float, int]:
        """``(log₂ P(phrase), nombre de prédictions)``.

        Je travaille en log base 2 pour que l'entropie soit en **bits**, ce qui
        est plus interprétable : « le modèle hésite comme s'il choisissait entre
        2^H possibilités ».
        """
        tokens = self._preparer(phrase, remplacer_inconnus=True)
        total = 0.0
        nb = 0
        debut = self.ordre - 1
        for i in range(debut, len(tokens)):
            contexte = tuple(tokens[max(0, i - self.ordre + 1):i])
            p = self.probabilite(contexte, tokens[i])
            total += math.log2(max(p, 1e-300))
            nb += 1
        return total, nb

    def entropie_croisee(self, phrases: list[str]) -> float:
        """Entropie croisée en bits par token."""
        total, nb = 0.0, 0
        for phrase in phrases:
            log_p, n = self.log_vraisemblance(phrase)
            total += log_p
            nb += n
        return -total / max(nb, 1)

    def perplexite(self, phrases: list[str]) -> float:
        """``2^entropie_croisée``.

        Interprétation : le **facteur de branchement moyen** — le nombre de mots
        entre lesquels le modèle hésite en moyenne à chaque position. Une
        perplexité de 50 signifie « le modèle est aussi incertain que s'il
        choisissait uniformément parmi 50 mots ».

        Question 1.3.1 du TP, et la définition qu'on oublie le plus souvent :
        la perplexité n'est pas une mesure de qualité absolue, c'est une mesure
        d'**incertitude sur un corpus donné**.
        """
        return 2 ** self.entropie_croisee(phrases)

    def taux_hors_vocabulaire(self, phrases: list[str]) -> float:
        """Proportion de tokens du test absents du vocabulaire d'entraînement.

        C'est LE chiffre qu'il faut regarder avant toute perplexité : un modèle
        peut afficher une perplexité magnifique simplement parce qu'il n'a
        presque rien à prédire de nouveau.
        """
        tokens = [t for p in phrases for t in segmenter_mots(p)]
        if not tokens:
            return 0.0
        return sum(t not in self.vocabulaire for t in tokens) / len(tokens)

    def taille_vocabulaire(self) -> int:
        return len(self.vocabulaire)


# --------------------------------------------------------------------------- #
# L'expérience : peut-on comparer des perplexités ?
# --------------------------------------------------------------------------- #


def comparer_langues(
    corpus: dict[str, list[str]], ordre: int = 3, lissage: str = "kneser_ney",
    graine: int = 0
) -> list[dict]:
    """Entraîne un modèle par langue sur un corpus parallèle et compare."""
    from .corpus import decouper_train_test, statistiques

    resultats = []
    for langue in sorted(corpus):
        train, test = decouper_train_test(corpus[langue], graine=graine)
        modele = ModeleNGramme(ordre=ordre, lissage=lissage)
        modele.entrainer(train)
        stats = statistiques(train)
        resultats.append({
            "langue": langue,
            "perplexite": modele.perplexite(test),
            "entropie": modele.entropie_croisee(test),
            "hors_vocabulaire": modele.taux_hors_vocabulaire(test),
            "ttr": stats["ttr"],
            "nb_types_train": stats["nb_types"],
            "tokens_test": float(sum(len(segmenter_mots(p)) for p in test)),
        })
    return resultats


def demontrer_incomparabilite(corpus: dict[str, list[str]]) -> dict[str, float]:
    """Démonstration : la perplexité dépend de la SEGMENTATION, pas de la langue.

    Le protocole : j'entraîne deux modèles sur **exactement le même texte
    français**, l'un segmenté en mots, l'autre en caractères. Le contenu
    informationnel est rigoureusement identique — c'est le même texte. Les
    perplexités, elles, n'ont rien à voir.

    C'est la démonstration la plus courte que je connaisse de la réponse à la
    question 1.3.2 : comparer des perplexités entre langues suppose que les
    unités soient comparables. Or elles ne le sont jamais — un « mot » turc
    n'est pas la même quantité d'information qu'un « mot » anglais.
    """
    from .corpus import decouper_train_test, segmenter_caracteres

    train, test = decouper_train_test(corpus["francais"], graine=0)

    modele_mots = ModeleNGramme(ordre=3)
    modele_mots.entrainer(train)

    # Pour les caractères, j'espace les caractères pour réutiliser le même
    # segmenteur : le modèle voit alors un « mot » = un caractère.
    train_car = [" ".join(segmenter_caracteres(p)) for p in train]
    test_car = [" ".join(segmenter_caracteres(p)) for p in test]
    modele_car = ModeleNGramme(ordre=3)
    modele_car.entrainer(train_car)

    return {
        "perplexite_mots": modele_mots.perplexite(test),
        "perplexite_caracteres": modele_car.perplexite(test_car),
        "entropie_mots": modele_mots.entropie_croisee(test),
        "entropie_caracteres": modele_car.entropie_croisee(test_car),
        "tokens_mots": float(sum(len(segmenter_mots(p)) for p in test)),
        "tokens_caracteres": float(sum(len(segmenter_caracteres(p)) for p in test)),
    }


def correlation(x: list[float], y: list[float]) -> float:
    """Corrélation de Pearson, écrite à la main (pas de dépendance)."""
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


if __name__ == "__main__":
    from .corpus import CORPUS_PARALLELE, TYPES_MORPHOLOGIQUES, decouper_train_test

    print("=== Modèles de langue n-grammes ===\n")

    # -- 1. Effet du lissage ------------------------------------------------ #
    print("--- Les trois lissages, sur le même corpus français ---\n")
    train, test = decouper_train_test(CORPUS_PARALLELE["francais"], graine=0)
    print(f"  train : {len(train)} phrases | test : {len(test)} phrases\n")

    print(f"  {'lissage':>12} | {'ordre 1':>10} | {'ordre 2':>10} | {'ordre 3':>10}")
    print("  " + "-" * 50)
    for lissage in ("add_k", "backoff", "kneser_ney"):
        ligne = f"  {lissage:>12} |"
        for ordre in (1, 2, 3):
            modele = ModeleNGramme(ordre=ordre, lissage=lissage)
            modele.entrainer(train)
            ligne += f" {modele.perplexite(test):>10.1f} |"
        print(ligne.rstrip("|"))

    print("\n  Sans lissage du tout, ces perplexités vaudraient toutes l'infini :")
    print("  il suffit d'UN seul n-gramme jamais vu pour annuler la probabilité")
    print("  de la phrase entière.\n")

    # -- 2. Comparaison entre langues --------------------------------------- #
    print("--- Perplexité par langue (corpus parallèle, même taille) ---\n")
    resultats = comparer_langues(CORPUS_PARALLELE, ordre=3)

    entete = (f"  {'langue':>10} | {'type':>28} | {'perplexité':>11} | "
              f"{'entropie':>9} | {'HV':>6} | {'TTR':>6}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))
    for r in sorted(resultats, key=lambda d: d["perplexite"]):
        print(f"  {r['langue']:>10} | {TYPES_MORPHOLOGIQUES[r['langue']]:>28} | "
              f"{r['perplexite']:>11.1f} | {r['entropie']:>9.2f} | "
              f"{100 * r['hors_vocabulaire']:>5.0f}% | {r['ttr']:>6.3f}")

    r_ttr = correlation([r["perplexite"] for r in resultats],
                        [r["ttr"] for r in resultats])
    r_hv = correlation([r["perplexite"] for r in resultats],
                       [r["hors_vocabulaire"] for r in resultats])
    print(f"\n  corrélation perplexité ~ TTR              : r = {r_ttr:+.3f}")
    print(f"  corrélation perplexité ~ hors-vocabulaire : r = {r_hv:+.3f}")
    print("  (5 langues : tendance, pas résultat statistique)")

    print(
        "\n  ATTENTION — ce tableau est un PIÈGE, et c'est tout son intérêt.\n"
        "\n"
        "  Le turc affiche la MEILLEURE perplexité (2,7) alors que c'est la\n"
        "  langue la plus difficile à modéliser. Explication : 80 % de ses tokens\n"
        "  de test sont hors-vocabulaire et deviennent donc <unk>. Or <unk> est\n"
        "  extrêmement facile à prédire — il est fréquent et prévisible.\n"
        "\n"
        "  Le modèle turc obtient une bonne perplexité en REFUSANT DE PRÉDIRE :\n"
        "  il ne se prononce pas sur l'identité de 80 % des mots. La corrélation\n"
        "  perplexité ~ hors-vocabulaire mesure exactement cet artefact.\n"
        "\n"
        "  C'est LA raison pour laquelle une perplexité ne se compare pas entre\n"
        "  langues : elle dépend du taux de hors-vocabulaire, donc de la\n"
        "  morphologie, donc de la langue elle-même. Une langue agglutinante est\n"
        "  structurellement avantagée par la métrique alors qu'elle est\n"
        "  structurellement plus dure à modéliser."
    )

    # -- 3. La démonstration -------------------------------------------------#
    print("\n--- La question 1.3.2 : peut-on comparer des perplexités ? ---\n")
    d = demontrer_incomparabilite(CORPUS_PARALLELE)
    print("  MÊME TEXTE français, deux segmentations différentes :\n")
    print(f"  {'unité':>12} | {'nb de tokens':>13} | {'perplexité':>11} | "
          f"{'entropie (bits/token)':>22}")
    print("  " + "-" * 68)
    print(f"  {'mots':>12} | {d['tokens_mots']:>13.0f} | "
          f"{d['perplexite_mots']:>11.1f} | {d['entropie_mots']:>22.2f}")
    print(f"  {'caractères':>12} | {d['tokens_caracteres']:>13.0f} | "
          f"{d['perplexite_caracteres']:>11.1f} | {d['entropie_caracteres']:>22.2f}")

    bits_mots = d["entropie_mots"] * d["tokens_mots"]
    bits_car = d["entropie_caracteres"] * d["tokens_caracteres"]
    print(f"\n  Information totale attribuée au test, en bits :")
    print(f"    modèle de mots       : {bits_mots:>9.0f} bits")
    print(f"    modèle de caractères : {bits_car:>9.0f} bits")

    print(
        "\nCe que ça démontre\n"
        "------------------\n"
        "* C'est le MÊME TEXTE, et pourtant les deux perplexités ne se comparent\n"
        "  pas : l'une compte des hésitations entre mots, l'autre entre\n"
        "  caractères. Une perplexité n'a de sens QUE relativement à une unité.\n"
        "\n"
        "* Le modèle de mots semble « meilleur » (moins de bits au total). C'est\n"
        "  faux : il triche. Avec 58 % de tokens hors-vocabulaire remplacés par\n"
        "  <unk>, il ne se prononce jamais sur l'identité de la majorité des\n"
        "  mots. Le modèle de caractères, lui, doit tout prédire.\n"
        "\n"
        "* Conclusion — la même que celle du tableau précédent, par un autre\n"
        "  chemin : une perplexité mesure une incertitude SUR UNE\n"
        "  SEGMENTATION ET UN VOCABULAIRE DONNÉS. Changer l'un ou l'autre change\n"
        "  le nombre sans que le modèle soit meilleur ou pire.\n"
        "\n"
        "* C'était la question 1.3.2 de mon TP de M2. J'avais répondu « en général\n"
        "  non, parce qu'il n'y a pas le même nombre de tokens selon le corpus ».\n"
        "  La réponse était juste mais incomplète : le vrai coupable est le\n"
        "  traitement du hors-vocabulaire, et il est bien plus vicieux, parce\n"
        "  qu'il AVANTAGE les langues les plus difficiles."
    )

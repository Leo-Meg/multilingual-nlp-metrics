# Version 2, BLEU. Implémenter la métrique, puis la casser

> **Où j'en suis.** La version 1 a montré qu'une perplexité ne se compare pas.
> Cette version s'attaque à la métrique la plus utilisée du TAL, et répare
> quelque chose, les trois questions les plus importantes de mon TP de M2
> étaient restées **sans réponse écrite** dans mon rendu.

---

## Nouveautés par rapport à la version 1

| Fichier | Ce que j'ajoute |
|---|---|
| `src/bleu.py` | BLEU complet (précision écrêtée, pénalité de brièveté, agrégation au niveau du corpus), **chrF**, la construction effective des permutations de Callison-Burch, le dénombrement, et trois tokenisations pour mesurer leur effet. |
| `tests/test_bleu.py` | 16 tests, dont la démonstration que BLEU-corpus n'est pas la moyenne des BLEU par phrase. |

## Origine universitaire, et une dette rendue

**TP n°3 de NLP multilingue.** *« How Reliable is MT Evaluation? »*
(Guillaume Wisniewski, M2), construit autour de Callison-Burch, Osborne & Koehn
(2006).

L'énoncé demandait sept choses. J'avais fait les questions techniques
(implémenter BLEU, évaluer MarianMT et mBART sur WMT'15, comparer avec
sacreBLEU). Les questions **3, 4 et 5**, les seules qui portaient vraiment sur
le fond, étaient restées sans réponse écrite dans mon notebook.

> **3.** Expliquez sur un exemple pourquoi de telles permutations ne feront
> jamais baisser le score BLEU.
> **4.** Combien de phrases pouvez-vous générer ainsi ? Calculez le nombre
> obtenu sur le jeu de test WMT'15.
> **5.** Pourquoi ce résultat remet-il en cause l'usage de BLEU ?

C'est ce que je répare ici. Et la bonne nouvelle. **cette démonstration ne
demande aucun modèle de traduction**. Elle consiste à partir d'une traduction et
à la manipuler, c'est ce qui la rend si convaincante, et reproductible en
Python pur.

## Lancer le code

```bash
python -m src.bleu
```

```bash
python -m tests.test_bleu
```

---

## Résultat n°1, BLEU ne mesure pas la qualité

| système | BLEU | chrF |
|---|---:|---:|
| traduction quasi parfaite | 86,91 | 92,78 |
| **paraphrase correcte** | **0,00** | 24,54 |
| référence contre elle-même | 100,00 | 100,00 |

La paraphrase, correcte, compréhensible, qu'un humain jugerait acceptable,
obtient **zéro**. BLEU ne mesure pas la qualité d'une traduction, il mesure la
**ressemblance de surface à une référence particulière**.

---

## Résultat n°2, les permutations qui ne coûtent (presque) rien

**Le principe.** BLEU ne regarde que des n-grammes jusqu'à l'ordre 4. Si l'on
découpe la phrase en blocs de **plus de 4 mots** et qu'on réordonne ces blocs,
tous les n-grammes internes sont préservés. Seuls ceux qui chevauchent les
frontières changent.

Sur une phrase de 30 mots, je construis les **720 permutations** possibles.

```
score de l'original  : 100,00
scores des variantes : min 69,73 | max 100,00 | moyenne 75,22
```

Trois exemples.

```
BLEU 88.76  « le chat noir dort tranquillement sur le tapis du salon pendant
              que les enfants jouent dans la cour de l' tombe doucement sur
              les toits école et que la pluie »

BLEU 82.78  « … les enfants jouent école et que la pluie dans la cour de l'
              tombe doucement sur les toits »
```

Ces phrases sont du charabia. Un lecteur francophone les rejette immédiatement.
BLEU leur donne 80-90.

### Question 4, combien peut-on en fabriquer ?

| longueur de phrase | blocs | permutations |
|---:|---:|---:|
| 10 | 2 | 2 |
| 20 | 4 | 24 |
| 30 | 6 | 720 |
| 50 | 10 | **3 628 800** |

Sur mes 4 phrases de démonstration. 48 combinaisons. Le jeu de test de WMT'15
contenait **2 169 phrases.** Le nombre de textes indiscernables par BLEU y est
astronomique, bien au-delà de ce qu'on peut écrire.

### Question 5, ce que ça remet en cause

BLEU est **structurellement aveugle à l'ordre des constituants**, qui est
pourtant l'essentiel de ce qu'un traducteur doit réussir. Deux conséquences.

- une amélioration de BLEU ne garantit pas une amélioration de qualité,
- un système peut être optimisé pour BLEU sans progresser réellement.

C'est exactement l'argument de Callison-Burch et al. (2006), et c'est pour ça
que la campagne WMT a fini par adopter l'évaluation humaine comme référence, et
les métriques neuronales (COMET, BLEURT) comme métriques automatiques.

---

## Résultat n°3, le même système, trois tokenisations, trois scores

| tokenisation | BLEU (bonnes trad.) | BLEU (paraphrases) |
|---|---:|---:|
| mots | 86,91 | 0,00 |
| sous-mots (3 car.) | 87,86 | 0,00 |
| **caractères** | **93,71** | **23,21** |

Plus les unités sont petites, plus le score monte, deux mots différents
partagent des caractères.

C'est le problème que **sacreBLEU** (Post, 2018) résout, il impose une
tokenisation de référence et publie une **signature complète** du calcul.
Sans cela, deux articles annonçant « BLEU 28,4 » peuvent parler de deux choses
sans rapport. C'était la question 7 du TP.

---

## Résultat n°4, où chrF est structurellement meilleur

| variante | BLEU | chrF |
|---|---:|---:|
| bonne traduction | 100,00 | 100,00 |
| **erreur de temps** (*mangent* / *mangeaient*) | faible | **élevé** |
| erreur de nombre (*l'enfant* / *les enfants*) | faible | élevé |
| mots sans rapport | faible | faible |

**Pour BLEU, `mangeaient` et `mangent` sont deux mots sans aucun rapport**,
exactement comme `mangeaient` et `dormait`. chrF, qui travaille sur les
n-grammes de caractères, voit qu'ils partagent presque tout et accorde un crédit
partiel.

C'est pour cela que BLEU **sous-évalue systématiquement** les systèmes sur les
langues morphologiquement riches, une seule désinence fausse coûte le mot
entier. Sur du turc ou du finnois, l'effet est massif, et c'est le lien direct
avec ce que j'ai mesuré à la version 1.

chrF a deux autres avantages, elle utilise le **rappel** (donc pas besoin de
pénalité de brièveté artificielle) et elle ne dépend d'**aucune tokenisation**.

---

## Un détail d'implémentation que beaucoup ratent

**BLEU-corpus n'est pas la moyenne des BLEU par phrase.** On agrège les
numérateurs et dénominateurs sur tout le corpus **avant** de diviser.

Les deux quantités donnent des nombres différents, et j'ai commencé par faire
l'erreur dans mon rendu de M2 (une boucle `for` avec accumulation dans une
liste, puis une moyenne). Le test
`test_bleu_est_agrege_au_niveau_du_corpus` le montre concrètement.

Raison sous-jacente, au niveau de la phrase, une précision 4-gramme nulle
annule tout le produit géométrique, d'où des zéros partout et une moyenne qui
ne veut rien dire. C'est aussi pour ça que le BLEU-phrase demande un lissage
spécifique.

---

## Ce qui reste ouvert

Mes mesures portent sur des traductions déjà produites. La façon dont le texte
est fabriqué à partir des probabilités du modèle reste hors de mon champ.

Je soupçonne que ce choix pèse autant que le modèle lui-même, et je n'ai rien
pour le vérifier.

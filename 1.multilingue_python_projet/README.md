# Version 1, modèles de langue et perplexité

> **Où j'en suis.** Je pars du TP qui ouvrait mon cours de NLP multilingue,
> entraîner un modèle de langue par langue et comparer les perplexités. La
> question centrale de l'énoncé était de savoir si des perplexités se comparent
> entre langues. J'avais répondu non sans pouvoir le démontrer. Ici je le
> démontre.

---

## Ce que contient cette version

| Fichier | Ce que j'y fais |
|---|---|
| `src/corpus.py` | Corpus parallèle de 40 phrases traduites en 5 langues, français, anglais, espagnol, allemand et turc. Segmentation en mots et en caractères avec normalisation Unicode, rapport type-occurrence, statistiques descriptives. |
| `src/ngrammes.py` | Modèle n-gramme complet avec trois lissages, additif, repli et Kneser-Ney interpolé. Gestion du hors-vocabulaire, perplexité, et les deux expériences méthodologiques. |
| `tests/test_ngrammes.py` | 18 tests, dont la vérification que Kneser-Ney privilégie bien les mots polyvalents. |

## Origine universitaire

TP n°1 de NLP multilingue, *One model to rule them all*
(Guillaume Wisniewski, M2 Linguistique Informatique).

Nous devions extraire 40 000 phrases d'entraînement et 3 000 de test pour
chacune des 41 langues de wiki40b, installer KenLM, entraîner un modèle par
langue, et comparer.

## Lancer le code

```bash
python -m src.corpus
```

```bash
python -m src.ngrammes
```

```bash
python -m tests.test_ngrammes
```

Aucune dépendance, pas même NumPy. Tout le corpus est écrit dans le code. Mes
notebooks de M2 commençaient par `tfds.load("wiki40b/fr", data_dir="gs://…")` et
un montage de Google Drive. Deux ans plus tard, plus rien ne s'exécute.

---

## Le protocole, un corpus parallèle

L'énoncé du TP imposait des jeux de taille identique, et demandait pourquoi à la
question 2.1.11. La réponse est que la langue doit être la seule variable.

Je vais un cran plus loin, mon corpus est parallèle, ce sont les mêmes 40 phrases
traduites. Cela neutralise aussi la variable du contenu.

| langue | type | tokens | types | TTR | car./token | hapax |
|---|---|---:|---:|---:|---:|---:|
| anglais | flexionnelle (pauvre) | 314 | 185 | 0,589 | 4,90 | 0,789 |
| français | flexionnelle | 322 | 190 | 0,590 | 5,24 | 0,763 |
| espagnol | flexionnelle | 304 | 189 | 0,622 | 5,33 | 0,794 |
| allemand | flexionnelle + composition | 287 | 193 | 0,672 | 5,94 | 0,777 |
| **turc** | **agglutinante** | **222** | 171 | **0,770** | **6,59** | **0,854** |

La typologie ressort nettement. Le turc dit la même chose en moins de tokens,
parce que les suffixes remplacent les mots grammaticaux, mais avec des tokens plus
longs et plus variés.

---

## Résultat n°1, le lissage n'est pas un détail

Perplexité sur le test, en français, selon le lissage et l'ordre.

| lissage | ordre 1 | ordre 2 | ordre 3 |
|---|---:|---:|---:|
| additif (`add_k`) | 8,8 | 8,3 | 13,7 |
| repli (`backoff`) | 8,7 | 6,8 | 8,2 |
| **Kneser-Ney** | 39,0 | 7,2 | **6,2** |

Sans lissage du tout, toutes ces valeurs vaudraient l'infini. Il suffit d'un seul
n-gramme jamais vu pour annuler la probabilité de la phrase entière.

### Le principe de Kneser-Ney

Pour la distribution de repli, on n'utilise pas la fréquence brute du mot mais son
nombre de contextes distincts.

L'exemple canonique est *Francisco*, qui est fréquent mais n'apparaît
pratiquement que dans *San Francisco*. Un lissage naïf lui donnera une forte
probabilité après n'importe quel mot. Kneser-Ney compte les contextes, voit qu'il
n'en a qu'un, et lui attribue une probabilité de continuation très faible.

La question posée n'est donc pas de savoir si un mot est fréquent, mais s'il est
polyvalent.

Je le vérifie sur le cas construit exprès dans
`test_kneser_ney_prefere_les_mots_polyvalents`.

---

## Résultat n°2, la perplexité récompense le refus de prédire

| langue | perplexité | entropie | hors-vocabulaire | TTR |
|---|---:|---:|---:|---:|
| **turc** | **2,7** | 1,44 | **80 %** | 0,798 |
| allemand | 3,9 | 1,95 | 73 % | 0,731 |
| anglais | 4,4 | 2,13 | 61 % | 0,632 |
| espagnol | 5,7 | 2,51 | 64 % | 0,661 |
| français | 6,2 | 2,63 | 58 % | 0,634 |

```
corrélation perplexité et hors-vocabulaire, r = −0,890
```

Le turc affiche la meilleure perplexité alors que c'est la langue la plus
difficile à modéliser.

L'explication est que 80 % de ses tokens de test sont hors-vocabulaire et
deviennent donc `<unk>`. Or `<unk>` est très facile à prédire, parce qu'il est
fréquent et prévisible. Le modèle turc obtient un bon score en refusant de se
prononcer sur l'identité de 80 % des mots.

C'est la vraie réponse à la question 1.3.2. En M2 j'avais répondu qu'on ne peut
pas comparer parce qu'il n'y a pas le même nombre de tokens. C'était juste mais
incomplet. Le vrai responsable est le traitement du hors-vocabulaire, et il est
plus gênant, parce qu'il avantage systématiquement les langues les plus dures.

---

## Résultat n°3, même texte, deux segmentations, deux résultats sans rapport

| unité | nb de tokens | perplexité | entropie (bits/token) |
|---|---:|---:|---:|
| mots | 84 | 6,2 | 2,63 |
| caractères | 480 | 8,7 | 3,13 |

```
Information totale attribuee au test
  modele de mots         221 bits
  modele de caracteres  1501 bits
```

C'est le même texte français. Le modèle de mots semble meilleur, avec sept fois
moins de bits, mais ce n'est pas le cas. Avec 58 % de hors-vocabulaire, il ne se
prononce jamais sur l'identité de la majorité des mots, alors que le modèle de
caractères doit tout prédire.

Une perplexité mesure une incertitude sur une segmentation et un vocabulaire
donnés. Changer l'un ou l'autre change le nombre sans que le modèle soit meilleur
ou pire.

---

## Le bug que je garde documenté

Ma première version ne traitait pas le hors-vocabulaire. Kneser-Ney donnait alors
des perplexités de l'ordre du million là où le lissage additif donnait 115.

Le diagnostic est que 46 % des tokens de test étaient inconnus, et que la
distribution de continuation de Kneser-Ney leur attribuait une probabilité quasi
nulle, parce que `contextes_du_mot[mot]` était vide.

La correction est la pratique standard, remplacer les mots rares du train par
`<unk>` pour que le modèle apprenne une vraie probabilité pour un mot inconnu.
Sans ce traitement, comparer deux lissages revient à comparer leurs façons
respectives de mal gérer l'inconnu, et non leur qualité de modèle.

Je documente l'erreur dans la docstring de `_preparer` plutôt que de la corriger
silencieusement. C'est le genre de piège qui rend une expérience ininterprétable
sans qu'aucun code ne plante.

---

## Une limite de mon protocole

J'utilise le même segmenteur pour les cinq langues. En M2 nous utilisions
`polyglot`, qui s'appuie sur l'algorithme Unicode Text Segmentation d'ICU et
applique des règles différentes selon la langue.

Mon choix est délibéré, je veux que la seule différence entre mes mesures vienne
des langues et non de segmenteurs différents. Il a un coût, les clitiques du
français comme `l'article` et les contractions de l'allemand comme `im` ne sont
pas traités. Je le documente plutôt que de le masquer.

---

## Ce qui reste ouvert

Je sais maintenant montrer qu'une perplexité ne se compare pas d'une langue à
l'autre, et pourquoi.

Ce que je ne sais pas, c'est si les autres métriques du domaine ont le même
défaut. J'en ai utilisé plusieurs pendant le master sans vérifier ce qu'elles
mesuraient réellement.

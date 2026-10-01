# NLP multilingue — la métrique qui ment

Modèles n-grammes et perplexité, écrits sans dépendance, pour répondre à une question restée sans réponse dans mon rendu de M2.

**Léo Mégret** — Master Linguistique Informatique, Université Paris Cité

> **État du dépôt : version 1.** C'est la première étape d'un travail que je
> mène par étapes, chacune dans son propre dossier. Seule la version 1 existe à
> ce jour. Je publie au fur et à mesure plutôt qu'une fois tout terminé, parce
> que l'intérêt de ce travail est justement l'enchaînement des questions.

---

## Pourquoi ce dépôt

Le TP qui ouvrait mon cours de NLP multilingue demandait d'entraîner un modèle de
langue par langue et de comparer les perplexités. La question centrale de
l'énoncé était : peut-on comparer des perplexités entre langues ?

J'avais répondu non, sans pouvoir le démontrer. Ce dépôt est la démonstration.

L'énoncé demandait 41 langues, 40 000 phrases d'entraînement chacune et
l'installation de KenLM. Je fais l'inverse : cinq langues, un corpus parallèle de
40 phrases écrit à la main, et tout dans le code. Le résultat tient sur un
ordinateur portable et il est reproductible dans dix ans.

---

## Ce qui existe aujourd'hui

### Version 1 — Modèles de langue et perplexité

| Fichier | Ce que j'y fais |
|---|---|
| `src/corpus.py` | Corpus parallèle de 40 phrases traduites en cinq langues, français, anglais, espagnol, allemand et turc. Segmentation en mots et en caractères avec normalisation Unicode, rapport type-occurrence, statistiques descriptives. |
| `src/ngrammes.py` | Modèle n-gramme complet avec trois lissages, additif, repli et Kneser-Ney interpolé. Gestion du hors-vocabulaire, perplexité, et les deux expériences méthodologiques. |
| `tests/test_ngrammes.py` | 18 tests, dont la vérification que Kneser-Ney privilégie bien les mots polyvalents. |

Origine universitaire : TP n°1 de NLP multilingue, *One model to rule them all*
(Guillaume Wisniewski, M2).

---

## Lancer le code

```bash
cd 1.multilingue_python_projet
python -m src.corpus
python -m src.ngrammes
python -m tests.test_ngrammes
```

Aucune dépendance, pas même NumPy. Le corpus est dans le code.

---

## Ce que je retiens de cette étape

**Le turc obtient la meilleure perplexité alors que c'est la langue la plus
difficile du corpus.** La raison est qu'il refuse de prédire 80 % de ses mots :
il les renvoie au hors-vocabulaire, qui ne compte pas dans la métrique. Une
perplexité peut donc récompenser l'abstention.

C'est exactement ce qui rend la comparaison entre langues illégitime, et ce que
je n'avais pas su démontrer en cours.

**Une limite que je documente plutôt que de la masquer.** Je segmente toutes les
langues avec la même règle, délibérément, pour que la seule différence entre mes
mesures vienne des langues. Cela a un coût : les clitiques du français comme
`l'article` et les contractions de l'allemand comme `im` ne sont pas traités.

---

## Ce qui reste ouvert

Je sais maintenant montrer qu'une perplexité ne se compare pas d'une langue à
l'autre, et pourquoi.

Ce que je ne sais pas, c'est si les autres métriques du domaine souffrent du même
défaut. J'en ai utilisé plusieurs pendant le master sans jamais vérifier ce
qu'elles mesuraient réellement.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, je change la conclusion, pas l'expérience.

**Le code est commenté en français.** C'est un dépôt à lire autant qu'à exécuter.

---

*Version anglaise : [README.md](README.md).*

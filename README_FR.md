# NLP multilingue : ce qu'une perplexité mesure vraiment

Modèles n-grammes, BLEU et décodage, écrits sans aucune dépendance, pour répondre à une question restée ouverte en cours.

**Léo Mégret** — implémentation, notes techniques en français.  
**Haeeul Hwang** — documentation anglaise, documentation coréenne, relecture.

> **État du dépôt** — étape 2 sur 3, publiée par Léo Mégret.

---

## Pourquoi ce dépôt

Le TP qui ouvrait mon cours de NLP multilingue demandait d'entraîner un modèle de
langue par langue et de comparer les perplexités. La question centrale de
l'énoncé était de savoir si des perplexités se comparent entre langues.

J'avais répondu non, sans pouvoir le démontrer. Ce dépôt est la démonstration.

L'énoncé demandait 41 langues, 40 000 phrases d'entraînement chacune et
l'installation de KenLM. Je fais l'inverse : cinq langues, un corpus parallèle de
40 phrases écrit à la main, et tout dans le code. Le résultat tourne sur un
ordinateur portable et reste exécutable dans dix ans.

---

## Les étapes

| | Étape | Modules | Tests | Publiée par | Note détaillée (français) |
|---|---|---|---:|---|---|
| **1** | Modèles de langue n-grammes et perplexité | `src/corpus.py`, `src/ngrammes.py` | 18 | Léo | [`ngrammes-et-perplexite`](notes/ngrammes-et-perplexite.md) |
| **2** | BLEU : implémenter la métrique, puis la casser | `src/bleu.py` | 16 | Léo | [`bleu`](notes/bleu.md) |

**34** tests au total.

---

## Ce que chaque étape a trouvé

**1. Modèles de langue n-grammes et perplexité**  
Le turc obtient la *meilleure* perplexité (2,7) alors que c'est la langue la plus dure du lot — parce qu'il refuse de prédire 80 % de ses mots. La note documente le bug à un million qui a mené à cette mesure.

**2. BLEU : implémenter la métrique, puis la casser**  
720 permutations des mots d'une même phrase donnent un BLEU moyen de 75,22. La métrique est largement aveugle à l'ordre des mots, c'est-à-dire à ce qu'on lui demande de juger.

---

## Lancer le code

```bash
python -m src.bleu

python -m tests.test_ngrammes
python -m tests.test_bleu
```

---

## Comment nous travaillons

**Rien à télécharger.** Le corpus est écrit dans le code. Tous mes
notebooks de master commençaient par un `wget` vers un serveur universitaire ou
un montage de Google Drive ; deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à
une commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu était faux ou incomplet, je le dis et je donne le bon.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français**, parce que c'est la langue dans laquelle je
l'ai pensé. La documentation est en anglais, et en coréen quand Haeeul est
passée dessus.

---

*English: [README.md](README.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*

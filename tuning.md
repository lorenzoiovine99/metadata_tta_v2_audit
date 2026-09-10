# Hyperparameter Tuning Protocol

## 1. Obiettivo

Il tuning viene organizzato come una procedura **gerarchica e staged**.

L'idea principale è:

> Prima si ottimizza il metodo più semplice che definisce una configurazione di base.
> I metodi più complessi ereditano tale configurazione e vengono successivamente ottimizzati solamente sui parametri aggiuntivi o specifici.

Questo evita di ripetere inutilmente la stessa grid search per modelli e metodi fortemente correlati.

La pipeline di tuning è:

```text
Single-head tuning
        ↓
best single-head configuration
        ↓
Double-head auxiliary tuning
        ↓
best double-head configuration
        ↓
Metadata TTA tuning
        ↓
best metadata TTA configuration
        ↓
Temporal Gradient specific tuning
        ↓
best Temporal Gradient configuration

        +
        ↓
Consensus specific tuning
        ↓
best Consensus configuration

TENT tuning independently
```

Gli anni OOD reali non vengono mai utilizzati per hyperparameter selection.

---

# 2. Principio generale

Il tuning è diviso in due blocchi:

1. **Baseline tuning**
2. **TTA tuning**

All'interno di ciascun blocco il tuning è sequenziale.

Non vengono quindi effettuate grid search completamente indipendenti per ogni metodo.

L'obiettivo è ridurre:

$$
N_{\text{trials}}
$$

senza perdere interpretabilità.

La procedura utilizzata rimane una **Grid Search**, inizialmente con 2-3 valori per parametro.

---

# 3. Baseline tuning

## 3.1 Split annuale

Per ogni anno ID:

$$
Y_1,Y_2,\dots,Y_T
$$

il dataset viene suddiviso in:

* train;
* validation;
* test.

Durante il tuning baseline vengono utilizzati solamente:

```text
train
validation
```

Il test split rimane escluso dalla model selection.

---

# 4. Training sequenziale delle baseline

Per ogni configurazione candidata il modello viene inizializzato una sola volta.

Successivamente continua ad allenarsi attraverso gli anni:

$$
Y_1^{train}
\rightarrow
Y_2^{train}
\rightarrow
\dots
\rightarrow
Y_T^{train}
$$

Il modello non viene reinizializzato tra gli anni.

Dopo il training di ogni anno viene misurata l'accuracy sul validation split dello stesso anno:

$$
Acc_t^{val}(h)
$$

dove \(h\) indica la configurazione corrente.

---

# 5. Baseline selection score

Per ogni configurazione:

$$
Score_{\text{baseline}}(h)
=
\frac{1}{T}
\sum_{t=1}^{T}
Acc_t^{val}(h)
$$

Viene quindi utilizzata una **macro-average sugli anni**.

La configurazione migliore è:

$$
h^*
=
\arg\max_h
Score_{\text{baseline}}(h)
$$

Ogni anno pesa allo stesso modo.

---

# 6. Stage 1 — Single-Head Tuning

La single-head rappresenta la configurazione baseline di base.

Vengono tunati i parametri comuni principali:

* `learning_rate`
* `weight_decay`
* `dropout`
* `shared_hidden_dim`
* `batch_size`

Possibile prima grid:

```yaml
learning_rate:
  - 0.0003
  - 0.001

weight_decay:
  - 0.0
  - 0.0001

dropout:
  - 0.0
  - 0.1

shared_hidden_dim:
  - 256
  - 512

batch_size:
  - 32
  - 64
```

Numero di trial:

$$
2^5 = 32
$$

Al termine viene selezionata:

$$
h^*_{\text{single}}
$$

e salvata come configurazione baseline comune di riferimento.

---

# 7. Stage 2 — Double-Head Tuning

La double-head non viene inizialmente tunata da zero.

Si parte dalla migliore configurazione trovata per la single-head:

$$
h^*_{\text{single}}
$$

I parametri comuni vengono quindi ereditati:

* learning rate;
* weight decay;
* dropout;
* hidden dimension;
* batch size.

Inizialmente viene tunato solamente:

* `aux_loss_weight`

perché rappresenta il principale elemento aggiuntivo della double-head.

Possibile grid:

```yaml
aux_loss_weight:
  - 0.1
  - 0.3
  - 0.5
```

La loss della double-head è:

$$
L
=
L_{\text{main}}
+
\lambda_{\text{aux}}
L_{\text{aux}}
$$

dove:

$$
\lambda_{\text{aux}}
=
aux\_loss\_weight
$$

La configurazione selezionata massimizza sempre la **main-task validation accuracy**.

L'auxiliary accuracy non viene utilizzata direttamente come selection criterion.

---

# 8. Perché `aux_loss_weight` è fondamentale

Il task ausiliario influenza la rappresentazione condivisa.

Un valore non adeguato di:

$$
\lambda_{\text{aux}}
$$

può ridurre significativamente la main-task accuracy.

Poiché i metodi metadata-based partono dalla double-head, è importante evitare che il confronto con la single-head parta da una baseline artificialmente più debole.

Per questo `aux_loss_weight` viene sempre tunato.

---

# 9. Optional Stage 3 — Double-Head Refinement

La configurazione single-head non viene considerata necessariamente ottimale anche per la double-head.

Essa viene utilizzata come **punto di partenza**.

Se, dopo aver ottimizzato `aux_loss_weight`, la double-head:

* rimane significativamente peggiore;
* mostra elevata sensibilità;
* oppure sembra under/over-regularized;

si effettua una piccola grid locale.

I parametri candidati principali sono:

* `learning_rate`
* `weight_decay`
* eventualmente `dropout`

Esempio:

```yaml
learning_rate:
  - 0.0003
  - 0.001
  - 0.003

aux_loss_weight:
  - 0.1
  - 0.3
  - 0.5
```

Se vengono tunati solo questi due parametri:

$$
3\times3=9
$$

trial.

Questa fase è opzionale.

Non viene effettuata automaticamente se la prima double-head è già competitiva.

---

# 10. Adapter Bottleneck

Il parametro:

```text
adapter_bottleneck_dim
```

rimane fisso.

Valore iniziale:

$$
64
$$

Non viene incluso nel tuning baseline.

Potrà essere analizzato successivamente tramite un ablation study dedicato.

---

# 11. Baseline tuning pipeline finale

```text
Grid search common baseline parameters
            ↓
        SINGLE-HEAD
            ↓
       best_single
            ↓
freeze common parameters
            ↓
tune aux_loss_weight
            ↓
        DOUBLE-HEAD
            ↓
       best_double
            ↓
optional local refinement
```

---

# 12. TTA Hyperparameter Tuning

Il tuning TTA utilizza solamente gli anni ID.

Gli anni ID vengono divisi temporalmente in:

```text
initial supervised source period
+
continuous pseudo-OOD period
```

Per esempio:

```text
2002 ... 2008
       ↓
 supervised source
       ↓
2009 → 2010 → 2011 → 2012
       continuous TTA stream
```

Lo stato TTA viene inizializzato una sola volta all'inizio dello pseudo-stream.

Non viene resettato tra gli anni.

---

# 13. Causalità

Durante lo pseudo-OOD stream viene mantenuta la stessa causalità della valutazione finale.

Per ogni campione:

$$
x_t
$$

la sequenza è:

```text
predict
   ↓
record prediction
   ↓
observe metadata
   ↓
adapt
   ↓
next observation
```

Formalmente:

$$
Predict(x_t)
\rightarrow
Observe(metadata_t)
\rightarrow
Adapt
\rightarrow
Predict(x_{t+1})
$$

La main-task label non viene fornita al metodo TTA.

---

# 14. TTA objective

Per ogni anno dello pseudo-stream viene calcolata la differenza rispetto al corrispondente frozen source model:

$$
\Delta Acc_t(h)
=
Acc_{\text{TTA},t}(h)
-
Acc_{\text{Frozen},t}
$$

Lo score principale è:

$$
Score_{\text{TTA}}(h)
=
\frac{1}{T}
\sum_t
\Delta Acc_t(h)
$$

La configurazione migliore è:

$$
h^*
=
\arg\max_h
Score_{\text{TTA}}(h)
$$

Vengono inoltre salvati:

* TTA accuracy per anno;
* frozen accuracy per anno;
* delta accuracy per anno;
* mean delta;
* standard deviation;
* worst-year delta;
* numero di anni migliorati rispetto al frozen model.

---

# 15. Stage 4 — Metadata TTA Tuning

Metadata TTA rappresenta il metodo base per i successivi metodi metadata-based.

Vengono tunati i principali parametri comuni:

* `learning_rate`
* `regularization`
* `gradient_clip`
* `normalized_aux_loss_threshold`

Possibile grid iniziale:

```yaml
learning_rate:
  - 1.0e-7
  - 1.0e-6

regularization:
  - 0.001
  - 0.01

gradient_clip:
  - 0.1
  - 0.5

normalized_aux_loss_threshold:
  - 0.5
  - 1.0
```

Numero massimo:

$$
2^4=16
$$

trial.

Vengono mantenuti fissi:

```yaml
batch_size: 1
steps: 1
episodic: false
reset_each_year: false
```

Al termine viene ottenuta:

$$
h^*_{\text{metadata}}
$$

---

# 16. Stage 5 — Temporal Gradient TTA

Temporal Gradient parte dalla migliore configurazione Metadata:

$$
h^*_{\text{metadata}}
$$

I parametri comuni vengono quindi inizialmente mantenuti fissi:

* `learning_rate`
* `regularization`
* `gradient_clip`
* `normalized_aux_loss_threshold`

Si tunano solamente i parametri specifici del metodo:

* `beta`
* `orthogonal_scale`
* `warmup_updates`

Esempio:

```yaml
beta:
  - 0.95
  - 0.99

orthogonal_scale:
  - 0.1
  - 0.5

warmup_updates:
  - 4
  - 16
```

Numero di trial:

$$
2^3=8
$$

La configurazione completa sarà quindi:

$$
h_{\text{temporal}}
=
h^*_{\text{metadata}}
+
h_{\text{temporal-specific}}
$$

---

# 17. Optional Temporal Gradient Refinement

Poiché Temporal Gradient modifica la geometria dell'update, il learning rate ottimale potrebbe differire da Metadata.

Se necessario, dopo il tuning dei parametri specifici si può effettuare una piccola refinement locale di:

* `learning_rate`

per esempio:

```yaml
learning_rate:
  - best_metadata_lr / 3
  - best_metadata_lr
  - best_metadata_lr * 3
```

insieme alla migliore configurazione specifica già trovata.

Questa fase è opzionale.

---

# 18. Stage 6 — Consensus TTA

Consensus parte anch'esso da:

$$
h^*_{\text{metadata}}
$$

I parametri comuni vengono inizialmente congelati.

Vengono tunati solamente:

* `consensus_beta`
* `class_warmup_updates`
* current/memory gradient mixing;
* `min_consensus_cosine`

Il mixing viene rappresentato tramite:

$$
\alpha
$$

con:

$$
g
=
\alpha g_{\text{current}}
+
(1-\alpha)g_{\text{memory}}
$$

Possibile grid:

```yaml
consensus_beta:
  - 0.95
  - 0.99

class_warmup_updates:
  - 4
  - 8

alpha:
  - 0.25
  - 0.5
  - 0.75

min_consensus_cosine:
  - 0.0
  - 0.25
```

Numero di trial:

$$
2\times2\times3\times2
=
24
$$

Molto inferiore rispetto a una grid completa contenente anche tutti i parametri Metadata.

---

# 19. Optional Consensus Refinement

Anche Consensus può successivamente effettuare una refinement locale di:

* `learning_rate`
* eventualmente `regularization`

se il comportamento indica che i valori ottimali ereditati da Metadata non sono adeguati.

La refinement deve essere locale e successiva al tuning dei parametri specifici.

---

# 20. Stage 7 — TENT

TENT viene tunato separatamente.

Non eredita gli hyperparameter Metadata perché utilizza un meccanismo di adaptation differente.

Parte dalla migliore single-head:

$$
h^*_{\text{single}}
$$

e vengono tunati inizialmente:

* `learning_rate`
* `batch_size`

Possibile grid:

```yaml
learning_rate:
  - 0.0001
  - 0.001
  - 0.01

batch_size:
  - 32
  - 64
```

Numero di trial:

$$
3\times2=6
$$

Rimangono fissi:

```yaml
optimizer: adam
weight_decay: 0.0
steps: 1
reset_each_year: false
```

---

# 21. Full hierarchical tuning pipeline

La pipeline completa diventa:

```text
──────────────────────────────────────────────
            BASELINE TUNING
──────────────────────────────────────────────

Single-head
common hyperparameters grid
        ↓
best_single
        ↓
        ├──────────────────────┐
        │                      │
        ↓                      ↓
Double-head                  TENT
inherit best_single          inherit best_single
tune aux_loss_weight        tune TENT params
        ↓
best_double
        ↓
optional local refinement


──────────────────────────────────────────────
               TTA TUNING
──────────────────────────────────────────────

best_double
        ↓
Metadata TTA
tune common adaptation params
        ↓
best_metadata
        ↓
   ┌────┴─────────────────┐
   │                      │
   ↓                      ↓
Temporal Gradient      Consensus
inherit metadata       inherit metadata
common params          common params
   │                      │
tune method-specific   tune method-specific
params                 params
   ↓                      ↓
best_temporal          best_consensus
   ↓                      ↓
optional LR            optional LR /
refinement             regularization refinement
```

---

# 22. Search strategy

La prima implementazione utilizza:

```text
Grid Search
```

Il vantaggio principale è l'interpretabilità.

Ogni configurazione testata è nota esplicitamente.

Non viene utilizzata una grande grid cartesiana globale.

La grid viene invece scomposta in più fasi.

In generale:

$$
\text{large global grid}
$$

viene sostituita da:

$$
\text{base grid}
+
\text{small conditional grids}
$$

Questo riduce significativamente il costo computazionale.

---

# 23. Esempio di riduzione del numero di trial

Approccio completamente indipendente:

Single:

$$
32
$$

Double:

$$
96
$$

Metadata:

$$
16
$$

Temporal Gradient:

$$
128
$$

Consensus:

$$
384
$$

Totale:

$$
656
$$

trial circa.

Con il protocollo staged:

Single:

$$
32
$$

Double aux:

$$
3
$$

Metadata:

$$
16
$$

Temporal specific:

$$
8
$$

Consensus specific:

$$
24
$$

TENT:

$$
6
$$

Totale:

$$
89
$$

trial prima delle eventuali refinement.

La riduzione è quindi molto significativa.

---

# 24. Result inheritance

Ogni stage produce:

```text
trials.csv
best.yaml
```

Il successivo stage legge direttamente il:

```text
best.yaml
```

del metodo da cui dipende.

Esempio:

```text
single_head/best.yaml
        ↓
double_head tuning
```

oppure:

```text
metadata/best.yaml
        ↓
temporal_gradient tuning
```

Questo comportamento deve essere esplicito e auditabile.

---

# 25. Best configurations

Gli artifact finali saranno indicativamente:

```text
results/tuning/
├── baselines/
│   ├── single_head/
│   │   ├── trials.csv
│   │   └── best.yaml
│   │
│   └── double_head/
│       ├── trials.csv
│       └── best.yaml
│
├── tta/
│   ├── metadata/
│   │   ├── trials.csv
│   │   └── best.yaml
│   │
│   ├── temporal_gradient/
│   │   ├── trials.csv
│   │   └── best.yaml
│   │
│   ├── consensus/
│   │   ├── trials.csv
│   │   └── best.yaml
│   │
│   └── tent/
│       ├── trials.csv
│       └── best.yaml
│
└── tuned_experiment.yaml
```

---

# 26. Separazione ID / OOD

Regola fondamentale:

> Gli anni OOD finali non devono mai influenzare la scelta degli hyperparameter.

Prima della valutazione OOD devono essere congelati:

* single-head configuration;
* double-head configuration;
* Metadata configuration;
* Temporal Gradient configuration;
* Consensus configuration;
* TENT configuration.

Solo successivamente viene eseguita la valutazione OOD.

---

# 27. Principio metodologico finale

Il protocollo segue una logica di complessità crescente.

Per le baseline:

$$
Single
\rightarrow
Double
$$

Per i metodi metadata-based:

$$
Metadata
\rightarrow
TemporalGradient
$$

e:

$$
Metadata
\rightarrow
Consensus
$$

I metodi più complessi ereditano quindi un insieme di hyperparameter già validato dal metodo più semplice.

La ricerca aggiuntiva si concentra principalmente sui gradi di libertà introdotti dal nuovo metodo.

Questo rende il tuning:

* più economico;
* più interpretabile;
* più facilmente auditabile;
* più semplice da modificare;
* compatibile con successive refinement;
* compatibile con una futura sostituzione della Grid Search con Optuna.

---

# 28. Future Optuna migration

La logica staged non dipende dalla Grid Search.

In futuro ogni stage potrà semplicemente sostituire:

```text
GridSearchStrategy
```

con:

```text
OptunaSearchStrategy
```

mantenendo invariati:

* objective;
* dataset split;
* temporal protocol;
* parameter inheritance;
* source checkpoints;
* metriche;
* result storage;
* dependency tra gli stage.

Il protocollo sperimentale rimarrà quindi identico.

# Model Compression for Edge AI

Progetto  dedicato allo studio e al confronto di tecniche di compressione di reti neurali, con focus di analisi per dispositivi **Edge / TinyML**.

L'obiettivo è partire da una CNN di riferimento (*teacher*), costruire una versione più leggera (*student*) e valutare come diverse tecniche, singolarmente e in combinazione, influenzino accuratezza, dimensione del modello, costo computazionale e memoria richiesta durante l'inferenza.

Il progetto usa **PyTorch** e implementa direttamente le principali trasformazioni, evitando API di alto livello che nascondano il funzionamento delle tecniche studiate.

---

## Indice

- [Obiettivi](#obiettivi)
- [Setup sperimentale](#setup-sperimentale)
- [Tecniche studiate](#tecniche-studiate)
- [Metriche](#metriche)
- [Limiti dell'esperimento](#limiti-dellesperimento)

## Obiettivi

Quattro domande principali:

1. **Quanto si può comprimere una CNN?** Ridurre pesi, parametri e memoria occupata dal modello.
2. **Qual è il costo in accuratezza?** Misurare il compromesso tra compressione e qualità delle predizioni.
3. **La compressione riduce davvero il costo di inferenza?** Confrontare MACs, memoria delle attivazioni, latenza ed energia stimata, distinguendo i benefici teorici da quelli ottenibili su hardware reale.
4. **Dove conviene eseguire il modello?** Studiare uno scenario di *split computing*, in cui una parte della rete gira sul dispositivo edge e la parte restante su un fog node.

Le configurazioni vengono confrontate usando lo stesso dataset e lo stesso protocollo di valutazione. 
## Setup sperimentale

### Dataset

Si utilizza **CIFAR-10**, composto da immagini RGB di dimensione `32 × 32` appartenenti a 10 classi.

### Modelli

Si utilizza una VGG11 costruita in `src/models.py`, adatta ad input 32x32.

Il teacher costituisce il modello di riferimento. Lo student mantiene la stessa struttura generale ma ridimensionato.


## Tecniche studiate

Le implementazioni sono raccolte nei moduli di `src/`; i notebook si occupano principalmente di configurare gli esperimenti, eseguirli e confrontarne i risultati.

### 1. Quantizzazione post-training (W8A8)

La quantizzazione implementata è una simulazione **Post-Training Quantization (PTQ)** a 8 bit per pesi e attivazioni:

- **Pesi (W8):** quantizzazione simmetrica `int8` per-channel, con una scala per canale di output.
- **Attivazioni (A8):** `FakeQuant` asimmetrico `int8` per-tensor, con scala e zero-point ricavati da calibrazione min/max.
- **Calibrazione:** eseguita su batch del training set; nella configurazione usata nei notebook, 4 batch da 128 immagini corrispondono a 512 campioni.


### 2. Pruning non strutturato

Il pruning non strutturato azzera singoli pesi in base al loro valore assoluto. La soglia può essere scelta globalmente oppure per-layer, in funzione della sparsità desiderata.

La valutazione considera diverse percentuali di sparsità e seleziona la configurazione sulla validation set. Il modello mantiene però le dimensioni dense originali: per ottenere un'accelerazione effettiva è necessario hardware o un runtime capace di sfruttare la sparsità.

La dimensione del formato sparso e i MACs ottenuti saltando gli zeri sono pertanto **stime ideali**, non misure di accelerazione del forward PyTorch denso.

### 4. Pruning strutturato

Il pruning strutturato rimuove interi filtri convoluzionali o neuroni fully connected e costruisce un modello fisicamente più piccolo, copiando le slice dei pesi rimasti.

Le configurazioni possono specificare separatamente il rapporto di pruning dei layer convoluzionali e fully connected. È inoltre possibile proteggere i primi layer tramite `skip_first_ratios`.

A differenza del pruning non strutturato, questa tecnica riduce direttamente le dimensioni dei layer e può quindi diminuire realmente parametri e MACs del modello ricostruito.

### 5. Low-rank decomposition

La decomposizione low-rank approssima alcuni layer tramite fattorizzazioni a rango ridotto basate su SVD:

- `Linear(in, out)` viene sostituito da due layer lineari intermedi;
- una convoluzione viene fattorizzata in una convoluzione `k × k` con un numero ridotto di canali e una convoluzione `1 × 1`.

Il rango viene selezionato in base all'energia singolare cumulata (`τ`). La fattorizzazione viene applicata soltanto quando riduce effettivamente il numero di parametri; i layer esclusi dalla trasformazione restano invariati.

### 6. Knowledge distillation

La distillation trasferisce allo student parte dell'informazione contenuta nelle predizioni del teacher. La loss combina la cross-entropy sulle etichette reali e la KL divergence tra le distribuzioni prodotte da teacher e student a temperatura `T`:

- Loss = alpha * CE(hard label) + (1 - alpha) * T^2 * KL(teacher_T || student_T).

L'esperimento valuta quanto lo student riesca a recuperare del divario di accuratezza rispetto al teacher.

### 7. Combinazione delle tecniche

Le tecniche vengono anche valutate in pipeline, per verificare se i benefici siano cumulativi o se una trasformazione renda più difficile applicarne un'altra.

### 8. Partial quantization e split computing

Lo split computing considera una rete divisa in due parti:

- **Edge:** prefisso della rete eseguito sul dispositivo, con quantizzazione `int8`;
- **Fog:** suffisso della rete eseguito su un nodo più potente, mantenuto in `fp32`.

Variando il punto di taglio si studia il compromesso tra lavoro eseguito sull'edge, risorse richieste al fog e dimensione del tensore intermedio da trasferire.




### Convenzioni e limiti delle metriche

- **MACs:** un MAC rappresenta una moltiplicazione e un accumulo; nel conteggio standard vengono profilati i layer `Conv2d` e `Linear`.
- **Energia:** la stima usa un modello aritmetico ispirato ai valori di Horowitz (ISSCC 2014). Nella configurazione documentata vengono assunti circa `4.6 pJ/MAC` per FP32 e `0.3 pJ/MAC` per INT8. È una stima del solo costo aritmetico: non include in modo completo accessi alla memoria, trasferimenti di dati o overhead del runtime.
- **Latenza:** viene misurata su CPU, con batch size 1 e un thread, usando warm-up e ripetizioni multiple. È utile per confronti relativi nella stessa macchina e sessione, ma non rappresenta la latenza su ESP32 o altro hardware embedded.
- **Peak activation memory:** è una stima basata sulle dimensioni di input e output dei layer profilati. Non coincide necessariamente con la SRAM totale richiesta dal sistema: workspace, buffer, accumulatori e memoria del runtime possono aumentare il fabbisogno reale.
- **Quantizzazione simulata:** il FakeQuant mantiene l'esecuzione floating point e non misura direttamente i tempi di un kernel integer ottimizzato.
- **Pruning non strutturato:** MACs ed energia calcolati saltando gli zeri rappresentano un caso ideale; il normale forward denso di PyTorch non sfrutta automaticamente la sparsità.

Le dimensioni riportate come `MB` seguono le convenzioni implementate nelle funzioni di profiling del progetto. Quando si confrontano dimensioni di checkpoint FP32, stime di formato INT8 e formati sparsi, è importante tenere conto del diverso criterio di calcolo.




## Limiti dell'esperimento

I risultati vanno interpretati tenendo presenti alcuni limiti:

1. La quantizzazione è simulata con FakeQuant in floating point e non è un'inferenza integer bit-exact.
2. Le stime energetiche considerano principalmente il costo aritmetico; il movimento dei dati può incidere in modo rilevante sul consumo reale.
3. La latenza misurata su CPU desktop non predice direttamente le prestazioni su un microcontrollore.
4. Il pruning non strutturato non accelera automaticamente un runtime denso; i vantaggi sui MACs sono effettivi solo se l'hardware o il software sfruttano la sparsità.
5. La memoria di picco stimata per le attivazioni non include necessariamente tutti i buffer e gli overhead di un deployment reale.
6. Le stime di split computing dipendono dai parametri del collegamento e dai costi assegnati all'edge e al fog; per una valutazione definitiva servono misure su hardware e rete reali.


---

**Nota:** Le metriche teoriche servono soprattutto a confrontare le configurazioni sotto ipotesi uniformi; la validazione del deployment richiede misure sul dispositivo embedded e sul collegamento di comunicazione effettivamente utilizzati.

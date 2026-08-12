# Vergelijkingschecklist: onze simulator naast Sama's R-code

Doel: de verschillen tussen beide implementaties systematisch vinden, zodat we weten
of afwijkende resultaten uit de **data-generatie** of uit het **modelfitten** komen.

Onze kolom is ingevuld uit de code (`sim.py`, `metc_simulation_report.py`,
`dlt_mitigation_analysis.py`). De rechterkolom is voor Sama's implementatie.

Volgorde is op afnemende verwachte impact — bovenaan staan de punten die een groot
verschil in MTD-accuracy kunnen verklaren.

---

## Al gevonden, vóór het overleg

### ✅ Onze MTD-definitie en scenario's kloppen met die van Sama

De ware MTD op haar slides is te reproduceren als *"hoogste dosis met ware acute
toxiciteit ≤ target"*, in **1-based** dosisnummering:

| Scenario | Ware acute kansen | Onze MTD (0-based) | Sama's notatie (1-based) |
|---|---|---|---|
| Acute low | 0.01 0.02 0.06 0.10 0.15 | L4 | 5 |
| Acute middle | 0.01 0.04 0.12 0.17 0.27 | L3 | 4 |
| Acute high | 0.01 0.05 0.15 0.22 0.30 | L2 | 3 |
| Acute steep | 0.01 0.02 0.08 0.24 0.34 | L2 | 3 |
| Acute shallow | 0.10 0.13 0.15 0.18 0.24 | L3 | 4 |

Haar `MTD_late = 5` in alle scenario's klopt ook: alle subacute kansen liggen onder 0.33.

**Actie:** geen — maar spreek expliciet af welke nummering we in verslaglegging
gebruiken. `L2` bij ons is `dosis 3` bij haar. Eén off-by-one in een tabel is genoeg
om een hele discussie te vertroebelen.

### ⚠️ Onze subacute prior wijkt af van wat per mail is doorgegeven

| | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| Mail van 8 juli aan Sama | **0.010** | 0.036 | 0.084 | 0.157 | 0.250 |
| Onze code | **0.012** | 0.036 | 0.084 | 0.157 | 0.250 |

**Actie:** uitzoeken welke de bedoelde is en één van de twee corrigeren. Het effect is
klein (het gaat om het laagste niveau), maar het is een echte inconsistentie tussen wat
wij simuleren en wat Sama gebruikt.

---

## 1. Data-generatie — hier verwachten we de grootste verschillen

### 1.1 Tijd tot DLT

| | Onze code | Sama |
|---|---|---|
| Acute (tox1) | **Uniform** over het hele acute venster | **Log-normaal** (per haar mail 31 juli) |
| Subacute (tox2) | **Uniform** over het subacute venster | **Log-normaal** |

*Waarom het uitmaakt:* alleen via de TITE-gewichten. Een log-normale verdeling
concentreert events eerder of later in het venster, waardoor het partiële gewicht op
het beslismoment anders uitvalt. Effect op MTD-accuracy is waarschijnlijk beperkt, op
trialduur groter.

**Vraag aan Sama:** welke parameters (meanlog/sdlog), en gedefinieerd vanaf welk
moment — inclusie, RT-start, of RT-einde?

### 1.2 Tijd tot operatie en uitstel van de OK

| | Onze code | Sama |
|---|---|---|
| Vindt OK plaats? | Bernoulli met p = 0.80 | ? |
| Tijd RT-einde → OK | **Vaste constante van 42 dagen** | **Log-normaal** |
| Kans op uitstel | **Niet gemodelleerd** | Aparte kans + verdeling van de uitstelduur |

*Waarom het uitmaakt:* dit is het duidelijkste verschil dat we kennen. Bij ons varieert
de OK-datum helemaal niet, dus het subacute venster opent bij iedere patiënt op precies
dezelfde relatieve dag. Bij Sama varieert dat, wat de weging van partiële subacute
follow-up direct beïnvloedt.

**Actie aan onze kant:** overwegen dit toe te voegen zodra we haar parametrisatie hebben.

### 1.3 Accrual

| | Onze code | Sama |
|---|---|---|
| Proces | Poisson (exponentiële tussenaankomsttijden) | ? |
| Tempo | 0.75/maand = 1 patiënt per 4 weken | slides: 1 per 4 weken; Koen vroeg om 1.5/maand |

**Actie:** één tempo afspreken vóór de definitieve run. 1.5/maand ≈ 1 per 3 weken.

---

## 2. Beslisregels — hier verwachten we het grootste effect op MTD-accuracy

### 2.1 Dosistoewijzing tijdens de trial

| | Onze code | Sama |
|---|---|---|
| EWOC aan | Hoogste dosis waar P(tox>target) < α voor **beide** eindpunten | ? |
| EWOC uit | Dosis met posterior gemiddelde **dichtst bij** target (klassieke CRM) | **Hoogste dosis met puntschatting onder target** (mail 8 juli) |
| Dose skipping | Verboden: `max_step = 1` én niet hoger dan `hoogste_geprobeerd + 1` | verboden |

*Dit is vermoedelijk de belangrijkste bron van verschil.* Onze "EWOC uit" is een ander
algoritme dan haar "EWOC uit". Hare is strenger: één DLT die de puntschatting van een
hoog niveau net over de target duwt sluit dat niveau uit, terwijl onze argmin-regel dat
niveau nog kan kiezen.

**Vraag aan Sama:** gebruikt ze het posterior gemiddelde, de posterior mediaan, of een
plug-in schatter op de posterior mean van de parameter? Dat laatste geeft merkbaar
andere waarden.

### 2.2 Finale MTD-selectie

| | Onze code | Sama |
|---|---|---|
| Regel | Zelfde als toewijzing, plus `restrict_to_tried = True` | ? |
| Alleen uit geprobeerde doses? | **Ja** | ? |

*Waarom het uitmaakt:* als zij ook niet-geprobeerde doses als MTD kan selecteren, is haar
accuracy in het Acute low scenario structureel hoger dan de onze.

### 2.3 Burn-in

| | Onze code | Sama |
|---|---|---|
| Definitie | Escaleer 1 niveau per cohort tot de eerste **waargenomen** tox1 | idem |
| Strikte follow-up-eis | Optioneel: escaleren mag pas als ≥ cohortgrootte patiënten volledige acute FU zonder DLT hebben | ? |

*Let op:* in de simulaties met de DLT bij initialisatie is burn-in per definitie meteen
voorbij, dus dit verklaart hier niets. Wel relevant voor de eerdere vergelijkingen.

---

## 3. Model en posterior

| | Onze code | Sama |
|---|---|---|
| Modelvorm | Empirisch / power: p_d = skeleton_d^exp(θ) | ? (`dfcrm` default is empiric) |
| Prior op θ | Normaal, gemiddelde 0, **σ = 1.0** | ? |
| Integratie | Gauss-Hermite, 61 knopen (41 in het METC-rapport) | ? (`dfcrm` gebruikt numerieke integratie) |
| Acute skeleton | 0.004 0.021 0.066 0.150 0.266 | idem, per mail |
| Subacute skeleton | 0.012 0.036 0.084 0.157 0.250 | 0.010 … — zie inconsistentie hierboven |
| Skeleton-herkomst | Exacte getallen, niet uit halfwidth herleid | Koen adviseerde exacte getallen |

*Waarom het uitmaakt:* σ bepaalt hoe hard het model de skeleton loslaat. Als haar σ
afwijkt, verschuiven alle posteriors. Dit is het meest waarschijnlijke "fit"-verschil
naast de beslisregel.

**Vraag aan Sama:** welke waarde voor σ, en of dat de pakket-default is of een bewuste
keuze. In `dfcrm` heet deze parameter `scale`; de default daar is níet 1.0, dus als zij de
default heeft laten staan wijkt haar prior af van de onze. Belangrijk om expliciet te
maken of we het over de standaarddeviatie of de variantie hebben — dat is een klassieke
bron van een factor-verschil.

---

## 4. Follow-up vensters en TITE-weging

| | Onze code | Sama |
|---|---|---|
| Inclusie → RT-start | 21 dagen | ? |
| Duur RT | 14 dagen | ? |
| RT-einde → OK | 42 dagen (vast) | log-normaal |
| Acuut venster | 56 dagen, startend bij **RT-start** | ? |
| Subacuut venster | 30 dagen, startend bij de **OK** | ? |
| Gewichtsfunctie | Lineair: verstreken tijd / vensterlengte, afgekapt op 1 | ? (standaard TITE is ook lineair) |
| Waargenomen event | Gewicht 1 | idem verwacht |
| Nog niet in venster | Gewicht 0 | ? |

*Op te merken:* bij ons eindigt het acute venster **exact op de OK-datum**
(21 + 14 + 42 = 77 dagen na inclusie = RT-start + 56). Dat is geen toeval maar een
gevolg van `tox1_win = rt_dur + rt_to_surg`. Als Sama de OK-datum laat variëren, kan
haar acute venster niet ook automatisch op de OK eindigen — dan is dat een structureel
verschil in de definitie van het acute venster.

**Vraag aan Sama:** is haar acute venster een vaste kalenderduur, of loopt het tot de OK?

---

## 5. Beslismoment en timing

| | Onze code | Sama |
|---|---|---|
| Wanneer wordt de dosis bepaald? | Op het moment dat de **laatste patiënt van het cohort arriveert**; geen wachttijd tussen cohorten | ? |
| Cohortgrootte | 3 | 3 |
| Partiële data gebruikt? | Ja, dat is het punt van TITE | ja |

*Waarom het uitmaakt:* dit bepaalt hoeveel follow-up er per beslissing beschikbaar is,
en daarmee zowel de trialduur als de kwaliteit van elke beslissing. Als zij wél wacht
(bijvoorbeeld tot een minimum aan follow-up), verklaart dat een deel van het
duurverschil.

---

## 6. Initialisatie met de bestaande zes patiënten

| | Onze code | Sama |
|---|---|---|
| Aantal | 6 op L1 (`dosis 2` in haar nummering) | 6 |
| Acute DLT's | 1, bij de laatste van de zes | 1 |
| Tijdstip van de DLT | Op het einde van het (al verstreken) follow-up venster, dus vanaf dag 0 waargenomen | ? |
| Gewicht in de likelihood | Volledig (1.0), tenzij `hist_weight` anders staat | ? |
| OK-status | Getrokken uit p = 0.80 | ? |
| Subacute DLT's | 0 voor alle zes | ? |
| Tellen ze mee in max_n? | **Ja** — 30 totaal betekent 24 nieuwe patiënten | ? |

*Dit laatste punt is belangrijk:* als haar 30 patiënten **naast** de zes bestaande komen,
heeft haar trial 6 patiënten meer om mee te escaleren. Dat alleen al kan een groot deel
van het accuracy-verschil in Acute low verklaren.

---

## 7. Het 6+3 design — grootste onverklaarde gat

In Acute low vinden wij **3.2%** correcte MTD-selectie voor 6+3; haar slide toont
**32.3%**. Twee concrete hypothesen, beide te testen:

**Hypothese A — andere initialisatie.** Onze `run_tite_6plus3` kent geen voorgeschiedenis:
die start schoon op L2 met 30 patiënten. Haar 6+3 in deze run heeft mogelijk wél de zes
patiënten met de DLT. Onze vergelijking is dus niet één-op-één.

**Hypothese B — onze 6+3-regels zijn strenger.** Onze implementatie eist voor escalatie:

- minstens 6 patiënten **én** 6 patiënten met een OK op het niveau (bij p = 0.80 betekent
  dat gemiddeld ~7.5 inclusies per niveau);
- 0 acute DLT's in 6, of ≤ 1 in 9;
- ≤ 1 subacute DLT in 6, of ≤ 3 in 9;
- volledige follow-up van het hele cohort vóór de beslissing, met "bridging"-patiënten op
  een lager niveau tijdens het wachten — die verbruiken wél sample size.

Een klassiek 6+3 zonder OK-evaluabiliteitseis en zonder bridging klimt binnen 30
patiënten veel verder. **Actie:** vraag Sama's exacte escalatie- en stopregels op en zet
ze naast bovenstaande lijst.

---

## 8. Uitkomstmaten — appels met appels

| | Onze definitie | Sama |
|---|---|---|
| Correcte MTD | Geselecteerde dosis == hoogste met ware tox ≤ target | idem (geverifieerd) |
| Te hoge MTD | Geselecteerd > ware MTD | ? |
| Aantal DLT's | Alleen **nieuwe** patiënten, exclusief de zes bestaande | ? |
| Trialduur | Aantal nieuwe patiënten × 4 weken | ? |
| Trialduur bij ons | 96 weken (vast: er is geen early stopping, dus altijd max_n) | slides: 30 weken in Acute low |

*Het duurverschil van 96 tegen 30 weken is te groot om alleen uit accrual te komen.*
Waarschijnlijke oorzaak: haar design stopt vroeg (early stopping), of zij rekent de duur
tot de laatste **beslissing** in plaats van tot het einde van alle follow-up. Bij ons
staat early stopping uit (`p_stop = 1.0`).

**Vraag aan Sama:** zit er een stopregel in, en vanaf welk moment tot welk moment wordt
de duur gemeten?

---

## Praktische aanpak voor het overleg

1. **Wissel eerst één generating file uit.** Wij hebben er al één klaarstaan
   (`trial_export/`): één trial, 30 patiënten, met per patiënt dosis, acute DLT + dag,
   OK + dag, subacute DLT + dag. Laat Sama die door haar model halen.
2. **Vergelijk cohort voor cohort, niet alleen de eindconclusie.** Onze
   `tite_crm_output_trial1.csv` bevat per beslismoment de TITE-gewichten, de posterior
   gemiddelden per dosis en de gekozen dosis. Wijkt de eerste beslissing al af, dan zit
   het in de weging of de prior; wijkt pas een latere af, dan in de beslisregel.
3. **Zet vervolgens σ en de beslisregel gelijk** en kijk of de verschillen verdwijnen.
   Dat isoleert het effect van de data-generatie.
4. **Laat het 6+3 apart lopen.** Dat is een eigen vergelijking met een eigen oorzaak, en
   het vertroebelt de CRM-discussie als we het erbij houden.

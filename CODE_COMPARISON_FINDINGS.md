# Bevindingen: Sama's R-code naast onze simulator

Doorgenomen op 18-08-2026, op basis van de map `tite-crm-simulation-main` zoals gedeeld
via Google Drive. Dit document vult de rechterkolom in van `CODE_COMPARISON_CHECKLIST.md`.

Gelezen: `README.md`, `examples/example_run.R`, `R/run_titecrm_simulation.R`
(`run_simulation()`), `R/run_full_titecrm_trial_M1.R`, `R/run_burnin_phase.R`,
`R/run_trial_phase.R`, `R/run_final_analysis.R`, `R/Generat_Time_function.R`,
`R/Generate_DLT_Time_function.R`.

Niet gelezen: `run_rule_based_trial.R`, `Correlate_Bernoulli_data_Copula.R`,
`Generating correlated.R`, `plot_time_distributions.R`.

> **Correctie op een eerdere versie van dit document.** Daarin stond dat het niet-beperken
> tot geprobeerde doses "vermoedelijk de grootste enkele verklaring" was voor het verschil
> in MTD-accuracy. Dat is gemeten en **onjuist gebleken**. De dominante factor is de
> dosis-selectieregel. Zie sectie 1.

---

## 1. De dominante factor: de dosis-selectieregel

We hebben Sama's regel in onze eigen simulator geïmplementeerd (`dose_rule="highest_below"`)
en één voor één omgezet, 1000 simulaties per cel. Uitkomsten in
`dlt_attribution_sensitivity/rule_alignment.csv`, gegenereerd door
`alignment_experiment.py`.

| Variant | Correct gem. | Te hoog gem. | Acute low |
|---|---|---|---|
| A · onze code (argmin, restrict, sigma 1.0) | 39.0% | 17.3% | 16.8% |
| B · + niet beperkt tot geprobeerde doses | 39.2% | 21.3% | 25.3% |
| C · + Sama's dosisregel | **31.7%** | 8.4% | **0.0%** |
| D · + sigma √1.34 | 31.6% | 8.7% | 0.0% |

Twee dingen springen eruit.

**De beperking tot geprobeerde doses verklaart weinig.** In Acute low levert het 8.5
procentpunt op (16.8 → 25.3), gemiddeld vrijwel niets (39.0 → 39.2). Het is een reëel
verschil, maar niet de verklaring voor het gat.

**De dosisregel verklaart het wél, en precies.** Met Sama's regel zakt Acute low naar
**0.0%** — exact wat haar slides tonen. En de verdeling die wij dan krijgen (L2 48%, L3 50%,
L4 0%) ligt dicht bij de hare (L2 54.5%, L3 40.3%, L4 0.0%).

**Sigma doet er nauwelijks toe** (31.7 → 31.6). Het verschil is echt, maar verklaart niets
van de discrepantie.

### Waarom die regel L4 uitsluit

De acute skeleton is `0.004, 0.021, 0.066, 0.150, 0.266` bij een target van 0.20.
**De prior op L4 (0.266) ligt al boven de target.** Onder de regel "hoogste dosis waarvan
de puntschatting op of onder de target ligt" valt L4 dus vanaf het eerste beslismoment
buiten de toelaatbare verzameling, en moet de data die schatting eerst onder 0.20 trekken.
Met een DLT op L1 die de hele curve omhoog duwt, gebeurt dat binnen 30 patiënten nooit.

Wij hebben apart getest of het subacute eindpunt hierin meespeelt door target2 op 0.99 te
zetten: het resultaat blijft 0.0%. **Het is de acute regel alleen.**

### Een verschil in onze code dat we zelf niet scherp hadden

Dit hoort hierbij en verdient een eigen vermelding. In `crm_choose_next()` geldt met EWOC
uit:

```python
if ewoc_alpha is None:
    candidates = np.arange(n_levels)      # alle doses
...
    dist = np.abs(pm1[candidates] - float(target1))
    k = int(candidates[int(np.argmin(dist))])
```

`od2` — de subacute overdose-kans — wordt wél berekend maar **nergens gebruikt**. Met EWOC
uit heeft het subacute eindpunt dus **geen enkele invloed** op de dosiskeuze, niet tijdens
de trial en niet bij de finale selectie. Bij Sama begrenzen acuut en subacuut allebei, via
`min(max(acuut toelaatbaar), max(subacuut toelaatbaar))`.

Gevolg voor ons eigen werk: **alle DLT-attributieanalyses van de afgelopen dagen draaiden
met EWOC uit**, en daarin heeft het subacute eindpunt dus niets gedaan. Dat maakt die
resultaten niet ongeldig, maar het moet erbij vermeld worden.

---

## 2. Finale MTD kan een niet-behandelde dosis zijn

In `run_final_analysis.R`:

```r
p11 <- max(which(Tite.CRM.1$ptox <= early.target))
last_dose <- acute.DLT.1$dose_level[nrow(acute.DLT.1)]
p <- ifelse(last_dose >= p11, p11, min(last_dose + 1, Max.dose))
```

De zoektocht loopt over alle niveaus; bij ons staat `restrict_final_to_tried = True`.

Belangrijke nuance: de grens is de **laatst toegediende** dosis, niet de hoogste ooit
geprobeerde. Dat snijdt twee kanten op. Eindigt de trial op L4, dan kan L5 als MTD worden
aangewezen zonder dat daar iemand behandeld is. Maar is L4 eerder wel geprobeerd en eindigt
de trial op L2, dan sluit haar regel L4 juist uit.

Zoals in sectie 1 gemeten verklaart dit maar een beperkt deel van het verschil. Het blijft
een inhoudelijke keuze om aan Sama voor te leggen, maar niet de hoofdoorzaak.

---

## 3. Chirurgie, en hoe OK-uitstel bij haar een DLT wordt

Dit is het antwoord op de vraag die Koen in juli stelde over uitstel van de operatie.

**Er is geen kans op chirurgie.** In `Generate_DLT_Time_function.R`:

```r
Surgery_Time <- matrix(operation.Time, nrow = N.patient, ncol = Max.dose)
```

Iedere patiënt krijgt een operatie, standaard op `operation.Time`. Bij ons is dat een
Bernoulli met `p_surgery = 0.80`, en patiënten zonder operatie leveren géén subacute
informatie. Haar subacute model gaat dus over een andere populatie dan het onze.

**Uitstel wordt gemodelleerd als herstel na toxiciteit:**

```r
Surgery_Time[i, d] <- max(operation.Time, light_recovery_time)
```

waarbij `light_recovery_time = Time_light + Reco_light`, beide log-normaal. De OK schuift
dus op wanneer een toxiciteit vóór de geplande operatiedatum optreedt.

**En dan het belangrijkste:**

```r
if (light_recovery_time > time_para$Max.Surgery.Time) {
  acute.dlt[i, d]      <- 1
  acute.DLT_Time[i, d] <- time_para$Max.Surgery.Time
}
```

**Uitstel voorbij `Max.Surgery.Time` (16) telt bij haar als een acute DLT.** Dat pad loopt
via `Tox_light`, een niet-DLT toxiciteit met kans `non_dlt.Prop1 = 0.15` per patiënt, die
wij helemaal niet modelleren.

Hier zit een consequentie die goed doordacht moet worden: **de werkelijke acute
toxiciteitskans in haar simulatie is niet `P.early`.** Die is `P.early` plus de bijdrage
van dit uitstelpad. Daarmee is de "ware MTD" die uit `P.early` wordt afgeleid strikt
genomen niet de ware MTD van haar datagenererende proces. Dat raakt de interpretatie van
elk accuracy-getal aan beide kanten. Dit zouden we expliciet aan haar moeten voorleggen —
mogelijk is het bewust en klein, maar het moet benoemd.

---

## 4. Onze subacute prior op L0 was fout

In `run_titecrm_simulation.R` staan de skeletons hard gecodeerd, met de
`getprior`-aanroepen erboven uitgecommentarieerd:

```r
Prior.1 <- c(0.004, 0.021, 0.066, 0.150, 0.266)
Prior.2 <- c(0.010, 0.036, 0.084, 0.157, 0.250)
```

Zij gebruikt dus de exacte getallen uit de mail van 8 juli, niet de halfwidth-afleiding;
`halfwidth.def` wordt doorgegeven maar niet meer gebruikt. Haar subacute L0 is **0.010**,
precies zoals doorgegeven. Onze code had 0.012. **Onze waarde was de afwijkende, en is
gecorrigeerd.**

De impact is verwaarloosbaar — we starten op L2 en komen zelden op L0 — maar de rapporten
in `dlt_attribution_sensitivity/` zijn met 0.012 gegenereerd.

---

## 5. Sigma

Zij roept `dfcrm::titecrm()` aan zonder `scale`, dus met de pakket-default. Volgens de
CRAN-documentatie van `dfcrm` is dat **`scale = sqrt(1.34)` ≈ 1.158**, oftewel een normale
prior op β met gemiddelde 0 en variantie 1.34, bij hetzelfde empirische model
`F(d, β) = d^exp(β)` dat wij gebruiken. Wij hebben `sigma = 1.0`.

Het model is dus identiek; alleen de priorspreiding verschilt. Zoals gemeten in sectie 1 is
het effect klein, maar het is een van de weinige punten die je triviaal gelijk kunt zetten.

---

## 6. Accrual

Bij haar krijgt een heel cohort dezelfde instroomtijd:

```r
Final_DATA$acute.DLT_Time$Enter[next_start:next_end] <- next_entry_time
```

Drie patiënten tegelijk, en het volgende cohort `Wait.Time` later (in het voorbeeld 4).
Wij gebruiken een Poisson-proces met exponentiële tussenaankomsttijden, 0.75 per maand.

Twee gevolgen. De TITE-gewichten binnen een cohort zijn bij haar identiek en bij ons niet.
En het tempo verschilt: drie patiënten per `Wait.Time` tegenover gemiddeld één per vier
weken bij ons.

**Voorbehoud:** de tijdseenheid van `time_para` en `Wait.Time` staat nergens expliciet.
Uit `operation.Time = 6` en `dur.late.Time = 4` naast onze 42 en 30 dagen lijken het weken,
maar zolang dat niet bevestigd is kunnen we het tempoverschil niet kwantificeren. Dit
verklaart vermoedelijk wel het grootste deel van het duurverschil (onze 96 tegen haar 30).

---

## 7. Haar burn-in is een vaste run-in, geen escalatie

```r
patient <- 1
current.dose <- initial.dose      # verandert nergens
while (!CRM.Run) {
  composite_DLT <- (...) && patient >= 6
  ...
  if (!composite_DLT) { if (patient >= 6) break; patient <- patient + 1 }
  else { CRM.Run <- TRUE }
}
```

De dosis escaleert niet. Alle zes patiënten krijgen `initial.dose`. En omdat
`composite_DLT` pas vanaf patiënt 6 kan worden geactiveerd, loopt de run-in altijd door tot
zes — zelfs als patiënt 1 tot en met 5 een DLT heeft.

Bij ons is burn-in "escaleer één niveau per cohort tot de eerste waargenomen acute DLT",
zoals Koen het in juli beschreef. **De "burn-in aan/uit" reeksen aan beide kanten meten dus
niet hetzelfde** en zijn niet naast elkaar te leggen. Voorstel: hernoem er één in de
verslaglegging.

---

## 8. Twee waarschijnlijke defecten in haar code

**`real.data.early` en `real.data.late` bestaan niet.** In `run_titecrm_simulation.R`
worden die namen doorgegeven aan `run_full_titecrm_trial()`, terwijl de parameters
`real.dlt.early` en `real.dlt.late` heten. Het crasht niet, omdat R lui evalueert én omdat
die argumenten in de body van `run_full_titecrm_trial()` niet worden gebruikt. Zodra iemand
ze wel gebruikt, valt het om.

**De burn-in overschrijft de dosistoewijzing van de vaste geschiedenis.**
`run_burnin_phase()` schrijft `dose_level[patient] <- current.dose` voor patiënt 1 t/m 6,
ongeacht `real.dose`. Bij de MERGE-configuratie (`real.dose = 2`, `initial.dose = 2`)
vallen ze samen, dus het gaat toevallig goed. De *uitkomsten* van de historische patiënten
worden wel correct gebruikt, want die zitten in de potential-outcome matrix.

---

## 9. Overige bevestigde verschillen

| Onderwerp | Sama | Wij |
|---|---|---|
| Tijd tot DLT | `rlnorm()`, log-normaal | uniform over het venster |
| Tijd tot OK | `operation.Time`, uitgesteld bij herstel na toxiciteit | vaste constante van 42 dagen |
| Kans op OK | geen — iedereen krijgt chirurgie | Bernoulli, `p_surgery = 0.80` |
| OK-uitstel > 16 | **telt als acute DLT** | niet gemodelleerd |
| Niet-DLT toxiciteit | `non_dlt.Prop1 = 0.15`, voedt het uitstelpad | bestaat niet |
| Acuut venster | eindigt op de patiëntspecifieke OK-datum | vast op 56 dagen |
| Escalatie | maximaal +1 niveau | maximaal +1 niveau |
| De-escalatie | **onbeperkt**, mag meerdere niveaus in één stap | maximaal −1 (`max_step = 1`) |
| Dosisregel | hoogste dosis met puntschatting ≤ target | EWOC aan: hoogste toelaatbare; uit: argmin |
| Subacuut in dosiskeuze | begrenst altijd mee | alleen als EWOC aan staat |
| EWOC | Monte Carlo, 10.000 trekkingen; **vervangt** de ptox-regel | Gauss-Hermite; filtert, dan hoogste |
| Correlatie acuut/subacuut | Gaussische copula (`inde`/`neg`/`pos`) | niet gemodelleerd |
| Sample size 6+3 | **50**, met `target.patients.at.mtd = 12` | 30, geen equivalent |
| Ware MTD bij twee eindpunten | `min(acute MTD, late MTD)` | alleen acute MTD |
| Toxiciteitsmaat in output | proportie patiënten | aantal patiënten |

**Onze 6+3-vergelijking was niet eerlijk.** Haar voorbeeld draait het regelgebaseerde design
met 50 patiënten en een expansieconcept dat wij niet kennen; wij gaven onze 6+3 er 30. Onze
2.8% tegen haar 32.3% mag dus **niet** worden gelezen als weerlegging van haar cijfer.

**De toxiciteitsmaat verschilt maar is consistent.** Zij rapporteert een proportie, wij een
aantal. Haar 10.15% × 30 ≈ 3.0, en wij vonden ~3.0.

**De ware MTD-definitie komt overeen** voor onze vijf scenario's, omdat alle subacute kansen
onder 0.33 liggen en `min(early, late)` daar gelijk is aan `early`.

---

## 10. Wat ik zou voorleggen aan Sama

Op volgorde van gemeten of verwacht effect.

1. **De dosis-selectieregel.** Onze reconstructie laat zien dat "hoogste dosis met
   puntschatting ≤ target" bij deze skeleton L4 vanaf het begin uitsluit, omdat de prior
   daar al 0.266 is. Is dat bedoeld? Dit verklaart het grootste deel van het verschil en
   raakt de hele dosisallocatie, niet alleen de eindselectie.
2. **Telt uitstel van de operatie voorbij 16 als acute DLT?** Zo ja, dan is de werkelijke
   acute toxiciteit in de simulatie hoger dan `P.early`, en klopt de daaruit afgeleide
   "ware MTD" niet meer als referentiepunt.
3. **Moet het subacute eindpunt de dosiskeuze begrenzen?** Bij haar wel, bij ons alleen met
   EWOC aan. Dit is een ontwerpvraag, geen implementatiedetail.
4. **Geen kans op chirurgie.** Bewust, of moet dat 80% worden zoals in ons model?
5. **Mag de finale MTD één niveau boven de laatst behandelde dosis liggen**, en dus mogelijk
   nooit toegediend zijn?
6. **Is `scale = sqrt(1.34)` een keuze of de ongemerkte default?**
7. **De eerste fase: zes patiënten op dezelfde dosis, of escaleren tot de eerste DLT?**
8. **Accrual-structuur, tijdseenheid, en de 6+3-regels** die voor het amendement moeten
   gelden.
9. **De twee defecten** uit sectie 8.

---

## 11. Conclusie

Onze simulator en die van Sama zijn op dit moment **geen twee implementaties van hetzelfde
design**. Ze verschillen op de dosisregel, de rol van het subacute eindpunt, de
chirurgiekans, de definitie van een acute DLT, de accrualstructuur, de burn-in, de prior
en de 6+3-instellingen.

Daarom heeft het geen zin om nu op basis van accuracy-percentages vast te stellen wie
"gelijk" heeft. De volgorde zou moeten zijn: eerst met Sama vaststellen welke regels het
bedoelde MERGE-design representeren, dan beide implementaties daarop gelijkzetten, en pas
dan opnieuw vergelijken.

Wat dit wel oplevert: we weten nu waar het verschil vandaan komt, en we kunnen het
reproduceren. Dat is precies wat de vergelijking moest opleveren.

# Bevindingen: Sama's R-code naast onze simulator

Doorgenomen op 18-08-2026, op basis van de map `tite-crm-simulation-main` zoals gedeeld
via Google Drive. Dit document vult de rechterkolom in van `CODE_COMPARISON_CHECKLIST.md`.

Gelezen bestanden: `README.md`, `examples/example_run.R`, `R/run_titecrm_simulation.R`
(bevat `run_simulation()`), `R/run_full_titecrm_trial_M1.R`, `R/run_burnin_phase.R`,
`R/run_trial_phase.R`, `R/run_final_analysis.R`, `R/Generat_Time_function.R`.

Nog niet gelezen: `run_rule_based_trial.R`, `Generate_DLT_Time_function.R`,
`Correlate_Bernoulli_data_Copula.R`, `Generating correlated.R`, `plot_time_distributions.R`.

---

## De vier verschillen die er het meest toe doen

### 1. Haar finale MTD-selectie is NIET beperkt tot geprobeerde doses

In `run_final_analysis.R`:

```r
p11 <- max(which(Tite.CRM.1$ptox <= early.target))
```

Dit loopt over **alle** dosisniveaus, niet alleen over de niveaus waar daadwerkelijk
patiënten zijn behandeld. Bij ons staat `restrict_final_to_tried = True`.

Er zit wel een rem op, direct daarna:

```r
last_dose <- acute.DLT.1$dose_level[nrow(acute.DLT.1)]
p <- ifelse(last_dose >= p11, p11, min(last_dose + 1, Max.dose))
```

De finale MTD mag dus hooguit één niveau boven de laatst *toegediende* dosis liggen — maar
wel één niveau boven alles wat ooit geprobeerd is.

**Waarom dit ertoe doet:** in het Acute low scenario, waar de ware MTD het hoogste niveau
is, kan haar design L5 als MTD aanwijzen zonder daar ooit een patiënt te hebben behandeld,
zolang de trial op L4 eindigde. Onze code kan dat per definitie niet. Dit is vermoedelijk
de **grootste enkele verklaring** voor het verschil in MTD-accuracy bij lage ware
toxiciteit, en het is een echte ontwerpkeuze om te bespreken — niet zomaar een bug.

### 2. De prior-spreiding (sigma) verschilt

Zij roept `dfcrm::titecrm()` aan zonder het `scale`-argument, dus met de pakket-default.
Wij gebruiken `sigma = 1.0`. Dat zijn verschillende priors op dezelfde parameter, wat elke
posterior verschuift.

**Actie:** vraag haar de gebruikte `scale` expliciet te maken en spreek één waarde af.
Dit is een van de weinige punten die je één-op-één gelijk kunt trekken en dan opnieuw
vergelijken.

### 3. Onze subacute prior op L0 is verkeerd — die van haar klopt

In `run_titecrm_simulation.R` staan de skeletons hard gecodeerd, met de `getprior`-aanroepen
er uitgecommentarieerd boven:

```r
#Prior.1 <- dfcrm::getprior(halfwidth = halfwidth.def, ...)
#Prior.2 <- dfcrm::getprior(halfwidth = halfwidth.def, ...)
Prior.1 <- c(0.004, 0.021, 0.066, 0.150, 0.266)
Prior.2 <- c(0.010, 0.036, 0.084, 0.157, 0.250)
```

Twee dingen tegelijk opgelost:

- Zij gebruikt **wel** de exacte getallen uit de mail van 8 juli, niet de halfwidth-afleiding.
  Het `halfwidth.def = 0.10` argument wordt doorgegeven maar niet meer gebruikt.
- Haar subacute L0 is **0.010**, precies wat Koen doorgaf. Onze code heeft 0.012.
  **Onze waarde is de afwijkende — die moeten wij corrigeren.**

### 4. De accrual is fundamenteel anders gemodelleerd

Bij haar komt een heel cohort **tegelijk** binnen, met een vast interval:

```r
Final_DATA$acute.DLT_Time$Enter[next_start:next_end] <- next_entry_time
```

Drie patiënten delen dezelfde `Enter`-tijd, en het volgende cohort komt `Wait.Time` later.
Met `Wait.Time = 4` en `cohort = 3` is dat drie patiënten per vier tijdseenheden.

Wij gebruiken een Poisson-proces met exponentiële tussenaankomsttijden op 0.75 per maand,
oftewel gemiddeld één patiënt per vier weken.

**Twee gevolgen.** Ten eerste is haar instroom aanzienlijk sneller dan de 1.5 patiënten per
maand die Koen vroeg — afhankelijk van de tijdseenheid drie tot vier keer zo snel. Ten
tweede zijn de TITE-gewichten binnen een cohort bij haar identiek (zelfde instroommoment),
bij ons niet. Dat beïnvloedt elke beslissing.

**Actie:** vraag expliciet naar de tijdseenheid van `time_para` en `Wait.Time`. Uit
`operation.Time = 6` en `dur.late.Time = 4` naast onze 42 en 30 dagen lijken het weken,
maar dat moet bevestigd. Dit verklaart vermoedelijk het grootste deel van het
duurverschil (onze 96 tegen haar 30).

---

## Twee dingen die volgens mij fouten zijn

### A. `real.data.early` en `real.data.late` bestaan niet

In `run_titecrm_simulation.R` wordt aangeroepen:

```r
run_titecrm <- run_full_titecrm_trial(
  ...
  real.data       = real.data,
  real.data.early = real.data.early,   # bestaat niet
  real.data.late  = real.data.late,    # bestaat niet
  ...
)
```

De functieparameters heten `real.dlt.early` en `real.dlt.late`. De namen
`real.data.early` / `real.data.late` komen nergens anders voor.

Dit gaat nu niet stuk omdat R argumenten lui evalueert **en** omdat
`run_full_titecrm_trial()` deze drie argumenten in zijn body helemaal niet gebruikt — ze
worden doorgegeven en vervolgens genegeerd. Zodra iemand ze wel gebruikt, of `force()`
aanroept, valt het om met "object 'real.data.early' not found".

**Belangrijker dan de fout zelf:** het betekent dat de vaste patiëntgeschiedenis *niet*
via deze route de TITE-CRM-logica in gaat. Dat gebeurt wel indirect — zie punt B.

### B. De burn-in overschrijft de dosistoewijzing van de vaste geschiedenis

`run_burnin_phase()` begint bij `patient <- 1` en schrijft:

```r
Final_DATA$acute.dlt$dose_level[patient] <- current.dose
```

waarbij `current.dose <- initial.dose` en **nooit verandert**. De eerste zes patiënten
krijgen dus altijd `initial.dose` toegewezen, ongeacht wat er in `real.dose` stond.

Bij de MERGE-configuratie (`real.dose = c(2,2,2,2,2,2)` en `initial.dose = 2`) vallen die
samen, dus het gaat toevallig goed. Maar het is fragiel: bij elke andere combinatie wordt
de opgegeven geschiedenis stilzwijgend overschreven.

De *uitkomsten* van de historische patiënten worden wel correct gebruikt, want die zitten
in de potential-outcome matrix (`acute.dlt[i1, real.dose[i1]] <- real.dlt.early[i1]`).

---

## Haar burn-in is iets heel anders dan de onze

Dit verdient een eigen kopje, want de term dekt bij ons en bij haar niet dezelfde lading.

```r
patient <- 1
current.dose <- initial.dose      # verandert nooit
while (!CRM.Run) {
  composite_DLT <- (acute.dlt[patient, current.dose] == 1 ||
                    subacute.dlt[patient, current.dose] == 1) && patient >= 6
  ...
  if (!composite_DLT) {
    if (patient >= 6) break       # burn-in eindigt na patiënt 6
    patient <- patient + 1
  } else {
    CRM.Run <- TRUE
  }
}
```

- De dosis **escaleert niet** tijdens de burn-in. Alle zes patiënten krijgen `initial.dose`.
- `composite_DLT` vereist `patient >= 6`, dus de burn-in kan niet vroeg eindigen: hij loopt
  altijd exact zes patiënten.
- Er is geen optie om de burn-in over te slaan; hij zit altijd in het pad.

Bij ons is burn-in "escaleer één niveau per cohort tot de eerste waargenomen DLT" —
precies zoals Koen het in juli beschreef. **Dat is een wezenlijk ander mechanisme.**
Haar burn-in is feitelijk een vaste run-in van zes patiënten op de startdosis.

Praktisch gevolg: de "burn-in aan/uit" vergelijking uit haar eerdere slides vergelijkt iets
anders dan onze "burn-in aan/uit". Die twee reeksen zijn niet naast elkaar te leggen.

---

## Overige bevestigde verschillen

| Onderwerp | Sama | Wij |
|---|---|---|
| Tijd tot DLT | `rlnorm()`, log-normaal | Uniform over het venster |
| Tijd tot OK | log-normaal, met `Max.Surgery.Time = 16` als plafond | vaste constante van 42 dagen |
| Acuut venster | eindigt op de **patiëntspecifieke, random** OK-datum | vast op 56 dagen |
| Escalatie | maximaal +1 niveau | maximaal +1 niveau |
| De-escalatie | **onbeperkt** — mag in één stap meerdere niveaus zakken | maximaal −1 niveau (`max_step = 1`) |
| Dosisregel tijdens trial | `max(which(ptox <= target))` | EWOC aan: hoogste toelaatbare; EWOC uit: argmin,\|pm − target\| |
| Twee eindpunten combineren | `min(max(acuut toelaatbaar), max(subacuut toelaatbaar))` | gezamenlijk EWOC-filter op beide |
| EWOC-implementatie | Monte Carlo, 10.000 normale trekkingen op beta | Gauss-Hermite kwadratuur |
| EWOC-werking | **vervangt** de ptox-regel volledig | filtert, daarna hoogste toelaatbare |
| Correlatie acuut/subacuut | Gaussische copula, instelbaar (`inde`/`neg`/`pos`) | **niet gemodelleerd**, altijd onafhankelijk |
| Niet-DLT toxiciteit | `non_dlt.Prop1 = 0.15`, apart gemodelleerd, beïnvloedt de observatievensters | bestaat niet |
| Sample size 6+3 | **50** in haar voorbeeldscript | 30 |
| Sample size TITE-CRM | 30, inclusief de vaste geschiedenis | 30, inclusief de zes | 
| Ware MTD bij twee eindpunten | `min(acute MTD, late MTD)` | alleen acute MTD |
| Toxiciteitsmaat in output | *proportie* patiënten met DLT | *aantal* patiënten met DLT |

Twee opmerkingen bij die tabel.

**De 6+3-vergelijking was niet eerlijk.** Haar voorbeeld draait het regelgebaseerde design
met `sample_size = 50` en `target.patients.at.mtd = 12`; wij gaven onze 6+3 maar 30
patiënten en kenden dat tweede concept niet. Dat verklaart vermoedelijk een groot deel van
het gat tussen onze 2.8% en haar 32.3% correcte selectie in Acute low. Onze 6+3-cijfers
moeten dus niet als weerlegging van de hare worden gelezen.

**De toxiciteitsmaat verschilt maar is consistent.** Zij rapporteert
`mean(acute.DLT.1$response)`, een proportie; wij een gemiddeld aantal. Haar 10.15% × 30
patiënten ≈ 3.0, en wij vonden ~3.0. Die twee komen dus wél overeen zodra je omrekent.

**De ware MTD-definitie komt overeen** voor de vijf scenario's die we gebruiken, omdat alle
subacute kansen onder 0.33 liggen en `min(early, late)` daar dus gelijk is aan `early`.
Bij een scenario met een strengere subacute curve zouden ze uiteenlopen.

---

## Wat ik zou voorleggen aan Sama

Op volgorde van verwacht effect.

1. **Beperking tot geprobeerde doses bij de finale MTD.** Is het bewust dat een niet-geprobeerde
   dosis als MTD kan worden aangewezen? Dit is de belangrijkste inhoudelijke keuze en heeft
   waarschijnlijk het grootste effect op de accuracy-cijfers.
2. **Welke `scale` gebruikt ze in `dfcrm::titecrm()`?** Nu impliciet de default. Spreek één
   waarde af en draai beide codes opnieuw.
3. **Tijdseenheid van `time_para` en `Wait.Time`,** en het accrualtempo. Drie patiënten per
   `Wait.Time` is aanzienlijk sneller dan 1.5 per maand.
4. **De naamfout `real.data.early` / `real.data.late`,** en de vraag of de vaste
   geschiedenis daadwerkelijk werkt zoals bedoeld gegeven dat de burn-in de dosistoewijzing
   overschrijft.
5. **Onbeperkte de-escalatie** — bewust, of een omissie?
6. **De term "burn-in"** betekent bij ons iets anders dan bij haar. Voorstel: hernoem één van
   de twee in de verslaglegging, anders praten we langs elkaar heen.
7. **Wij corrigeren onze subacute prior op L0** van 0.012 naar 0.010.

Voor de vergelijking zelf: de generating file die klaarstaat in `trial_export/` is nu extra
bruikbaar, omdat we haar `Final_DATA`-structuur kennen en de mapping kunnen maken.

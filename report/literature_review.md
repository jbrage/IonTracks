# Literature review: modelling ion recombination in ionization chambers

*Prepared 2026-09-23 as input to the IonTracks refactor. Links point to the publisher page (DOI) or an open preprint. Summaries were checked against each paper's abstract. The "Relevance" lines are my own assessment for IonTracks.*

**Goal this review serves:** being able to calculate recombination in ion beams where tracks overlap in complex ways because of the beam's time structure (pulses, spills, spot scanning, RF micro-structure, UHDR/FLASH). The user provides the temporal structure, and the code returns *k*<sub>s</sub> and the time-resolved signal without the calculation blowing up.

---

## 0. Take-aways (read this first)

1. **No existing code covers the whole target space.** There are three families of models:
   - *Uniform-density 1D PDE solvers.* These handle arbitrary pulse shapes and durations, free electrons and space charge, but have no track structure: Gotz 2017, Kranzer 2021/2022, Paz-Martín 2022/2025, Fenwick & Kumar's transport code.
   - *Ion-track models.* These cover initial + general recombination, but only for "pulsed" or "continuous" beams, with little or no space charge: IonTracks 2016/2020, Kaiser 2012, Orts 2024.
   - *Analytic formulas.* These are valid only in limiting regimes: Jaffé, Boag (pulsed / continuous / free-electron), Fenwick & Kumar f<sub>exp</sub>.

   **Ion tracks + arbitrary user-defined time structure + (optionally) free electrons and space charge** is an open niche that a refactored IonTracks can fill.
2. **Standard corrections can fail exactly where the time structure is complex.**
   - Furutani 2022: a short-spot PBS field matched *neither* the continuous nor the pulsed Boag correction.
   - Gan 2026: misclassifying the time structure changed *k*<sub>s</sub> by up to 2.5 %.
   - Brualla-González 2016: when charge from successive pulses overlaps ("beyond the Boag limit"), the voltage dependence stops following Boag, which invalidates voltage-based extrapolation methods.
3. **Above roughly 10–100 mGy per pulse, free electrons and field distortion dominate.** This is shown by Gotz 2017, Kranzer 2021/2022, Paz-Martín 2022/2025, Fenwick & Kumar 2023/2024, Bancheri 2024 and Gérard Ortega 2026. A model that ignores them is not valid for UHDR pulsed beams. For quasi-continuous ion beams at clinical rates, the Orts 2024 authors report that both effects are negligible.
4. **Time-resolved current is the observable to validate against.** Paz-Martín 2022 reproduced oscilloscope traces of the instantaneous chamber current, and that is a stronger test than a single *k*<sub>s</sub>. The refactored code should output the electrode current *I*(*t*), not only *f*.
5. **Validation datasets exist.** The Paz-Martín 2022 dataset is on [Zenodo](https://zenodo.org/records/7673972), and PTB UHDR measurements are in Bourgouin 2023, Kranzer 2022 and Paz-Martín 2025.

---

## 1. Foundational theory

**Jaffé G (1913). *Zur Theorie der Ionisation in Kolonnen.* Annalen der Physik 347, 303–344.**
[doi:10.1002/andp.19133471205](https://onlinelibrary.wiley.com/doi/abs/10.1002/andp.19133471205)
Jaffé's column model is the original theory of *initial* (intra-track) recombination. A column of ion pairs recombines while it diffuses and is pulled apart by the field. For plane-parallel chambers it predicts a characteristic 1/*V* dependence of 1/*f*, and it depends on the angle between the track and the field.
*Relevance:* this is the analytic benchmark for single-track runs. It is implemented in `hadrons/functions.py::Jaffe_theory` and used as the test oracle.

**Boag JW, Hochhäuser E, Balk OA (1996). *The effect of free-electron collection on the recombination correction to ionization measurements of pulsed radiation.* Phys. Med. Biol. 41, 885–897.**
[doi:10.1088/0031-9155/41/5/005](https://iopscience.iop.org/article/10.1088/0031-9155/41/5/005)
The paper compares three models of charge collection in pulsed beams, each with a free-electron component (the part of the electrons collected before they attach to O₂). It gives the approximate formulas f′, f″ and f‴ that became the standard free-electron corrections.
*Relevance:* this is the analytic target for a 1D uniform solver with free electrons, and the baseline that Fenwick & Kumar improved on.

**Boutillon M (1998). *Volume recombination parameter in ionization chambers.* Phys. Med. Biol. 43, 2061–2072.**
[doi:10.1088/0031-9155/43/8/005](https://iopscience.iop.org/article/10.1088/0031-9155/43/8/005)
Boutillon measured the volume-recombination parameter *m*² under conditions where the basic continuous-beam theory strictly holds, using current ratios at two voltages extrapolated to zero current. The result was *m*² = 3.97×10¹⁴ s m⁻¹ C⁻¹ V² (1.7 %). The formalism separates initial from general recombination.
*Relevance:* the continuous-beam analytic benchmark. It is used by Christensen 2020 to separate the two recombination types that IonTracks computes together.

**Kanai T, Sudo M, Matsufuji N, Futami Y (1998). *Initial recombination in a parallel-plate ionization chamber exposed to heavy ions.* Phys. Med. Biol. 43(12).**
[doi:10.1088/0031-9155/43/12/012](https://iopscience.iop.org/article/10.1088/0031-9155/43/12/012)
The authors measured columnar recombination for C, Ne, Ar and Fe beams in air, CO₂, Ar and TE gas. 1/*f* is linear in 1/*E*, with a slope that grows strongly with LET, as Jaffé predicts.
*Relevance:* this is the source of the transport constants hard-coded in IonTracks (α = 1.6×10⁻⁶ cm³/s, *D* = 3.7×10⁻² cm²/s, µ = 1.65 cm²/Vs). It also provides high-LET validation data.

---

## 2. IonTracks and other ion-track recombination models

**Christensen JB, Tölli H, Bassler N (2016). *A general algorithm for calculation of recombination losses in ionization chambers exposed to ion beams.* Med. Phys. 43, 5484–5492.**
[doi:10.1118/1.4962483](https://doi.org/10.1118/1.4962483) · [arXiv:1610.06749](https://arxiv.org/abs/1610.06749)
This paper introduces IonTracks. Track-structure theory places the charge carriers, and coupled drift–diffusion–recombination PDEs are solved on a 3D grid. The paper states the aim of predicting combined initial + general recombination for any ion beam, with arbitrary particle-energy spectrum and temporal structure. Validation is against Jaffé (initial) and Boag (general).
*Relevance:* this is the code under review. The "arbitrary temporal structure" claim is the goal of the refactor, but the current implementation only realises *continuous* (Poisson-in-time) and *single-track* cases.

**Christensen JB, Almhagen E, Stolarczyk L, et al. (2020). *Mapping initial and general recombination in scanning proton pencil beams.* Phys. Med. Biol. 65, 115003.**
[doi:10.1088/1361-6560/ab8579](https://iopscience.iop.org/article/10.1088/1361-6560/ab8579)
The study measured *k*<sub>s</sub> in PBS at DCPT and Skandion (70–244 MeV, < 400 Gy/min in air). Boutillon's formalism separates the initial part (compared to Jaffé) from the general part (compared to IonTracks). The two-voltage method agrees with Boutillon and IonTracks above about 100 Gy/min but tracks Jaffé at lower rates, and the three-voltage (3VL) method agrees broadly.
*Relevance:* this is the existing experimental validation of the continuous-beam solver. The PBS "local dose rate" concept here is the seed of the spatiotemporal beam description proposed in the refactoring plan.

**Kaiser F-J, et al. (2012). *Initial recombination in the track of heavy charged particles: numerical solution for air filled ionization chambers.* Acta Oncologica 51, 368–375.**
[doi:10.3109/0284186X.2011.626452](https://doi.org/10.3109/0284186X.2011.626452)
The authors solve Jaffé's PDE numerically with realistic (non-Gaussian, amorphous track model) radial distributions around carbon tracks, using the Gascoigne 3D finite-element library. They show how the assumed radial dose distribution changes the predicted initial recombination.
*Relevance:* this justifies offering several radial models (Gauss, Geiss, …). It also warns that sharply peaked cores need finer grids, which is one source of the resolution blow-up.

**Orts M, Rossomme S, Souris K, Sterpin E (2024). *Monte Carlo simulation of initial and volume ion recombination correction factor in plane parallel ionization chambers exposed to ion beams.* Phys. Med. Biol. 69(16).**
[doi:10.1088/1361-6560/ad6730](https://iopscience.iop.org/article/10.1088/1361-6560/ad6730)
This is a particle-based Monte Carlo alternative to IonTracks. Individual ions drift and diffuse (MSD = 2*D*Δ*t*), and recombination is sampled stochastically inside a capture radius. Gaussian track structure is used, the z-component of space-charge screening is included, and free electrons follow Boag models I–III. Time structure is "pulsed" (tracks spread uniformly over the pulse) or "continuous". Agreement with Jaffé and Boag is ≤ 0.05 %, validated against p/He/C data in a PPC40. It is only validated at conventional dose rates, and only about 0.08 mm of chamber radius is simulated.
*Relevance:* this is the closest competitor and a useful cross-check. It has the same limitation as IonTracks: no general time structure. Its runtimes (about 2–3 minutes per case) set the bar for performance.

**Geiß OB, Krämer M, Kraft G (1998). *Efficiency of thermoluminescent detectors to heavy charged particles.* Nucl. Instrum. Meth. B 142, 592–598.**
[doi:10.1016/S0168-583X(98)00325-5](https://doi.org/10.1016/S0168-583X(98)00325-5)
This is the source of the Geiss radial dose distribution: a constant core of radius *a*₀ plus a 1/*r*² fall-off out to *r*<sub>max</sub> ∝ *E*<sup>1.5</sup>. IonTracks implements it in `geiss_utils.py`.
*Relevance:* an alternative track model. Because of its very high core density it is the main stress test for spatial resolution and for the explicit recombination step.

**Greilich S, Grzanka L, Bassler N, Andersen CE, Jäkel O (2010). *Amorphous track models: a numerical comparison study.* Radiation Measurements 45, 1406–1409.**
[DTU record](https://orbit.dtu.dk/en/publications/amorphous-track-models-a-numerical-comparison-study)
The paper introduces **libamtrack**, an open library of amorphous track models, radial dose distributions and stopping-power tables. It shows that the choice of sub-model changes predictions significantly.
*Relevance:* IonTracks optionally imports `pyamtrack` (`use_beta`). A refactor could source radial dose distributions and LET from libamtrack instead of the ad-hoc tables in `data_LET/`.

---

## 3. Ion-beam measurements, protocols and PBS / synchrotron / synchrocyclotron time structure

**Palmans H, Thomas R, Kacperek A (2006). *Ion recombination correction in the Clatterbridge Centre of Oncology clinical proton beam.* Phys. Med. Biol. 51, 903–917.**
[doi:10.1088/0031-9155/51/4/010](https://doi.org/10.1088/0031-9155/51/4/010)
The authors separated initial and volume recombination in a cyclotron proton beam (about 26 Gy/min) by varying dose rate and voltage. The volume-recombination parameters were consistent with photon beams, and a general recombination equation reproduced the data to 0.1 %.
*Relevance:* a quasi-continuous proton benchmark with separated components.

**Liszka M, et al. (2018). *Ion recombination and polarity correction factors for a plane-parallel ionization chamber in a proton scanning beam.* Med. Phys. 45, 391–401.**
[doi:10.1002/mp.12668](https://doi.org/10.1002/mp.12668)
Markus chamber measurements in Proteus-235 PBS (30–230 MeV). *k*<sub>s</sub> depends on energy and dose rate and reaches about 0.5 % (TVM) at 13.4 Gy/s. Volume recombination cannot be neglected in PBS, where local dose rates are orders of magnitude above conventional MV.
*Relevance:* this is PBS data where the local instantaneous dose rate, not the average, drives recombination. It motivates a spatiotemporal beam input.

**Rossomme S, et al. (2017). *Ion recombination correction factor in scanned light-ion beams for absolute dose measurement using plane-parallel ionisation chambers.* Phys. Med. Biol. 62, 5365–5382.**
[doi:10.1088/1361-6560/aa730f](https://iopscience.iop.org/article/10.1088/1361-6560/aa730f)
The study measured *k*<sub>s</sub> for PPC40 chambers in scanned p, He, C and O beams and found excellent agreement with Jaffé (initial) combined with Boag (volume). It is also the source of the LET→Gaussian track radius *b* relation used by `calc_b_cm`.
*Relevance:* the multi-ion validation set, and the origin of the *b*(LET) parametrisation.

**Rossomme S, Delor A, Lorentini S, et al. (2020). *Three-voltage linear method to determine ion recombination in proton and light-ion beams.* Phys. Med. Biol. 65, 045015.**
[doi:10.1088/1361-6560/ab3779](https://doi.org/10.1088/1361-6560/ab3779)
The paper defines the 3VL method: fit the saturation current from three voltages in the linear region. It checks that *k*<sub>s</sub> is linear in 1/*V*² (continuous) or 1/*V* (pulsed) for Farmer chambers above 100 V.
*Relevance:* the code could emulate these experimental procedures by running several voltages, so users can compare "measured-style" *k*<sub>s</sub> to the true simulated *k*<sub>s</sub>.

**Tansho R, Furukawa T, Hara Y, et al. (2016). *Experimental verification of gain drop due to general ion recombination for a carbon-ion pencil beam.* Med. Phys. 43, 635–642.**
[doi:10.1118/1.4939225](https://doi.org/10.1118/1.4939225)
Concentric-electrode and large-area parallel-plate measurements show that the *non-uniform* lateral charge density of a carbon pencil beam changes general recombination. The effect is small at conventional intensity but significant for small spots and high rates.
*Relevance:* the lateral beam profile matters, so the refactor should accept a spatial beam profile as well as a time structure.

***Effect of general ion recombination on dose measurement for high dose rate carbon-ion scanning beam* (2020). Nucl. Instrum. Meth. B.**
[ScienceDirect S0168583X20300586](https://www.sciencedirect.com/science/article/abs/pii/S0168583X20300586)
The study used 290 MeV/u carbon spot scanning at up to 4×10⁹ particles/s at 12 and 200 keV/µm. General recombination cannot be ignored at high intensity, particularly for the dose monitor chamber.
*Relevance:* synchrotron-based, high-LET, high-rate data where the initial × general cross term matters. IonTracks' 3D model is uniquely suited to this.

**Hamatani N, Yagi M, Shimizu S, et al. (2024). *Investigation of ionization chamber characteristics for ultrahigh-dose-rate scanned carbon-ion beams.* In Vivo 38, 2220–2227.**
[doi:10.21873/invivo.13686](https://doi.org/10.21873/invivo.13686) · see also Yagi M et al. (2024) *Sci. Rep.* 14, 11574, [doi:10.1038/s41598-024-62148-2](https://doi.org/10.1038/s41598-024-62148-2) (dedicated UHDR carbon dose monitor)
Four chamber types were tested in a 208 MeV/u carbon beam at 110 Gy/s average. The Advanced Markus was found most suitable, and the others saturated at specified voltages.
*Relevance:* UHDR high-LET data, the regime where track overlap is strongest.

**Furutani KM, Remmes NN, Kruse JJ, et al. (2022). *Ion chamber collection efficiencies for proton spot scanning calibration.* arXiv:2211.12300.**
[arXiv:2211.12300](https://arxiv.org/abs/2211.12300)
Farmer chamber measurements in a PBS SOBP at 10–500 V, extrapolated to infinite voltage. The standard calibration field follows the *continuous*-beam model, and using the pulsed model instead gives a 0.5 % calibration error. **A field with much shorter spot durations agreed with neither the continuous nor the pulsed model.**
*Relevance:* this is direct experimental evidence for the regime the refactor targets: spot dwell comparable to the ion collection time. It would make an excellent validation case.

**Gan JK, Lew KS, Chua CGA, et al. (2026). *Ion recombination correction in reference dosimetry for pencil beam scanned proton beams.* Med. Phys. 53, e70466.**
[doi:10.1002/mp.70466](https://doi.org/10.1002/mp.70466)
The authors evaluated TRS-398 Rev.1 (2024) recombination methods on a synchrotron PBS system (70–229 MeV, 6–200 MU; Farmer and Advanced Markus, 20–400 V). TVM agrees with Jaffé extrapolation within about 1 % at low energy and low MU but deviates by up to 6.25 % (Farmer) at high energy and high MU. **Misclassifying the time structure (pulsed vs continuous) gave deviations up to 2.49 %.** Charge multiplication above 150 V in the Advanced Markus was modelled with exponential fitting.
*Relevance:* the protocol leaves the time structure ambiguous for synchrotron PBS. A code that takes the actual time structure as input resolves that ambiguity.

**Leite AMM, Cavallone M, Ronga MG, et al. (2023). *Ion recombination correction factors and detector comparison in a very-high dose rate proton scanning beam.* Physica Medica 106, 102518.**
[doi:10.1016/j.ejmp.2022.102518](https://doi.org/10.1016/j.ejmp.2022.102518)
Measurements with Advanced Markus, PPC05, Razor Nano and CC01 in Orsay cyclotron PBS at 4–385 Gy/s mean (instantaneous up to 1000 Gy/s). *k*<sub>s</sub> was below 2 % for most chambers, using Niatel, TVM and Boag-continuous. The recombination loss is not negligible.
*Relevance:* VHDR quasi-continuous proton data where mean and instantaneous dose rates differ. Time-structure input is required to model it.

**Lee E, Lourenço AM, Speth J, et al. (2022). *Ultrahigh dose rate pencil beam scanning proton dosimetry using ion chambers and a calorimeter in support of first in-human FLASH clinical trial.* Med. Phys. 49, 6171–6182.**
[doi:10.1002/mp.15844](https://doi.org/10.1002/mp.15844)
Plane-parallel chambers were compared to a primary-standard calorimeter at 5–60 Gy/s PBS. The Advanced Markus/calorimeter ratio was 1.002 ± 0.007, and the recombination corrections from different methods agreed.
*Relevance:* calorimeter-referenced UHDR proton benchmark.

**Lourenço A, Subiel A, Lee N, et al. (2023). *Absolute dosimetry for FLASH proton pencil beam scanning radiotherapy.* Sci. Rep. 13, 2054.**
[doi:10.1038/s41598-023-28192-0](https://doi.org/10.1038/s41598-023-28192-0)
The paper establishes calorimeter-based absolute dosimetry for FLASH PBS with 0.9 % uncertainty, and derives the correction factors needed.
*Relevance:* reference data and uncertainty budget for UHDR PBS.

**Setianegara J, Wang A, Gerard N, et al. (2025). *Characterization of commercial detectors for absolute proton UHDR dosimetry on a compact clinical proton synchrocyclotron.* Med. Phys. 52, e17847.**
[doi:10.1002/mp.17847](https://doi.org/10.1002/mp.17847)
IBA ProteusONE (S2C2) synchrocyclotron, 228 MeV UHDR PBS, with a *pulsed* time structure: µA instantaneous currents and 129.5 nA mean at isocentre. *P*<sub>ion</sub> was 1.0097 (PPC05), 1.0214 (Razor) and 1.0294 (Razor Nano). TRS-398 and Niatel agree for suitable *V*<sub>H</sub>/*V*<sub>L</sub>.
*Relevance:* an example of a pulsed ion beam: µs pulses at ~kHz carrying many tracks per pulse. This is the "overlapping tracks from pulse structure" case in the goal statement.

**Folkerts MM, Abel E, Busold S, et al. (2020). *A framework for defining FLASH dose rate for pencil beam scanning.* Med. Phys. 47, 6396–6404.** [doi:10.1002/mp.14456](https://doi.org/10.1002/mp.14456)
**van de Water S, Safai S, Schippers JM, Weber DC, Lomax AJ (2019). *Towards FLASH proton therapy: the impact of treatment planning and machine characteristics on achievable dose rates.* Acta Oncol. 58, 1463–1469.** [doi:10.1080/0284186X.2019.1627416](https://doi.org/10.1080/0284186X.2019.1627416)
Both papers formalise the *spatiotemporal* delivery of PBS. Folkerts defines a per-voxel "effective irradiation time" dose rate. van de Water (PSI) defines a dose-averaged dose rate over spots and shows how scanning speed, spot dwell and dead times shape local dose-rate histories.
*Relevance:* this is exactly the input model a user would hand to IonTracks: a list of spots (time, position, weight, σ) from which the local fluence rate Φ̇(*x*, *y*, *t*) at the chamber follows.

**IAEA TRS-398 Rev. 1 (2024).** The revised code of practice recommends the two-voltage method (Boag & Currant 1980) for *k*<sub>s</sub>, with separate pulsed and continuous forms, and leaves the choice to the user. The Gan 2026 and Furutani 2022 results show where that choice is ambiguous. [IAEA/ICTP training material (2025)](https://indico.ictp.it/event/10876/session/0/contribution/4/material/slides/0.pdf)

---

## 4. Pulsed and UHDR beams: free electrons, space charge and pulse duration (mostly electrons)

**Laitano RF, Guerra AS, Pimpinella M, et al. (2006). *Charge collection efficiency in ionization chambers exposed to electron beams with high dose per pulse.* Phys. Med. Biol. 51, 6419–6436.**
[doi:10.1088/0031-9155/51/24/009](https://iopscience.iop.org/article/10.1088/0031-9155/51/24/009)
Boag two-voltage analysis for plane-parallel chambers in low and high dose-per-pulse (DPP) electron beams, explicitly accounting for free electrons.
*Relevance:* early high-DPP data including free electrons.

**Petersson K, Jaccard M, Germond J-F, et al. (2017). *High dose-per-pulse electron beam dosimetry — a model to correct for the ion recombination in the Advanced Markus ionization chamber.* Med. Phys. 44, 1157–1167.**
[doi:10.1002/mp.12111](https://doi.org/10.1002/mp.12111)
An empirical logistic *k*<sub>s</sub>(DPP) model for the Advanced Markus over 10⁻⁴–10 Gy per pulse. Recombination depends on DPP, not on mean dose rate.
*Relevance:* a pulse-by-pulse independence benchmark. With pulse period ≫ collection time, only the DPP matters, and the refactor's "episode" decomposition should reproduce this automatically.

**Gotz M, Karsch L, Pawelke J (2017). *A new model for volume recombination in plane-parallel chambers in pulsed fields of high dose-per-pulse.* Phys. Med. Biol. 62, 8634–8654.**
[doi:10.1088/1361-6560/aa8985](https://iopscience.iop.org/article/10.1088/1361-6560/aa8985)
The authors solve a **1D transport model** (positive and negative ions plus free electrons) with self-consistent field distortion. They measured an Advanced Markus with 4 µs pulses up to about 1 Gy/pulse and with **variable pulse durations up to 308 µs**. The model reproduces the data over the full DPP and voltage range, including **arbitrary pulse durations and repetition rates**, which existing theory could not.
*Relevance:* this is the reference design for the proposed 1D uniform solver and its time-structure handling. It shows that arbitrary pulse durations are tractable in 1D.

**McManus M, Romano F, Lee ND, et al. (2020). *The challenge of ionisation chamber dosimetry in ultra-short pulsed high dose-rate very high energy electron beams.* Sci. Rep. 10, 9089.**
[doi:10.1038/s41598-020-65819-y](https://www.nature.com/articles/s41598-020-65819-y)
A graphite calorimeter against a plane-parallel chamber in a 200 MeV electron beam at 0.03–5.26 Gy/pulse. Collection efficiency ranged from 97 % down to 4 %. The recombination models available at the time were compared and found wanting.
*Relevance:* extreme-DPP, ultra-short-pulse benchmark.

**Kranzer R, Poppinga D, Weidner J, et al. (2021). *Ion collection efficiency of ionization chambers in ultra-high dose-per-pulse electron beams.* Med. Phys. 48, 819–830.**
[doi:10.1002/mp.14620](https://doi.org/10.1002/mp.14620)
Advanced Markus and custom 0.5/1/2 mm end-window chambers at 24 MeV and 0.01–3 Gy/pulse. The authors extend the Gotz model to time- and position-resolved field distortion. Jaffé plots and TVM *underestimate* efficiency at 3–130 mGy and *overestimate* it above 1 Gy.
*Relevance:* space-charge implementation details, and a warning that fitted-voltage methods are biased at high DPP.

**Kranzer R, Schüller A, Gómez Rodríguez F, et al. (2022). *Charge collection efficiency, underlying recombination mechanisms, and the role of electrode distance of vented ionization chambers under ultra-high dose-per-pulse conditions.* Physica Medica 104, 10–17.**
[doi:10.1016/j.ejmp.2022.10.021](https://doi.org/10.1016/j.ejmp.2022.10.021)
Chambers with three electrode spacings measured up to 5.4 Gy/pulse against alanine and simulated with a PDE model. Losses stem from a **temporal and spatial charge-carrier imbalance** and the resulting field distortion. Gaps down to 0.25 mm give CCE > 99 %.
*Relevance:* this explains *why* 1D uniform models need the Poisson equation at UHDR. The same mechanism appears in ion beams when many tracks arrive within one pulse.

**Paz-Martín J, Schüller A, Bourgouin A, et al. (2022). *Numerical modeling of air-vented parallel plate ionization chambers for ultra-high dose rate applications.* Physica Medica 103, 147–156.**
[doi:10.1016/j.ejmp.2022.10.006](https://doi.org/10.1016/j.ejmp.2022.10.006) · data: [Zenodo 7673972](https://zenodo.org/records/7673972)
The authors solve transport equations numerically for Roos chambers at PTB (4 µs pulses, up to about 1 Gy/pulse) and **measured the instantaneous current during and after the pulse**. The free-electron fraction varies with DPP, which Boag cannot explain. The effective free-electron fraction agreed within 5 %. The paper also studies the **effect of pulse structure on CCE**.
*Relevance:* the best available validation target for time-resolved output, with public data.

**Di Martino F, et al. (2022). *A new calculation method for the free electron fraction of an ionization chamber in the ultra-high-dose-per-pulse regimen.* Physica Medica 103.**
[ScienceDirect S1120179722020919](https://www.sciencedirect.com/science/article/abs/pii/S1120179722020919)
The method derives the free-electron fraction from charge-vs-voltage measurements alone, and gives saturation factors for the Advanced Markus up to 0.5 Gy/pulse.
*Relevance:* this provides the free-electron-fraction parameter that the solver needs as input or output.

**Fenwick JD, Kumar S (2023). *Collection efficiencies of ionization chambers in pulsed radiation beams: an exact solution of an ion recombination model including free electron effects.* Phys. Med. Biol. 68(1).**
[doi:10.1088/1361-6560/aca74e](https://iopscience.iop.org/article/10.1088/1361-6560/aca74e)
The authors derive an exact solution (f<sub>exp</sub>) of the Boag 1996 model, where the negative-ion density varies exponentially from electron attachment. It agrees with an ion-transport code to **5 decimal places**, while Boag's approximations differ by up to 6.1 % at 1 Gy/pulse and 2 mm. It excludes field distortion, which becomes substantial at ≥ 100 mGy/pulse.
*Relevance:* **an exact analytic oracle for unit-testing the 1D solver with free electrons** to 1e-5 precision.

**Fenwick JD, Kumar S, Pardo-Montero J, et al. (2024). *Collection efficiencies of cylindrical and plane parallel ionization chambers: analytical and numerical results and implications for experimentally determined correction factors.* Phys. Med. Biol. 69(15).**
[doi:10.1088/1361-6560/ad63ed](https://iopscience.iop.org/article/10.1088/1361-6560/ad63ed)
The paper derives f<sub>Gauss</sub> for cylindrical chambers and an equivalence condition between parallel-plate and cylindrical geometry. Charge screening changes efficiencies by ≤ 0.03 %, ≤ 1.1 % and ≤ 21.3 % at 1, 10 and 100 mGy/pulse.
*Relevance:* quantifies when space charge can be ignored (below about 1–10 mGy/pulse), which is the threshold a planner can use to switch Poisson off. The cylindrical formula also opens a path to thimble chambers.

**Bourgouin A, Paz-Martín J, Gedik YC, et al. (2023). *Charge collection efficiency of commercially available parallel-plate ionisation chambers in ultra-high dose-per-pulse electron beams.* Phys. Med. Biol. 68(23).**
[doi:10.1088/1361-6560/ad0a58](https://iopscience.iop.org/article/10.1088/1361-6560/ad0a58)
22 chambers of 6 models at PTB, 0.1–6.4 Gy/pulse (2.5 µs). Intra-model variation averaged 4 % (up to 20 %) and inter-model variation about 10 %, with large polarity effects (0.914–1.201).
*Relevance:* manufacturing tolerance (gap *d*) dominates at UHDR, so outputs should include sensitivity to *d* and *V*.

**Bancheri J, Seuntjens J (2024). *A semi-analytical procedure to determine the ion recombination correction factor in high dose-per-pulse beams.* Med. Phys. 51, 4458–4471.**
[doi:10.1002/mp.17005](https://doi.org/10.1002/mp.17005)
A homotopy-perturbation solution of the full PDE system with space charge and field-dependent electron velocity and attachment. It predicts a 1/*V*² dependence of *k*<sub>s</sub> caused by free electrons and fits data up to 1 Gy/pulse (1 mm gap) or 3 Gy/pulse (0.5 mm gap).
*Relevance:* this supplies an extrapolation law that the code could emit for users.

**Paz-Martín J, Schüller A, Bourgouin A, et al. (2025). *Evaluation of the two-voltage method for parallel-plate ionization chambers irradiated with pulsed beams.* Med. Phys. 52, 4894–4909.**
[doi:10.1002/mp.17814](https://doi.org/10.1002/mp.17814)
Advanced Markus and PPC05 chambers at PTB (50–400 V, pulse durations 0.5–2.9 µs). The Boag models disagree with each other and with the numerical model even below 30 mGy/pulse. Fenwick–Kumar matches the numerical model at low DPP. The numerical model matches experiment within 0.5–0.7 %. **Classical TVM systematically overestimates *k*<sub>sat</sub>**, even at low DPP. The paper gives polynomial parametrisations per pulse duration.
*Relevance:* strong evidence that a transport model with the real pulse duration is needed. It also shows the value of emitting *k*<sub>s</sub> vs voltage-ratio tables.

**Paz-Martín J, Schüller A, Apel M, et al. (2025/2026). *Charge collection efficiency of thimble ionization chambers exposed to ultra-high dose per pulse.* arXiv:2512.14525.**
[arXiv:2512.14525](https://arxiv.org/abs/2512.14525)
A new finite-element code for 1D and 2D geometries. Thimble chambers show up to 7 % polarity-dependent differences, and the model agrees within 1.4–1.6 %.
*Relevance:* 2D/cylindrical extension, relevant if IonTracks ever moves beyond parallel plates.

**Flores-Mancera MA, Schüller A, Paz-Martín J, et al. (2026). *Atmospheric pressure influence on the charge collection efficiency of air-vented ionization chambers in ultra-high dose per pulse electron beams for FLASH radiotherapy.* Phys. Med. Biol. 71 (8).**
[doi:10.1088/1361-6560/ae556b](https://iopscience.iop.org/article/10.1088/1361-6560/ae556b)
CCE falls by about 6–8 % per 100 hPa beyond the standard air-density correction. A squared-pressure scaling rule works, and the numerical model predicts the residuals within 3 %.
*Relevance:* gas parameters (µ, *D*, α, attachment) must scale with *p* and *T*. Today they are hard-coded constants in IonTracks.

**Gérard Ortega P, De Lentdecker G, Flandroy Q, Nys J (2026). *Charge collection efficiency in air-vented plane-parallel ionisation chambers at ultra-high dose rates: a self-consistent Garfield++ Monte Carlo model including space-charge effects and ion recombination.* arXiv:2606.24384.**
[arXiv:2606.24384](https://arxiv.org/abs/2606.24384)
Garfield++ extended with ion–ion recombination and self-consistent space-charge fields. It agrees with Fenwick–Kumar, Kranzer and Paz-Martín. Space charge can raise the local field more than 4× or cancel it. The CCE loss is driven mainly by a falling free-electron fraction through field-dependent attachment.
*Relevance:* the most recent independent model and a cross-check target. It shows that field-dependent attachment matters.

**Subiel A, Bourgouin A, Kranzer R, et al. (2024). *Metrology for advanced radiotherapy using particle beams with ultra-high dose rates.* Phys. Med. Biol. 69(14).**
[doi:10.1088/1361-6560/ad539d](https://iopscience.iop.org/article/10.1088/1361-6560/ad539d)
A review of UHDR/FLASH dosimetry standards, primary and secondary standards, and open needs.
*Relevance:* context for which outputs a metrology user would want.

---

## 5. Overlapping pulses and arbitrary time structure (beyond the Boag limit)

**Brualla-González L, Aguiar P, González-Castaño DM, et al. (2016). *Recombination in liquid-filled ionization chambers beyond the Boag limit.* Med. Phys. 43, 4142.**
[doi:10.1118/1.4953452](https://doi.org/10.1118/1.4953452)
When the pulse period is shorter than the collection time, charge from successive pulses overlaps. Overlap increases recombination. The DPP dependence keeps its form, but the **voltage dependence no longer follows Boag**, because changing *V* changes the collection time and therefore the overlap. Voltage-based corrections (three-voltage, modified TVM) fail in this regime, while the two-dose-rate method still works at a fixed repetition frequency. The work combines experiment and numerical simulation (liquid chambers).
*Relevance:* this is the clearest statement of the "complex overlap from time structure" physics. The same logic applies to air chambers when the pulse period is below *t*<sub>c</sub> (for example high-rep-rate beams, PBS spot sequences, RF micro-structure in dense bunches). It shows that correct handling requires simulating **several consecutive pulses until a periodic steady state is reached**.

**Related liquid-chamber numerical work:** Andersson J, Kaiser F-J, Gómez F, et al. (2012) *A comparison of different experimental methods for general recombination correction for liquid ionization chambers*, Phys. Med. Biol. 57, 7161, [doi:10.1088/0031-9155/57/21/7161](https://iopscience.iop.org/article/10.1088/0031-9155/57/21/7161). Also Gotz M et al. (2019) *Correction for volume recombination in liquid ionization chambers at high dose-per-pulse*, Med. Phys., [doi:10.1002/mp.13600](https://doi.org/10.1002/mp.13600). Both use the same drift–diffusion–recombination framework with multiple pulses. They show that the numerical strategies carry over, and that pulse overlap is routine in liquids, where mobilities are low.

**Richter C, et al. (2011). *A dosimetric system for quantitative cell irradiation experiments with laser-accelerated protons.* Phys. Med. Biol. 56, 1529.**
[doi:10.1088/0031-9155/56/6/002](https://iopscience.iop.org/article/10.1088/0031-9155/56/6/002)
Laser-driven protons arrive in nanosecond bursts. A transmission chamber cross-calibrated to a Faraday cup needs a saturation correction that becomes non-negligible above about 10⁴ Gy/s.
*Relevance:* the extreme case of "all tracks at once". The ion-track overlap is total within a pulse, which is the limit where a 1D uniform model with free electrons should take over from the 3D track model.

---

## 6. Numerical methods

**Dehghan M (2004). *Numerical solution of the three-dimensional advection–diffusion equation.* Applied Mathematics and Computation 150, 5–19.**
[ScienceDirect S0096300303001930](https://www.sciencedirect.com/science/article/abs/pii/S0096300303001930)
The paper compares explicit and implicit two-level finite-difference schemes for 3D constant-coefficient advection–diffusion, including the Lax–Wendroff-type explicit scheme and von Neumann criteria.
*Relevance:* the source of IonTracks' current stencil and stability test (`2(sx+sy+sz)+cx²+cy²+cz² ≤ 1`, `cx²cy²cz² ≤ 8 sx sy sz`). It does **not** address positivity, reaction terms or sharp fronts, all of which matter here (see the code review).

**Hundsdorfer W, Verwer JG (2003). *Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations.* Springer Series in Computational Mathematics.**
[doi:10.1007/978-3-662-09017-6](https://link.springer.com/book/10.1007/978-3-662-09017-6)
The standard reference for operator splitting (Strang), positivity-preserving limited advection schemes, IMEX and implicit treatment of stiff reaction terms, and splitting error analysis.
*Relevance:* the proposed solver (split drift / diffusion / exact recombination / source) follows this book's recipes. Chapter IV (splitting) and the chapters on limiters and positivity are the relevant ones.

---

## 7. Summary table: which physics matters where

| Regime (air, parallel plate) | Initial (track) rec. | General rec. | Free electrons | Space charge | Model that suffices |
|---|---|---|---|---|---|
| Continuous proton, clinical rate | small (*k*<sub>s</sub>−1 ~10⁻⁴) | dominant | no | no | Boag continuous + Jaffé, or IonTracks continuous |
| Continuous carbon / high LET | significant | significant, **cross term** | no | no | **IonTracks 3D** (Tansho 2020 regime) |
| PBS, spot dwell ≫ *t*<sub>c</sub> | as above | quasi-static per spot | no | no | per-spot steady state (adiabatic) |
| PBS, spot dwell ~ *t*<sub>c</sub> (Furutani short-spot) | as above | **transient** | no | no | **time-resolved IonTracks** |
| Synchrocyclotron (µs pulses, ~ms period) | yes | pulsed, many tracks/pulse | at high DPP | at high DPP | 3D tracks per pulse → uniform limit at high fluence/pulse |
| Linac, conventional DPP | n/a | pulsed (Boag) | small | no | Boag pulsed / 1D |
| Electron UHDR (≥ 0.1 Gy/pulse) | n/a | dominant | **yes** | **yes** | 1D PDE with Poisson + attachment (Gotz/Kranzer/Paz-Martín) |
| Pulse period < *t*<sub>c</sub> (overlap) | — | **pulse-overlap** | depends | depends | multi-pulse simulation to periodic steady state (Brualla-González) |
| Laser-driven ions (ns bunch, huge DPP) | total overlap | extreme | yes | yes | 1D uniform + free e⁻ + Poisson |

*t*<sub>c</sub> is the ion collection time, *d*/(µ<sub>+</sub>*E*): about 25–150 µs for typical parallel-plate chambers.

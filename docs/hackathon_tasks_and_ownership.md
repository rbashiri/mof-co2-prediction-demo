# Hackathon tasks and ownership

## Repository checkpoint

GitHub and local `main` matched at `6ba883e` when this plan was prepared.
The working tree was clean before adding this document.

Susan completed the scientific and technical foundation:

- Prepared shared datasets and selected Phase 3 Model B.
- Connected geometry + CIF extraction, 48 model features, five-pressure
  prediction, and SHAP to Streamlit.
- Separated runtime and development dependency declarations.
- Validated 20 focused tests in both existing and clean Python environments.
- Verified identical predictions across 25,928 rows at all five pressures.
- Froze `phase3-model-b-v1` with checksums, schema, metrics, and environment
  records in the [manifest](PHASE3/phase3-model-b-v1.yaml).

See the [validation report](PHASE3/pre_freeze_validation.md) for scope and
limitations. These checks are not certification of arbitrary CIFs, unseen
MOF families, experimental accuracy, or hosted deployment.

## Non-negotiable baseline rule

**Preserve the frozen models and feature definitions. Raise any necessary
changes with Susan before modifying them.**

Do not overwrite `phase3-model-b-v1`, retrain in its folder, change the
48-feature order/meaning, reintroduce oxidation fields, or silently alter
predictions. Approved scientific/model changes require a new version and
revalidation. Interface, hosting, and LLM work must wrap the existing pipeline.
LLM text must never replace, adjust, or generate the numeric predictions.

## Work allocation

Owners below are roles until individual collaborators are assigned at kickoff.
All remaining items start as pending; earlier local checks do not complete
deployment acceptance.

| Task | Accountable owner | Dependencies | Acceptance criteria |
|------|-------------------|--------------|---------------------|
| Interface design | Collaborator | Current Streamlit app | Consistent layout, clear units/tooltips, readable predictions and SHAP, loading states, useful errors, and clear distinction between simulated prediction and generated narrative |
| Browser testing | Collaborator | UI and hosted build | Real CIF/CSV upload, downloads, all pressure choices, invalid inputs, and stale-result hiding tested in agreed desktop browsers; smaller-screen layout checked |
| Reproducible Docker + MOFid | Collaborator | Frozen artifacts and environment records | Clean build includes native Open Babel libraries/resources and MOFid; no personal paths; reference extraction and app tests pass; review upstream license obligations before distribution |
| Hugging Face Docker Space | Collaborator | Working container and artifact delivery | Agreed access, secrets configured outside Git, stable demo URL, startup and end-to-end reference run verified after restart |
| Frozen bundle delivery | Susan uploads; collaborator verifies | Local versioned bundle | Five models and accompanying records uploaded; downloaded artifacts match manifest hashes; online update confirmed |
| LLM provider/model decision | Susan + collaborator | Budget, hosting, and data policy | Agree model/provider, hosted vs local execution, cost limit, timeout, credentials owner, and allowed data before implementation |
| LLM summary integration | Collaborator; Susan approves content | Provider decision, predictions and SHAP | Grounded short summary, explicit generated-text label, failure handling, stale-summary invalidation, and acceptance tests described below |
| Scientific results and SHAP review | Susan; collaborator prepares examples | Frozen outputs and sample explanations | Approve demo examples, units, feature meanings, attribution interpretation, limitations, and final LLM wording |
| Slides and demo script | Collaborator drafts; Susan reviews science | Stable outputs and agreed claims | Concise problem/approach/results/demo/limitations story, roles, screenshots, and timed script |
| Rehearsal and backup | Whole team; collaborator prepares backup | Hosted demo and reviewed slides | Full timed run completed; backup recording/screenshots and local demo instructions available |

## LLM summary is included in the hackathon scope

The provider is undecided; no LLM integration currently exists. A local,
explicitly rule-based summary is now available as a preparation step; see the
[review workflow](../Readme/readme_webinterface.md#stage-1-local-summary-implemented).
It does not complete the LLM integration or scientific approval tasks. Agree
the provider and data policy together before implementing external calls.

### Scientific content contract

- Input: frozen version, pressure/uptake pairs with units, selected SHAP
  pressure, feature names/values, signed SHAP contributions, and base value
  from the same feature row and model.
- Summarize predictions at the five trained pressures. Explain the strongest
  selected-pressure contributions relative to the model baseline, not
  changes caused by experimentally modifying a material.
- Target a short paragraph (approximately 80-150 words), with a visible
  "AI-generated summary" label and simulated-data limitation.
- Do not invent mechanisms, citations, oxidation states, experimental
  validation, confidence intervals, or predictions at untrained pressures.
- Use only approved structured results. Do not send raw CIFs, filenames,
  full datasets, model files, credentials, or source code to a hosted
  provider. Agree any external transmission policy before enabling requests.
- Keep the numeric prediction table and SHAP plot as the source of truth.

### Implementation and testing contract

- Generate only on an explicit user action, with a visible loading state.
- On timeout/provider failure, show an explicit error and allow retry.
  Predictions and SHAP must remain usable; never disguise canned text as
  a successful LLM response.
- Hide/invalidate the summary when CIF, geometry, model, or selected SHAP
  pressure changes. Tie it to the exact result context.
- Keep API credentials in hosting secrets/environment variables, not Git.
- Test grounded inputs, correct pressure/units/signs, changed-input behavior,
  failure paths, and credential handling with mocked calls.
- Susan reviews actual provider outputs on the agreed demo examples.
  If content invents claims, do not display it as approved scientific text.
- No external request is authorized by this planning document itself.

## Preliminary scientific review of recorded results

The frozen manifest records these test-set results:

| Pressure (bar) | R2 | MAE (mol/kg) | RMSE (mol/kg) |
|----------------|----|-------------|--------------|
| 0.01 | 0.7331 | 0.0504 | 0.1137 |
| 0.05 | 0.8299 | 0.1256 | 0.2162 |
| 0.1 | 0.8503 | 0.1812 | 0.2868 |
| 0.5 | 0.8877 | 0.3572 | 0.4893 |
| 2.5 | 0.9118 | 0.5760 | 0.7593 |

These support reporting performance on the evaluated simulated dataset.
R2 is not a percentage accuracy. Higher-pressure R2 rises while absolute
error also rises; do not describe every aspect of accuracy as improving.
The 60/20/20 random row split does not establish generalization to unseen
linker or MOF families. Test-set errors are aggregate metrics, not per-MOF
uncertainty intervals. Exact training invocation commit was not recorded;
artifact identity is established by the manifest hashes.

SHAP additivity was checked for hMOF-0 at every pressure, but this verifies
numerical consistency, not scientific causation or correctness of an LLM
interpretation. No feature-ranking or chemistry narrative is approved by
this document without inspecting the actual example explanations.

Susan's remaining review checklist:

- Check predictions and SHAP together for 2-3 reference examples using
  matching CIFs and supplied geometry.
- Check baseline, sign, selected pressure, units, and descriptor labels.
- Treat linker fragment counts as extracted-descriptor counts, not
  necessarily physical unit-cell abundances.
- Inspect negative or non-monotonic predictions if present. Separate
  pressure models do not enforce an adsorption isotherm; raise concerns
  rather than silently clipping or smoothing frozen outputs.
- Confirm source simulation conditions before stating temperature or other
  conditions in slides; do not infer missing metadata.
- Approve scientific captions, generated summaries, and limitations.

## Suggested schedule

| Date | Milestone |
|------|-----------|
| Sunday-Monday, Oct 11-12 | Short kickoff, assign named owners, choose LLM/provider policy, verify artifact access, agree UI sketch and demo examples |
| Tuesday, Oct 13 | Container builds cleanly; first styled UI; LLM input/output contract and mocked tests ready |
| Wednesday, Oct 14 | First Hugging Face deployment; LLM integration connected; Susan reviews initial outputs |
| Thursday, Oct 15 | Browser and failure-path testing; slide draft and backup demo prepared |
| Friday, Oct 16 | Fix blockers, scientific sign-off, full rehearsal; stop adding features |
| Saturday, Oct 17 | Ready setup: hosted demo, reviewed LLM summary, slides, backup, and handoff instructions |

If the LLM or hosting is blocked, raise it immediately; do not silently remove
the LLM from the agreed scope. Keep a local/core-prediction backup while the
team decides the workaround.

## Short kickoff meeting (20-30 minutes)

1. Confirm deadline/timezone and individual owners (5 minutes).
2. Review frozen baseline, artifact access, and protected definitions (5 minutes).
3. Choose LLM/provider, budget, and data policy (5-10 minutes).
4. Agree UI, demo examples, slides, and milestone check-ins (5-10 minutes).

Record decisions and owner names in this document. Each collaborator reports
what is done, what is blocked, and the next delivery date. Use focused pull
requests; flag changes affecting science or frozen artifacts before coding.

## Saturday readiness checklist

- [ ] Frozen bundle delivered and downloaded hashes verified.
- [ ] Clean Docker/native MOFid build and reference checks pass.
- [ ] Hugging Face URL works after restart.
- [ ] UI, real uploads, downloads, SHAP, and invalid inputs pass browser review.
- [ ] LLM summary is integrated, labelled, tested, and scientifically approved.
- [ ] Susan approves example predictions, explanations, and slide claims.
- [ ] Slides, timed demo script, recording/screenshots, and local backup ready.
- [ ] Setup instructions, remaining limitations, and named owners documented.

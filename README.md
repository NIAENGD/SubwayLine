## Project title

Synthetic Polycentric City Transit Optimizer

## Objective

Build a modular Python application that generates a synthetic 64×64 U.S.-style city, simulates jobs and population distributions, estimates commuting demand and transit mode choice, and searches for the optimal transit network for 1 through 8 lines. The application must launch from `main.py`, open a GUI, expose comprehensive settings, support reproducible experiments via saved configs and random seeds, and export results for research use.

## External calibration assumptions

The default city should be polycentric, not CBD-dominated. Large U.S. metros are heavily decentralized: Brookings found only about 21% of jobs within 3 miles of downtown and about 45% more than 10 miles away in the largest 98 metro areas. Transit should also start from a conservative U.S. baseline: in 2024, 3.7% of workers used public transportation to work and the mean one-way commute was 27.2 minutes. The 2022 NHTS reports average commute travel time of 26.94 minutes by privately owned vehicle and 43.05 minutes by public transit. Default station access assumptions should follow common FTA planning guidance: about 1/4 mile for bus stops and 1/2 mile for rail stations. ([Brookings][1])

These values are calibration anchors only. The simulator is not required to reproduce any one real metro exactly.

---

## 1. Core functional requirements

The system must do the following:

1. Generate a 64×64 city grid with 4096 tiles.
2. Store, for every tile:

   * population
   * jobs
   * optional derived attributes later: worker count, accessibility score, station coverage, etc.
3. Force the center 5×5 area to be the downtown employment center.
4. Support exactly 5 major employment centers by default, with user-editable presets for more concentrated or more distributed employment patterns.
5. Model employment centers as:

   * a hard center/core with fixed job share and exact footprint
   * a surrounding round magnetic halo with distance decay
   * halo jobs capped so halo + core does not exceed 150% of core jobs
6. Generate a population surface that is explicitly uneven and non-monotonic, rather than simple downtown-high / suburb-low radial decay.
7. Generate home-to-work OD demand from population and jobs.
8. Compute probability of transit use for each OD pair.
9. Optimize transit networks for:

   * 1 line
   * 2 lines
   * 3 lines
   * ...
   * 8 lines
10. Enforce line geometry rules:

* lines may go in any direction
* maximum direction change between consecutive segments is 90 degrees
* no U-turns except allowed terminal reversal logic

11. Provide a desktop GUI with comprehensive settings and visualization.
12. Export maps, network results, metrics, and config files.
13. Be modular enough that each subsystem can be developed and tested independently.

---

## 2. Scope boundaries

Version 1 should include:

* synthetic city generation
* OD demand generation
* transit mode share estimation
* network optimization
* GUI
* export and reproducibility

Version 1 should not include:

* real GTFS import
* detailed time-of-day scheduling
* exact vehicle blocking / crew scheduling
* road network microsimulation
* income/race demographic microdata
* bus bunching or reliability simulation
* land-use feedback loops over time

Those can be Phase 2 or Phase 3 extensions.

---

## 3. City model

### 3.1 Grid and units

Use a 64×64 raster grid.

Default tile size: 0.5 miles per tile.
Default total city size: 32×32 miles.

Tile size must be user-configurable.

Each tile stores:

```python
class Tile:
    x: int
    y: int
    population: float
    jobs: float
    is_downtown: bool
    center_id: int | None
```

### 3.2 Employment centers

Default center configuration:

* Center 1: downtown, 30% of total jobs, footprint 5×5
* Center 2: 15% of total jobs, footprint 4×4
* Center 3: 10% of total jobs, footprint 3×3
* Center 4: 8% of total jobs, footprint 3×3
* Center 5: 5% of total jobs, footprint 3×3
* Remaining 32% distributed outside these centers

Downtown is fixed at map center.

Other centers must be placed subject to:

* no overlap
* minimum separation distance
* optional ring constraints so edge-city locations are plausible
* user seed reproducibility

### 3.3 Employment halo / magnetic effect

Each employment center has:

* a core footprint
* a circular or near-circular halo outside the footprint
* distance decay that ends after a configurable radius

Rules:

* the halo is weaker than the core
* the total jobs in halo + core must not exceed 150% of core jobs
* for 3×3 centers, no halo effect beyond the 4th block outside the edge
* the halo should be round-ish, not square

Implementation requirement:

For each center:

1. Allocate core jobs exactly to core tiles.
2. Build a truncated radial decay field around the core edge.
3. Normalize halo weights so total halo jobs hit the configured halo amount exactly.
4. Add halo jobs to affected tiles.

Suggested default:

* downtown halo ratio: 0.50 of core
* secondary center halo ratio: 0.25 to 0.40 of core

### 3.4 Background jobs

The remaining jobs outside the five core centers should not be assigned as pure random noise.

Use:

* a smoothed random field
* optional weak attraction toward large centers
* optional corridor-like streaking if desired later

This background layer should create dispersed employment while preserving a realistic polycentric structure.

### 3.5 Employment presets

Required presets:

1. Monocentric
2. Concentrated polycentric
3. Moderate polycentric
4. Distributed polycentric
5. Highly decentralized / edge-city heavy

Each preset controls:

* center job shares
* center sizes
* allowable placement zones
* halo strength
* background job share

---

## 4. Population model

Population must be generated independently from jobs.

### 4.1 Design goal

The population surface should look uneven and U.S.-like:

* not a clean downtown-to-suburb gradient
* mixed high / medium / low bands
* some high-density outer clusters
* some low-density inner gaps
* irregular geometry

### 4.2 Population generation components

Build population density from four multiplied or blended layers:

1. Radial ring profile
   Use configurable ring intensities, intentionally non-monotonic.

2. Angular distortion field
   Break circular symmetry using sinusoidal or noise-based direction-dependent scaling.

3. Suburban cluster bumps
   Add 8–12 local Gaussian residential peaks, mostly outside downtown.

4. Correlated noise
   Add low-frequency random variation so the map looks organic.

### 4.3 Jobs-housing interaction

Add a jobs-housing interaction parameter:

* low value = jobs and housing largely independent
* medium value = mild repulsion from pure job cores
* high value = stronger residential avoidance of major employment centers

This prevents unrealistic population piling directly into every employment center unless explicitly requested by a user preset.

### 4.4 Population presets

Required presets:

1. Inner-ring dense
2. Mixed ringed U.S. style
3. Suburban clustered
4. Outer-clustered sprawl
5. Randomized uneven

Default preset should be “Mixed ringed U.S. style”.

---

## 5. Demand model

### 5.1 Workers

For each tile:

* workers = worker_ratio × population

Default worker ratio should be configurable. Suggested default: 0.42.

### 5.2 Destination choice

Generate home-to-work demand using a gravity-style destination model.

For each origin tile (i) and destination tile (j):

* attractiveness increases with jobs at destination
* attractiveness decreases with travel impedance

Recommended form:

* attraction ∝ jobs_j^alpha × exp(-beta × car_time_ij)

Normalize across all destinations so the origin’s worker total is preserved exactly.

Output:

* OD matrix of expected commute flows

### 5.3 Car generalized cost

Transit must compete against a baseline car cost.

Car generalized cost should include:

* travel time from synthetic speed surface
* downtown congestion penalty
* major center congestion penalty
* optional parking penalty at large job centers

Version 1 can use a synthetic impedance field rather than a full road network.

### 5.4 Transit mode share probability

Transit probability must be computed as a percentage, not by threshold logic.

Use binary logit:

* utility depends on difference between car generalized cost and transit generalized cost
* include penalties for transfers and long access walks
* include bonuses for good station coverage

Required outputs:

* probability of transit use for each OD pair
* expected transit riders by OD pair
* expected car users by OD pair

---

## 6. Transit network representation

### 6.1 Network objects

```python
class Station:
    id: str
    x: float
    y: float
    name: str
    score: float

class Line:
    id: str
    stations: list[str]
    mode: str
    headway_min: float

class Segment:
    from_station: str
    to_station: str
    geometry: list[tuple[float, float]]
    length_miles: float
```

### 6.2 Geometry rules

Every line is an ordered station sequence.

Rules:

* segments may be any direction
* turning angle between consecutive segments must be <= 90 degrees
* backtracking / U-turns prohibited except terminal reversal allowance
* shared stations allowed
* shared track allowed but penalized if excessive

### 6.3 Station candidate generation

Do not search over every tile blindly.

Generate candidate stations from:

* employment centers
* large residential clusters
* high combined jobs + population tiles
* high OD-throughput tiles
* likely interchange locations

Apply minimum station spacing.

### 6.4 Access radius

Expose separate defaults for:

* local bus-like networks
* rapid transit / rail-like networks

Initial default for rapid transit:

* 1 tile catchment if tile = 0.5 mile

Later extension:

* support network walk-sheds instead of simple Euclidean access.

---

## 7. Passenger assignment

### 7.1 Transit generalized cost

Transit generalized cost must include:

* access walk time
* waiting time
* in-vehicle time
* transfer penalty
* egress walk time

### 7.2 Assignment graph

Use a graph where the state includes line context so transfer penalties are applied correctly.

Recommended node type:

* `(station_id, line_id)` plus access and egress pseudo-nodes

### 7.3 Assignment constraints

Defaults:

* max transfers = 2
* no circular wandering
* choose minimum generalized cost path

Outputs:

* path chosen for each OD pair with nonzero transit probability
* boardings by station
* passenger loads by segment
* transfer volumes at stations

---

## 8. Network optimization

### 8.1 Optimization targets

The system must solve for best networks with exactly:

* 1 line
* 2 lines
* ...
* 8 lines

Each case should output:

* line geometries
* metrics
* maps
* score summary

### 8.2 Candidate solver design

Do not brute-force.

Use staged optimization:

#### Stage A: Anchor extraction

Identify:

* job hubs
* residential demand clusters
* transfer opportunity nodes

#### Stage B: Corridor scoring

Estimate strongest travel corridors between anchors using OD demand.

#### Stage C: Initial network generation

Build initial candidate lines using:

* greedy corridor connection
* beam search
* or seeded path construction

#### Stage D: Improvement search

Refine networks using metaheuristics.

Recommended first implementation:

* greedy warm start
* simulated annealing improvement

Optional later:

* NSGA-II for multi-objective Pareto optimization

### 8.3 Mutation operators

Required mutation operators:

* add station at line end
* remove station at line end
* insert intermediate station
* delete intermediate station
* reroute one bend
* split one line into two
* merge two lines
* shift transfer node
* reduce overlap
* create partial circumferential connector

### 8.4 Warm-start across line counts

Solve 1-line first.
Then solve 2-line using the 1-line solution as seed.
Continue through 8 lines.

This creates realistic network growth and improves runtime.

---

## 9. Objective function

Use a weighted composite score with multiple selectable profiles.

### 9.1 Default balanced objective

Suggested balanced score:

* 35% ridership capture
* 25% accessibility improvement
* 15% population coverage
* 10% directness / competitiveness vs car
* -10% operating cost
* -5% equity penalty

### 9.2 Metric definitions

Ridership capture:

* expected transit commute trips

Accessibility improvement:

* increase in jobs reachable within threshold time by transit

Population coverage:

* share of population within station catchment

Directness:

* ratio of transit generalized time to car generalized time for captured trips

Operating cost:

* line length × service factor × line count
* optionally frequency-adjusted

Equity penalty:

* penalize highly uneven accessibility distribution across neighborhoods

### 9.3 Alternative objective profiles

Required presets:

1. Ridership-max
2. Accessibility-max
3. Balanced
4. Low-cost
5. Coverage-max

---

## 10. GUI requirements

Use PySide6.

### 10.1 Required tabs

#### City tab

Controls:

* total population
* total jobs
* tile size
* employment preset
* population preset
* halo strength
* center shares
* center sizes
* jobs-housing interaction
* residential cluster count
* random seed

#### Demand tab

Controls:

* worker ratio
* destination choice alpha
* impedance beta
* car speed parameters
* congestion penalties
* parking penalty
* logit parameters

#### Transit rules tab

Controls:

* line count
* station spacing
* access radius
* transfer penalty
* max turn angle
* shared track penalty
* maximum transfers
* service template
* headway

#### Optimization tab

Controls:

* solver type
* iterations
* temperature schedule
* beam width
* candidate pool size
* random seed
* parallel workers

#### Results tab

Must show:

* jobs heatmap
* population heatmap
* OD desire lines
* generated line map
* coverage map
* accessibility map
* boardings / loads
* score breakdown
* per-line metrics

#### Batch runs tab

Must support:

* multiple seeds
* multiple presets
* line-count sweeps
* export of aggregated metrics

### 10.2 Required UX features

* one-click “Generate City”
* one-click “Solve Selected Line Count”
* one-click “Solve 1–8”
* save config
* load config
* export images
* export metrics CSV
* reset to defaults

---

## 11. File and module structure

Recommended repository structure:

```text
project_root/
  main.py
  requirements.txt
  README.md

  config/
    schema.py
    defaults.py

  city/
    jobs_generator.py
    population_generator.py
    presets.py
    fields.py
    validators.py

  demand/
    worker_generation.py
    impedance.py
    destination_choice.py
    mode_choice.py
    od_matrix.py

  network/
    stations.py
    line_geometry.py
    rasterization.py
    graph_builder.py
    assignment.py
    metrics.py

  optimize/
    anchors.py
    corridor_scoring.py
    initial_builder.py
    mutations.py
    annealing.py
    objectives.py
    sweep_runner.py

  gui/
    app.py
    main_window.py
    state.py
    tabs/
      city_tab.py
      demand_tab.py
      transit_tab.py
      optimization_tab.py
      results_tab.py
      batch_tab.py
    plots/
      heatmaps.py
      network_plot.py
      charts.py

  io/
    config_io.py
    export_csv.py
    export_json.py
    export_images.py

  tests/
    test_city.py
    test_population.py
    test_jobs.py
    test_od.py
    test_mode_choice.py
    test_geometry.py
    test_assignment.py
    test_optimizer.py
    test_regression_seeded.py
```

---

## 12. Data persistence

Use JSON for configuration and experiment metadata.

Use NumPy arrays or xarray for raster data.

Suggested saved experiment bundle:

* config.json
* jobs.npy
* population.npy
* od_summary.npy or compressed matrix
* best_networks.json
* metrics.csv
* rendered PNGs

---

## 13. Performance requirements

Version 1 target:

* city generation under 2 seconds
* OD generation under 10 seconds for default settings
* single line-count optimization under 30–90 seconds depending on search settings
* full 1–8 sweep should be batch-capable, not necessarily instant

Use:

* NumPy heavily
* SciPy where useful
* optional multiprocessing for solver evaluation

---

## 14. Testing and validation

### 14.1 Unit tests

Required tests:

* jobs sum exactly to total jobs
* population sum exactly to total population
* center footprints do not overlap
* halo cap never exceeds configured maximum
* OD matrix preserves worker total
* transit probabilities are in [0, 1]
* turn-angle rule is never violated
* transfer penalty is applied in path assignment
* exports serialize and reload correctly

### 14.2 Regression tests

Seed-locked scenarios must produce identical:

* city fields
* objective values within tolerance
* network geometry within tolerance

### 14.3 Sanity checks

The system should raise warnings when:

* total workers far exceed total jobs
* catchment radius is too small to serve anyone
* optimization fails to find feasible routes
* line overlap becomes excessive
* solution degenerates into nearly identical lines

---

## 15. Deliverables

The code team must deliver:

1. Working Python application launched by `main.py`
2. GUI with all core settings
3. Modular city generator
4. Demand model
5. Transit network representation
6. Path assignment engine
7. Optimizer for 1–8 lines
8. Export pipeline
9. README with setup and usage
10. Basic test suite

---

## 16. Development phases

### Phase 1 — Foundation

Deliver:

* config system
* raster grid
* jobs generator
* population generator
* plotting of synthetic city

Acceptance:

* user can generate and visualize jobs and population maps from GUI

### Phase 2 — Demand

Deliver:

* workers
* OD generation
* car impedance model
* initial metrics

Acceptance:

* OD demand matrix generated and summarized correctly

### Phase 3 — Transit evaluation

Deliver:

* station candidates
* line objects
* geometry validation
* transit path assignment
* transit generalized cost

Acceptance:

* any user-drawn or auto-built line can be evaluated for coverage and travel cost

### Phase 4 — Optimization

Deliver:

* anchor extraction
* corridor scoring
* greedy initial line builder
* annealing-based refinement
* 1–8 line sweep

Acceptance:

* solver returns feasible networks for 1–8 lines and outputs metrics

### Phase 5 — GUI polish and export

Deliver:

* batch runs
* save/load config
* export CSV/JSON/PNG
* result comparison views

Acceptance:

* user can run experiments and export a complete result package

### Phase 6 — QA and refinement

Deliver:

* tests
* performance tuning
* parameter calibration
* documentation cleanup

Acceptance:

* reproducible outputs and stable runtime

---

## 17. Acceptance criteria for version 1

Version 1 is complete when all of the following are true:

1. Launching `python main.py` opens the GUI without manual code edits.
2. The user can generate a city with configurable job and population presets.
3. The city always contains five employment centers, including fixed downtown.
4. The halo logic obeys the 150% total cap.
5. The population surface is irregular and non-monotonic by default.
6. The system can compute OD demand and transit probabilities.
7. The system can optimize and output feasible networks for each line count from 1 to 8.
8. All generated lines obey the max-90-degree turn rule.
9. The user can save and reload a full experiment.
10. The user can export maps and metrics.
11. The core tests pass.

---

## 18. Recommended implementation defaults

Use these defaults unless changed in GUI:

* tile size: 0.5 miles
* total population: 2,000,000
* total jobs: 900,000
* employment preset: moderate polycentric
* population preset: mixed ringed U.S. style
* worker ratio: 0.42
* station catchment: 0.5 mile equivalent for rapid-transit template
* max transfers: 2
* line overlap penalty: moderate
* objective profile: balanced
* optimizer: greedy seed + simulated annealing
* seed: fixed integer default for reproducibility

---

## 19. Non-negotiable implementation principles

1. Modular architecture only. No monolithic single-file prototype.
2. Reproducibility is mandatory.
3. All totals must balance exactly.
4. Transit share must be probabilistic, not rule-based.
5. Network design must be constraint-aware, especially turn-angle rules.
6. GUI must be usable by non-developers.
7. Export format must support later academic analysis.

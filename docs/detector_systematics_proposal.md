# Configurable detector physics and systematics

## Design

The implementation separates choosing effective inputs from using those inputs:

```text
Read nominal YAML properties
             |
Select module-specific input values
             |
Resolve requested systematics on a fresh copy
             |
Set effective constants and calculate derived quantities
             |
Load dependent response data and refresh compiled consumers
             |
Run the simulation with a fixed effective configuration
```

The resolver never changes simulation globals. The detector and physics setters
do not interpret systematic requests. Quenching and drifting retain their
original kernel interfaces and physics equations.

The scope is the independent inputs from the parameter survey, not unrestricted
assignment to every name in the constants folder.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| [config.yaml](../larndsim/config/config.yaml) | Select input files, profile and requested variations |
| [physics.yaml](../larndsim/config/physics.yaml) | Nominal recombination/ionisation parameters and model |
| [detector_defaults.yaml](../larndsim/config/detector_defaults.yaml) | Fixed fallback inputs when a detector YAML omits a supported property |
| [ndlar-module.yaml](../larndsim/detector_properties/ndlar-module.yaml) | ND-LAr's nominal detector properties |
| [systematics.yaml](../larndsim/config/systematics.yaml) | Supported parameter definitions and detector-specific uncertainties |
| [properties.py](../larndsim/config/properties.py) | Read, copy, select module values, validate and resolve |
| [consts/__init__.py](../larndsim/consts/__init__.py) | Install effective physics/detector/light properties |
| [runtime.py](../larndsim/config/runtime.py) | Recreate configuration-dependent CPU/CUDA consumers |
| [simulate_pixels.py](../cli/simulate_pixels.py) | Coordinate loading, preparation, execution and provenance |

## Example presets

Only Birks A_b and electric field have example systematic presets:

```yaml
ndlar_birks_up_demo:
    <<: *ndlarmain
    SYSTEMATICS_PROFILE: ndlar
    DETECTOR_SYSTEMATICS:
        physics.BIRKS_Ab: up

ndlar_birks_down_efield_up_demo:
    <<: *ndlarmain
    SYSTEMATICS_PROFILE: ndlar
    DETECTOR_SYSTEMATICS:
        physics.BIRKS_Ab: down
        detector.E_FIELD: up

ndlar_birks_custom_demo:
    <<: *ndlarmain
    DETECTOR_SYSTEMATICS:
        physics.BIRKS_Ab:
            value: 0.780
```

`<<: *ndlarmain` inherits the original ND-LAr file selections and flags.
The combined preset produces A_b = 0.780 and E_FIELD = 0.510 kV/cm from the
ND-LAr nominal values 0.800 and 0.500.

```sh
python cli/simulate_pixels.py \
    --input_filename=input.h5 \
    --output_filename=birks_down_field_up.h5 \
    --config=ndlar_birks_down_efield_up_demo
```

The usual simulation environment, input and response files are still required.
`ndlar` remains nominal. An empty `DETECTOR_SYSTEMATICS` mapping also means
nominal. Each preset selects one simulation, not an automatic parameter scan.

## Supported inputs

| Public parameter name | Nominal source key | Units | Validation |
| --- | --- | --- | --- |
| `detector.E_FIELD` | `e_field` | kV/cm | Positive |
| `detector.TEMPERATURE` | `temperature` | K | Positive |
| `detector.ELECTRON_LIFETIME` | `lifetime` | us | Positive |
| `detector.LONG_DIFF` | `long_diff` | cm^2/us | Non-negative |
| `detector.TRAN_DIFF` | `tran_diff` | cm^2/us | Non-negative |
| `physics.BIRKS_Ab` | `BIRKS_Ab` | Dimensionless | In (0, 1] |
| `physics.BIRKS_kb` | `BIRKS_kb` | (kV/cm)(g/cm^2)/MeV | Positive |
| `physics.W_ION` | `W_ION` | MeV | Positive |
| `physics.BOX_ALPHA` | `BOX_ALPHA` | Dimensionless | Positive |
| `physics.BOX_BETA` | `BOX_BETA` | (kV/cm)(g/cm^2)/MeV | Positive |
| `physics.RECOMBINATION_MODEL` | `RECOMBINATION_MODEL` | Model name | `birks` or `box` |

All numerical values must be finite numbers, not booleans or numeric strings.
All supported inputs accept `{value: ...}` replacements. Numerical inputs also
accept `up`/`down` once an uncertainty is registered for the selected profile.
The model accepts only an explicit replacement, not a numerical shift.

The simulation translates the configured model name into the existing
`physics.BIRKS` or `physics.BOX` identifier when launching quenching.
Both Box parameters are configurable; the unchanged nominal beta is **0.207**,
not the alternative 0.212 discussed in the survey. Uncertainties for temperature,
lifetime, diffusion, k_b, W_ION and Box parameters are deliberately not invented.

`V_DRIFT` and `DRIFT_MAX_TIME` are not independent registered inputs:

```python
V_DRIFT = E_FIELD * electron_mobility(E_FIELD, TEMPERATURE)
DRIFT_MAX_TIME = DRIFT_LENGTH / V_DRIFT
```

They are recalculated by the detector setter. Diffusion coefficients remain
independent inputs; the code does not infer them from field or temperature.

## Registry-driven resolution

Each supported input has a declarative definition, for example:

```yaml
parameters:
    detector.E_FIELD:
        section: detector
        key: e_field
        units: kV/cm
        minimum_exclusive: 0
        per_module: true

profiles:
    ndlar:
        detector.E_FIELD:
            relative_uncertainty: 0.02
```

The same generic loop handles every registered input. A definition describes
its source, bounds or allowed model names, and whether a list represents
module-specific values. There are no Birks-specific or field-specific resolver
functions.

`up` means `nominal * (1 + relative_uncertainty)` and `down` means
`nominal * (1 - relative_uncertainty)`. Explicit values replace the nominal;
they never stack on top of a named shift. Unknown names, missing uncertainty
definitions and invalid values raise errors before property installation.

The shipped ND-LAr registry defines only:

- Electric field: 2%, as requested for this example.
- Birks A_b: 2.5%, explicitly marked illustrative and warning when selected.

**The Birks uncertainty is not an approved production physics prescription.**
Production definitions need an approved source and validation.

## Loading and setter interfaces

Nominal physics inputs always start from `config/physics.yaml`. A preset may
optionally supply `PHYSICS_PROPERTIES` to select an override YAML containing
any subset of those physics keys. A bare filename is resolved within the config
directory, following existing preset path conventions.

Detector inputs start from fixed YAML defaults, then the selected detector YAML.
This avoids inheriting a previous run's shifted field, lifetime or coefficients.

Conceptually, the CLI does:

```python
nominal = read_properties(detector_properties, cfg.get('PHYSICS_PROPERTIES'))

# For each module, always start again from nominal:
effective, metadata = resolve_properties(nominal, cfg, i_module=i_mod)
derived = consts.set_properties(effective, pixel_layout, response_file, i_module=i_mod)
response = consts.detector.load_response(response_file)
prepare_simulation()

quenching.quench[BPG, TPB](tracks, physics.recombination_mode())
drifting.drift[BPG, TPB](tracks)
```

The detector and light setters accept either their existing filename arguments
or already-loaded dictionaries. Layout input also accepts a dictionary.
They defensively copy mappings instead of mutating the nominal configuration.
The existing `consts.load_properties(...)` entry point remains available as a
read/resolve/set convenience wrapper; its optional physics file and systematic
configuration arguments enable the same path for library callers.

All five supported detector inputs accept a scalar, a singleton list, or one
value per module in module-mapping order. Selection precedes the variation.
With module variation disabled, a multi-value list uses its first value and
warns, matching the established field/lifetime convention. Physics inputs are
run-wide. Independent per-module systematic requests are not implemented.

## Why central preparation is necessary

Changing Python globals alone does not reliably update compiled Numba functions.
The driver synchronises outstanding GPU work before replacing effective
constants and then calls `prepare_simulation()` before invoking consumers.

Preparation reloads the imported configuration-dependent modules in dependency
order. This recreates CPU/CUDA dispatchers and import-time function defaults,
including quenching and drifting, without adding configuration arguments to
their physics kernels. Providers such as pixel-ID helpers precede consumers
that import those functions directly.

Library callers changing properties in an existing process must follow the same
synchronise/install/prepare sequence. Keep module references, not directly
imported old kernel/function references, across preparation boundaries.
This is a sequential global-state design: simultaneous configurations in
different threads are not supported. Separate processes are recommended for
independent systematic runs.

The preparation list is explicit. Supporting new classes of constants,
particularly arrays, geometry or unit definitions, still requires auditing
their consumers and dependent resource allocation. This is not a guarantee
that every arbitrary Python name can safely be overridden.

## Provenance and response assumptions

The HDF5 `configs` object's `detector_systematics` attribute stores JSON:

```text
schema_version: 1
modules:
    <module ID or all_modules>:
        parameters:
            <parameter name>: nominal, requested, value, units, profile
        derived:
            detector: V_DRIFT, DRIFT_LENGTH, DRIFT_MAX_TIME
```

Named shifts also record their uncertainty definition, including illustrative
status. Module IDs follow the existing one-based numbering. Existing software
version and Git metadata are retained.

Charge-response and light LUTs are held fixed. The existing charge-response
padding/chopping uses the updated drift velocity, but the tables are not
regenerated for the changed field or model. Production use requires validating
this approximation or supplying suitable alternate response files.

## Validation

- [test_systematics.py](../tests/test_systematics.py): all supported inputs,
  example presets, generic registered shifts, missing uncertainties, validation,
  module selection, immutable nominal properties and fresh defaults.
- [test_property_runtime.py](../tests/test_property_runtime.py): actual setters,
  derived drift quantities, original quenching/drifting interfaces, both models,
  expected charge/light yields, lifetime/diffusion/timing, CPU dispatcher refresh,
  repeated configurations and module-specific fields in an isolated Numba CUDA
  simulator process.

The simulator test uses NumPy for host-side response-file access and packet-class
placeholders. It does **not** validate real CUDA compilation/device memory or a
full detector run. Those remain required production checks, along with output
provenance and response compatibility in an appropriately equipped environment.

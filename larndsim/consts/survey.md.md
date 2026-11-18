Read , lines 1 to 240
Completed with input: {
  "path": "/nashome/e/emuhamma/work/NDSimReco/plots_shifted/larnd-sim/larndsim/consts/detector.py",
  "view_range": [
    1,
    240
  ]
}

Read 
Completed with input: {
  "path": "/nashome/e/emuhamma/work/NDSimReco/plots_shifted/larnd-sim/larndsim/consts/physics.py"
}

Read , lines 310 to 512
Completed with input: {
  "path": "/nashome/e/emuhamma/work/NDSimReco/plots_shifted/larnd-sim/larndsim/consts/detector.py",
  "view_range": [
    310,
    512
  ]
}

Read 
Completed with input: {
  "path": "/nashome/e/emuhamma/work/NDSimReco/plots_shifted/larnd-sim/larndsim/consts"
}

Yes. Below is an inventory of the current `consts folder`, including **values created during loading**, not just assignments at the top of each file.

**Important:** these are architectural categories, not guarantees that each parameter is currently YAML-configurable or safe to change independently.

## Categories

| Code | Kind | Intended treatment |
|---|---|---|
| **P** | Independent physical/model input | Configurable; candidate for a physics systematic |
| **D** | Derived quantity | Recalculate from effective inputs |
| **G** | Geometry, mapping or array-valued data | Configure/load with shape and consistency validation |
| **S** | Structural, algorithmic or operational setting | Configure before preparing buffers, kernels and execution |
| **I** | Implementation definition or unit convention | Usually not a detector systematic |

Some values belong to more than one category. For example, pixel pitch is a physical geometry input, but changing it also changes array dimensions and response compatibility.

---

## 1. Physics constants

Source: `physics.py`

| Names | Category | Meaning / treatment |
|---|---|---|
| `BOX_ALPHA`, `BOX_BETA` | **P** | Box recombination parameters |
| `BIRKS_Ab`, `BIRKS_kb` | **P** | Birks recombination parameters |
| `W_ION` | **P** | Mean energy per ion pair; material/model input |
| `E_CHARGE` | **I** | Physical constant, electron-charge magnitude in coulombs |
| `BOX`, `BIRKS` | **I** | Model-selection identifiers, not physical quantities |

**Current configuration:** these are defined in Python. Our demo passes an effective Birks A_b into quenching without changing the nominal global.

**Natural YAML candidates:** the four recombination parameters and `W_ION`.

---

## 2. Detector constants

Source: `detector.py`

### Transport and material properties

| Names | Category | Meaning / treatment |
|---|---|---|
| `TEMPERATURE` | **P** | Detector temperature |
| `LAR_DENSITY` | **P** | Liquid-argon density |
| `E_FIELD` | **P** | Effective electric field |
| `ELECTRON_MOBILITY_PARAMS` | **P / G** | Tuple of mobility-model coefficients |
| `ELECTRON_LIFETIME` | **P** | Electron survival timescale |
| `LONG_DIFF`, `TRAN_DIFF` | **P** | Longitudinal/transverse diffusion coefficients |
| `DEFAULT_E_FIELD` | **I** | Fallback introduced by our demo, not another physical field |
| `V_DRIFT` | **D** | Calculated from field, temperature and mobility coefficients |

**Dependency distinction:** the current code calculates velocity from field and temperature, but density and diffusion coefficients remain independent supplied values. It does **not** automatically recalculate every material property when temperature or field changes.

### Geometry and channel mapping

| Names | Category | Meaning / treatment |
|---|---|---|
| `DRIFT_LENGTH` | **G / D** | Read from layout, or inferred from anode positions |
| `DRIFT_MAX_TIME` | **D** | Drift length divided by drift velocity |
| `TPC_OFFSETS` | **G** | Loaded offsets, converted to internal axis convention |
| `TPC_BORDERS` | **G / D** | Constructed from offsets, layout and drift length |
| `TILE_BORDERS` | **G / D** | Calculated from pixel positions and pitch |
| `PIXEL_PITCH` | **G** | Loaded from layout with unit conversion |
| `PIXEL_CONNECTION_DICT` | **G / D** | Constructed from layout channel mapping |
| `N_PIXELS`, `N_PIXELS_PER_TILE` | **D / S** | Calculated pixel counts |
| `TILE_POSITIONS`, `TILE_ORIENTATIONS` | **G** | Layout inputs |
| `TILE_MAP`, `TILE_CHIP_TO_IO` | **G** | Tile and readout mappings |
| `MODULE_TO_IO_GROUPS`, `MODULE_TO_TPCS` | **G** | Module/readout associations |
| `TPC_TO_MODULE` | **G / D** | Reverse mapping |
| `MOD_IDS` | **D / S** | Created by the loader from module mapping keys |

Changing these often means rebuilding geometry, pixel selections, buffers and potentially selecting different response tables.

### Charge-response and numerical settings

| Names | Category | Meaning / treatment |
|---|---|---|
| `TIME_SAMPLING` | **S** | Simulation time sampling |
| `RESPONSE_SAMPLING` | **S** | Loaded from response-file metadata |
| `RESPONSE_BIN_SIZE` | **G / S** | Loaded spatial response-bin size |
| `RESPONSE_MAX_TIME` | **D** | Calculated from non-zero response samples |
| `MAX_RADIUS` | **D / S** | Inferred from response dimensions, optionally overridden; far-field settings can replace it |
| `MIN_STEP_SIZE` | **S** | Segment subdivision scale |
| `DIFF_N_SIGMAS` | **S** | Diffusion padding extent |
| `NEIGHBORING_PIX_DIST` | **D / G** | Constructed from the neighbour radius |
| `DEFAULT_PLANE_INDEX` | **I** | Sentinel indicating no valid pixel plane |

Notice that response sampling is **not simply a free detector scalar**: the actual response file defines its discretisation.

### Electronics response

| Names | Category | Meaning / treatment |
|---|---|---|
| `DISCRIMINATION_THRESHOLD` | **P** | Charge discrimination threshold |
| `GAIN` | **P** | Front-end gain |
| `BUFFER_RISETIME` | **P** | Analogue response timescale |
| `V_CM`, `V_REF`, `V_PEDESTAL` | **P** | Electronics voltages |
| `RESET_NOISE_CHARGE` | **P** | Reset noise |
| `UNCORRELATED_NOISE_CHARGE` | **P** | Uncorrelated noise |
| `DISCRIMINATOR_NOISE` | **P** | Discriminator noise |
| `ADC_HOLD_DELAY`, `ADC_BUSY_DELAY` | **S** | Electronics timing settings |
| `PERIODIC_RESET_CYCLES`, `RESET_CYCLES` | **S** | Reset configuration |
| `ADC_COUNTS` | **S** | ADC resolution/range structure |
| `ADC_SCALE_FACTOR`, `ADC_OFFSET` | **S** | Digitisation controls |

The **P** entries here are detector-response inputs rather than fundamental material properties, but they are natural systematic candidates.

### Clocks and event timing

| Names | Category | Meaning / treatment |
|---|---|---|
| `CLOCK_CYCLE` | **S** | Clock period |
| `ROLLOVER_CYCLES` | **S** | Counter rollover setting |
| `USE_PPS_ROLLOVER` | **S** | Reset-policy selection |
| `PPS_CYCLES` | **D / S** | Initially calculated from clock period; can also be supplied explicitly |
| `CLOCK_RESET_PERIOD` | **D / S** | Initially selected from PPS or rollover settings; can also be supplied explicitly |
| `EVENT_RATE` | **S** | Despite its name, documented as an average inter-event time |
| `NON_BEAM_EVENT_GAP` | **S** | Event timing offset |

**Dependency concern:** the initial clock formulas and subsequent YAML assignments are separate. A future builder should establish whether these are independent overrides or derived values and consistently enforce that policy.

### Packet implementation

| Names | Category | Meaning / treatment |
|---|---|---|
| `LARPIX_VERSION` | **S** | Selects hardware/protocol family |
| `LARPIX_REGISTRY` | **I** | Supported packet classes and format definitions |
| `PACKET_VERSION` | **D / I** | Selected from registry; becomes a packet class after loading |
| `PACKET_TYPE` | **D / I** | Selected packet-type identifier |
| `LARPIX_HDF5_VERSION` | **D / I** | Selected output-format version |

These are discrete compatibility choices—not values to vary by ±2%.

---

## 3. Light constants

Source: `light.py`

### Light production and analogue response

| Names | Category | Meaning / treatment |
|---|---|---|
| `SCINT_PRESCALE` | **P** | Scintillation-yield scale |
| `W_PH` | **P** | Ionisation-plus-excitation work function |
| `SINGLET_FRACTION` | **P** | Singlet contribution |
| `TAU_S`, `TAU_T` | **P** | Singlet/triplet decay times |
| `DEFAULT_LIGHT_GAIN` | **P** | Default gain used when expanding channel gains |
| `LIGHT_GAIN` | **P / G** | Effective per-channel gain array |
| `OP_CHANNEL_EFFICIENCY` | **P / G** | Per-channel efficiency array |
| `LIGHT_RESPONSE_TIME` | **P** | Response timescale |
| `LIGHT_OSCILLATION_PERIOD` | **P** | Response oscillation period |
| `IMPULSE_MODEL` | **G** | Sampled response, optionally loaded from a file |
| `IMPULSE_TICK_SIZE` | **S** | Sample spacing for the impulse model |

**Current gap:** `SCINT_PRESCALE` and `W_PH` are defined here but are not assigned by the existing light-property setter.

### Channel structure and mappings

| Names | Category | Meaning / treatment |
|---|---|---|
| `N_OP_CHANNEL` | **S** | Optical-channel count |
| `OP_CHANNEL_TO_TPC` | **G / D** | Constructed channel-to-TPC mapping |
| `TPC_TO_OP_CHANNEL` | **G / D** | Loaded/constructed reverse association |
| `OP_CHANNEL_PER_TRIG` | **S** | Trigger grouping size |

### Sampling, trigger and model settings

| Names | Category | Meaning / treatment |
|---|---|---|
| `LIGHT_SIMULATED` | **S** | Enable/disable light simulation |
| `ENABLE_LUT_SMEARING` | **S** | LUT-smearing option |
| `LIGHT_TICK_SIZE`, `LIGHT_WINDOW` | **S** | Internal light sampling/window |
| `SIPM_RESPONSE_MODEL` | **S** | Response-model selection |
| `LIGHT_DET_NOISE_SAMPLE_SPACING` | **S** | Noise-spectrum sampling |
| `LIGHT_TRIG_MODE` | **S** | Trigger-mode selection |
| `LIGHT_TRIG_THRESHOLD` | **P / G** | Trigger thresholds expanded into an array |
| `LIGHT_TRIG_WINDOW` | **S** | Trigger window |
| `LIGHT_DIGIT_SAMPLE_SPACING` | **S** | Digitiser sampling |
| `LIGHT_NBIT` | **S** | Digitiser resolution |

**Important:** a scalar YAML gain or efficiency can become a full array. Resolution must happen before that expansion, or explicitly support elementwise array changes.

---

## 4. Simulation settings

Source: `sim.py`

These are all primarily **S: structural/algorithmic/operational**, rather than detector physics constants.

| Group | Names |
|---|---|
| Batching | `SEGMENT_BATCH_SIZE`, `EVENT_BATCH_SIZE`, `WRITE_BATCH_SIZE` |
| Event organisation | `EVENT_SEPARATOR`, `IS_SPILL_SIM`, `SPILL_PERIOD`, `MAX_EVENTS_PER_FILE`, `TRACKS_DSET_NAME` |
| Time selection | `MAX_SEGMENT_T0` |
| Capacity and truth handling | `MAX_TRACKS_PER_PIXEL`, `ASSOCIATION_COUNT_TO_STORE`, `MAX_ADC_VALUES`, `MAX_MC_TRUTH_IDS`, `MC_TRUTH_THRESHOLD` |
| Execution features | `FARFIELD_ENABLED` |
| Runtime-assigned setting | `MOD2MOD_VARIATION` |

Some affect selection or truncation and therefore can affect results. Others mainly affect resource use. Changing array capacities requires allocations and compiled consumers to agree.

`MOD2MOD_VARIATION` is assigned by the simulation driver; it is not given a normal top-level default in this module.

---

## 5. Far-field induction settings

Source: `ff_induction.py`

| Names | Category | Meaning / treatment |
|---|---|---|
| `FARFIELD_MODE` | **S** | Segment versus voxel calculation |
| `FARFIELD_SIGNAL_MODEL` | **S** | Induction-model selection |
| `COARSE_VOXEL_SIZE_X`, `COARSE_VOXEL_SIZE_Y`, `COARSE_VOXEL_SIZE_Z` | **S** | Numerical discretisation |
| `CHARGE_COLLECTION_RADIUS`, `CHARGE_NEIGHBOR_RADIUS` | **S** | Pixel classification/extent settings |
| `INDUCTION_CUTOFF_RADIUS` | **S** | Maximum induction consideration distance |
| `FAR_FIELD_SEGMENT_STEP_CM` | **S** | Segment subdivision |
| `INDUCTION_SIGNAL_THRESHOLD` | **S** | Numerical inclusion threshold |
| `DIPOLE_N_TERMS` | **S** | Expansion truncation order |
| `DIPOLE_SCALE` | **P** | Empirical model normalisation |

All have existing YAML loading. Several numerical settings need convergence studies rather than interpretation as measured detector uncertainties.

---

## 6. Units and definitions

Source: `units.py`

These are **I: implementation/unit conventions**, including aliases and quantities calculated from base units. They should remain separate from normal detector-property overrides.

The complete set of module-level assigned names, grouped by purpose:

| Group | Names |
|---|---|
| Base length / miscellaneous | `euro`, `millimeter`, `millimeter2`, `millimeter3` |
| Length units | `centimeter`, `centimeter2`, `centimeter3`, `decimeter`, `decimeter2`, `decimeter3`, `meter`, `meter2`, `meter3`, `kilometer`, `kilometer2`, `kilometer3`, `micrometer`, `nanometer`, `angstrom`, `fermi` |
| Volume | `liter`, `l`, `ml`, `mul`, `nl`, `pl` |
| Length aliases | `nm`, `mum`, `micron`, `micron2`, `micron3`, `mm`, `mm2`, `mm3`, `cm`, `cm2`, `cm3`, `m`, `m2`, `m3`, `km`, `km2`, `km3`, `ft` |
| Cross-section | `barn`, `millibarn`, `microbarn`, `nanobarn`, `picobarn` |
| Angle | `radian`, `milliradian`, `degree`, `steradian`, `rad`, `mrad`, `sr`, `deg` |
| Time | `nanosecond`, `second`, `millisecond`, `microsecond`, `picosecond`, `femtosecond`, `year`, `day`, `minute`, `hour`, `s`, `ms`, `ps`, `fs`, `mus`, `ns` |
| Frequency | `hertz`, `kilohertz`, `megahertz`, `gigahertz`, `MHZ`, `kHZ`, `kHz`, `GHZ` |
| Charge | `e`, `e_SI`, `coulomb` |
| Energy | `megaelectronvolt`, `electronvolt`, `milielectronvolt`, `kiloelectronvolt`, `gigaelectronvolt`, `teraelectronvolt`, `petaelectronvolt`, `meV`, `eV`, `keV`, `MeV`, `GeV`, `TeV`, `PeV`, `eV2` |
| Energy in joules | `joule`, `J`, `milijoule`, `microjoule`, `nanojoule`, `picojoule`, `femtojoule`, `mJ`, `muJ`, `nJ`, `pJ`, `fJ` |
| Mass | `kilogram`, `gram`, `milligram`, `ton`, `kiloton`, `kg`, `g`, `mg` |
| Power | `watt`, `W`, `milliwatt`, `microwatt`, `nanowatt`, `mW`, `muW` |
| Force | `newton` |
| Pressure | `hep_pascal`, `pascal`, `Pa`, `kPa`, `MPa`, `GPa`, `bar`, `milibar`, `atmosphere` |
| Linear density | `denier` |
| Current | `ampere`, `milliampere`, `microampere`, `nanoampere`, `mA`, `muA`, `nA` |
| Voltage | `megavolt`, `kilovolt`, `volt`, `millivolt`, `V`, `mV`, `kV`, `MV` |
| Resistance | `ohm` |
| Capacitance | `farad`, `millifarad`, `microfarad`, `nanofarad`, `picofarad`, `nF`, `pF` |
| Magnetic quantities | `weber`, `tesla`, `gauss`, `kilogauss`, `henry` |
| Temperature | `kelvin`, `K` |
| Amount of substance | `mole`, `mol`, `milimole`, `micromole`, `nanomole`, `picomole` |
| Activity | `becquerel`, `curie`, `Bq`, `mBq`, `muBq`, `kBq`, `MBq`, `cks`, `U238ppb`, `Th232ppb` |
| Dose | `gray` |
| Photometry | `candela`, `lumen`, `lux` |
| Scale/count conventions | `perCent`, `perThousand`, `perMillion`, `pes`, `adc` |

The file also contains the temperature-conversion function `celsius`; that is a function, not a configurable constant.

These names are inventoried **as written**, without asserting that every numerical definition has been independently validated.

---

## 7. Package coordinator

Source: `init.py`

No separate physical constants are defined here. It coordinates loading simulation, far-field, detector and light properties.

---

## What this tells us about the redesign

**The folder is really a mixture of inputs, computed state and implementation definitions—not simply a collection of constants.**

The clean separation would be:

```text
Configurable inputs
  Physics / material / electronics parameters
  Geometry and mappings
  Algorithmic and structural settings
                    ↓
          Resolve and validate
                    ↓
Computed state
  Drift velocity, borders, counts, expanded arrays,
  reverse mappings, clock relationships, response extents
                    ↓
          Initialise simulation
```

For the first refactor, I would therefore:

1. **Make independent inputs configurable across all relevant modules.**
2. **Keep derived state explicitly calculated**, not blindly overwritten.
3. **Keep units, sentinels, identifiers and packet-class definitions separate.**
4. **Preserve provenance for both inputs and derived results.**

That provides broad coverage without promising that assigning an arbitrary name in this folder is automatically physically or computationally meaningful.
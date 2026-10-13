# API reference

Generated from the docstrings of `bcpnm` 0.1.0 by `paper/tools/build_api_docs.py`. All names below are importable from the top-level package (`import bcpnm as bc`) unless marked otherwise.

## `bcpnm.core`

Core data structures: `StateSpace`, `Node`, `Rule` and `PNMGraph`.

A model is a set of nodes (psychological constructs or persons), each with a finite,
ordered set of integer-coded states, and one update rule per node. A rule maps the
current values of the node's inputs (and the node's own current value) to a
probability distribution over the node's next state. Deterministic rules (Boolean,
threshold, table) put all probability on one state.

### class `StateSpace(values: 'Sequence[int]', labels: 'Optional[Mapping[int, str]]' = None, name: 'str' = '')`

An ordered, finite set of integer-coded states with optional labels.

Parameters
----------
values : sequence of int
    The admissible states in increasing order, e.g. `(-1, 0, 1)`.
labels : mapping, optional
    Human-readable label per value, e.g. `{-1: "conceal", 0: "silent", 1: "acknowledge"}`.
name : str, optional
    Name used in configuration files and reports.

- **boolean**(cls, off: 'str' = 'off', on: 'str' = 'on') -> "'StateSpace'" (method): States `(0, 1)`.
- **ternary**(cls, labels: 'Sequence[str]' = ('negative', 'neutral', 'positive')) -> "'StateSpace'" (method): Blume–Capel states `(-1, 0, +1)`.
- **ordinal**(cls, k: 'int', labels: 'Optional[Sequence[str]]' = None) -> "'StateSpace'" (method): States `(0, 1, ..., k-1)`.
- **size** (property)
- **array** (property)
- **index**(self, value) -> 'np.ndarray' (method): Position(s) of state value(s) in the ordered state space.
- **to_dict**(self) -> 'dict' (method)

### class `Node(name: 'str', states: 'StateSpace', description: 'str' = '', construct: 'str' = '', measurement: 'str' = '', initial: 'Optional[int]' = None) -> None`

A node in a `PNMGraph`.

`construct` and `measurement` document what the node represents and how
its state would be measured; they are carried into generated reports.


### class `Rule(inputs: 'Iterable[str]' = ())`

Base class for update rules.

Subclasses implement either `next_value` (deterministic rules) or
`probs` (stochastic rules). All methods are vectorized over `R`
replicates: `parents` has shape `(R, n_inputs)`, `current` has shape
`(R,)` and `values` is the node's ordered state values `(S,)`.

- **next_value**(self, parents: 'np.ndarray', current: 'np.ndarray', values: 'np.ndarray') -> 'np.ndarray' (method)
- **probs**(self, parents: 'np.ndarray', current: 'np.ndarray', values: 'np.ndarray') -> 'np.ndarray' (method): Probability of each next state (R x S).
- **sample**(self, parents, current, values, rng: 'np.random.Generator') -> 'np.ndarray' (method)
- **params**(self) -> 'dict' (method): Parameters for reporting and configuration export.
- **copy**(self) -> "'Rule'" (method)

### class `PNMGraph(name: 'str' = 'model', meta: 'Optional[dict]' = None)`

A polytomous network model: nodes, state spaces and one update rule per node.

Parameters
----------
name : str
    Model name used in reports.
meta : dict, optional
    Free-form metadata (domain, version, sources of rules, seed).

Examples
--------
>>> from bcpnm import PNMGraph, StateSpace
>>> from bcpnm.rules import BooleanRule
>>> g = PNMGraph("toggle")
>>> g.add_node("A", StateSpace.boolean()); g.add_node("B", StateSpace.boolean())
>>> g.set_rule("A", BooleanRule("not B")); g.set_rule("B", BooleanRule("not A"))
>>> g.N
2

- **add_node**(self, name: 'str', states: 'StateSpace', **kwargs) -> 'Node' (method)
- **set_rule**(self, node: 'str', rule: 'Rule') -> 'None' (method)
- **names** (property)
- **N** (property)
- **index**(self, name: 'str') -> 'int' (method)
- **sizes**(self) -> 'List[int]' (method)
- **state_space_size**(self) -> 'int' (method)
- **parents**(self, node: 'str') -> 'Tuple[str, ...]' (method)
- **edges**(self) -> 'List[Tuple[str, str, dict]]' (method): Directed edges `(source, target, attributes)` implied by the rules.
- **to_networkx**(self) (method): Return a `networkx.DiGraph` with node metadata and edge attributes.
- **validate**(self) -> 'None' (method): Check that every node has a rule and every rule input is a node.
- **deterministic** (property)
- **value_arrays**(self) -> 'List[np.ndarray]' (method)
- **encode**(self, X: 'np.ndarray') -> 'np.ndarray' (method): Encode state vectors (R x N values) as integers.
- **decode**(self, codes) -> 'np.ndarray' (method): Decode integers into state vectors (R x N values).
- **all_states**(self) -> 'np.ndarray' (method): Every configuration of the network (use only for small state spaces).
- **random_states**(self, n: 'int', rng=None) -> 'np.ndarray' (method)
- **initial_state**(self, rng=None) -> 'np.ndarray' (method): Initial states from node metadata; nodes without `initial` are drawn at random.
- **as_dict**(self, x: 'Sequence') -> 'Dict[str, int]' (method)
- **from_dict**(self, d: 'Mapping[str, int]') -> 'np.ndarray' (method)
- **copy**(self) -> "'PNMGraph'" (method)
- **summary**(self) -> 'str' (method)

## `bcpnm.rules`

Update-rule builders.

Deterministic (logical) rules
    `BooleanRule`, `CaseRule`, `ThresholdRule`, `TableRule`,
    `ClampRule`, `IdentityRule`.
Stochastic rules
    `SoftmaxRule` (the Blume–Capel transition rule), `NoisyRule`
    (random perturbation, as in probabilistic Boolean networks with perturbation) and
    `PBNRule` (random choice among several functions, as in probabilistic
    Boolean networks).
Wrappers
    `FixedInputRule` removes the influence of chosen inputs (edge knockout).

The Blume–Capel rule with `beta=np.inf` is deterministic: it becomes a ternary
threshold rule. This is how logical networks arise as the zero-noise limit of the
probabilistic model.

### class `CaseRule(cases: 'Sequence[Tuple[str, Number]]', default: 'Union[Number, str]' = 'keep')`

Multi-valued logical rule: the first condition that holds sets the next state.

Parameters
----------
cases : list of (str, int)
    Pairs `(condition, value)`; conditions use `and`, `or`, `not`,
    comparisons (`>=`, `==`, ...) and `+ - *`. Node names are variables;
    a bare name is true when its value is nonzero.
default : int or "keep"
    Next state when no condition holds; `"keep"` retains the current state.

Example
-------
`CaseRule([("Distress >= 2 and not Help", 2), ("Distress >= 1", 1)], default=0)`

- **next_value**(self, parents, current, values) (method)
- **params**(self) (method): Parameters for reporting and configuration export.

### class `BooleanRule(expr: 'str', true_value: 'Number' = 1, false_value: 'Number' = 0)`

Logical rule `expr` returning `true_value` when it holds and `false_value` otherwise.

Example: `BooleanRule("Stigma and not Support")`.

- **params**(self) (method): Parameters for reporting and configuration export.

### function `AND(*names: 'str') -> 'BooleanRule'`

Boolean conjunction of the named inputs.

### function `OR(*names: 'str') -> 'BooleanRule'`

Boolean disjunction of the named inputs.

### function `NOT(name: 'str') -> 'BooleanRule'`

Boolean negation of one input.

### class `ThresholdRule(weights: 'Mapping[str, float]', thresholds: 'Sequence[float]' = (0.0,), outputs: 'Sequence[Number]' = (0, 1), bias: 'float' = 0.0, tie: 'str' = 'keep')`

Weighted-sum threshold rule for Boolean, ternary or ordinal nodes.

The activation is $a = b + \sum_j w_j x_j$. With cut points
$\theta_1 < \dots < \theta_K$ and outputs $o_0, \dots, o_K$, the next
state is $o_k$ where $k$ is the number of cut points below $a$.
If $a$ equals a cut point, `tie="keep"` keeps the current state when it is
one of the two adjacent outputs (the convention of Hopfield-type threshold networks).

Parameters
----------
weights : mapping {input: weight}
thresholds : sequence of float, default `(0,)`
outputs : sequence of int, default `(0, 1)`; must have `len(thresholds) + 1` entries
bias : float
tie : {"keep", "low", "high"}

- **activation**(self, parents) (method): Weighted input sum; rounded to 9 decimals so that ties at a cut point do not
- **next_value**(self, parents, current, values) (method)
- **params**(self) (method): Parameters for reporting and configuration export.

### function `majority(inputs: 'Sequence[str]', weights: 'Optional[Sequence[float]]' = None, ternary: 'bool' = True) -> 'ThresholdRule'`

Weighted majority rule: sign of the weighted input sum (ties keep the current state).

### class `TableRule(inputs: 'Sequence[str]', table: 'Mapping[Tuple, Number]', default: 'Union[Number, str]' = 'keep')`

Explicit lookup table from input configurations to the next state.

Parameters
----------
inputs : sequence of str
table : mapping {tuple of input values: next value}
default : int or "keep"
    Used for configurations not listed in the table.

- **next_value**(self, parents, current, values) (method)
- **params**(self) (method): Parameters for reporting and configuration export.

### class `ClampRule(value: 'Number')`

Fix a node at a constant state (knockout or forced expression).

- **next_value**(self, parents, current, values) (method)
- **params**(self) (method): Parameters for reporting and configuration export.

### class `IdentityRule()`

Keep the current state (an exogenous node that changes only by intervention).

- **next_value**(self, parents, current, values) (method)

### class `SoftmaxRule(weights: 'Optional[Mapping[str, float]]' = None, lam: 'Optional[Mapping[str, float]]' = None, h: 'float' = 0.0, delta: 'float' = 0.0, rho: 'float' = 0.0, beta: 'float' = 1.0)`

Blume–Capel (polytomous) transition rule.

For candidate next state $s$, the utility is

$$u(s) = \beta\,[\, s\,\eta - \delta s^2 + \zeta\,\mathbb{1}(s=0) + \rho\,\mathbb{1}(s = x_{\text{current}}) \,],$$

with polarity field $\eta = h + \sum_j w_j x_j$ and zero-neighbour field
$\zeta = \sum_j \lambda_j\,\mathbb{1}(x_j = 0)$, and
$P(s) \propto \exp u(s)$. For states $\{-1,0,+1\}$,
$\mathbb{1}(x=0) = 1 - x^2$, which is the zero–zero coordination term of the
Blume–Capel probabilistic network model. With symmetric `weights` and `lam`
and `rho = 0`, asynchronous updating with this rule is heat-bath (Gibbs)
sampling of the equilibrium model. With `beta = np.inf` the rule is
deterministic (arg-max; ties keep the current state, then prefer 0).

- **deterministic** (property)
- **utilities**(self, parents, current, values, scale: 'Optional[float]' = None) -> 'np.ndarray' (method)
- **probs**(self, parents, current, values) (method): Probability of each next state (R x S).
- **next_value**(self, parents, current, values) (method)
- **with_beta**(self, beta: 'float') -> "'SoftmaxRule'" (method)
- **params**(self) (method): Parameters for reporting and configuration export.

### class `NoisyRule(base: 'Rule', p: 'float')`

With probability `p` the node takes a uniformly random state; otherwise `base` applies.

This is the random-perturbation mechanism of probabilistic Boolean networks
(Shmulevich, Dougherty, & Zhang, 2002) and makes every state reachable.

- **probs**(self, parents, current, values) (method): Probability of each next state (R x S).
- **params**(self) (method): Parameters for reporting and configuration export.

### class `PBNRule(components: 'Sequence[Rule]', probabilities: 'Sequence[float]')`

Probabilistic Boolean (or multi-valued) network rule.

At each update one of the `components` is chosen with the given selection
probabilities (Shmulevich, Dougherty, Kim, & Zhang, 2002). Components may have
different inputs; the rule's inputs are their union.

- **probs**(self, parents, current, values) (method): Probability of each next state (R x S).
- **params**(self) (method): Parameters for reporting and configuration export.

### class `FixedInputRule(base: 'Rule', fixed: 'Mapping[str, Number]')`

Wrap a rule so that selected inputs are replaced by constants (edge knockout).

- **deterministic** (property)
- **next_value**(self, parents, current, values) (method)
- **probs**(self, parents, current, values) (method): Probability of each next state (R x S).
- **params**(self) (method): Parameters for reporting and configuration export.

### function `deterministic_limit(rule: 'Rule') -> 'Rule'`

Zero-noise limit of a rule: Softmax with `beta=inf`; noise and PBN mixing removed.

For a `PBNRule` the component with the largest selection probability is kept.

## `bcpnm.dynamics`

Simulation of network dynamics.

Update schemes
--------------
`"synchronous"`
    All nodes update at once from the previous state (Kauffman, 1969).
`"asynchronous"`
    General asynchronous updating: at each step one node, chosen uniformly at random,
    updates (Garg et al., 2008). With symmetric Blume–Capel rules this is Gibbs sampling.
`"sequential"`
    Every node updates once per step in a fresh random order (a "sweep").

All simulations are vectorized over replicate runs and fully determined by `seed`.

### function `step(graph: 'PNMGraph', X: 'np.ndarray', scheme: 'str' = 'synchronous', rng=None, _cache=None) -> 'np.ndarray'`

Advance replicate states `X` (R x N) by one update step.

### class `Intervention(perturbation: 'object', start: 'int' = 0, end: 'Optional[int]' = None, name: 'str' = '') -> None`

A perturbation that is active during `start <= t < end` (`end=None`: until the end).

- **active**(self, t: 'int') -> 'bool' (method)

### class `Trajectory(states: 'np.ndarray', names: 'List[str]', scheme: 'str', seed: 'Optional[int]' = None, meta: 'dict' = <factory>) -> None`

Simulated states with shape `(T + 1, R, N)`: time x replicate run x node.

- **T** (property)
- **runs** (property)
- **node**(self, name: 'str') -> 'np.ndarray' (method): States of one node, shape `(T + 1, R)`.
- **prob**(self, value: 'float', node: 'Optional[str]' = None) -> 'np.ndarray' (method): Proportion of runs in `value` at each time: shape `(T + 1,)` for one node,
- **occupancy**(self, value: 'float') -> 'np.ndarray' (method): Proportion of nodes in `value` at each time, averaged over runs.
- **final**(self) -> 'np.ndarray' (method)
- **to_long**(self) (method): Long-format `pandas.DataFrame` with columns time, run, node, state.
- **to_panel**(self, run: 'int' = 0) -> 'np.ndarray' (method): States of one run as a `(T + 1) x N` array (e.g. for inference).

### function `simulate(graph: 'PNMGraph', x0: 'Optional[np.ndarray]' = None, steps: 'int' = 100, runs: 'int' = 1, scheme: 'str' = 'synchronous', seed: 'Optional[int]' = None, interventions: 'Sequence[Intervention]' = (), record_every: 'int' = 1) -> 'Trajectory'`

Simulate `runs` replicate trajectories of length `steps`.

Parameters
----------
graph : PNMGraph
x0 : array, optional
    Initial state: a vector (N,) used for every run, an array (runs x N), or
    `None` for node `initial` values (random where not set), drawn per run.
steps : int
runs : int
scheme : {"synchronous", "asynchronous", "sequential"}
seed : int, optional
    Seed of the random stream; the same seed reproduces the same trajectories.
interventions : sequence of `Intervention`
    Time-windowed perturbations (see `perturbation`).
record_every : int
    Keep every `k`-th state to save memory.

### function `successors(graph: 'PNMGraph', X: 'np.ndarray') -> 'np.ndarray'`

Deterministic synchronous successors of states `X` (requires deterministic rules).

## `bcpnm.attractors`

Attractors, basins and stationary distributions.

* Deterministic networks: exhaustive or sampled attractor search under synchronous
  updating (fixed points and cycles; Kauffman, 1969) and exhaustive search under
  general asynchronous updating (terminal strongly connected components of the state
  transition graph; Garg et al., 2008).
* Stochastic networks: exact stationary distribution for small state spaces and the
  probability mass carried by the basins of the zero-noise ("quasi-") attractors.

### class `Attractor(states: 'np.ndarray', kind: 'str', basin: 'int', basin_fraction: 'float', weak_basin: 'Optional[int]' = None, label: 'str' = '') -> None`

A fixed point, synchronous cycle, or asynchronous complex attractor.

- **size** (property)
- **as_dicts**(self, names) -> 'List[Dict[str, int]]' (method)

### class `AttractorResult(attractors: 'List[Attractor]', names: 'List[str]', scheme: 'str', method: 'str', n_states: 'int', basin_map: 'Optional[np.ndarray]' = None) -> None`

Attractors of a deterministic network with basin information; iterate over it, or
use `summary()`, `table()` and `label_by()`.

- **table**(self) (method): Attractors as a `pandas.DataFrame` (one row per attractor).
- **label_by**(self, fn: 'Callable[[np.ndarray], str]') -> "'AttractorResult'" (method): Attach labels computed from each attractor's states (e.g. 'all silent').
- **summary**(self) -> 'str' (method)

### function `deterministic_graph(graph: 'PNMGraph') -> 'PNMGraph'`

Copy of `graph` with every rule replaced by its zero-noise limit.

### function `async_transition_graph(graph: 'PNMGraph') -> 'sparse.csr_matrix'` *(in `bcpnm.attractors`)*

Sparse adjacency of the asynchronous state transition graph (deterministic rules).

### function `find_attractors(graph: 'PNMGraph', scheme: 'str' = 'synchronous', method: 'str' = 'auto', samples: 'int' = 2000, max_steps: 'int' = 500, seed=None, exhaustive_limit: 'int' = 2000000) -> 'AttractorResult'`

Find the attractors of a deterministic network.

Parameters
----------
scheme : {"synchronous", "asynchronous"}
method : {"auto", "exhaustive", "sampling"}
    `"auto"` enumerates the full state space when it has at most
    `exhaustive_limit` configurations (200,000 for asynchronous search).
samples, max_steps : int
    Random starts and maximal path length for synchronous sampling.
seed : int, optional

### class `StationaryResult(pi: 'np.ndarray', graph: 'PNMGraph', scheme: 'str', iterations: 'int', X: 'np.ndarray' = None) -> None`

Stationary distribution over all configurations of a stochastic network.

- **marginal**(self, node: 'str') -> 'Dict[int, float]' (method)
- **prob**(self, predicate: 'Callable[[np.ndarray], np.ndarray]') -> 'float' (method): Probability of the event `predicate(X)` (a boolean vector over configurations).
- **expectation**(self, fn: 'Callable[[np.ndarray], np.ndarray]') -> 'float' (method)
- **top**(self, k: 'int' = 5) (method)

### function `transition_matrix(graph: 'PNMGraph', scheme: 'str' = 'asynchronous') -> 'sparse.csr_matrix'`

Exact Markov transition matrix over all configurations (small networks only).

### function `stationary_distribution(graph: 'PNMGraph', scheme: 'str' = 'asynchronous', tol: 'float' = 1e-12, max_iter: 'int' = 200000) -> 'StationaryResult'`

Stationary distribution by power iteration on the lazy chain `(I + P) / 2`.

### function `quasi_attractors(graph: 'PNMGraph', stationary: 'Optional[StationaryResult]' = None, scheme: 'str' = 'asynchronous') -> 'AttractorResult'`

Attractors of the zero-noise limit, with basin mass under the stochastic model.

`basin_fraction` is replaced by the stationary probability of the attractor's
(strong) basin, i.e. the long-run share of time spent in that regime.

## `bcpnm.equilibrium`

Equilibrium Blume–Capel probabilistic network model (BC PNM).

States $x_i \in \{-1, 0, +1\}$; symmetric polarity couplings $K$,
symmetric zero–zero coordination $\Lambda$, zero-state stability
$\delta_i$, field $h_i$ and concentration $\beta$:

$$H(\mathbf{x}) = -\sum_{i<j} K_{ij} x_i x_j - \sum_{i<j} \Lambda_{ij}(1-x_i^2)(1-x_j^2) + \sum_i \delta_i x_i^2 - \sum_i h_i x_i, \qquad P(\mathbf{x}) = \exp[-\beta H(\mathbf{x})] / Z.$$

The module provides exact enumeration (small N), Gibbs sampling, conditional
probabilities, homogeneous mean-field equations and continuation sweeps for hysteresis.

### class `BlumeCapelModel(names: 'Sequence[str]', K=None, Lam=None, delta=0.0, h=0.0, beta: 'float' = 1.0, labels: 'Sequence[str]' = ('negative', 'neutral', 'positive'))`

Equilibrium Blume–Capel PNM.

Parameters
----------
names : sequence of str
K, Lam : (N, N) symmetric arrays, optional
    Polarity coupling and zero–zero coordination (zero diagonal).
delta, h : float or (N,) arrays
beta : float
labels : sequence of three str
    Labels for the states -1, 0, +1.

- **N** (property)
- **from_edges**(cls, names, edges: 'Sequence[Tuple[str, str, float, float]]', **kwargs) -> "'BlumeCapelModel'" (method): Build from `(i, j, K_ij, Lam_ij)` tuples (undirected).
- **energy**(self, X) -> 'np.ndarray' (method): Energy $H(\mathbf{x})$ of configurations (R x N).
- **configurations**(self) -> 'np.ndarray' (method)
- **exact**(self) -> 'Tuple[np.ndarray, np.ndarray]' (method): All configurations and their exact Boltzmann probabilities.
- **conditional**(self, X, i: 'int') -> 'np.ndarray' (method): $P(X_i = s \mid \mathbf{x}_{-i})$ for s in (-1, 0, 1); shape (R, 3).
- **log_odds_zero**(self, X, i: 'int') -> 'np.ndarray' (method): Log-odds of state 0 versus +1 and versus -1 (columns), i.e.
- **beg_form**(self) -> 'Dict[str, np.ndarray]' (method): Equivalent Blume–Emery–Griffiths parameters.
- **occupancy**(X, value: 'float' = 0.0, weights=None) -> 'float' (method): Expected proportion of nodes in `value` (e.g. 0 = non-disclosure).
- **co_occupancy**(self, X, value: 'float' = 0.0, weights=None, adjacency=None) -> 'float' (method): Expected proportion of ties whose two ends are both in `value` (connected co-silence).
- **marginals**(self) -> 'np.ndarray' (method): Exact marginal probabilities P(x_i = s), shape (N, 3).
- **to_graph**(self, name: 'str' = 'blume_capel') -> 'PNMGraph' (method): Equivalent `PNMGraph` with heat-bath `SoftmaxRule` per node.
- **gibbs**(self, sweeps: 'int' = 1000, burn: 'int' = 100, chains: 'int' = 4, seed=None, x0=None) -> 'np.ndarray' (method): Heat-bath Gibbs samples after burn-in, shape `(chains * (sweeps - burn), N)`.

### function `mean_field(beta: 'float', J: 'float', Lam: 'float', delta: 'float', h: 'float' = 0.0, m0: 'float' = 0.0, q0: 'float' = 0.67, tol: 'float' = 1e-11, max_iter: 'int' = 20000, damping: 'float' = 0.5) -> 'Tuple[float, float]'`

Homogeneous mean-field fixed point `(m, q)` for a complete network with total
couplings `J` and `Lam` (each normalized by N - 1).

$m = 2e^{-\beta d}\sinh(\beta\eta)/[1 + 2e^{-\beta d}\cosh(\beta\eta)]$,
$q = 2e^{-\beta d}\cosh(\beta\eta)/[1 + 2e^{-\beta d}\cosh(\beta\eta)]$,
with $\eta = h + Jm$ and $d = \delta + \Lambda(1-q)$.

### function `mean_field_free_energy(m: 'float', q: 'float', beta: 'float', J: 'float', Lam: 'float', delta: 'float', h: 'float' = 0.0) -> 'float'` *(in `bcpnm.equilibrium`)*

Variational free energy per node at the product distribution implied by `(m, q)`.

### function `mean_field_global(beta, J, Lam, delta, h=0.0) -> 'Tuple[float, float]'`

Globally stable mean-field solution (lowest free energy over several starts).

### class `Sweep(values: 'np.ndarray', up: 'np.ndarray', down: 'np.ndarray', parameter: 'str') -> None` *(in `bcpnm.equilibrium`)*

Result of a continuation sweep: parameter values and (m, q) going up and down.

- **jumps**(self, column: 'int' = 0, min_jump: 'float' = 0.2) -> 'Tuple[Optional[float], Optional[float]]' (method): Parameter values at which the upward and downward branches jump.

### function `continuation(parameter: 'str', values: 'Sequence[float]', beta: 'float', J: 'float', Lam: 'float', delta: 'float', h: 'float' = 0.0) -> 'Sweep'`

Follow the mean-field solution as `parameter` ("h" or "delta") increases, then decreases.

Differences between the two branches indicate hysteresis (path dependence).

### function `phase_diagram(betas: 'Sequence[float]', deltas: 'Sequence[float]', J: 'float' = 1.0, Lam: 'float' = 0.0, h: 'float' = 0.0) -> 'np.ndarray'`

|m*| of the globally stable mean-field solution on a (delta x beta) grid.

## `bcpnm.perturbation`

Perturbations and intervention analysis.

A perturbation returns a modified copy of a `PNMGraph`; the original
is never changed. Perturbations can be applied permanently, compared through their
effect on attractors and basins, or scheduled in time with
`Intervention`.

Psychological readings
    * `Clamp` - hold a construct or person at a state (e.g. a facilitator who
      always acknowledges; a symptom suppressed by treatment). With `value=0` in a
      Boolean model this is a "knockout"; with the highest value it is "overexpression".
    * `RemoveEdge`, `ScaleEdge`, `FlipEdge` - remove, weaken or reverse
      an influence (e.g. a confidential channel that cuts a hierarchical pressure).
    * `ShiftField` - change the contextual field or bias of a node (e.g. reduced
      anticipated stigma), and `SetParameter` for any rule parameter.

### class `Perturbation()` *(in `bcpnm.perturbation`)*

Base class: a perturbation maps a graph to a modified copy (`apply` or call).

- **apply**(self, graph: 'PNMGraph') -> 'PNMGraph' (method)

### class `Clamp(node: 'str', value: 'float') -> None`

Fix `node` at `value`.

- **apply**(self, graph) (method)

### function `knockout(node: 'str', value: 'float' = 0) -> 'Clamp'`

Hold a node at its 'off' (or other) state.

### function `overexpress(graph: 'PNMGraph', node: 'str') -> 'Clamp'`

Hold a node at its highest state.

### class `RemoveEdge(source: 'str', target: 'str', neutral: 'float' = 0.0) -> None`

Remove the influence of `source` on `target`.

For weighted rules the weight (and zero–zero term) is set to zero; for other rules
the input is replaced by the constant `neutral` value.

- **apply**(self, graph) (method)

### class `ScaleEdge(source: 'str', target: 'str', factor: 'float', include_lam: 'bool' = True, include_w: 'bool' = True) -> None`

Multiply the influence of `source -> target` by `factor` (weighted rules only).

`include_w` and `include_lam` choose whether the polarity weight, the zero–zero
term, or both are scaled (e.g. `include_w=False` weakens only the coordination of
silence along a tie).

- **apply**(self, graph) (method)

### class `FlipEdge(source: 'str', target: 'str') -> None`

Reverse the sign of `source -> target` (weighted rules only).

- **apply**(self, graph) (method)

### class `ShiftField(node: 'str', amount: 'float') -> None`

Add `amount` to the field `h` (SoftmaxRule) or `bias` (ThresholdRule) of `node`.

- **apply**(self, graph) (method)

### class `SetParameter(node: 'str', parameter: 'str', value: 'float') -> None`

Set a scalar rule attribute, e.g. `SetParameter("Mother", "delta", 0.2)`.

- **apply**(self, graph) (method)

### class `Combined(parts: 'Sequence[Perturbation]') -> None`

Apply several perturbations in order.

- **apply**(self, graph) (method)

### function `compare_attractors(base: 'AttractorResult', perturbed: 'AttractorResult', key: 'Callable[[np.ndarray], str]' = None)`

Basin fractions before and after a perturbation, matched by attractor label
(or by the attractor's states when no labels are set). Returns a DataFrame.

### function `leverage_scan(graph: 'PNMGraph', target: 'Callable[[np.ndarray], bool]', value_fn: 'Callable[[str], float]', scheme: 'str' = 'synchronous', nodes: 'Optional[Sequence[str]]' = None)`

Clamp each node in turn and report the share of initial states whose attractor
satisfies `target` (e.g. 'most members acknowledge').

`value_fn(node)` gives the clamp value for each node. Returns a DataFrame sorted
by the resulting share.

### function `intervention_effect(graph: 'PNMGraph', perturbation: 'Perturbation', outcome: 'Callable[[np.ndarray], np.ndarray]', steps: 'int' = 200, runs: 'int' = 2000, scheme: 'str' = 'asynchronous', seed: 'int' = 0, burn: 'int' = 50, x0=None) -> 'Dict[str, float]'`

Mean of `outcome(states)` after burn-in without and with a sustained perturbation,
using common random numbers (the same seed) for both conditions.

## `bcpnm.inference`

Learning update rules and parameters from data.

* `reveal` - REVEAL-style search for the smallest input sets that determine each
  node's next state (Liang, Fuhrman, & Somogyi, 1998), returning lookup-table rules.
* `fit_transition_model` - penalized maximum likelihood for the directed
  Blume–Capel transition rule (multinomial logit per node) from time series.
* `fit_equilibrium` - nodewise pseudo-likelihood for the symmetric equilibrium
  model from cross-sectional data (cf. Waldorp et al., 2026), symmetrized afterwards.
* `log_score` - mean one-step log predictive probability, for held-out comparison.

### class `RevealResult(parents: 'Dict[str, Tuple[str, ...]]', determination: 'Dict[str, float]', accuracy: 'Dict[str, float]', tables: 'Dict[str, TableRule]') -> None` *(in `bcpnm.inference`)*

Inferred input sets, information-theoretic determination, in-sample accuracy and
lookup-table rules per node; `to_graph(template)` builds a deterministic model.

- **to_graph**(self, template: 'PNMGraph', name: 'str' = 'inferred') -> 'PNMGraph' (method)

### function `reveal(data: 'ArrayList', names: 'Sequence[str]', max_parents: 'int' = 3, criterion: 'str' = 'bic', min_determination: 'float' = 0.95) -> 'RevealResult'`

REVEAL-style inference of input sets and lookup tables from synchronous time series.

Parameters
----------
data : array (T x N), list of arrays, or array (T x R x N)
names : node names in column order
max_parents : int
    Largest input set searched.
criterion : {"reveal", "bic"}
    `"reveal"`: the smallest set whose mutual information with the next state is at
    least `min_determination` of that state's entropy (exact for noise-free data).
    `"bic"`: minimizes `n * H(y | inputs) * ln 2 + 0.5 * n_params * ln n` and is
    preferable for noisy data.

### class `TransitionFit(names: 'List[str]', W: 'np.ndarray', Lam: 'np.ndarray', h: 'np.ndarray', delta: 'np.ndarray', rho: 'np.ndarray', converged: 'bool', meta: 'dict' = <factory>) -> None` *(in `bcpnm.inference`)*

Estimates of the directed Blume–Capel rule (on the scale beta * parameter):
`W[i, j]` and `Lam[i, j]` are influences of j on i; `to_graph()` builds the model.

- **to_graph**(self, states: 'StateSpace' = None, name: 'str' = 'fitted', beta: 'float' = 1.0, threshold: 'float' = 0.0) -> 'PNMGraph' (method): Graph with `SoftmaxRule` per node; edges with |w| and |lam| at or below

### function `fit_transition_model(data: 'ArrayList', names: 'Sequence[str]', values: 'Sequence[float]' = (-1, 0, 1), lam: 'bool' = True, rho: 'bool' = True, alpha: 'float' = 0.001) -> 'TransitionFit'`

Fit the directed Blume–Capel transition rule to synchronous time series.

The concentration $\beta$ is fixed at 1 for identification, so estimates are
on the scale $\beta \cdot$ parameter. `alpha` is an L2 penalty on the
network terms (`W` and `Lam`).

### class `EquilibriumFit(names: 'List[str]', K: 'np.ndarray', Lam: 'np.ndarray', h: 'np.ndarray', delta: 'np.ndarray', converged: 'bool') -> None` *(in `bcpnm.inference`)*

Symmetrized pseudo-likelihood estimates of the equilibrium model; `to_model()`
returns a `BlumeCapelModel`.

- **to_model**(self, beta: 'float' = 1.0) (method)

### function `fit_equilibrium(samples: 'np.ndarray', names: 'Sequence[str]', lam: 'bool' = True, alpha: 'float' = 0.001) -> 'EquilibriumFit'`

Nodewise pseudo-likelihood for the equilibrium Blume–Capel model (beta = 1).

Each node's full conditional (Equation 5 of the tutorial) is a multinomial logit in
the other nodes; the two estimates of each symmetric coupling are averaged.

### function `log_score(graph: 'PNMGraph', data: 'ArrayList') -> 'float'`

Mean log predictive probability of observed synchronous transitions (nats per node).

## `bcpnm.io`

Configuration files (YAML/JSON) and data import/export.

A configuration fully specifies a model so that it can be shared, versioned and
re-run. Minimal example (YAML)::

    meta: {name: toy, domain: family communication, seed: 1}
    state_spaces:
      talk: {values: [-1, 0, 1], labels: {-1: conceal, 0: silent, 1: acknowledge}}
    nodes:
      - {name: A, states: talk, construct: "parent's communication"}
      - {name: B, states: talk}
    model:
      type: softmax          # softmax | threshold | logical | table
      beta: 1.0
      defaults: {delta: 0.3, h: 0.0, rho: 0.0}
      edges:                 # weights w and zero–zero terms lam, source -> target
        - {source: A, target: B, w: 0.8, lam: 0.5}
      symmetric: true        # also add target -> source with the same values
    dynamics: {scheme: asynchronous, steps: 200, runs: 500}

See `examples/*.yaml` for complete Boolean, threshold and Blume–Capel models.

### function `load_config(path: 'PathLike') -> 'Dict[str, Any]'`

Read a YAML or JSON configuration file into a dictionary.

### function `save_config(cfg: 'Dict[str, Any]', path: 'PathLike') -> 'None'`

Write a configuration dictionary to YAML (`.yaml`/`.yml`) or JSON.

### function `build_model(cfg: 'Dict[str, Any]') -> 'PNMGraph'`

Construct a `PNMGraph` from a configuration dictionary.

Two styles are supported and can be mixed:

* `model.type: softmax` with an `edges` list builds Blume–Capel transition rules
  for every node (`symmetric: true` makes the network undirected, i.e. an
  equilibrium Blume–Capel model under asynchronous updating);
* a `rules` mapping gives an explicit rule per node (`boolean`, `case`,
  `threshold`, `table`, `clamp`, `identity` or `softmax`), overriding the
  edge-based rule for that node.

### function `load_model(path: 'PathLike') -> 'Tuple[PNMGraph, Dict[str, Any]]'`

Load a configuration file and build its model. Returns `(graph, config)`.

### function `graph_to_config(graph: 'PNMGraph') -> 'Dict[str, Any]'`

Export a graph to a configuration dictionary (rules written explicitly per node).

### function `save_trajectory(traj, path: 'PathLike') -> 'None'` *(in `bcpnm.io`)*

Write a `Trajectory` in long CSV format.

### function `read_panel(path: 'PathLike', names=None) -> 'np.ndarray'` *(in `bcpnm.io`)*

Read a wide CSV (rows = measurement occasions, columns = nodes) into a T x N array.

## `bcpnm.validation`

Checks of model validity, sensitivity and robustness.

* `expert_consistency` - does each specified influence have the sign experts expect?
* `input_activities` / `average_sensitivity` - how strongly each input moves
  each rule (Shmulevich & Kauffman, 2004), generalized to multi-valued and stochastic rules.
* `canalizing_inputs` - input values that fix a rule's output regardless of the
  other inputs (Kauffman et al., 2003).
* `derrida_curve` - one-step spreading of small differences (Derrida & Pomeau, 1986).
* `attractor_robustness` - share of single-node perturbations of an attractor after
  which the network returns to the same attractor.
* `threshold_ties` - input configurations at which a threshold rule sits exactly on
  a cut point, so that results depend on the tie-breaking convention.
* `parameter_recovery` and `oat_sensitivity` - simulation-based checks.

### function `expected_next(graph: 'PNMGraph', node: 'str', X: 'np.ndarray') -> 'np.ndarray'` *(in `bcpnm.validation`)*

Expected next value of `node` given states `X` (R x N).

### function `expert_consistency(graph: 'PNMGraph', expectations: 'Sequence[Tuple[str, str, int]]', samples: 'int' = 2000, seed=0)`

Compare model-implied influence signs with expert expectations.

For each `(source, target, expected_sign)` the source is raised by one state
(where possible) in random reference configurations, and the average change in the
target's expected next value is computed. Returns a DataFrame with the implied
effect, its sign and agreement, and the proportion of agreements in `df.attrs`.

### function `input_activities(graph: 'PNMGraph', samples: 'int' = 4000, seed=0) -> 'Dict[str, Dict[str, float]]'`

Activity of each input on each rule.

For deterministic rules: probability that changing the input to another random
value changes the output (the activity of Shmulevich & Kauffman, 2004). For
stochastic rules: mean total-variation distance between next-state distributions.

### function `average_sensitivity(graph: 'PNMGraph', samples: 'int' = 4000, seed=0) -> 'Dict[str, float]'`

Sum of input activities per node, and their mean over nodes (key `"mean"`).

### function `canalizing_inputs(graph: 'PNMGraph', node: 'str') -> 'list'`

Input values that alone determine a deterministic rule's output.

Returns a list of `(input, input_value, forced_output)`. The node's own current
value is included as an input when the rule depends on it.

### function `derrida_curve(graph: 'PNMGraph', max_distance: 'Optional[int]' = None, samples: 'int' = 2000, seed=0, scheme: 'str' = 'synchronous') -> 'Dict[int, float]'`

Mean Hamming distance after one update between states that initially differ in
`d` nodes. Values above `d` (slope > 1 at d = 1) indicate that small
differences grow (sensitive regime); values below indicate that they shrink.

### function `attractor_robustness(graph: 'PNMGraph', result: 'Optional[AttractorResult]' = None) -> 'Dict[int, float]'`

For each synchronous attractor, the share of single-node changes to its states
after which the network returns to the same attractor (requires exhaustive search).

### function `threshold_ties(graph: 'PNMGraph')`

Count input configurations at which a `ThresholdRule` activation
equals a cut point. Nonzero counts mean that results depend on the `tie` convention;
shifting the bias by a small amount removes such knife-edge cases.

### function `parameter_recovery(true: 'Mapping[str, np.ndarray]', estimated: 'Mapping[str, np.ndarray]')`

Correlation, RMSE and sign agreement between true and estimated parameter arrays.

### function `oat_sensitivity(outcome: 'Callable[[Dict[str, float]], float]', params: 'Mapping[str, float]', rel_change: 'float' = 0.1)`

One-at-a-time elasticities: % change in `outcome` per % change in each parameter.

## `bcpnm.visualization`

Plots for networks, trajectories, attractors and phase behaviour (Matplotlib).

Colours: for ternary models, blue = +1, grey = 0, red = -1 (a diverging scheme with a
neutral midpoint); for Boolean and ordinal models, a single-hue blue ramp. Every
function returns the Matplotlib `Axes` so figures can be customized and saved.

### function `state_colors(values: 'Sequence[float]') -> 'Dict[float, str]'` *(in `bcpnm.visualization`)*

Colours per state value: diverging for (-1, 0, +1), light-to-dark otherwise.

### function `plot_network(graph: 'PNMGraph', state: 'Optional[Sequence[float]]' = None, ax=None, pos=None, weight_attr: 'str' = 'weight', seed: 'int' = 1, node_size: 'int' = 1100, font_size: 'int' = 9, labels: 'Optional[Dict[str, str]]' = None)` *(in `bcpnm.visualization`)*

Draw the network; arrows show influence, line width |weight|, colour the sign
(dark = positive, red dashed = negative). Nodes are coloured by `state` if given.

### function `plot_trajectory(traj, run: 'int' = 0, ax=None, labels: 'Optional[Dict[float, str]]' = None)` *(in `bcpnm.visualization`)*

Raster of node states over time for one run.

### function `plot_probabilities(traj, value: 'float', nodes: 'Optional[Sequence[str]]' = None, ax=None, shade=None)` *(in `bcpnm.visualization`)*

Proportion of runs in `value` over time, one line per node.

### function `plot_basins(result, ax=None, labels: 'Optional[Sequence[str]]' = None, title: 'str' = '')` *(in `bcpnm.visualization`)*

Horizontal bars of basin fractions (or stationary basin mass) per attractor.

### function `plot_state_transition_graph(graph: 'PNMGraph', scheme: 'str' = 'synchronous', ax=None, max_states: 'int' = 300, result=None, seed: 'int' = 2)` *(in `bcpnm.visualization`)*

State transition graph of a small deterministic network, attractor states highlighted.

### function `plot_hysteresis(sweep, column: 'int' = 0, ax=None, xlabel: 'Optional[str]' = None, ylabel: 'Optional[str]' = None)` *(in `bcpnm.visualization`)*

Upward and downward continuation branches of a mean-field sweep.

### function `plot_phase_diagram(M: 'np.ndarray', betas, deltas, ax=None)` *(in `bcpnm.visualization`)*

Heat map of |m*| over (beta, delta) from `phase_diagram`.

### function `plot_derrida(curve: 'Dict[int, float]', ax=None)` *(in `bcpnm.visualization`)*

Derrida plot with the diagonal as reference.

## `bcpnm.reporting`

Reporting helpers: model cards, draft methods/results text and the PNM reporting checklist.

The checklist adapts reporting items from STROBE (von Elm et al., 2007), PRISMA 2020
(Page et al., 2021) and the ODD protocol for simulation models (Grimm et al., 2020)
to probabilistic network models. `write_report` produces a Markdown file with
the model card, analysis settings, software environment and the completed checklist,
which can be attached as supplementary material.

### function `checklist(as_frame: 'bool' = True)`

The PNM reporting checklist as a `pandas.DataFrame` (or list of dicts).

### function `node_table(graph: 'PNMGraph')` *(in `bcpnm.reporting`)*

Nodes, state spaces and constructs as a `pandas.DataFrame`.

### function `rule_table(graph: 'PNMGraph')` *(in `bcpnm.reporting`)*

Rules, inputs and parameters as a `pandas.DataFrame`.

### function `model_card(graph: 'PNMGraph') -> 'str'`

Markdown description of a model (nodes, state spaces, rules).

### function `methods_text(graph: 'PNMGraph', scheme: 'str' = 'asynchronous', runs: 'Optional[int]' = None, steps: 'Optional[int]' = None, seed: 'Optional[int]' = None, extra: 'str' = '') -> 'str'`

Draft a methods paragraph describing the model and simulation settings.

The text is a starting point: authors should edit it and add the substantive
justification of nodes and rules.

### function `results_text(attractors=None, stationary=None, value_labels: 'Optional[Mapping[int, str]]' = None) -> 'str'`

Draft a results sentence from an `AttractorResult` and/or a
`StationaryResult`.

### function `write_report(path, graph: 'PNMGraph', config: 'Optional[dict]' = None, sections: 'Optional[Mapping[str, str]]' = None, completed: 'Optional[Mapping[str, str]]' = None) -> 'Path'`

Write a Markdown report: model card, extra sections, environment and checklist.

Parameters
----------
path : output file (`.md`)
graph : the model
config : the configuration used (embedded as JSON for reproducibility)
sections : mapping of heading -> Markdown text (e.g. results)
completed : mapping of checklist item number (e.g. "5") -> where/how it is reported

## `bcpnm.utils`

Small utilities: random number generation, state encoding, softmax, environment capture.

### function `make_rng(seed: 'int | np.random.Generator | None' = None) -> 'np.random.Generator'`

Return a NumPy `Generator`.

Passing an integer gives a reproducible stream; passing an existing generator
returns it unchanged so that functions can share one stream.

### function `softmax(u: 'np.ndarray', axis: 'int' = -1) -> 'np.ndarray'` *(in `bcpnm.utils`)*

Numerically stable softmax along `axis`.

### function `sample_categorical(p: 'np.ndarray', rng: 'np.random.Generator') -> 'np.ndarray'` *(in `bcpnm.utils`)*

Draw one index per row from row-wise probability vectors `p` (shape R x S).

### function `radices(sizes: 'Sequence[int]') -> 'np.ndarray'` *(in `bcpnm.utils`)*

Place values for mixed-radix encoding of states (last node varies fastest).

### function `encode_indices(idx: 'np.ndarray', sizes: 'Sequence[int]') -> 'np.ndarray'` *(in `bcpnm.utils`)*

Encode per-node state indices (R x N) into single integers (R,).

### function `decode_indices(codes: 'np.ndarray | int', sizes: 'Sequence[int]') -> 'np.ndarray'` *(in `bcpnm.utils`)*

Decode integers into per-node state indices (R x N).

### function `environment_report(packages: 'Iterable[str]' = ('numpy', 'scipy', 'networkx', 'matplotlib', 'yaml')) -> 'dict'`

Versions of Python, the operating system and key packages, for reporting.

# Design notes

These notes give the derivations behind the code and the reason for each choice, so that every
decision can be defended. Module names refer to `src/splat_navmap/`.

## 1. What is measured, and against what

A map is only as good as the decisions it supports. The benchmark therefore scores two things for
every extraction setting:

* **cells** (`evaluate.score_map`): precision, recall and IoU of the occupied cells against the
  exact occupancy of the height band swept by the robot. Cells strictly inside a solid are ignored
  (all eight neighbours occupied), because a splat only models visible surfaces and nothing can be
  planned through them.
* **paths** (`evaluate.evaluate_paths`): for start-goal pairs that are feasible in the true
  geometry, plan on the extracted grid and classify the outcome as *unreachable* (no path, or an
  endpoint is blocked), *collision* (the robot disc swept along the path comes closer than its
  radius to a true solid) or *valid*. Valid paths also report their length relative to the
  optimum on the exact grid.

The ground truth is analytic (`geometry.py`): boxes and vertical cylinders. A cell is occupied when
its square overlaps a footprint with positive area, and the clearance of a path is the exact
distance between its segments and the footprints. Nothing in the truth goes through sampling.

## 2. Why the discretisation cannot cause a collision

`grid.inflate` blocks a cell when *any point of its square* is closer than the robot radius to *any
point of an occupied square*. The distance between two cell squares offset by `(dx, dy)` cells is
`res * hypot(max(|dx| - 1, 0), max(|dy| - 1, 0))`, which is the Euclidean distance transform to the
occupied set grown by one cell in the 8-neighbourhood. If every true obstacle lies inside occupied
cells, a robot centre anywhere in an unblocked cell then keeps a true clearance of at least the
radius. Paths planned on the *exact* grid therefore never collide (this is a test, and the
benchmark asserts it for every scene), so every collision measured on an extracted grid is caused by
the map and not by rounding. The price is a conservative robot: it needs about one cell more room
than strictly necessary.

Planner paths are polylines through cell centres. For A*, with corner cutting forbidden, consecutive
cells are 8-neighbours and the segment stays inside unblocked cells. For Theta* the segment between
two path vertices must pass the supercover line-of-sight test (every cell whose closed square meets
the segment is unblocked), so the same guarantee holds.

## 3. The synthetic flat

`scene.py` splits a rectangle of 9-12 m by 7-9 m into up to 4-6 rooms with a binary space partition. Each
cut adds one wall with one door of 0.85-1.0 m, so the room graph is a tree and the flat is connected.
Furniture is chosen to exercise the height band of the extraction rather than for realism: a table
whose top (0.72 m) is above the robot but whose legs are not, low crates, a wall cabinet floating
between 0.25 and 0.65 m, a plant whose foliage overhangs the robot, stools (curved footprint) and a
1 cm rug below the robot's ground clearance. Layouts are accepted only when at least 97 % of the
free space (after inflation) is mutually reachable.

## 4. The Gaussian "reconstruction" and its defects

`synth.reconstruct` covers every visible face with flat, surface-aligned Gaussians (surfels) on a
jittered lattice: two tangential standard deviations proportional to the sampling step and a thin
one along the normal. It is a stand-in for a trained scene, not a trained scene, and the conclusions
hold for this defect model only. Four defects are controlled by explicit knobs, each drawn from its
own random stream so that switching one on or off leaves the others unchanged:

* **floaters**: semi-transparent anisotropic blobs placed uniformly in the volume of the flat;
* **holes**: balls in which nothing was reconstructed, centred on low surface points (regions no
  camera saw);
* **weak surfaces**: spatially coherent patches whose opacity is drawn in `[0.05, 0.45]` instead of
  `[0.90, 0.99]`; the patches come from a random-Fourier-feature approximation of a
  squared-exponential field (Rahimi and Recht, 2007);
* **thickness**: the density profile across a surface has standard deviation `t`. Half of the
  variance comes from the scale of each Gaussian along the normal and half from scattering the
  centres by `t / sqrt(2)`, so neither a centre-based nor an extent-based extractor is favoured by
  construction.

## 5. Two ways to turn Gaussians into a grid

Both return a score in `[0, 1]` per cell and the grid is `score >= tau`.

**Centre counting** (`extract.centre_score`) is the first thing one writes: the score of a cell is the
largest opacity among the Gaussians whose centre lies in the cell and in the height band.

**Footprint accumulation** (`extract.footprint_score`) renders the band from above with the 3DGS
image-formation model. Along the vertical ray through `(x, y)`, a Gaussian with opacity `alpha`
contributes

    a(x, y) = alpha * exp(-1/2 d^T S_xy^-1 d) * w(x, y),      d = (x, y) - mu_xy

where `S_xy` is the 2x2 marginal covariance (orthographic EWA splatting, Zwicker et al. 2001) and
`w` is the share of the ray's density that lies in the band `[z_lo, z_hi]`. Conditioning the 3D
Gaussian on `(x, y)` (Schur complement) gives a 1D Gaussian along the ray,

    m = mu_z + S_z,xy S_xy^-1 d,        s^2 = S_zz - S_z,xy S_xy^-1 S_xy,z

so `w = Phi((z_hi - m) / s) - Phi((z_lo - m) / s)` exactly (`Phi` is the standard normal CDF). A
test compares this closed form with numerical quadrature of the 3D density along each ray, and the
band mass with Monte-Carlo sampling.

Opacities accumulate as `1 - prod_i (1 - a_i)`. This is front-to-back alpha compositing with the
order dropped, which is exact here because only the total is needed. Rays are cast on a lattice
three times finer than the grid and a cell takes the value of its most opaque ray. A surface
thinner than the ray spacing could fall between two rays, so every footprint is widened by half
that spacing (`S_xy + (step / 2)^2 I`), the same low-pass dilation reference rasterisers apply in
screen space. A thin wall midway between two rays keeps `exp(-1/2) = 61 %` of its opacity.

The two methods behave differently by construction. Centre counting sees a wall through its few
centres per cell and is blind to a Gaussian that extends into a cell from outside; footprint
accumulation sees the whole extent, which makes it more complete and also fatter by about one cell
at low thresholds.

## 6. Planners

`planners.py` implements A* with the octile heuristic, which is consistent on an 8-connected grid,
so the first expansion of the goal is optimal (tested against Dijkstra on random grids). A diagonal
move needs both orthogonal neighbours free. Because of that rule, 8-connected planning reaches
exactly the 4-connected component of the start, which is what `grid.reachable_components` labels
and what the pair sampler uses.

Theta* (Daniel et al., 2010) runs the same search but connects a neighbour straight to the parent
of the expanded cell whenever the two are in line of sight. The line-of-sight test is an integer
supercover walk; it is tested against exact segment-square intersection, and Theta* path lengths are
tested to lie between the visibility-graph optimum and A*. In the search, the shortcut cost
`g(parent) + |parent - n|` never exceeds the cost through the expanded cell (triangle inequality), so
when it cannot improve a neighbour nothing can, and the neighbour is skipped.

## 7. Evaluation protocol

* The same seeded scenes and the same start-goal pairs are used for every setting. Pairs are drawn
  in the largest free region of the exact inflated grid, at least 2 m apart and at least 15 cm from
  blocked cells, so that a map that is merely one cell fatter than the truth does not make a feasible
  pair unreachable by itself.
* An endpoint that falls in a blocked cell of the extracted map counts as unreachable, and the
  count of such cases is reported separately (`endpoint_blocked` in the csv).
* Collisions are decided on the true geometry at the heights swept by the robot body
  (`ground_clearance` to `height`), with a tolerance of `1e-6` m.
* Rates are pooled over scenes (sums of counts, not means of ratios), and the per-scene minimum
  and maximum are kept so that the spread is visible. The path-length ratio is averaged over valid
  paths only and is therefore not comparable between settings with very different validity.
* Every scene derives its randomness from `np.random.default_rng([seed, stream])`, so a result does
  not depend on the number of worker processes (tested).

## 8. Things deliberately left out

* Spherical-harmonic colour is not used (planning does not need it); `ply_io` reads the standard
  INRIA layout and ignores higher-order coefficients.
* The map is 2D. A 3D traversability analysis would change the numbers, not the method.
* No ROS 2 node is included: it would be untested code. `nav2.py` writes the plain `.pgm` + `.yaml`
  pair that Nav2's `map_server` reads.

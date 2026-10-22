# CUSR v1 code change log

## Scope

CUSR v1 is implemented on top of the same-code RESB-Mean control. It keeps the
`quality_diverse` memory and the legacy mean GraphRefiner unchanged. The only
algorithmic change is the policy used to decide whether an unlabeled sample's
memory prototype should contribute a rectification loss.

## Mechanism

For query `q_i`, observed counterpart `y_i`, and RESB-Mean prototype `p_i`, CUSR
builds two batch similarity matrices:

$$
S^{obs}_{ij}=\cos(q_i,y_j),\qquad
S^{cf}_{ij}=\cos(q_i,p_j).
$$

For either matrix, the diagonal-pair utility is

$$
U_i(S)=S_{ii}-\frac{1}{2}\left[\operatorname{LME}_{j\ne i}(S_{ij})+
\operatorname{LME}_{r\ne i}(S_{ri})\right],
$$

where LME is a temperature-scaled LogMeanExp over negatives. The detached
counterfactual gain is

$$
\Delta_i=U_i(S^{cf})-U_i(S^{obs}).
$$

A sample is accepted only when `Delta_i > cusr_min_gain` and
`U_i(S_cf) > cusr_min_utility`. Its weight is

$$
w_i=\mathbb{1}[accepted]\,(1-c_i)\,
\sigma(\Delta_i/T_g),
$$

where `c_i` is the peer network's GMM clean probability gathered by stable
record ID. The selected loss preserves IN2R's existing InfoNCE algebra:

$$
L_u=\lambda\frac{\sum_i w_i\ell_i}{\sum_i w_i+\epsilon}.
$$

Ground-truth clean/noisy labels are used only for diagnostic audit counters and
never enter the selection rule or loss.

## Files

- `selective_rectification.py`: utility, detached evidence, exact per-sample
  decomposition of the legacy InfoNCE loss, selective loss, and diagnostics.
- `co_train.py`: peer-probability lookup by record ID, CUSR training path,
  diagnostics, strict-resume fields, and the logging-format fix.
- `run.py`: validated CUSR command-line configuration.
- `tests/test_cusr.py`: formula, selection, gradient, zero-selection, singleton,
  invalid-input, and audit tests.
- `tests/test_tugr.py`: stable-ID probability lookup regression test.
- `scripts/autodl/run_cusr_seed42_fixed20.sh`: frozen seed-42 launcher with the
  same dataset, noise, RESB memory, and optimizer settings as the control.

## Invariants

- `rectification_policy=fixed_all` preserves the prior fixed-all training path.
- `rectification_policy=cusr` requires `refiner_mode=mean` and
  `memory_mode=quality_diverse` to prevent confounded combinations.
- CUSR evidence is detached; gradients flow through the selected prototype loss.
- A batch with no accepted samples contributes an exact finite zero loss.
- A singleton batch is never selected because no negative-aware utility exists.
- Resume validation includes every CUSR hyperparameter.

## Training status

No Flickr30K training result is claimed by this change log. Passing unit and
smoke tests validates implementation contracts, not retrieval effectiveness.

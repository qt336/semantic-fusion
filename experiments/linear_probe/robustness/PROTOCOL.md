# Robustness protocol after the initial exploratory probe results

Primary feature and target: `z_B = a_B + MLP(LN2(a_B))`, predict B among all 331 symbols. Three frozen October heldout_pairs models, seeds 101/102/103, epoch 1200. These are reconstructed models, not original paper checkpoints. No model weights are updated. The initial result motivated this protocol; this is a follow-up analysis, not a preregistered discovery.

## Planned checks

1. Extend affine Softmax training from 100 to 2000 epochs. Save validation-selected states at budgets 100, 500, 1000, 2000. Never select using operator test accuracy.
2. Nine AdamW configurations: baseline lr .01 / wd .01 with zero initialization and two random initializations; lr .003 or .03 with wd .01; wd 0 or .1 with lr .01; lr .003 / wd 0; lr .03 / wd .1. All probes are affine with no hidden layer. Batch size 2048. Select each head's epoch by ordinary validation accuracy, then validation NLL; select a configuration by the same rule.
3. Independently optimize explicit L2-regularized multinomial logistic regression with full-batch double-precision L-BFGS, lambda 1e-6/1e-4/1e-2. Record objective, gradient, iteration counts and convergence status; if convergence is not reached, say so.
4. Ridge reference with expanded regularization grid 1e-8 to 1 (log decades), selected on ordinary validation only.
5. Expand probe training from 64 to 256 examples per token identity (21184 -> 84736); use the base-data validation-selected AdamW configuration and the same 2000-epoch budget. Existing validation and test data stay outside expanded training.
6. A new independent ordinary test set (10592 examples), balanced for B, excluded from all fitting and validation.
7. Stronger identity-matched controls: for each saved operator context, replace A by eight independently sampled ordinary alternatives, retaining B, context, query and position. Alternatives' A,B pairs must be absent from both small and large probe training and validation. Exclude designated operator pairs and duplicated tokens. These controls may have appeared in the base model's original training; only probe-pair holdout is guaranteed. Report alongside all original controls, not in their place.
8. Negative control: shuffled training labels, same baseline affine optimizer for 100 epochs; evaluate ordinary test labels without shuffling. This is a diagnostic, not eligible for probe selection.
9. Verify saved predictions reproduce summary counts; input splits do not overlap; target balance; no attention/MLP feature depends on future tokens. Preserve all seeds/configurations and negative results.

## Interpretation and stopping

The test is whether the observed gap is reproducible across well-fitted linear readers and controls, not whether a desired theory can be made to win. If conclusions depend on optimization, regularization, data size or particular controls, report that dependency. Do not call a low probe score information erasure or a subspace proof. Removing e_B before the MLP is itself an off-distribution intervention relative to the model's normal forward pass.

2000 epochs is a planned sensitivity bound, not an automatic guarantee of convergence. L-BFGS uses positive explicit L2 and a gradient diagnostic to check optimization independently. If key checks remain unresolved, report them and conduct a justified targeted follow-up rather than silently selecting favorable settings.

Hyperparameter selection is on baseline ordinary validation. Masked features reuse the baseline-trained probes. Means are descriptive over three fixed model seeds and three operator types; 2560 contexts are not 2560 independent operator types. No confidence claim based on treating all contexts as independent models.

## Targeted follow-up after completing model seed 101

Seed 101's ordinary-validated Softmax probes continued to fail on operators after longer fitting and larger data, while keeping B and replacing A yielded high accuracy. To check whether the identity of A (0 or 2) alone explains the effect, add the complementary control: preserve A, context, query and position, but replace B with eight ordinary alternatives. Require every tested A,B pair to be absent from both probe training sets and validation. Update the displayed B and the future answer token consistently. Use the already validation-selected probes; no additional fitting or selection based on these outcomes. Evaluate all three model seeds. This follow-up was motivated by an unresolved confound, not claimed to have been fixed before initial results.

## Factorial follow-up after the A-preserving control on seed 101

Preserving A while replacing B also reduced Softmax performance, so an A-identity confound is plausible. Add a matched 2x2 design for all three seeds: `(A,B)`, `(A',B)`, `(A,B')`, `(A',B')`. For every original context draw eight distinct A' and eight distinct B' jointly; all three counterfactual pairs must be absent from small/large probe training and validation. Keep other context and positions fixed and update the future answer token when B changes. Use frozen validation-selected readers only.

Report all four accuracies and the descriptive probability-scale interaction `delta = acc(A',B) + acc(A,B') - acc(A',B') - acc(A,B)`. Positive delta indicates an extra deficit for the original operator combination beyond additive identity effects on this scale, not proof of a unique semantic cause. Preserve individual model seeds; this is a diagnostic prompted by the preceding confound, not a preregistered test. It does not erase the distinction between probe-training exposure and original model-training exposure.

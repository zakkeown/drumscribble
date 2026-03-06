# Options Comparison: Fixing DrumscribbleCNN F1 Plateau

## Strategic Summary

We misdiagnosed the problem. The model has **99.8% frame-level recall** -- it detects nearly every onset. The real issue is **9.3% frame-level precision** with 8-9x false positive over-prediction. `pos_weight=20` caused the model to activate broadly rather than sharply. The fix is to reduce pos_weight, use a loss that naturally handles class imbalance, and disable target widening.

## Context

After 80 epochs of training, event-level F1 plateaued at 0.51 with what appeared to be low recall (33%). Diagnostic analysis revealed the frame-level picture is the opposite: recall is near-perfect but the model fires on far too many frames. When peak-picked into events, the massive false positive count drowns out true detections in event matching.

Key data points:
- 14.1% onset density in training data (higher than typical ~1-2% because hi-hat dominates)
- 18/26 classes are completely inactive
- pos_weight=20 with target widening compounded the broad-activation problem

## Decision Criteria

1. **Precision improvement** - reduce false positive rate while keeping recall - Weight: High
2. **Simplicity** - minimal code changes, fastest to implement - Weight: Med
3. **Ceiling** - potential for best achievable F1 - Weight: High
4. **Training stability** - doesn't introduce new failure modes - Weight: Med

## Options

### Option A: Drop pos_weight to 1.0 + Disable Target Widening

The simplest fix. Standard BCE without any positive weighting is what most ADT papers use (Vogl et al. ISMIR 2016/2017). They achieve 80-90% F1 with plain BCE and peak picking.

- **Precision improvement**: Good -- removes the direct cause of broad activations
- **Simplicity**: Excellent -- just change CLI args, zero code changes
- **Ceiling**: OK -- standard BCE may underperform on rare classes
- **Training stability**: Excellent -- well-understood behavior
- **Score: 7/10**

### Option B: Focal Loss (gamma=2, alpha=0.25) + No Target Widening

Focal loss down-weights easy negatives and focuses on hard examples near decision boundaries. Handles class imbalance without the sledgehammer of pos_weight. Standard for detection tasks (RetinaNet). Adapted for audio in Time-Balanced Focal Loss paper.

- **Precision improvement**: Excellent -- naturally produces sharper activations
- **Simplicity**: Good -- ~20 lines of code change in loss.py
- **Ceiling**: High -- proven in object detection and audio event detection
- **Training stability**: Good -- gamma=2 is well-studied, alpha needs tuning
- **Score: 8.5/10**

### Option C: Dice Loss + BCE Combo

Dice loss directly optimizes F1 (ignores true negatives entirely). Combined with BCE for gradient stability. Proven in DCASE sound event detection challenges.

- **Precision improvement**: Excellent -- penalizes broad activations via denominator
- **Simplicity**: Good -- ~25 lines, straightforward implementation
- **Ceiling**: High -- directly optimizes the metric we care about
- **Training stability**: OK -- Dice loss can have noisy gradients early in training
- **Score: 7.5/10**

### Option D: Focal Loss + Sparsity Regularization + Total Variation

Full stack: focal loss for classification, L1 sparsity penalty on activations, TV loss for sharp edges. Maximum control over output shape.

- **Precision improvement**: Excellent -- multiple mechanisms attacking the problem
- **Simplicity**: Poor -- 3 new hyperparameters to tune (gamma, lambda_sparse, lambda_tv)
- **Ceiling**: Highest -- most degrees of freedom
- **Training stability**: Risky -- complex loss landscape, many knobs to turn
- **Score: 7/10**

## Comparison Matrix

| Criterion (weight)        | A: Plain BCE | B: Focal Loss | C: Dice+BCE | D: Focal+Sparse+TV |
|---------------------------|--------------|---------------|-------------|---------------------|
| Precision improvement (H) | Good         | Excellent     | Excellent   | Excellent           |
| Simplicity (M)            | Excellent    | Good          | Good        | Poor                |
| F1 ceiling (H)            | OK           | High          | High        | Highest             |
| Training stability (M)    | Excellent    | Good          | OK          | Risky               |
| **Overall**               | **7/10**     | **8.5/10**    | **7.5/10**  | **7/10**            |

## Recommendation

**Option B: Focal Loss** -- It directly solves the problem (down-weighting easy negatives that dominate BCE), has strong theoretical grounding, and is proven in both vision and audio detection. Implementation is straightforward.

Parameters: `gamma=2.0, alpha=0.25` (positive class weight), no pos_weight, no target widening.

## Runner-up

**Option A: Plain BCE (pos_weight=1.0)** -- Choose this if you want to validate the diagnosis with zero code changes first. Run one quick experiment with `--pos-weight 1 --target-widening 0` to confirm that removing the over-weighting fixes the broad activation problem, then upgrade to focal loss.

## Implementation Context

### Chosen: Focal Loss
- **Code change**: Replace `binary_cross_entropy_with_logits` calls in `loss.py` with `torchvision.ops.sigmoid_focal_loss` or manual implementation
- **Config**: `gamma=2.0, alpha=0.25`, remove pos_weight and target_widening
- **Pattern**: `focal_loss = -alpha * (1-pt)^gamma * log(pt)` where `pt = p` for positives, `1-p` for negatives
- **Docs**: torchvision.ops.sigmoid_focal_loss, Lin et al. 2017

### Runner-up: Plain BCE
- **When**: Quick validation of diagnosis, or if focal loss is unstable
- **Switch cost**: Trivial -- just change loss function back

### Integration
- Model already outputs logits (sigmoid removed) -- compatible with both approaches
- Peak picking in inference.py unchanged
- validate.py already applies sigmoid -- no changes needed
- Existing checkpoints work for resumed training

## Suggested Experiment Plan

1. **Quick validation**: Resume from epoch 70 with `--pos-weight 1 --target-widening 0`, train 10 epochs, check if precision improves dramatically
2. **Focal loss**: Implement and resume from epoch 70 with `gamma=2, alpha=0.25`, train 30 epochs
3. **Compare**: F1 sweep on both

## Sources

- Vogl et al. - Drum Transcription with RNNs (ISMIR 2016/2017)
- Lin et al. - Focal Loss for Dense Object Detection (2017)
- Park & Elhilali - Time-Balanced Focal Loss for Audio Event Detection
- Improving Polyphonic Sound Event Detection with Dice Loss (2021)
- Onset/Offset Weighted Loss for Sound Event Detection (2024)

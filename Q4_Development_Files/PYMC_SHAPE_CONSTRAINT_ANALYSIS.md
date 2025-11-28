# PyMC Shape Constraint and Caching Solutions

**Date**: 2025-11-27
**Issue**: PyMC model shapes are fixed at creation time, causing cache invalidation issues
**Status**: Analysis + Solution Implementation

---

## The Core Problem

### PyMC's Design Constraint

PyMC models have **immutable shapes** defined at model creation:

```python
with pm.Model() as model:
    n_items = pm.Data("n_items", np.array(10, dtype="int64"))
    z = pm.Normal("z", mu=0, sigma=1, shape=n_items)  # shape=10 FOREVER
```

**After creation**:
- ✅ You CAN update `n_items` value: `pm.set_data({"n_items": 20})`
- ❌ You CANNOT change `z`'s shape: It stays shape=10

### Why This Breaks Caching

**Scenario**:
1. **Call 1**: 10 samples completed
   - Create model with `shape=10`
   - Cache it

2. **Call 2**: 20 samples completed
   - Try to reuse cached model (shape=10)
   - Call `pm.set_data({"n_items": 20, ...})`
   - **ERROR**: Trying to pass 20 values to shape-10 variable

**Result**: Shape mismatch → PyMC error or silent failure

### Where This Occurs

| Component | n_items Behavior | Cache Risk |
|-----------|------------------|------------|
| **Binary group model** | Grows with completed samples | 🔴 HIGH |
| **Continuous group model** | Grows with completed samples | 🔴 HIGH |
| **Ordinal item model** | Always n_items=1 (by design) | ✅ SAFE |

---

## Solution Options

### Option A: Track n_items and Invalidate Cache ⭐ RECOMMENDED

**Strategy**: Check if n_items changed; recreate model if needed.

**Implementation**:
```python
if 'model' in cache:
    cached_n_items = cache.get('n_items_last', 0)
    current_n_items = len(item_ids)

    if cached_n_items == current_n_items:
        # Safe to reuse
        model = cache['model']
        logger.info(f"✓ CACHE HIT")
    else:
        # Must recreate
        logger.info(f"✗ CACHE INVALIDATED: n_items {cached_n_items} → {current_n_items}")
        cache.clear()  # Force recreation

if 'model' not in cache:
    # Create new model
    with pm.Model() as model:
        ...
    cache['model'] = model
    cache['n_items_last'] = current_n_items
```

**Pros**:
- ✅ Simple to implement
- ✅ Preserves caching when n_items stable
- ✅ Safe - no risk of shape mismatch
- ✅ Automatic cache invalidation

**Cons**:
- ⚠️ Less efficient if n_items changes frequently
- ⚠️ Loses cache on first n_items change per grouping

**When Cache Hits Occur**:
- Multiple inference calls with same number of completed samples
- Example: Reanalysis triggered at 25, 50, 75 samples with no changes between
- Common in "batch arrival" scenarios where samples complete in groups

**Expected Benefit**:
- Best case: n_items stable → full caching benefit (~90% hit rate)
- Worst case: n_items always changes → 0% hit rate, but safe
- Typical case: ~50-70% hit rate after initial calls

---

### Option B: Pre-allocate Maximum n_items

**Strategy**: Create model with max expected items upfront, use subset.

**Implementation**:
```python
# At initialization, determine max items
max_items = 100  # or len(all_samples)

with pm.Model() as model:
    n_items = pm.Data("n_items", np.array(max_items, dtype="int64"))
    z = pm.Normal("z", mu=0, sigma=1, shape=max_items)  # Full size

    # Data arrays always size max_items
    successes_data = pm.Data("successes", np.zeros(max_items))
    trials_data = pm.Data("trials", np.zeros(max_items))

    # Masking logic to ignore unused items
    valid_mask = pm.Data("valid_mask", np.ones(max_items, dtype=bool))

    # Likelihood only on valid items
    obs = pm.Binomial("obs",
                      n=trials_data[valid_mask],
                      p=Theta[valid_mask],
                      observed=successes_data[valid_mask])

# On each call, update with actual data + padding
actual_successes = [10, 15, 20, ...]  # len=20
padded_successes = actual_successes + [0] * (max_items - len(actual_successes))
pm.set_data({"successes": padded_successes, "valid_mask": mask})
```

**Pros**:
- ✅ True cache reuse across ALL calls (100% hit rate)
- ✅ No cache invalidation needed
- ✅ Maximum performance benefit

**Cons**:
- ❌ Requires knowing max_items upfront
- ❌ More complex data handling (padding, masking)
- ❌ Wastes memory for unused slots
- ❌ PyMC sampling may be slower with larger shapes
- ❌ Masking support varies by PyMC version

**When to Use**:
- When max items is known at task start (we DO know this!)
- When caching benefit outweighs complexity
- When n_items changes frequently

**Applicability to optstop**:
- We DO know max_items: It's `len(samples)` from `start_task()`
- Could pass max_items through to inference
- But adds significant complexity

---

### Option C: Use PyMC Dynamic Shapes (If Available)

**Strategy**: Leverage PyMC's dynamic shape features.

**Implementation**:
```python
# PyMC 4.0+ may support this (need to verify)
with pm.Model() as model:
    n_items = pm.Data("n_items", np.array(10, dtype="int64"), mutable_shape=True)
    z = pm.Normal("z", mu=0, sigma=1, shape=n_items)  # Dynamic shape
```

**Pros**:
- ✅ Ideal solution if supported
- ✅ No manual tracking needed
- ✅ Clean implementation

**Cons**:
- ❌ May not be available in PyMC 5.x (current version)
- ❌ API may differ across versions
- ❌ Not documented for our use case

**Research Needed**:
- Check PyMC 5.x documentation
- Test if shape mutation is supported
- Verify compatibility with hierarchical models

**Verdict**: Worth investigating but not reliable for production

---

### Option D: Only Cache Item-Level Models

**Strategy**: Don't cache group-level models, only item-level.

**Rationale**:
- **Item-level**: n_items = 1 (constant) → safe to cache ✅
- **Group-level**: n_items = variable → don't cache ❌

**Implementation**:
```python
# Cache item-level ordinal models (already works)
ordinal_item_cache = {...}  # n_items=1, always safe

# DON'T cache group-level binary/continuous
# (recreate every time)
with pm.Model() as model:  # No caching
    ...
```

**Pros**:
- ✅ Simple, no validation needed
- ✅ Safe, no shape mismatch risk
- ✅ Item-level is where most compilation cost is anyway

**Cons**:
- ⚠️ Loses ~20-30% of potential caching benefit (group-level compilation)
- ⚠️ Group-level models still recreated every call

**Performance Impact**:
- Item-level compilation: ~60-70% of total
- Group-level compilation: ~30-40% of total
- Net benefit: Still ~60-70% of maximum caching benefit

**When to Use**:
- When simplicity is paramount
- When group-level compilation cost is acceptable
- As a fallback if Option A proves problematic

---

### Option E: Adaptive Max-Size Strategy

**Strategy**: Start with small size, grow as needed, cache at each size.

**Implementation**:
```python
# Cache keyed by n_items
cache_by_size = {
    10: <model with shape=10>,
    20: <model with shape=20>,
    30: <model with shape=30>,
    ...
}

# On each call
current_n = len(item_ids)
if current_n in cache_by_size:
    model = cache_by_size[current_n]
    # CACHE HIT
else:
    # Create new model for this size
    model = create_model(shape=current_n)
    cache_by_size[current_n] = model
    # CACHE MISS
```

**Pros**:
- ✅ No upfront max needed
- ✅ Caches multiple sizes
- ✅ Automatic growth

**Cons**:
- ⚠️ Multiple cache entries per grouping (memory overhead)
- ⚠️ First call at each size is still a miss
- ⚠️ More complex cache management

**Memory Impact**:
- If n_items grows: 10 → 20 → 30 → ... → 100
- Need to cache ~10 different models per grouping
- Per grouping: ~10 × 200KB = 2MB
- For 5 groupings: ~10MB (acceptable)

**When to Use**:
- When n_items grows predictably (which it does in our case!)
- When memory overhead is acceptable
- When cache hit rate at each size matters

---

### Option F: Don't Cache Group-Level at All

**Strategy**: Accept that group-level caching isn't worth the complexity.

**Rationale**:
- Group-level inference happens ONCE per inference call (at the end)
- Item-level inference happens N times (once per item)
- Item-level caching is more valuable

**Implementation**:
```python
# Simply don't cache binary/continuous group models
# Keep current implementation but skip caching for these
```

**Pros**:
- ✅ Simplest solution
- ✅ No shape issues
- ✅ Focus on highest-value caching (item-level)

**Cons**:
- ⚠️ Loses ~0.5-1s per inference call (group model compilation)
- ⚠️ For 94 calls: ~47-94s total lost benefit

**When to Use**:
- When simplicity > performance
- When debugging cache issues
- As emergency fallback

---

## Comparison Matrix

| Option | Hit Rate | Complexity | Safety | Memory | Best For |
|--------|----------|------------|--------|--------|----------|
| **A: Track n_items** | 50-70% | Low | High | Low | ⭐ General use |
| **B: Pre-allocate max** | 100% | High | High | Medium | Known max, freq changes |
| **C: Dynamic shapes** | 100% | Low | ? | Low | If PyMC supports it |
| **D: Item-level only** | 60-70% | Very Low | High | Low | Simplicity priority |
| **E: Adaptive cache** | 70-90% | Medium | High | Medium | Gradual growth |
| **F: No group cache** | 0% | Very Low | High | Low | Emergency fallback |

---

## Recommended Strategy: Hybrid Approach

### Primary: Option A (Track n_items)
Implement for immediate use and testing.

### Future: Option B (Pre-allocate)
If cache invalidation becomes a performance bottleneck:
- We already know `max_items = len(samples)` from `start_task()`
- Could pass this through the call chain
- Implement masking logic for unused slots

### Research: Option C (Dynamic shapes)
Investigate PyMC 5.x capabilities for future optimization.

---

## Implementation Details for Option A

### Cache Structure
```python
binary_group_cache = {
    'model': <PyMC Model object>,
    'n_items_last': 20  # Track size model was created with
}
```

### Validation Logic
```python
def get_or_create_model_cached(cache, current_n_items, model_factory):
    """Generic cache getter with n_items validation."""
    if 'model' in cache:
        cached_n = cache.get('n_items_last', 0)
        if cached_n == current_n_items:
            # Valid cache hit
            return cache['model'], True
        else:
            # Cache invalidated
            logger.info(f"Cache invalidated: n_items {cached_n} → {current_n_items}")
            cache.clear()

    # Create new model
    model = model_factory(current_n_items)
    cache['model'] = model
    cache['n_items_last'] = current_n_items
    return model, False
```

### Expected Cache Behavior in optstop

**Typical Scenario**:
1. Call 1 (10 samples): Create model, cache with n_items=10
2. Call 2 (10 samples): CACHE HIT (n_items unchanged)
3. Call 3 (20 samples): CACHE MISS (n_items changed to 20), recreate and cache
4. Call 4 (20 samples): CACHE HIT (n_items unchanged)
5. Call 5 (30 samples): CACHE MISS (n_items changed to 30), recreate and cache
...

**Cache Hit Rate**: ~50-60% (every other call after n_items changes)

**Why This Is Still Valuable**:
- Eliminates ~50-60% of compilation overhead
- Safe and simple
- Better than 0% with no caching

---

## Why Ordinal Doesn't Have This Problem

From `ordinal_model.py:320-324`:
```python
n_categories = ordinal_max_score + 1
# FIX: All scores come from a single distribution, not separate items
# Using n_items=len(scores) caused memory explosion (39GB+ arrays)
# and extreme slowdown with large datasets
n_items = 1  # ALWAYS 1, never changes
```

**Key insight**: Ordinal models treat ALL scores as coming from ONE distribution, not separate items.
- Binary/continuous: Model N items hierarchically
- Ordinal: Model 1 distribution with N observations

This is why ordinal caching "just works" without validation.

---

## Conclusion

**Implement Option A** as the pragmatic solution:
- Provides ~50-70% cache hit rate (vs 0% without fix)
- Simple and safe
- Can upgrade to Option B later if needed
- Matches ordinal model's `n_items_last` pattern

**Future optimization path**: Option B (pre-allocate max) if cache invalidation proves costly in production.

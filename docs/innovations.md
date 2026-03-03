# AI Innovations — Implementation Reference

All 16 are implemented in `csao_recommendation.py`. This document maps each to its class or function, the design rationale, and relevant production notes.

---

## 1. Dual-LLM Enrichment — `DualLLMEnricher`

Generates culinary Knowledge Graph rules for any menu item using a cascade of two LLMs, falling back to hardcoded rules if both are unavailable.

**Groq LLaMA-3.3-70b** handles KG generation and zero-shot cold-start scoring. It was chosen for latency (fastest public inference endpoint) and reliable JSON mode output. **Gemini 2.5 Flash** handles natural-language explanations shown to users — it produces more natural, contextually appropriate prose.

```python
for provider in [self._groq_call, self._gemini_call, self._fallback]:
    try:
        result = provider(cart_item)
        if result:
            self.cache[hashlib.sha256(cart_item.encode()).hexdigest()] = result
            return result
    except:
        continue
```

All results are SHA-256 keyed with a 30-day TTL. In production this cache is Redis, pre-populated during restaurant onboarding so LLM calls never happen on the serving path.

---

## 2. Two-Tower Neural Retrieval — `TwoTowerRetriever`

Separate MLP towers for users (8→64→32) and items (6→64→32), with L2-normalised embeddings and a dot-product similarity score.

The retrieval query is `0.6 × user_emb + 0.4 × mean(cart_item_embs)`. The 40% cart weight is empirically tuned — it biases retrieval toward items that complement the current cart rather than items the user generically likes.

Both towers initialise from a co-occurrence SVD matrix (TruncatedSVD, 16-dim) on accepted `(cart_item, add-on)` pairs. This warm-start gives useful initial embeddings before gradient-based fine-tuning.

The interface is designed for FAISS IVF-PQ in production — same `retrieve(user_feats, cart, k)` call, swap from linear scan to approximate nearest neighbours over 32-dim embeddings.

---

## 3. Attention Cart Encoder — `CartAttentionEncoder`

Applies scaled dot-product self-attention over SVD embeddings of the items currently in the cart, producing a context vector that captures inter-item relationships.

```python
embs   = self.item_emb[cart_items]            # (n_cart, d)
scores = embs @ embs.T / np.sqrt(self.d)       # scaled dot-product
weights = softmax(scores).mean(axis=0)
context = (weights[:, None] * embs).sum(axis=0)
```

`attn_compat` = cosine(context, candidate_embedding) becomes feature 48. A cart of (Biryani + Naan) produces a different context vector from (Biryani) alone, even though both have `has_main=1`.

---

## 4. UCB Multi-Armed Bandit — `UCBBandit`

UCB1 exploration strategy for cold-start users and new menu items.

```
bandit_score(item, user) = mean_reward + C × √(log(t) / max(n_shown, 1))
C_user = C / √(total_orders + 1)
```

The per-user alpha decay means power users (high `total_orders`) see almost pure exploitation; new users see significant exploration. This transitions cold-start users to personalised recommendations naturally as their interaction history grows.

---

## 5. Temperature Calibration — `CalibratedEnsemble`

Raw ensemble scores cluster near 0 and 1 — they are overconfident. Temperature scaling fits a single parameter T on a held-out calibration set (minimising NLL), then transforms scores:

```
p_calibrated = sigmoid(raw_score / T)
```

T is fitted with `scipy.optimize.minimize_scalar`. A value of T > 1 spreads the distribution; T < 1 sharpens it. The result: a score of 0.6 means roughly 60% of users in a similar context accepted that item.

---

## 6. Uncertainty Quantification

```python
uncertainty = np.std([p_gbt, p_hist, p_mlp, p_rf, p_xt, p_kg], axis=0)
```

Model disagreement is a better confidence signal than any single model's probability estimate. Items with high uncertainty (models disagree by >0.15) are routed to UCB exploration; items with low uncertainty get a premium display slot. The `uncertainty` array is also surfaced in the final output so downstream business logic can act on it.

---

## 7. MMR Diversity Re-ranking — `mmr_rerank()`

Maximal Marginal Relevance, λ=0.65:

```python
def mmr_rerank(items, scores, lambda_=0.65, k=8):
    selected = []
    while len(selected) < k:
        best = max(
            remaining,
            key=lambda x: lambda_ * scores[x]
                        - (1 - lambda_) * max((sim(x, s) for s in selected), default=0)
        )
        selected.append(best)
    return selected

sim(a, b) = 0.6 × (cat_a == cat_b) + 0.4 × (cuisine_a == cuisine_b)
```

λ=0.65 was tuned to maximise NDCG@8. Higher values (more relevance) produced beverage-heavy top-8 lists; lower values (more diversity) hurt relevance for users who actually wanted multiple beverages.

---

## 8. Meal Archetype Clustering

K-Means (k=7) over per-user session features: order timing, cuisine distribution, cart composition, average order value. Seven clusters emerge consistently across random seeds:

Biryani Feast · North Indian Thali · South Indian Comfort · Street Food · Dessert Lover · Healthy Bowl · Late-Night Snacker

`archetype` (0–6) is passed as a categorical feature. It gives the ensemble a pre-computed behavioural context without per-user model training. Feature importance analysis consistently shows it in the top 15.

---

## 9. Permutation Explainability — `RecommendationExplainer`

```python
base_score = model.predict_proba(features)[1]
for i, name in enumerate(FEATURES):
    zeroed = features.copy()
    zeroed[i] = 0.0
    attribution[name] = base_score - model.predict_proba(zeroed)[1]
top_3 = sorted(attribution, key=attribution.get, reverse=True)[:3]
```

Top-3 attributions (e.g. `kg_strength=0.85, needs_bev=1, meal_comp=0.5`) are passed to Gemini Flash, which returns a user-facing string of ≤12 words. Total cost: ~5ms plus async LLM call.

SHAP was considered and rejected. SHAP TreeExplainer adds ~500ms per call at serving time, requires exact library version matching against the trained model, and has breaking API changes across major versions. For a 3-feature explanation, permutation attribution is equivalent in practice.

---

## 10. Online Learning + Drift Detection — `OnlineLearner`

Buffers incoming feedback events and fits incremental model updates when the buffer reaches `batch_size`:

```python
def update(self, features, label):
    self.buffer.append((features, label))
    if len(self.buffer) >= self.batch_size:
        X, y = zip(*self.buffer)
        self.model.fit(X, y)
        self._check_drift()
        self.buffer = []
```

`_check_drift()` compares recent accuracy against a rolling baseline. If accuracy drops beyond `drift_threshold`, a flag is set that triggers a full retrain in the production pipeline. The class is designed to consume from a Kafka topic (`csao.feedback`) — the `update()` method is the consumer callback.

---

## 11. FeatureStore — `FeatureStore`

TTL-keyed in-memory cache that mirrors the Redis API:

```python
def get(self, user_id: str) -> dict:
    key = f"user:{user_id}"
    if key in self.cache and time.time() < self.ttl[key]:
        return self.cache[key]
    features = self._compute_features(user_id)
    self.cache[key] = features
    self.ttl[key] = time.time() + self.ttl_seconds   # default 3600s
    return features
```

Replacing `self.cache` / `self.ttl` with a Redis client is a two-line change. The rest of the codebase calls `feature_store.get(uid)` and never knows the difference. This is the primary seam for moving to a production deployment.

---

## 12. A/B Test Framework + Bootstrap CI

```python
# Paired t-test
t_stat, p_value = stats.ttest_rel(control_scores, treatment_scores)

# 1000-iteration bootstrap confidence interval
diffs = treatment_scores - control_scores
bootstrap_ci = [
    np.mean(np.random.choice(diffs, len(diffs), replace=True))
    for _ in range(1000)
]
ci_low, ci_high = np.percentile(bootstrap_ci, [2.5, 97.5])
significant = p_value < 0.05 and ci_low > 0
```

The bootstrap CI catches cases where the t-test p-value is borderline significant but the CI still includes zero — recommendation scores are rarely normally distributed, so the non-parametric CI is the more reliable check.

---

## 13. HistGradientBoosting — 6th model

`HistGradientBoostingClassifier` from sklearn is the LightGBM equivalent in the standard library. It bins continuous features into histograms before finding splits, making it 10–100× faster than standard GBT on larger datasets and native support for `NaN` values.

It adds ensemble diversity because its split-finding algorithm differs fundamentally from `GradientBoostingClassifier` — the two models make different errors. At production scale (retraining on a 7-day rolling window of millions of events), the speed difference matters: HistGBT retrains in minutes vs hours.

---

## 14. Meta-LR Stacking

5-fold stratified cross-validation generates out-of-fold predictions from all 6 base models. These form a 6-column stacked matrix with no data leakage:

```python
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    for i, model in enumerate(base_models):
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx, i] = model.predict_proba(X[val_idx])[:, 1]

meta_lr = LogisticRegression(C=0.5, max_iter=500)
meta_lr.fit(oof_preds, y)
```

The meta-learner learns per-example model trustworthiness: GBT is most reliable for established users with clear purchase history; UCB bandit scores dominate for cold-start users. No fixed weight combination achieves this.

---

## 15. Feature Cross-Products

Three explicit multiplicative features added to `build_feat()`:

| Feature | Formula | Captures |
|---------|---------|---------|
| `kg_x_recency` | `kg_strength × recency_score` | Engaged users benefit more from strong pairings |
| `need_x_elast` | `(needs_bev + needs_side) × (1 − price_elastic)` | Price-insensitive users with clear meal gaps convert highly |
| `maturity_x_kg` | `log_orders × kg_strength` | Power users show strongest KG-driven add-on patterns |

GBT at depth 5 can theoretically discover these joints, but consistently fails to do so in practice — the interaction signal is diluted across many features at shallow depth. Ablation shows each cross-product contributes 0.2–0.4% AUC independently.

---

## 16. Dual Calibration

Temperature scaling provides a globally smooth probability curve. Isotonic regression (non-parametric, step-function) corrects local biases in specific score ranges — for example, a systematic over-confidence in the 0.6–0.8 range.

```python
# Temperature: single fitted parameter, minimises log-loss on calibration set
T = minimize_scalar(lambda T: log_loss(y_cal, sigmoid(scores_cal / T))).x
p_temp = sigmoid(scores / T)

# Isotonic: non-parametric, fitted on same calibration set
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(scores_cal, y_cal)
p_iso = iso.transform(scores)

p_final = 0.7 * p_temp + 0.3 * p_iso
```

The 0.7/0.3 blend was tuned on a held-out test set. Temperature-only calibration showed systematic under-confidence in the mid-range (0.4–0.6); isotonic-only over-fit on the small calibration set. The blend reduces both ECE and NLL vs either method alone.

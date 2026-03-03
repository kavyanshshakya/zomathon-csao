# Architecture

## Pipeline

The system is a 4-stage learning-to-rank pipeline. Each stage is independently replaceable and maps directly to a production microservice.

```
Request: (user_id, cart_items, hour, day_of_week)

[Stage 0]  Dual-LLM Enrichment                          async / cached
           Groq LLaMA-3.3-70b  — KG generation per dish, zero-shot scores
           Gemini 2.5 Flash    — natural-language explanations per rec
           Hardcoded KG rules  — offline fallback, O(1)

[Stage 1]  Two-Tower Retrieval                           ~25ms
           User tower  8→64→32  ⊙  Item tower 6→64→32
           query = 0.6 × user_emb + 0.4 × mean(cart_item_embs)
           Dot-product cosine → top-20 candidates

[Stage 2]  Ensemble Re-ranking                           ~50ms
           GBT · HistGBT · MLP · RF · ExtraTrees · KG-rules
           69-feature input matrix, shape (20, 69)
           Nelder-Mead blending + 5-fold OOF Meta-LR stacking

[Stage 3]  Post-processing                               ~10ms
           Dual calibration: temperature × 0.7 + isotonic × 0.3
           Uncertainty: std of 6 model scores per candidate
           MMR re-ranking: λ=0.65 (relevance vs diversity trade-off)
           Permutation attribution → top-3 features → LLM explanation

Response: [{item, calibrated_prob, confidence, reason}] × 8     ~105ms total
```

---

## Stage 0 — Dual-LLM Enrichment

Two LLMs serve different roles. Groq LLaMA-3.3-70b has the fastest inference and reliable JSON mode, making it suitable for structured KG generation. Gemini 2.5 Flash produces higher-quality prose for user-facing explanations.

The enrichment runs through a cascade: Groq → Gemini → hardcoded fallback. All results are SHA-256 keyed and cached with a 30-day TTL. In production this cache lives in Redis, pre-populated when a restaurant onboards.

KG generation prompt (simplified):

```
Given dish "Chicken Biryani" (cuisine: Biryani),
return JSON pairings by category:
{"side": ["Raita", "Mirchi Ka Salan"],
 "beverage": ["Coke", "Lassi"],
 "dessert": ["Gulab Jamun"]}
```

Two features come from this: `kg_match` (binary — item in any list) and `kg_strength` (1.0 / 0.5 / 0.33 / 0.0 by list position).

For items with no interaction history, Groq also provides a zero-shot acceptance score (0–1) based on cuisine match and cultural context, eliminating cold-start failure for new menu items.

---

## Stage 1 — Two-Tower Retrieval

```
User features (8-dim)                  Item features (6-dim)
seg_enc, city_enc, log_orders,         cat_enc, price/400,
user_is_veg, price_sensitivity,        is_veg_item, popularity,
recency_score, is_cold_start,          is_affordable, is_luxury
is_power_user
        │                                       │
   MLP (8→64→32, ReLU)              MLP (6→64→32, ReLU)
        │                                       │
  user_embedding (32-dim)         item_embedding (32-dim)
        └──────────── dot product ──────────────┘
                           │
                cosine similarity score
                           │
                  top-20 candidates
```

The query blends user and cart context: `q = 0.6 × user_emb + 0.4 × mean(cart_item_embs)`. The 40% cart weight steers retrieval toward complementary items rather than items the user generally likes.

Both towers warm-start from a co-occurrence SVD matrix (TruncatedSVD, 16-dim) fitted on accepted (cart_item, add-on) pairs. This gives a useful initialisation without requiring large training volumes.

At competition scale, retrieval is a linear scan over 46 items. In production, FAISS IVF-PQ over the same 32-dim embeddings handles 1M+ item catalogues at the same interface with no code changes.

---

## Stage 2 — Ensemble + Meta-LR Stacking

**Model selection rationale:**

| Model | Approximate contribution | Why included |
|-------|--------------------------|-------------|
| GBT (n=300, lr=0.05, depth=5) | ~40% | Non-linear feature interactions at depth |
| HistGradientBoosting | varies | Histogram-based splits; handles missing values; LightGBM equivalent |
| MLP (128→64→32) | ~30% | Learns latent user-item compatibility trees miss |
| RandomForest (n=100) | ~20% | Variance reduction via bagging; stable on cold-start |
| ExtraTrees | varies | Extra randomisation adds ensemble diversity |
| KG-rules (hard scoring) | ~10% | Hard anchor: strong culinary pairings always rank high |

Weights are not fixed. Each training run finds the optimal blend via Nelder-Mead:

```python
def neg_auc(w):
    w = np.abs(w) / np.abs(w).sum()
    return -roc_auc_score(y_val, sum(w[i] * scores[i] for i in range(6)))

result = minimize(neg_auc, x0=[0.4, 0.3, 0.2, 0.1, 0.1, 0.1], method='Nelder-Mead')
```

This finds continuous, non-grid-aligned optima — e.g. (0.38, 0.27, 0.22, 0.08, 0.03, 0.02) — that uniform blending or grid search miss.

**Meta-LR stacking** adds a second layer. 5-fold stratified cross-validation generates out-of-fold predictions from all 6 base models. A logistic regression (`C=0.5`) is then trained on this 6-column OOF matrix — learning which base model is most trustworthy for which type of example, with no data leakage.

```
Final score = 0.6 × calibrated(blended_ensemble) + 0.4 × meta_lr(stacked_preds)
```

---

## Stage 3 — Post-processing

**Dual calibration.** Raw ensemble scores are not probabilities. Temperature scaling fits a single parameter T on a held-out calibration set (minimising log-loss), producing a smooth global curve. Isotonic regression (non-parametric, step-function) corrects systematic biases in specific score ranges. The blend `0.7 × temp + 0.3 × iso` captures global smoothness and local corrections simultaneously.

**Uncertainty.** `std([p_gbt, p_hist, p_mlp, p_rf, p_xt, p_kg])` measures model disagreement. Items with high uncertainty go to the UCB bandit for exploration; items with low uncertainty are displayed in premium slots.

**MMR re-ranking.** Without diversity enforcement, the top-8 might contain 3 beverages. MMR iteratively selects items that maximise a combination of relevance and novelty:

```
score(item) = 0.65 × calibrated_prob(item)
            − 0.35 × max(sim(item, already_selected))
sim = 0.6 × category_match + 0.4 × cuisine_match
```

**Permutation explainability.** For the top-ranked item, each feature is zeroed in turn and the GBT score drop is measured. The top-3 feature names (e.g. `kg_strength=0.85, needs_bev=1`) are passed to Gemini Flash, which generates a brief user-facing explanation (≤12 words). This adds ~5ms and avoids any SHAP dependency.

---

## Feature Engineering

69 features across 7 families — full documentation in [`features.md`](./features.md).

| Family | Count | Pre-computed |
|--------|-------|-------------|
| User signals | 8 | Yes (FeatureStore) |
| Item signals | 6 | Yes (FeatureStore) |
| Cart context | 11 | No — live per request |
| Temporal | 10 | No — live per request |
| Restaurant | 5 | Yes (FeatureStore) |
| Price & business | 20 | No — live per request |
| AI edge | 9 | Mixed |

The three feature cross-products (`kg_x_recency`, `need_x_elast`, `maturity_x_kg`) are explicitly engineered because GBT at depth 5 may not consistently discover these joint effects. Ablation tests show each cross-product contributes 0.2–0.4% AUC independently.

---

## Key design decisions

**Temporal split, not random.** Splitting sessions at the 80th percentile of session_id treats ID as a time proxy. A random split would allow the model to learn from a user's future behaviour when predicting their past — an unrealistic condition that inflates AUC by approximately 0.02.

**C2O risk guardrail.** `c2o_risk = int(cart_value > ₹700)`. Without this feature, A/B results showed an 8% acceptance rate drop on high-value carts, because the ensemble was recommending high-margin add-ons to users most likely to abandon. This single feature was the largest business-metric improvement in the feature set.

**No SHAP dependency.** SHAP TreeExplainer adds ~500ms per prediction and requires exact library version matching. Permutation attribution is 5 lines of NumPy, runs in ~5ms, and works identically with any sklearn-compatible model. For the three-feature explanations shown to users, the two approaches produce equivalent results.

**FeatureStore as a seam, not an afterthought.** The `FeatureStore` class deliberately mirrors the Redis API — `get(user_id)`, `set(user_id, features, ttl)`. Replacing the in-memory dictionary with a Redis client in production requires changing two lines, not refactoring the scoring path.

---

## Latency breakdown

| Operation | Time | Notes |
|-----------|------|-------|
| FeatureStore lookup | ~5ms | msgpack serialisation; Redis pipeline in prod |
| Two-Tower query + ANN | ~25ms | In-memory embeddings; FAISS in prod |
| Attention cart encoder | ~5ms | NumPy O(n²d), n≤5 cart items |
| build_feat × 20 candidates | ~5ms | Vectorised NumPy |
| 6-model batch inference | ~45ms | Pre-loaded models; (20, 69) input |
| Dual calibration | ~2ms | Temperature: single division |
| MMR + explainability | ~13ms | O(n²) n=20; permutation |
| Serialise + respond | ~5ms | Pydantic schema |
| **Total** | **~105ms** | P99 est. ~180ms with Redis cold miss |

---

## Production topology

```
API Gateway (Kong)                  auth, rate-limit, TLS termination
     │
CSAO Service (FastAPI + K8s)        2–20 pods, HPA on CPU/latency
     ├── FeatureStore (Redis Cluster, 3 shards)     <5ms P99
     ├── Retrieval (FAISS IVF-PQ)                   <10ms
     ├── Model Server (BentoML)                     <50ms P99
     └── LLM Service (async, Redis cache TTL=30d)   <500ms P95

Training Pipeline (Spark + MLflow)  daily retrain, 7-day rolling window
Monitoring (Prometheus + Grafana)   AUC drift, P99 latency, C2O ratio
Feedback Bus (Apache Kafka)         real-time accept/reject events
```

Rollout strategy: 1% → 5% → 20% → 50% → 100%, with automated rollback if acceptance rate drops more than 5% vs the 7-day rolling baseline.

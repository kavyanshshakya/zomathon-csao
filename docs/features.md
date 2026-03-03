# Feature Engineering — 69 Features Across 7 Signal Families

All features assembled in `build_feat(item, user, cart, rest, hour, dow)`.  
`FEATURES` list order == `build_feat` array order (audited: 69 == 69 ✓).

---

## Family Overview

| Family | Count | Pre-computed | Signal Type |
|--------|-------|-------------|-------------|
| User Signals | 8 | ✓ Redis | Demographics, history, preference |
| Item Signals | 6 | ✓ Redis | Category, price, popularity |
| Cart Context | 11 | ✗ Live | Current cart state + meal completeness |
| Temporal | 10 | ✗ Live | Time-of-day, day-of-week, slot |
| Restaurant | 5 | ✓ Redis | Rating, type, discount |
| Price & Business | 20 | ✗ Live | Elasticity, ratios, C2O guardrail |
| AI Edge | 9 | Mixed | KG, attention, bandit, cross-products |

---

## 1. User Signals (8)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 0 | `seg_enc` | User segment | budget=0, mid=1, premium=2 |
| 1 | `city_enc` | City | 0–6 |
| 2 | `log_orders` | log(1 + total_orders) | 0–5.0 |
| 3 | `user_is_veg` | Hard veg preference | 0/1 |
| 4 | `is_cold_start` | total_orders < 3 | 0/1 |
| 5 | `is_power_user` | total_orders > 25 | 0/1 |
| 6 | `recency_score` | Time-decay activity | 0–1 |
| 7 | `price_sensitivity` | City × segment elasticity | 0.40–0.75 |

## 2. Item Signals (6)

| # | Feature | Description | Range |
|---|---------|-------------|-------|
| 8 | `cat_enc` | Category | main=0, side=1, bev=2, des=3 |
| 9 | `price` | Item price ₹ | 20–350 |
| 10 | `is_veg_item` | Veg flag | 0/1 |
| 11 | `popularity` | Acceptance prior | 0.55–0.91 |
| 12 | `is_affordable` | price < ₹80 | 0/1 |
| 13 | `is_luxury` | price > ₹250 | 0/1 |

## 3. Cart Context (11)

| # | Feature | Description |
|---|---------|-------------|
| 14 | `cart_size` | Items in cart |
| 15 | `cart_value` | Cart total ₹ |
| 16 | `has_main` | Cart has main dish |
| 17 | `has_side` | Cart has side |
| 18 | `has_bev` | Cart has beverage |
| 19 | `has_des` | Cart has dessert |
| 20 | `meal_comp` | (has_main+has_side+has_bev+has_des)/4 |
| 21 | `meal_gap` | 1 − meal_comp |
| 22 | `needs_bev` | No beverage in cart |
| 23 | `needs_side` | Has main, no side |
| 24 | `needs_des` | ≥2 items, no dessert |

## 4. Temporal (10)

| # | Feature | Description |
|---|---------|-------------|
| 25 | `hour_sin` | sin(2π × hour/24) — cyclical |
| 26 | `hour_cos` | cos(2π × hour/24) |
| 27 | `dow_sin` | sin(2π × dow/7) |
| 28 | `dow_cos` | cos(2π × dow/7) |
| 29 | `is_lunch` | hour ∈ [12,14] |
| 30 | `is_dinner` | hour ∈ [19,21] |
| 31 | `is_late_night` | hour ∈ [22,23,0,1,2] |
| 32 | `is_weekend` | Sat or Sun |
| 33 | `slot_enc` | breakfast/brunch/lunch/snack/dinner/late_night |
| 55 | `weekend_dinner` | is_weekend × is_dinner |

## 5. Restaurant (5)

| # | Feature | Description |
|---|---------|-------------|
| 34 | `rest_rating` | Rating 2.5–5.0 |
| 35 | `rest_is_chain` | Chain vs independent |
| 36 | `rest_discount` | Active discount |
| 37 | `rest_quality` | (rating − 2.5)/2.5 |
| 38 | `discount_cheap` | discount AND price < ₹100 |

## 6. Price & Business (20)

| # | Feature | Description |
|---|---------|-------------|
| 39 | `price_ratio` | item_price / cart_value |
| 40 | `price_per_item` | cart_value / cart_size |
| 41 | `price_rel` | item_price / menu_avg |
| 42 | `price_elastic` | price_sensitivity × (price/200) |
| 43 | `veg_conflict` | veg_user × non_veg_item |
| 44 | `budget_block` | budget_user × price > ₹150 |
| 45 | `prem_bonus` | premium_user × price < ₹120 |
| 50 | `cart_value_bucket` | [0–200)=0, [200–400)=1, [400–700)=2, [700+)=3 |
| 52 | `c2o_risk` | cart_value > ₹700 |
| 53 | `addon_cart_pct` | item_price / (cart_value + item_price) |
| 54 | `is_micro_addon` | addon_cart_pct < 15% |
| 56 | `premium_dessert` | premium × dessert category |
| 57 | `budget_beverage` | budget × beverage category |
| 58 | `cooccur_score` | Co-occurrence from training data |
| 59 | `completion_score` | Meal pattern completion likelihood |
| 60 | `order_velocity` | Recency × frequency |
| 61 | `city_cuisine_affinity` | City-level cuisine preference |
| 62 | `price_vs_cat_median` | item_price / category_median |
| 63 | `kg_pos_weighted` | kg_strength² |
| 64 | `cart_diversity` | Distinct cuisines in cart |

## 7. AI Edge (9)

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 46 | `kg_match` | DualLLMEnricher | Binary KG pairing match |
| 47 | `kg_strength` | DualLLMEnricher | Position-weighted: 1.0/0.5/0.33/0.0 |
| 48 | `attn_compat` | CartAttentionEncoder | Cosine(attention_cart_vec, item_emb) |
| 49 | `bandit_score` | UCBBandit | UCB1 exploration score |
| 50 | `archetype` | K-Means k=7 | Meal cluster ID |
| 65 | `premium_in_cart` | Cart analysis | Any cart item > ₹250 |
| 66 | `kg_x_recency` | Cross-product | kg_strength × recency_score |
| 67 | `need_x_elast` | Cross-product | (needs_bev+needs_side) × (1−price_elastic) |
| 68 | `maturity_x_kg` | Cross-product | log_orders × kg_strength |

---

## Feature Importance (Top 10)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `kg_strength` | ~18% |
| 2 | `kg_match` | ~15% |
| 3 | `bandit_score` | ~12% |
| 4 | `needs_bev` | ~10% |
| 5 | `price_elastic` | ~8% |
| 6 | `c2o_risk` | ~7% |
| 7 | `attn_compat` | ~6% |
| 8 | `meal_gap` | ~5% |
| 9 | `recency_score` | ~4% |
| 10 | `is_cold_start` | ~3% |

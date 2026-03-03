"""
================================================================================
ZOMATHON — CSAO RAIL RECOMMENDATION SYSTEM
Dataset Generator — Synthetic Interaction Data
================================================================================
Run: python3 generate_dataset.py
Output: ./data/interactions.csv, ./data/users.csv, ./data/restaurants.csv
================================================================================
"""
import numpy as np
import pandas as pd
import os, json
from collections import defaultdict
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CITIES = ['Mumbai','Delhi','Bangalore','Hyderabad','Chennai','Pune','Kolkata']
CITY_PROFILE = {
    'Mumbai':    {'late_night':0.35,'aov':380,'veg':0.45,'price_sens':0.60},
    'Delhi':     {'late_night':0.25,'aov':420,'veg':0.40,'price_sens':0.50},
    'Bangalore': {'late_night':0.40,'aov':460,'veg':0.50,'price_sens':0.40},
    'Hyderabad': {'late_night':0.20,'aov':350,'veg':0.35,'price_sens':0.65},
    'Chennai':   {'late_night':0.15,'aov':310,'veg':0.65,'price_sens':0.70},
    'Pune':      {'late_night':0.30,'aov':340,'veg':0.55,'price_sens':0.60},
    'Kolkata':   {'late_night':0.20,'aov':290,'veg':0.60,'price_sens':0.75},
}

# (category, price, is_veg, cuisine, popularity)
MENU = {
    'Chicken Biryani':      ('main',280,False,'Biryani',0.88),
    'Veg Biryani':          ('main',220,True, 'Biryani',0.75),
    'Mutton Biryani':       ('main',340,False,'Biryani',0.82),
    'Butter Chicken':       ('main',320,False,'North Indian',0.91),
    'Dal Makhani':          ('main',240,True, 'North Indian',0.80),
    'Paneer Tikka Masala':  ('main',280,True, 'North Indian',0.84),
    'Chole Bhature':        ('main',180,True, 'North Indian',0.77),
    'Masala Dosa':          ('main',120,True, 'South Indian',0.86),
    'Idli Sambar':          ('main',90, True, 'South Indian',0.79),
    'Chicken 65':           ('main',260,False,'South Indian',0.83),
    'Margherita Pizza':     ('main',320,True, 'Pizza',0.87),
    'Chicken Pizza':        ('main',380,False,'Pizza',0.90),
    'BBQ Pizza':            ('main',400,False,'Pizza',0.88),
    'Veg Burger':           ('main',150,True, 'Burger',0.72),
    'Chicken Burger':       ('main',200,False,'Burger',0.85),
    'Zinger Burger':        ('main',230,False,'Burger',0.88),
    'Hakka Noodles':        ('main',180,True, 'Chinese',0.78),
    'Chicken Fried Rice':   ('main',220,False,'Chinese',0.83),
    'Pasta Arrabbiata':     ('main',260,True, 'Continental',0.76),
    'Chicken Pasta':        ('main',300,False,'Continental',0.81),
    'Raita':                ('side',60, True, 'Biryani',0.72),
    'Salan':                ('side',80, True, 'Biryani',0.68),
    'Mirchi Ka Salan':      ('side',70, True, 'Biryani',0.65),
    'Garlic Naan':          ('side',40, True, 'North Indian',0.80),
    'Butter Roti':          ('side',30, True, 'North Indian',0.73),
    'Sambar':               ('side',40, True, 'South Indian',0.77),
    'Coconut Chutney':      ('side',30, True, 'South Indian',0.74),
    'Caesar Salad':         ('side',160,True, 'Continental',0.55),
    'Garlic Bread':         ('side',90, True, 'Continental',0.78),
    'Coleslaw':             ('side',60, True, 'Burger',0.65),
    'Onion Rings':          ('side',80, True, 'Burger',0.68),
    'Spring Rolls':         ('side',140,True, 'Chinese',0.70),
    'Manchurian':           ('side',160,True, 'Chinese',0.72),
    'Coke':                 ('beverage',60, True,'Universal',0.82),
    'Lassi':                ('beverage',80, True,'North Indian',0.77),
    'Masala Chai':          ('beverage',50, True,'Universal',0.71),
    'Cold Coffee':          ('beverage',120,True,'Universal',0.68),
    'Fresh Lime Soda':      ('beverage',70, True,'Universal',0.74),
    'Mango Shake':          ('beverage',110,True,'Universal',0.72),
    'Watermelon Juice':     ('beverage',90, True,'Universal',0.65),
    'Gulab Jamun':          ('dessert',90, True,'North Indian',0.76),
    'Ice Cream':            ('dessert',110,True,'Universal',0.70),
    'Rasmalai':             ('dessert',120,True,'North Indian',0.68),
    'Brownie':              ('dessert',140,True,'Universal',0.65),
    'Kheer':                ('dessert',80, True,'North Indian',0.72),
    'Payasam':              ('dessert',70, True,'South Indian',0.69),
}
ITEMS = list(MENU.keys())
ITEM_IDX = {item:i for i,item in enumerate(ITEMS)}

KG = {
    'Chicken Biryani':      {'side':['Raita','Salan','Mirchi Ka Salan'],'beverage':['Lassi','Coke'],'dessert':['Gulab Jamun','Kheer']},
    'Veg Biryani':          {'side':['Raita','Salan'],'beverage':['Lassi','Fresh Lime Soda'],'dessert':['Gulab Jamun','Kheer']},
    'Mutton Biryani':       {'side':['Raita','Mirchi Ka Salan'],'beverage':['Lassi','Coke'],'dessert':['Gulab Jamun','Kheer']},
    'Butter Chicken':       {'side':['Garlic Naan','Butter Roti'],'beverage':['Lassi','Coke'],'dessert':['Gulab Jamun','Rasmalai']},
    'Dal Makhani':          {'side':['Garlic Naan','Butter Roti'],'beverage':['Lassi'],'dessert':['Gulab Jamun','Kheer']},
    'Paneer Tikka Masala':  {'side':['Garlic Naan'],'beverage':['Lassi','Cold Coffee'],'dessert':['Rasmalai','Gulab Jamun']},
    'Chole Bhature':        {'side':['Sambar'],'beverage':['Lassi','Masala Chai'],'dessert':['Gulab Jamun']},
    'Masala Dosa':          {'side':['Sambar','Coconut Chutney'],'beverage':['Masala Chai'],'dessert':['Payasam']},
    'Idli Sambar':          {'side':['Coconut Chutney'],'beverage':['Masala Chai'],'dessert':['Payasam']},
    'Chicken 65':           {'side':['Sambar'],'beverage':['Fresh Lime Soda','Coke'],'dessert':['Payasam','Ice Cream']},
    'Margherita Pizza':     {'side':['Garlic Bread','Caesar Salad'],'beverage':['Coke','Cold Coffee'],'dessert':['Brownie','Ice Cream']},
    'Chicken Pizza':        {'side':['Garlic Bread','Onion Rings'],'beverage':['Coke'],'dessert':['Brownie','Ice Cream']},
    'BBQ Pizza':            {'side':['Garlic Bread','Onion Rings'],'beverage':['Coke','Cold Coffee'],'dessert':['Brownie']},
    'Veg Burger':           {'side':['Coleslaw','Onion Rings'],'beverage':['Coke','Mango Shake'],'dessert':['Ice Cream','Brownie']},
    'Chicken Burger':       {'side':['Coleslaw','Onion Rings'],'beverage':['Coke','Cold Coffee'],'dessert':['Ice Cream']},
    'Zinger Burger':        {'side':['Coleslaw'],'beverage':['Coke','Cold Coffee'],'dessert':['Ice Cream','Brownie']},
    'Hakka Noodles':        {'side':['Spring Rolls','Manchurian'],'beverage':['Coke'],'dessert':['Ice Cream']},
    'Chicken Fried Rice':   {'side':['Spring Rolls','Manchurian'],'beverage':['Coke'],'dessert':['Ice Cream']},
    'Pasta Arrabbiata':     {'side':['Garlic Bread','Caesar Salad'],'beverage':['Cold Coffee'],'dessert':['Brownie']},
    'Chicken Pasta':        {'side':['Garlic Bread'],'beverage':['Cold Coffee','Watermelon Juice'],'dessert':['Brownie']},
}

def get_slot(h):
    if h < 6:   return 'late_night'
    elif h < 9: return 'breakfast'
    elif h < 12: return 'brunch'
    elif h < 15: return 'lunch'
    elif h < 18: return 'snack'
    elif h < 22: return 'dinner'
    else: return 'late_night'

# ─────────────────────────────────────────────────────────────────────────────
# GENERATORS
# ─────────────────────────────────────────────────────────────────────────────
def gen_users(n=2500):
    print(f"  Generating {n} users...")
    cities = np.random.choice(CITIES, n, p=[.25,.20,.20,.10,.10,.08,.07])
    segs   = np.random.choice(['budget','mid','premium'], n, p=[.35,.45,.20])
    orders = np.clip(np.random.exponential(12, n).astype(int)+1, 1, 150)
    veg    = np.array([np.random.random() < CITY_PROFILE[c]['veg'] for c in cities])
    ps     = np.array([CITY_PROFILE[c]['price_sens'] * (1.3 if s=='budget' else 0.7 if s=='premium' else 1.0)
                       for c, s in zip(cities, segs)])
    df = pd.DataFrame({
        'user_id': range(n), 'city': cities, 'segment': segs,
        'total_orders': orders, 'is_veg': veg,
        'price_sensitivity': np.clip(ps, 0.2, 0.95),
        'recency_score': np.random.beta(2, 3, n)
    })
    return df

def gen_restaurants(n=600):
    print(f"  Generating {n} restaurants...")
    cuisines = np.random.choice(['North Indian','South Indian','Chinese','Pizza','Burger','Biryani','Continental'],
                                n, p=[.20,.15,.15,.15,.15,.12,.08])
    return pd.DataFrame({
        'restaurant_id': range(n), 'cuisine': cuisines,
        'city': np.random.choice(CITIES, n),
        'rating': np.round(np.clip(np.random.normal(4., 0.5, n), 2.5, 5.), 1),
        'is_chain': (np.random.random(n) > 0.65).astype(int),
        'price_bracket': np.random.choice(['budget','mid','premium'], n, p=[.4,.45,.15]),
        'discount': (np.random.random(n) > 0.6).astype(int)
    })

def simulate_acceptance(cart, cand, user, hour, rest):
    """14-factor probabilistic acceptance simulation — production-grade label generation."""
    cat, price, is_veg_i, cuisine, pop = MENU[cand]
    cart_cats = set(MENU[i][0] for i in cart)
    cart_val  = sum(MENU[i][1] for i in cart)
    prob = 0.12

    # 1. Knowledge Graph pairing (primary signal, position-weighted)
    mains = [i for i in cart if MENU[i][0]=='main' and i in KG]
    if mains:
        rules = KG[mains[0]]
        if cand in rules.get(cat, []):
            pos = rules[cat].index(cand)
            prob += 0.38 - pos * 0.05
        if len(mains) > 1 and cand in KG.get(mains[1], {}).get(cat, []):
            prob += 0.10

    # 2. Meal completion signals
    if cat == 'beverage' and 'beverage' not in cart_cats: prob += 0.22
    if cat == 'side' and 'side' not in cart_cats and 'main' in cart_cats: prob += 0.18
    if cat == 'dessert' and 'dessert' not in cart_cats and len(cart) >= 2: prob += 0.12

    # 3. Item popularity
    prob += (pop - 0.7) * 0.15

    # 4. Price elasticity
    pf = 1.0 - user['price_sensitivity'] * (price / 300.0)
    prob *= max(0.28, pf)

    # 5. Veg hard constraint
    if user['is_veg'] and not is_veg_i:
        prob = 0.01

    # 6. Meal-slot context
    slot = get_slot(hour)
    if cat == 'beverage' and slot in ['lunch','dinner']: prob += 0.10
    if cat == 'beverage' and slot == 'breakfast': prob += 0.08
    if cat == 'dessert' and slot == 'late_night': prob -= 0.07
    if cat == 'dessert' and slot in ['dinner','lunch']: prob += 0.06

    # 7. Discount sensitivity
    if rest['discount'] and price < 120: prob += 0.10

    # 8. User maturity
    if user['total_orders'] < 3: prob *= 0.65
    if user['segment'] == 'premium': prob *= 1.18
    elif user['segment'] == 'budget' and price > 150: prob *= 0.55

    # 9. Cart saturation (diminishing returns after 3+ items)
    prob *= max(0.45, 1.0 - 0.09 * len(cart))

    # 10. Recency — active users more receptive
    prob *= (0.82 + 0.32 * user['recency_score'])

    # 11. Restaurant quality
    prob *= (0.70 + 0.30 * (rest['rating'] - 2.5) / 2.5)

    # 12. C2O risk — very high cart value signals ordering friction
    if cart_val > 700: prob *= 0.80
    if cart_val > 1000: prob *= 0.70

    # 13. Cuisine alignment bonus
    main_cuisine = MENU[mains[0]][3] if mains else 'Universal'
    if cuisine == main_cuisine or cuisine == 'Universal': prob += 0.05

    # 14. Cart size saturation curve (secondary)
    prob *= max(0.50, 1.0 - 0.07 * max(0, len(cart) - 1))

    prob = np.clip(prob, 0.01, 0.95)
    return int(np.random.random() < prob), prob

def gen_interactions(users_df, rests_df, n_sessions=20000):
    print(f"  Generating {n_sessions} sessions (~{n_sessions*10:,} interaction rows)...")
    records = []
    mains = [k for k, v in MENU.items() if v[0] == 'main']
    for sid in range(n_sessions):
        user = users_df.sample(1).iloc[0]
        rest = rests_df.sample(1).iloc[0]
        cp   = CITY_PROFILE[user['city']]

        # City-realistic hour distribution
        if np.random.random() < cp['late_night']:
            hour = np.random.choice([23, 0, 1, 2])
        elif np.random.random() < 0.55:
            hour = np.random.choice(list(range(12, 15)) + list(range(19, 22)))
        else:
            hour = np.random.randint(6, 23)
        dow = np.random.randint(0, 7)

        # Cart composition
        nm   = np.random.choice([1, 2, 3], p=[.62, .30, .08])
        cart = list(np.random.choice(mains, min(nm, len(mains)), replace=False))
        if user['is_veg']:
            cart = [i for i in cart if MENU[i][2]]
            if not cart:
                veg_mains = [k for k, v in MENU.items() if v[0]=='main' and v[2]]
                cart = [np.random.choice(veg_mains)]
        cv  = sum(MENU[i][1] for i in cart)
        ccs = set(MENU[i][0] for i in cart)

        # Candidate items
        cands = [i for i in ITEMS if i not in cart and MENU[i][0] != 'main']
        if user['is_veg']:
            cands = [i for i in cands if MENU[i][2]]
        cands = list(np.random.choice(cands, min(10, len(cands)), replace=False))

        for cand in cands:
            cat, price, is_veg_i, cuisine, pop = MENU[cand]
            mc  = [i for i in cart if i in KG]
            kg, kgs = 0, 0.0
            if mc:
                rules = KG.get(mc[0], {})
                if cand in rules.get(cat, []):
                    kg  = 1
                    pos = rules[cat].index(cand)
                    kgs = 1.0 / (pos + 1)
            lbl, tp = simulate_acceptance(cart, cand, user, hour, rest)
            meal_c  = (int('main' in ccs) + int('side' in ccs) + int('beverage' in ccs) + int('dessert' in ccs)) / 4.0
            records.append({
                'session_id': sid, 'user_id': user['user_id'], 'rest_id': rest['restaurant_id'],
                'cart_items': '|'.join(cart), 'cart_size': len(cart), 'cart_value': cv,
                'candidate': cand, 'cat': cat, 'price': price, 'is_veg_item': int(is_veg_i),
                'popularity': pop, 'cuisine': cuisine,
                'segment': user['segment'], 'city': user['city'],
                'total_orders': user['total_orders'], 'user_is_veg': int(user['is_veg']),
                'price_sensitivity': user['price_sensitivity'], 'recency_score': user['recency_score'],
                'hour': hour, 'dow': dow, 'meal_slot': get_slot(hour), 'is_weekend': int(dow >= 5),
                'rest_rating': rest['rating'], 'rest_discount': int(rest['discount']),
                'rest_is_chain': int(rest['is_chain']),
                'kg_match': kg, 'kg_strength': kgs,
                'has_main': int('main' in ccs), 'has_side': int('side' in ccs),
                'has_bev': int('beverage' in ccs), 'has_des': int('dessert' in ccs),
                'meal_comp': meal_c,
                'label': lbl, 'true_prob': tp
            })
        if (sid + 1) % 5000 == 0:
            print(f"    {sid+1}/{n_sessions} sessions done...")

    df = pd.DataFrame(records)
    print(f"  ✓ {len(df):,} rows | {df['label'].mean():.2%} positive rate")
    return df

def main():
    print("=" * 60)
    print("  CSAO Dataset Generator — Zomathon")
    print("=" * 60)
    os.makedirs('./data', exist_ok=True)

    users_df        = gen_users(2500)
    rests_df        = gen_restaurants(600)
    interactions_df = gen_interactions(users_df, rests_df, n_sessions=20000)

    users_df.to_csv('./data/users.csv', index=False)
    rests_df.to_csv('./data/restaurants.csv', index=False)
    interactions_df.to_csv('./data/interactions.csv', index=False)

    print("\n✓ Saved:")
    print(f"  ./data/users.csv ({len(users_df):,} rows)")
    print(f"  ./data/restaurants.csv ({len(rests_df):,} rows)")
    print(f"  ./data/interactions.csv ({len(interactions_df):,} rows)")

    print("\nPositive rate breakdown:")
    print(interactions_df.groupby('segment')['label'].mean().to_string())
    print("\nCity-wise positive rate:")
    print(interactions_df.groupby('city')['label'].mean().to_string())
    print("\nRun csao_recommendation.py to train models on this dataset.")

    # Save metadata
    meta = {
        'n_users': len(users_df), 'n_restaurants': len(rests_df),
        'n_sessions': int(interactions_df['session_id'].nunique()),
        'n_interactions': len(interactions_df),
        'positive_rate': float(interactions_df['label'].mean()),
        'n_menu_items': len(MENU),
        'cities': CITIES,
        'features': list(interactions_df.columns),
    }
    with open('./data/metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print("  ./data/metadata.json (schema + stats)")

if __name__ == '__main__':
    main()
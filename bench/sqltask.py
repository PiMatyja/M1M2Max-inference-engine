# sqltask.py: the SQL migration task, its fixture database, and an exact checker (pure Python reference).
import re, sqlite3

SCHEMA = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE, country TEXT NOT NULL);
CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT NOT NULL UNIQUE, name TEXT NOT NULL, category TEXT NOT NULL);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(id),
  created_at TEXT NOT NULL, status TEXT NOT NULL CHECK (status IN ('paid','refunded','pending')));
CREATE TABLE order_items (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(id),
  product_id INTEGER NOT NULL REFERENCES products(id), quantity INTEGER NOT NULL, unit_price_cents INTEGER NOT NULL);
CREATE TABLE legacy_discounts (id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(id), discount_cents INTEGER NOT NULL);
"""

CUSTOMERS = [(1, "Ana", "ana@x.test", "PL"), (2, "Ben", "ben@x.test", "DE"), (3, "Cy", "cy@x.test", "US"),
             (4, "Dee", "dee@x.test", "PL"), (5, "Eve", "eve@x.test", "FR"), (6, "Fox", "fox@x.test", "US"),
             (7, "Gus", "gus@x.test", "DE"), (8, "Hal", "hal@x.test", "PL")]
PRODUCTS = [(1, "BK-1", "Book", "books"), (2, "BK-2", "Atlas", "books"), (3, "TL-1", "Hammer", "tools"),
            (4, "TL-2", "Saw", "tools"), (5, "GM-1", "Chess", "games")]
ORDERS = [  # (id, customer, created_at, status)
    (1, 1, "2026-01-05 10:00:00", "paid"), (2, 1, "2026-02-10 09:30:00", "paid"), (3, 1, "2026-03-01 12:00:00", "refunded"),
    (4, 2, "2026-01-20 08:00:00", "paid"), (5, 2, "2026-04-02 16:45:00", "pending"),
    (6, 3, "2026-05-05 11:11:00", "pending"),                       # only pending: kept, zero stats
    (7, 4, "2026-02-14 14:00:00", "paid"), (8, 4, "2026-02-15 14:00:00", "paid"),  # category tie
    (9, 6, "2026-06-01 10:00:00", "refunded"),                      # only refunded: kept, zero stats
    (10, 8, "2026-07-07 07:07:00", "paid"),                          # discount bigger than items is allowed, value may go negative
    (11, 2, "2026-03-03 03:03:00", "paid"),
]
ITEMS = [  # (id, order, product, qty, unit_price_cents)
    (1, 1, 1, 2, 1500), (2, 1, 3, 1, 4000), (3, 2, 2, 1, 3000), (4, 3, 4, 5, 9900),
    (5, 4, 3, 2, 4000), (6, 4, 5, 1, 2500), (7, 5, 1, 10, 1500), (8, 6, 2, 1, 3000),
    (9, 7, 1, 2, 2000), (10, 8, 5, 1, 4000), (11, 9, 4, 1, 9900), (12, 10, 5, 1, 500),
    (13, 11, 4, 1, 7000), (14, 11, 1, 1, 1000),
]
DISCOUNTS = [(1, 1, 500), (2, 1, 250), (3, 4, 1000), (4, 3, 2000), (5, 10, 800), (6, 11, 0), (7, 7, 300)]
# Customers 5 (Eve) and 7 (Gus) have no orders at all and must be deleted.

PROMPT = """You are working on a SQLite 3.40+ database with this schema:

```sql
""" + SCHEMA.strip() + """
```

Write ONE SQLite script that does all of the following, inside a single transaction:

1. Add a column `discount_cents INTEGER NOT NULL DEFAULT 0` to `orders`. Fill it from `legacy_discounts`: an order can have several discount rows, so use their sum. Orders without discount rows keep 0.
2. Drop the `legacy_discounts` table once its data is moved. Do not drop or recreate any other table.
3. Create a table `customer_stats` with columns `customer_id INTEGER PRIMARY KEY`, `paid_orders INTEGER NOT NULL`, `lifetime_value_cents INTEGER NOT NULL`, `first_paid_at TEXT`, `last_paid_at TEXT`, `top_category TEXT`, and fill it with one row per customer that still exists after step 4:
   - Only orders with status 'paid' count.
   - `lifetime_value_cents` = sum over paid orders of (sum of quantity * unit_price_cents of its items) minus that order's discount_cents. It may be negative.
   - `first_paid_at` / `last_paid_at` = earliest / latest `created_at` of the customer's paid orders.
   - `top_category` = the product category with the highest item spend (quantity * unit_price_cents, before discounts) across the customer's paid orders; break ties by the alphabetically first category name.
   - Customers with no paid orders get paid_orders 0, lifetime_value_cents 0 and NULL for the other three columns.
4. Delete customers who have no orders at all (of any status). Do not delete any other rows from any table.

Reply with the complete script in a single ```sql code block."""


def build(path=":memory:"):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
    db.executemany("INSERT INTO products VALUES (?,?,?,?)", PRODUCTS)
    db.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    db.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", ITEMS)
    db.executemany("INSERT INTO legacy_discounts VALUES (?,?,?)", DISCOUNTS)
    db.commit()
    return db


def expected():
    disc = {}
    for _, o, d in DISCOUNTS:
        disc[o] = disc.get(o, 0) + d
    cat = {p[0]: p[3] for p in PRODUCTS}
    have_orders = {o[1] for o in ORDERS}
    stats = {}
    for c in CUSTOMERS:
        if c[0] not in have_orders:
            continue
        paid = [o for o in ORDERS if o[1] == c[0] and o[3] == "paid"]
        if not paid:
            stats[c[0]] = (c[0], 0, 0, None, None, None)
            continue
        value, spend = 0, {}
        for o in paid:
            for it in ITEMS:
                if it[1] == o[0]:
                    value += it[3] * it[4]
                    spend[cat[it[2]]] = spend.get(cat[it[2]], 0) + it[3] * it[4]
            value -= disc.get(o[0], 0)
        top = sorted(spend.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if spend else None
        dates = sorted(o[2] for o in paid)
        stats[c[0]] = (c[0], len(paid), value, dates[0], dates[-1], top)
    return {
        "tables": {"customers", "products", "orders", "order_items", "customer_stats"},
        "customers": sorted(c for c in CUSTOMERS if c[0] in have_orders),
        "orders": sorted(o + (disc.get(o[0], 0),) for o in ORDERS),
        "order_items": sorted(ITEMS),
        "products": sorted(PRODUCTS),
        "customer_stats": sorted(stats.values()),
    }


def extract_sql(text):
    blocks = re.findall(r"```sql\s*\n(.*?)```", text or "", re.S | re.I)
    return max(blocks, key=len) if blocks else None


def check(sql):
    """Returns (passed, list of problems)."""
    if not sql:
        return False, ["no ```sql block"]
    db = build()
    try:
        db.executescript(sql)
    except Exception as e:
        return False, [f"script error: {e}"]
    exp, problems = expected(), []
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
    if tables != exp["tables"]:
        problems.append(f"tables {sorted(tables)}")
    if db.in_transaction:
        problems.append("left a transaction open")
    def rows(q):
        try:
            return sorted(tuple(r) for r in db.execute(q))
        except Exception as e:
            return [f"query error: {e}"]
    for name, q in [("customers", "SELECT id,name,email,country FROM customers"),
                    ("orders", "SELECT id,customer_id,created_at,status,discount_cents FROM orders"),
                    ("order_items", "SELECT id,order_id,product_id,quantity,unit_price_cents FROM order_items"),
                    ("products", "SELECT id,sku,name,category FROM products"),
                    ("customer_stats", "SELECT customer_id,paid_orders,lifetime_value_cents,first_paid_at,last_paid_at,top_category FROM customer_stats")]:
        got = rows(q)
        if got != exp[name]:
            problems.append(f"{name} wrong: got {got[:6]} expected {exp[name][:6]}")
    try:
        col = [r for r in db.execute("PRAGMA table_info(orders)") if r[1] == "discount_cents"]
        if not col or col[0][3] != 1 or str(col[0][4]) != "0":
            problems.append(f"discount_cents column definition {col}")
    except Exception as e:
        problems.append(f"pragma error {e}")
    return not problems, problems


if __name__ == "__main__":
    good = """BEGIN;
ALTER TABLE orders ADD COLUMN discount_cents INTEGER NOT NULL DEFAULT 0;
UPDATE orders SET discount_cents = COALESCE((SELECT SUM(discount_cents) FROM legacy_discounts d WHERE d.order_id = orders.id), 0);
DROP TABLE legacy_discounts;
DELETE FROM customers WHERE id NOT IN (SELECT customer_id FROM orders);
CREATE TABLE customer_stats (customer_id INTEGER PRIMARY KEY, paid_orders INTEGER NOT NULL, lifetime_value_cents INTEGER NOT NULL,
  first_paid_at TEXT, last_paid_at TEXT, top_category TEXT);
WITH ov AS (SELECT o.id, o.customer_id, o.created_at, COALESCE((SELECT SUM(quantity*unit_price_cents) FROM order_items i WHERE i.order_id=o.id),0) - o.discount_cents AS v
            FROM orders o WHERE o.status='paid'),
cs AS (SELECT o.customer_id, p.category, SUM(i.quantity*i.unit_price_cents) AS s FROM orders o JOIN order_items i ON i.order_id=o.id
       JOIN products p ON p.id=i.product_id WHERE o.status='paid' GROUP BY 1,2),
top AS (SELECT customer_id, category FROM (SELECT customer_id, category, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY s DESC, category) rn FROM cs) WHERE rn=1)
INSERT INTO customer_stats SELECT c.id, COUNT(ov.id), COALESCE(SUM(ov.v),0), MIN(ov.created_at), MAX(ov.created_at), (SELECT category FROM top WHERE top.customer_id=c.id)
FROM customers c LEFT JOIN ov ON ov.customer_id=c.id GROUP BY c.id;
COMMIT;"""
    print("good:", check(good))
    print("no tiebreak:", check(good.replace("ORDER BY s DESC, category", "ORDER BY s DESC, category DESC"))[0])
    print("drops orders too:", check(good.replace("DROP TABLE legacy_discounts;", "DROP TABLE legacy_discounts; DROP TABLE order_items;"))[0])
    print("counts refunded:", check(good.replace("WHERE o.status='paid' GROUP", "GROUP"))[0])
    print("deletes zero-paid customers:", check(good.replace("SELECT customer_id FROM orders)", "SELECT customer_id FROM orders WHERE status='paid')"))[0])
    print("no sum of discounts:", check(good.replace("SELECT SUM(discount_cents)", "SELECT MAX(discount_cents)"))[0])
    print(extract_sql("text\n```sql\nSELECT 1;\n```\n") == "SELECT 1;\n")

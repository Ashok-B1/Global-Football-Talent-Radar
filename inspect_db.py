import duckdb

db = r"C:\Users\banot\Downloads\transfermarkt-datasets.duckdb"
con = duckdb.connect(db, read_only=True)

print("\n=== COMPETITIONS ===")
print(
    con.sql("""
        SELECT competition_id, name, country_name, confederation, total_clubs
        FROM competitions
        ORDER BY country_name, name
    """).df().to_string(index=False)
)

print("\n=== APPEARANCES ===")
print(
    con.sql("""
        SELECT MIN(date) AS earliest_date,
               MAX(date) AS latest_date,
               COUNT(*) AS rows
        FROM appearances
    """).df().to_string(index=False)
)

print("\n=== PLAYER VALUATIONS ===")
print(
    con.sql("""
        SELECT MIN(date) AS earliest_date,
               MAX(date) AS latest_date,
               COUNT(*) AS rows
        FROM player_valuations
    """).df().to_string(index=False)
)

print("\n=== TRANSFERS ===")
print(
    con.sql("""
        SELECT MIN(transfer_date) AS earliest_date,
               MAX(transfer_date) AS latest_date,
               COUNT(*) AS rows
        FROM transfers
    """).df().to_string(index=False)
)

con.close()

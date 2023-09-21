import pandas as pd
import snowflake.connector
from authenticate import user, password, account, warehouse, database, schema

# connect to Snowflake
conn = snowflake.connector.connect(
    user=user,
    password = password,
    account = account,
    warehouse = warehouse,
    database = database,
    scheme=schema
)
query = 'SELECT * FROM AMAZON_AND_ECOMMERCE_WEBSITES_PRODUCT_VIEWS_AND_PURCHASES.DATAFEEDS.PRODUCT_VIEWS_AND_PURCHASES LIMIT 1000000'

cur = conn.cursor()
cur.execute(query)

chunks = []
while True:
    batch = cur.fetchmany(10000) # fetch 10k rows at a time
    if not batch:
        break
    chunk = pd.DataFrame(batch, columns=[desc[0] for desc in cur.description])

    # add chunk to list of chunks
    chunks.append(chunk)

# close curosr and connection
cur.close()
# Close the connection
conn.close()

# concetant chunks and save as compressed Parquet file
df = pd.concat(chunks, axis=0)
df.to_parquet('compressed.parquet')
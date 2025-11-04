import pandas as pd
import pyodbc

# Update this string if your SQL Server uses a different instance/name
connection_string = (
    r"DRIVER={ODBC Driver 17 for SQL Server};"
    r"SERVER=localhost;"
    r"DATABASE=AdventureWorks2022;"
    r"Trusted_Connection=yes;"
)

# Example query (same as your previous export)
query = """
SELECT
    c.CustomerID,
    DATEDIFF(day, MIN(soh.OrderDate), MAX(soh.OrderDate)) AS TenureDays,
    DATEDIFF(day, MAX(soh.OrderDate), GETDATE()) AS RecencyDays,
    COUNT(soh.SalesOrderID) AS Frequency,
    SUM(soh.TotalDue) AS MonetaryValue
FROM Sales.Customer c
JOIN Sales.SalesOrderHeader soh ON c.CustomerID = soh.CustomerID
GROUP BY c.CustomerID
ORDER BY MonetaryValue DESC


"""

try:
    # Create connection
    conn = pyodbc.connect(connection_string)
    # Read SQL query into DataFrame
    df = pd.read_sql(query, conn)
    # Print summary
    print(df.head())
    # Optional: save (overwrite) to CSV automatically
    df.to_csv("high_value_orders.csv", index=False)
    print("Data exported to high_value_orders.csv")
    conn.close()
except Exception as e:
    print("Error:", e)

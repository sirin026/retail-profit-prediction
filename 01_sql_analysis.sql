-- ============================================================
-- Retail Profit Prediction — SQL Exploratory Analysis
-- Dataset: US Superstore Sales (2014–2017)
-- Author: Siri Namala
-- ============================================================

-- Q1: Overall KPIs
SELECT
    COUNT(*)                                AS total_orders,
    ROUND(SUM(sales), 2)                    AS total_sales,
    ROUND(SUM(profit), 2)                   AS total_profit,
    ROUND(AVG(profit), 2)                   AS avg_profit_per_order,
    ROUND(AVG(discount) * 100, 1)           AS avg_discount_pct,
    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) AS loss_orders,
    ROUND(SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS loss_rate_pct
FROM superstore;

-- Q2: Profit by Product Category
SELECT
    category,
    COUNT(*)                        AS order_count,
    ROUND(SUM(sales), 2)            AS total_sales,
    ROUND(SUM(profit), 2)           AS total_profit,
    ROUND(AVG(profit), 2)           AS avg_profit,
    ROUND(SUM(profit)/SUM(sales)*100, 1) AS profit_margin_pct
FROM superstore
GROUP BY category
ORDER BY total_profit DESC;

-- Q3: Profit by Region
SELECT
    region,
    COUNT(*)                        AS order_count,
    ROUND(SUM(profit), 2)           AS total_profit,
    ROUND(AVG(profit), 2)           AS avg_profit,
    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) AS loss_orders
FROM superstore
GROUP BY region
ORDER BY total_profit DESC;

-- Q4: Discount Impact on Profit (Bucketed)
SELECT
    CASE
        WHEN discount = 0             THEN 'No Discount'
        WHEN discount <= 0.10         THEN '1-10%'
        WHEN discount <= 0.20         THEN '11-20%'
        WHEN discount <= 0.40         THEN '21-40%'
        WHEN discount <= 0.60         THEN '41-60%'
        ELSE '60%+'
    END AS discount_bucket,
    COUNT(*)                          AS order_count,
    ROUND(AVG(profit), 2)             AS avg_profit,
    SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END) AS loss_orders,
    ROUND(SUM(CASE WHEN profit < 0 THEN 1 ELSE 0 END)*100.0/COUNT(*), 1) AS loss_rate_pct
FROM superstore
GROUP BY discount_bucket
ORDER BY avg_profit DESC;

-- Q5: Top 10 Most Profitable Sub-Categories
SELECT
    sub_category,
    COUNT(*)                        AS order_count,
    ROUND(SUM(profit), 2)           AS total_profit,
    ROUND(AVG(profit), 2)           AS avg_profit,
    ROUND(AVG(discount)*100, 1)     AS avg_discount_pct
FROM superstore
GROUP BY sub_category
ORDER BY total_profit DESC
LIMIT 10;

-- Q6: Bottom 10 Loss-Making Sub-Categories
SELECT
    sub_category,
    COUNT(*)                        AS order_count,
    ROUND(SUM(profit), 2)           AS total_profit,
    ROUND(AVG(profit), 2)           AS avg_profit,
    ROUND(AVG(discount)*100, 1)     AS avg_discount_pct
FROM superstore
GROUP BY sub_category
ORDER BY total_profit ASC
LIMIT 10;

-- Q7: Profit by Shipping Mode
SELECT
    ship_mode,
    COUNT(*)                        AS order_count,
    ROUND(AVG(profit), 2)           AS avg_profit,
    ROUND(AVG(JULIANDAY(ship_date) - JULIANDAY(order_date)), 1) AS avg_ship_days,
    ROUND(SUM(profit), 2)           AS total_profit
FROM superstore
GROUP BY ship_mode
ORDER BY avg_profit DESC;

-- Q8: Monthly Profit Trend
SELECT
    STRFTIME('%Y-%m', order_date)   AS year_month,
    COUNT(*)                        AS order_count,
    ROUND(SUM(profit), 2)           AS monthly_profit,
    ROUND(AVG(profit), 2)           AS avg_profit
FROM superstore
GROUP BY year_month
ORDER BY year_month;

-- Q9: High-Value Customers (Top 10 by Profit Contribution)
SELECT
    customer_name,
    segment,
    COUNT(*)                        AS order_count,
    ROUND(SUM(profit), 2)           AS total_profit,
    ROUND(SUM(sales), 2)            AS total_sales
FROM superstore
GROUP BY customer_name, segment
ORDER BY total_profit DESC
LIMIT 10;

-- Q10: Category × Region Profitability Matrix
SELECT
    category,
    region,
    COUNT(*)                        AS order_count,
    ROUND(SUM(profit), 2)           AS total_profit,
    ROUND(AVG(profit), 2)           AS avg_profit,
    ROUND(AVG(discount)*100, 1)     AS avg_discount_pct
FROM superstore
GROUP BY category, region
ORDER BY category, total_profit DESC;

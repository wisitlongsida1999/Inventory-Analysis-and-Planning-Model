# การคำนวณในระบบวิเคราะห์และวางแผนสต็อก

## 1. การคำนวณ Service Level และ Safety Stock

```python
z_score = {0.90: 1.28, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}.get(self.service_level, 1.96)
safety_stock = int(z_score * std_monthly_sales * np.sqrt(self.lead_time_months))
```

- ใช้ z-score ในการคำนวณระดับความเชื่อมั่น (Service Level) 
- Safety Stock คำนวณจาก: z-score × ค่าเบี่ยงเบนมาตรฐานของยอดขาย × รากที่สองของ lead time
- ยิ่ง Service Level สูง Safety Stock จะยิ่งมาก เพื่อป้องกันของขาด

## 2. การคำนวณ Reorder Point

```python
reorder_point = int((avg_monthly_sales * self.lead_time_months) + safety_stock)
```

- จุดสั่งซื้อ = (ยอดขายเฉลี่ยต่อเดือน × ระยะเวลานำ) + Safety Stock
- เมื่อสต็อกลดลงถึงจุดนี้ควรสั่งซื้อเพิ่ม

## 3. การพยากรณ์ยอดขาย 3 เดือนล่วงหน้า

```python
model = ExponentialSmoothing(
    product_sales['productQuantity'],
    seasonal_periods=4,
    trend='add',
    seasonal='add',
    initialization_method="estimated"
).fit()
forecast = model.forecast(3)
predicted_sales_3m = int(forecast.sum())
```

- ใช้ Holt-Winters Exponential Smoothing ในการพยากรณ์
- คำนึงถึงทั้ง trend และ seasonality ของข้อมูล
- ถ้าข้อมูลน้อยเกินไป จะใช้ค่าเฉลี่ยแทน

## 4. การคำนวณ Stock Coverage

```python
stock_coverage = current_stock / avg_monthly_sales if avg_monthly_sales > 0 else float('inf')
```

- บอกว่าสต็อกปัจจุบันจะอยู่ได้กี่เดือน
- คำนวณจาก: สต็อกปัจจุบัน ÷ ยอดขายเฉลี่ยต่อเดือน

## 5. การคำนวณ Excess Stock

```python
excess_stock = max(0, current_stock - (predicted_sales_3m + safety_stock))
```

- สต็อกส่วนเกิน = สต็อกปัจจุบัน - (ยอดขายที่พยากรณ์ + Safety Stock)
- ถ้าติดลบให้นับเป็น 0

## 6. การคำนวณ Inventory Turnover

```python
inventory_value = current_stock * unit_value
annual_sales_value = (avg_monthly_sales * 12) * unit_value
turnover_ratio = annual_sales_value / inventory_value if inventory_value > 0 else 0
```

- อัตราการหมุนเวียนสินค้า = มูลค่าขายต่อปี ÷ มูลค่าสต็อกปัจจุบัน
- บ่งบอกประสิทธิภาพในการบริหารสต็อก

## การแสดงผลและการแจ้งเตือน

### กราฟวิเคราะห์ 3 แบบ
1. ระยะเวลาที่สต็อกจะอยู่ได้
2. เปรียบเทียบสต็อกปัจจุบันกับยอดขายที่คาดการณ์
3. มูลค่าสต็อกส่วนเกิน

### ระบบแจ้งเตือน
- สต็อกสูงเกินไป (> 5 เดือน)
- สต็อกต่ำเกินไป (< 1.5 เดือน)

### การวิเคราะห์เพิ่มเติม
- มีการวิเคราะห์แบบเชิงลึกสำหรับแต่ละสินค้า 
- แสดงเทรนด์การขายเพื่อช่วยในการตัดสินใจวางแผนสต็อก
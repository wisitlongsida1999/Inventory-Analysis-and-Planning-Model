# 📊 ระบบวิเคราะห์และวางแผนสต็อก

ระบบช่วยวิเคราะห์และวางแผนการจัดการสต็อกสินค้าอัจฉริยะ สร้างด้วย Python และ Streamlit

## ✨ ความสามารถหลัก

- 📈 วิเคราะห์ข้อมูลสต็อกและยอดขายแบบอัตโนมัติ
- 🤖 พยากรณ์ยอดขายล่วงหน้า 3 เดือนด้วย Holt-Winters Algorithm
- 🎯 คำนวณจุดสั่งซื้อและ Safety Stock ที่เหมาะสม
- 📊 แสดงผลด้วยกราฟแบบ Interactive
- ⚡ แจ้งเตือนสินค้าที่มีสต็อกสูง/ต่ำเกินไป
- 💹 วิเคราะห์อัตราการหมุนเวียนสินค้า

## 🛠 การติดตั้ง

1. Clone repository:
```bash
git clone https://github.com/yourusername/Inventory-Analysis-and-Planning-Model.git
cd Inventory-Analysis-and-Planning-Model
```

2. สร้าง virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # สำหรับ Linux/Mac
# หรือ
venv\Scripts\activate  # สำหรับ Windows
```

3. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Requirements

```
streamlit
pandas
numpy
scikit-learn
statsmodels
plotly
```

## 🚀 การใช้งาน

1. รันแอปพลิเคชัน:
```bash
streamlit run app.py
```

2. เตรียมไฟล์ข้อมูล:
   - `inventory.csv`: ข้อมูลสต็อกปัจจุบัน
   - `sales.csv`: ประวัติการขาย

3. อัปโหลดไฟล์และตั้งค่าพารามิเตอร์:
   - Service Level (90-99%)
   - Lead Time (1-6 เดือน)
   - เกณฑ์แจ้งเตือนสต็อกสูง/ต่ำ

## 📊 ฟีเจอร์การวิเคราะห์

### ภาพรวมสต็อก
- จำนวนรายการสินค้าทั้งหมด
- จำนวนสินค้าที่มีสต็อกสูง/ต่ำ
- มูลค่าสต็อกส่วนเกิน
- กราฟวิเคราะห์ 3 รูปแบบ

### การวิเคราะห์เชิงลึก
- ข้อมูลสต็อกปัจจุบัน
- การขายเฉลี่ย
- จุดสั่งซื้อที่แนะนำ
- กราฟแนวโน้มการขาย

### เทรนด์การขาย
- เทรนด์การขายรวม
- เทรนด์การขายแยกตามสินค้า

## 📐 การคำนวณที่สำคัญ

- **Safety Stock**: คำนวณจาก z-score × ค่าเบี่ยงเบนมาตรฐานของยอดขาย × √lead time
- **Reorder Point**: (ยอดขายเฉลี่ยต่อเดือน × lead time) + safety stock
- **Stock Coverage**: สต็อกปัจจุบัน ÷ ยอดขายเฉลี่ยต่อเดือน
- **Excess Stock**: สต็อกปัจจุบัน - (ยอดขายที่พยากรณ์ + safety stock)

## 📄 รูปแบบไฟล์ข้อมูล

### inventory.csv
```csv
รายการสินค้า,จำนวน,มูลค่าต่อหน่วย
สินค้า A,100,500
สินค้า B,150,300
```

### sales.csv
```csv
Date,product,productQuantity,totalPrice
2024-01-01,สินค้า A,10,5000
2024-01-02,สินค้า B,15,4500
```

## 👥 การมีส่วนร่วม

ยินดีรับ Pull Requests และ Issues! สามารถแจ้งปัญหาหรือเสนอแนะการปรับปรุงได้ที่ Issues tab

## 📝 License

MIT License

## 📧 ติดต่อ

หากมีคำถามหรือข้อเสนอแนะ สามารถติดต่อได้ที่ [wisitlongsida1999@gmail.com]
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class InventoryOptimizer:
    def __init__(self, service_level=0.95, lead_time_months=3):
        self.inventory_df = None
        self.sales_df = None
        self.monthly_sales = None
        self.service_level = service_level
        self.lead_time_months = lead_time_months
        
    def load_data(self, inventory_df, sales_df):
        """
        โหลดข้อมูลจาก DataFrame
        """
        self.inventory_df = inventory_df
        self.sales_df = sales_df
        self.sales_df['documentDate'] = pd.to_datetime(self.sales_df['documentDate'])
        
    def preprocess_data(self):
        """
        เตรียมข้อมูลสำหรับการวิเคราะห์
        """
        self.monthly_sales = (self.sales_df.groupby([
            pd.Grouper(key='documentDate', freq='M'),
            'product'
        ])['productQuantity']
        .sum()
        .reset_index())
        
    def calculate_metrics(self, high_stock_months=6, low_stock_months=1):
        """
        คำนวณ metrics สำหรับการวิเคราะห์สต็อก
        """
        # แปลง service level เป็น Z-score
        z_score = {
            0.90: 1.28,
            0.95: 1.96,
            0.98: 2.33,
            0.99: 2.58
        }.get(self.service_level, 1.96)
        
        results = []
        for product in self.inventory_df['รายการสินค้า'].unique():
            current_stock = self.inventory_df[
                self.inventory_df['รายการสินค้า'] == product
            ]['จำนวน'].values[0]
            
            product_sales = self.monthly_sales[
                self.monthly_sales['product'] == product
            ].sort_values('documentDate')
            
            if len(product_sales) > 0:
                avg_monthly_sales = product_sales['productQuantity'].mean()
                std_monthly_sales = product_sales['productQuantity'].std()
                
                stock_coverage = current_stock / avg_monthly_sales if avg_monthly_sales > 0 else float('inf')
                
                safety_stock = z_score * std_monthly_sales * np.sqrt(self.lead_time_months)
                reorder_point = (avg_monthly_sales * self.lead_time_months) + safety_stock
                
                if len(product_sales) >= 4:
                    try:
                        model = ExponentialSmoothing(
                            product_sales['productQuantity'],
                            seasonal_periods=3,
                            trend='add',
                            seasonal='add',
                            initialization_method="estimated"
                        ).fit()
                        forecast = model.forecast(3)
                        predicted_sales_3m = forecast.sum()
                    except:
                        predicted_sales_3m = avg_monthly_sales * 3
                else:
                    predicted_sales_3m = avg_monthly_sales * 3
                
                excess_stock = max(0, current_stock - (predicted_sales_3m + safety_stock))
                
                unit_value = float(self.inventory_df[
                    self.inventory_df['รายการสินค้า'] == product
                ]['มูลค่าต่อหน่วย'].values[0])
                
                results.append({
                    'product': product,
                    'current_stock': current_stock,
                    'avg_monthly_sales': round(avg_monthly_sales, 2),
                    'stock_coverage_months': round(stock_coverage, 2),
                    'safety_stock': round(safety_stock, 2),
                    'reorder_point': round(reorder_point, 2),
                    'predicted_sales_3m': round(predicted_sales_3m, 2),
                    'excess_stock': round(excess_stock, 2),
                    'excess_stock_value': round(excess_stock * unit_value, 2)
                })
        
        return pd.DataFrame(results)
    
    def get_recommendations(self, results_df, high_stock_months=6, low_stock_months=1):
        """
        สร้างคำแนะนำสำหรับการจัดการสต็อก
        """
        recommendations = []
        
        for _, row in results_df.iterrows():
            if row['stock_coverage_months'] > high_stock_months:
                status = "สต็อกสูงเกินไป"
                action = "ควรระงับการสั่งซื้อและพิจารณาทำโปรโมชั่นเพื่อระบายสต็อก"
                risk_level = "สูง"
            elif row['stock_coverage_months'] < low_stock_months:
                status = "สต็อกต่ำ"
                action = f"ควรสั่งซื้อเพิ่มอย่างน้อย {int(row['reorder_point'])} ชิ้น"
                risk_level = "สูง"
            elif row['current_stock'] < row['reorder_point']:
                status = "ถึงจุดสั่งซื้อ"
                action = f"ควรสั่งซื้อเพิ่มให้ถึงระดับ {int(row['reorder_point'] + row['safety_stock'])} ชิ้น"
                risk_level = "ปานกลาง"
            else:
                status = "ปกติ"
                action = "ติดตามยอดขายและสต็อกตามปกติ"
                risk_level = "ต่ำ"
            
            recommendations.append({
                'product': row['product'],
                'status': status,
                'risk_level': risk_level,
                'action': action,
                'excess_stock_value': f"{row['excess_stock_value']:,.2f} บาท"
            })
        
        return pd.DataFrame(recommendations)

    def plot_stock_analysis(self, results_df, high_stock_months=6, low_stock_months=1):
        """
        สร้างกราฟวิเคราะห์สต็อก
        """
        fig = plt.figure(figsize=(15, 12))
        
        # Plot 1: Stock Coverage
        plt.subplot(3, 1, 1)
        colors = ['red' if x > high_stock_months else 'orange' if x < low_stock_months else 'green' 
                 for x in results_df['stock_coverage_months']]
        sns.barplot(data=results_df, x='product', y='stock_coverage_months', palette=colors)
        plt.axhline(y=high_stock_months, color='r', linestyle='--', alpha=0.5, 
                   label=f'เกณฑ์สต็อกสูงเกินไป ({high_stock_months} เดือน)')
        plt.axhline(y=low_stock_months, color='orange', linestyle='--', alpha=0.5, 
                   label=f'เกณฑ์สต็อกต่ำ ({low_stock_months} เดือน)')
        plt.title('ระยะเวลาที่สต็อกจะอยู่ได้ (เดือน)')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Plot 2: Current Stock vs Predicted Sales
        plt.subplot(3, 1, 2)
        width = 0.35
        x = np.arange(len(results_df))
        
        plt.bar(x - width/2, results_df['current_stock'], width, label='สต็อกปัจจุบัน')
        plt.bar(x + width/2, results_df['predicted_sales_3m'], width, label='ยอดขายที่คาดการณ์ (3 เดือน)')
        
        plt.xlabel('สินค้า')
        plt.ylabel('จำนวน')
        plt.title('เปรียบเทียบสต็อกปัจจุบันกับยอดขายที่คาดการณ์')
        plt.xticks(x, results_df['product'], rotation=45)
        plt.legend()
        
        # Plot 3: Excess Stock Value
        plt.subplot(3, 1, 3)
        sns.barplot(data=results_df, x='product', y='excess_stock_value')
        plt.title('มูลค่าสต็อกส่วนเกิน (บาท)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        return fig

def main():
    st.set_page_config(page_title="ระบบวิเคราะห์และวางแผนสต็อก", layout="wide")
    
    st.title("📊 ระบบวิเคราะห์และวางแผนสต็อก")
    
    # Sidebar
    st.sidebar.header("⚙️ ตั้งค่าพารามิเตอร์")
    
    service_level = st.sidebar.select_slider(
        "Service Level (ระดับการให้บริการ)",
        options=[0.90, 0.95, 0.98, 0.99],
        value=0.95,
        help="ระดับความเชื่อมั่นในการรักษาสต็อกสำรอง สูงขึ้น = สต็อกสำรองมากขึ้น"
    )
    
    lead_time_months = st.sidebar.slider(
        "Lead Time (ระยะเวลารอสินค้า, เดือน)",
        min_value=1,
        max_value=6,
        value=3,
        help="ระยะเวลาที่ต้องรอตั้งแต่สั่งจนได้รับสินค้า"
    )
    
    high_stock_months = st.sidebar.slider(
        "เกณฑ์สต็อกสูงเกินไป (เดือน)",
        min_value=3,
        max_value=12,
        value=6,
        help="จำนวนเดือนที่ถือว่ามีสต็อกมากเกินไป"
    )
    
    low_stock_months = st.sidebar.slider(
        "เกณฑ์สต็อกต่ำ (เดือน)",
        min_value=0.5,
        max_value=3.0,
        value=1.0,
        step=0.5,
        help="จำนวนเดือนที่ถือว่ามีสต็อกน้อยเกินไป"
    )
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        inventory_file = st.file_uploader("อัพโหลดไฟล์ inventory.csv", type=['csv'])
        
    with col2:
        sales_file = st.file_uploader("อัพโหลดไฟล์ sales.csv", type=['csv'])
    
    if inventory_file and sales_file:
        try:
            # อ่านไฟล์
            inventory_df = pd.read_csv(inventory_file)
            sales_df = pd.read_csv(sales_file)
            
            # สร้าง optimizer
            optimizer = InventoryOptimizer(service_level, lead_time_months)
            optimizer.load_data(inventory_df, sales_df)
            optimizer.preprocess_data()
            
            # วิเคราะห์ข้อมูล
            results = optimizer.calculate_metrics(high_stock_months, low_stock_months)
            recommendations = optimizer.get_recommendations(results, high_stock_months, low_stock_months)
            
            # แสดงผลการวิเคราะห์
            st.header("📈 ผลการวิเคราะห์")
            
            # Metrics overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "จำนวนสินค้าทั้งหมด",
                    f"{len(results)} รายการ"
                )
            
            with col2:
                high_stock_items = len(recommendations[recommendations['status'] == 'สต็อกสูงเกินไป'])
                st.metric(
                    "สินค้าที่มีสต็อกสูงเกินไป",
                    f"{high_stock_items} รายการ"
                )
            
            with col3:
                total_excess_value = results['excess_stock_value'].sum()
                st.metric(
                    "มูลค่าสต็อกส่วนเกินรวม",
                    f"{total_excess_value:,.2f} บาท"
                )
            
            # Detailed results
            st.subheader("รายละเอียดการวิเคราะห์")
            st.dataframe(results)
            
            # Recommendations
            st.subheader("คำแนะนำ")
            st.dataframe(recommendations)
            
            # Plots
            st.subheader("กราฟวิเคราะห์")
            fig = optimizer.plot_stock_analysis(results, high_stock_months, low_stock_months)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")
            
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
        # จัดการค่า NaN ในข้อมูล inventory
        self.inventory_df = inventory_df.copy()
        self.inventory_df['จำนวน'] = self.inventory_df['จำนวน'].fillna(0)
        self.inventory_df['มูลค่าต่อหน่วย'] = self.inventory_df['มูลค่าต่อหน่วย'].fillna(0)
        
        # จัดการค่า NaN ในข้อมูล sales
        self.sales_df = sales_df.copy()
        numeric_columns = ['productQuantity', 'totalPrice', 'Price per unit', 'productCost', 'Cost per unit']
        for col in numeric_columns:
            if col in self.sales_df.columns:
                self.sales_df[col] = pd.to_numeric(self.sales_df[col], errors='coerce').fillna(0)
        
        # แปลง Date เป็น datetime
        self.sales_df['Date'] = pd.to_datetime(self.sales_df['Date'], errors='coerce')
        self.sales_df = self.sales_df.dropna(subset=['Date'])
        
    def preprocess_data(self):
        self.monthly_sales = (self.sales_df.groupby([
            pd.Grouper(key='Date', freq='M'),
            'product'
        ])['productQuantity']
        .sum()
        .reset_index())

    def calculate_metrics(self, high_stock_months=5, low_stock_months=1.5):
        z_score = {0.90: 1.28, 0.95: 1.96, 0.98: 2.33, 0.99: 2.58}.get(self.service_level, 1.96)
        results = []
        
        for product in self.inventory_df['รายการสินค้า'].unique():
            try:
                current_stock = int(self.inventory_df[
                    self.inventory_df['รายการสินค้า'] == product
                ]['จำนวน'].values[0])
            except:
                current_stock = 0
                
            product_sales = self.monthly_sales[
                self.monthly_sales['product'] == product
            ].sort_values('Date')
            
            if len(product_sales) > 0:
                # แปลง productQuantity เป็น numeric และแทนที่ค่า NaN ด้วย 0
                product_sales['productQuantity'] = pd.to_numeric(product_sales['productQuantity'], errors='coerce').fillna(0)
                avg_monthly_sales = product_sales['productQuantity'].mean()
                std_monthly_sales = product_sales['productQuantity'].std()
                if pd.isna(avg_monthly_sales): avg_monthly_sales = 0
                if pd.isna(std_monthly_sales): std_monthly_sales = 0
                
                stock_coverage = current_stock / avg_monthly_sales if avg_monthly_sales > 0 else float('inf')
                safety_stock = int(z_score * std_monthly_sales * np.sqrt(self.lead_time_months))
                reorder_point = int((avg_monthly_sales * self.lead_time_months) + safety_stock)
                
                if len(product_sales) >= 4:
                    try:
                        model = ExponentialSmoothing(
                            product_sales['productQuantity'],
                            seasonal_periods=4,
                            trend='add',
                            seasonal='add',
                            initialization_method="estimated"
                        ).fit()
                        forecast = model.forecast(3)
                        predicted_sales_3m = int(forecast.sum())
                    except:
                        predicted_sales_3m = int(avg_monthly_sales * 3)
                else:
                    predicted_sales_3m = int(avg_monthly_sales * 3)
                
                excess_stock = max(0, current_stock - (predicted_sales_3m + safety_stock))
                unit_value = float(self.inventory_df[
                    self.inventory_df['รายการสินค้า'] == product
                ]['มูลค่าต่อหน่วย'].values[0])
                
                # เพิ่มการคำนวณ ROI และ Turnover Ratio
                inventory_value = current_stock * unit_value
                annual_sales_value = (avg_monthly_sales * 12) * unit_value
                turnover_ratio = annual_sales_value / inventory_value if inventory_value > 0 else 0
                
                results.append({
                    'product': product,
                    'current_stock': current_stock,
                    'avg_monthly_sales': int(avg_monthly_sales),
                    'stock_coverage_months': round(stock_coverage, 2),
                    'safety_stock': safety_stock,
                    'reorder_point': reorder_point,
                    'predicted_sales_3m': predicted_sales_3m,
                    'excess_stock': int(excess_stock),
                    'excess_stock_value': int(excess_stock * unit_value),
                    'inventory_turnover': round(turnover_ratio, 2)
                })
        
        return pd.DataFrame(results)

    def create_interactive_plots(self, results_df, high_stock_months=5, low_stock_months=1.5):
        # สร้าง subplot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'ระยะเวลาที่สต็อกจะอยู่ได้ (เดือน)',
                'เปรียบเทียบสต็อกปัจจุบันกับยอดขายที่คาดการณ์',
                'มูลค่าสต็อกส่วนเกิน (บาท)'
            ),
            vertical_spacing=0.15
        )
        
        # Plot 1: Stock Coverage
        colors = ['red' if x > high_stock_months else 'orange' if x < low_stock_months else 'green' 
                 for x in results_df['stock_coverage_months']]
        
        fig.add_trace(
            go.Bar(
                x=results_df['product'],
                y=results_df['stock_coverage_months'],
                marker_color=colors,
                name='ระยะเวลาสต็อก',
                hovertemplate="สินค้า: %{x}<br>ระยะเวลา: %{y:.1f} เดือน<extra></extra>"
            ),
            row=1, col=1
        )
        
        # เพิ่มเส้นขีดเกณฑ์
        fig.add_hline(y=high_stock_months, line_dash="dash", line_color="red", 
                     annotation_text="เกณฑ์สต็อกสูง", row=1, col=1)
        fig.add_hline(y=low_stock_months, line_dash="dash", line_color="orange", 
                     annotation_text="เกณฑ์สต็อกต่ำ", row=1, col=1)
        
        # Plot 2: Current Stock vs Predicted Sales
        fig.add_trace(
            go.Bar(
                x=results_df['product'],
                y=results_df['current_stock'],
                name='สต็อกปัจจุบัน',
                marker_color='#3498db',
                hovertemplate="สินค้า: %{x}<br>จำนวน: %{y:,.0f} ชิ้น<extra></extra>"
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=results_df['product'],
                y=results_df['predicted_sales_3m'],
                name='ยอดขายที่คาดการณ์ (3 เดือน)',
                marker_color='#e74c3c',
                hovertemplate="สินค้า: %{x}<br>จำนวน: %{y:,.0f} ชิ้น<extra></extra>"
            ),
            row=2, col=1
        )
        
        # Plot 3: Excess Stock Value
        fig.add_trace(
            go.Bar(
                x=results_df['product'],
                y=results_df['excess_stock_value'],
                name='มูลค่าสต็อกส่วนเกิน',
                marker_color='#9b59b6',
                hovertemplate="สินค้า: %{x}<br>มูลค่า: %{y:,.2f} บาท<extra></extra>"
            ),
            row=3, col=1
        )
        
        # ปรับแต่ง layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="การวิเคราะห์สต็อกสินค้า",
            hovermode='x unified',
            barmode='group'
        )
        
        # ปรับแต่ง x-axis
        for i in range(1, 4):
            fig.update_xaxes(tickangle=45, row=i, col=1)
        
        return fig

    def analyze_sales_trend(self):
        """วิเคราะห์เทรนด์การขายรายเดือน"""
        monthly_trend = self.sales_df.groupby([
            pd.Grouper(key='Date', freq='M'),
            'product'
        ]).agg({
            'productQuantity': 'sum',
            'totalPrice': 'sum'
        }).reset_index()
        
        return monthly_trend

def main():
    st.set_page_config(
        page_title="ระบบวิเคราะห์และวางแผนสต็อก",
        page_icon="📊",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 1400px;
            margin: 0 auto;
        }
        .main {
            padding: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .warning {
            color: #ff4b4b;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🔍 ระบบวิเคราะห์และวางแผนสต็อก")
    st.markdown("*เครื่องมือช่วยวิเคราะห์และวางแผนการจัดการสต็อกสินค้าอัจฉริยะ*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ ตั้งค่าการวิเคราะห์")
        
        # เพิ่มการเลือกมุมมองการวิเคราะห์
        analysis_view = st.radio(
            "📊 มุมมองการวิเคราะห์",
            ["ภาพรวมสต็อก", "การวิเคราะห์เชิงลึก", "เทรนด์การขาย"]
        )
        
        st.markdown("---")
        
        service_level = st.select_slider(
            "🎯 Service Level",
            options=[0.90, 0.95, 0.98, 0.99],
            value=0.95,
            format_func=lambda x: f"{int(x*100)}%"
        )
        
        lead_time_months = st.slider(
            "⏱️ Lead Time (เดือน)",
            min_value=1,
            max_value=6,
            value=3
        )
        
        st.markdown("---")
        
        st.subheader("เกณฑ์การแจ้งเตือน")
        high_stock_months = st.number_input(
            "🔴 สต็อกสูง (เดือน)",
            min_value=3.0,
            max_value=12.0,
            value=5.0,
            step=0.5
        )
        
        low_stock_months = st.number_input(
            "🟡 สต็อกต่ำ (เดือน)",
            min_value=0.5,
            max_value=3.0,
            value=1.5,
            step=0.5
        )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        inventory_file = st.file_uploader(
            "อัพโหลดไฟล์ inventory.csv",
            type=['csv']
        )
        
    with col2:
        sales_file = st.file_uploader(
            "อัพโหลดไฟล์ sales.csv",
            type=['csv']
        )
    
    if inventory_file and sales_file:
        try:
            inventory_df = pd.read_csv(inventory_file)
            sales_df = pd.read_csv(sales_file)
            
            optimizer = InventoryOptimizer(service_level, lead_time_months)
            optimizer.load_data(inventory_df, sales_df)
            optimizer.preprocess_data()
            
            results = optimizer.calculate_metrics(high_stock_months, low_stock_months)
            
            if analysis_view == "ภาพรวมสต็อก":
                # ภาพรวมสต็อก
                st.header("ภาพรวมสต็อก")
                
                total_items = len(results)
                high_stock_items = len(results[results['stock_coverage_months'] > high_stock_months])
                low_stock_items = len(results[results['stock_coverage_months'] < low_stock_months])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📦 จำนวนทั้งหมด", f"{total_items} รายการ")
                
                with col2:
                    st.metric("⚠️ สินค้าสต็อกสูง", f"{high_stock_items} รายการ")
                    
                with col3:
                    st.metric("⚡ สินค้าสต็อกต่ำ", f"{low_stock_items} รายการ")
                    
                with col4:
                    total_excess_value = results['excess_stock_value'].sum()
                    st.metric("💰 มูลค่าส่วนเกิน", f"{total_excess_value:,.2f} บาท")
                
                # กราฟวิเคราะห์
                st.header("กราฟวิเคราะห์")
                fig = optimizer.create_interactive_plots(results, high_stock_months, low_stock_months)
                st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_view == "การวิเคราะห์เชิงลึก":
                st.header("การวิเคราะห์เชิงลึก")
                
                # เลือกสินค้าที่ต้องการวิเคราะห์
                selected_product = st.selectbox(
                    "เลือกสินค้าที่ต้องการวิเคราะห์",
                    results['product'].tolist()
                )
                
                # แสดงข้อมูลเชิงลึกของสินค้าที่เลือก
                product_data = results[results['product'] == selected_product].iloc[0]
                
                # สร้าง KPI cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### สต็อกปัจจุบัน")
                    st.markdown(f"**{product_data['current_stock']:,}** ชิ้น")
                    st.markdown(f"ระยะเวลาสต็อก: **{product_data['stock_coverage_months']:.1f}** เดือน")
                
                with col2:
                    st.markdown("### การขายเฉลี่ย")
                    st.markdown(f"**{product_data['avg_monthly_sales']:,}** ชิ้น/เดือน")
                    st.markdown(f"คาดการณ์ 3 เดือน: **{product_data['predicted_sales_3m']:,}** ชิ้น")
                
                with col3:
                    st.markdown("### จุดสั่งซื้อที่แนะนำ")
                    st.markdown(f"Reorder Point: **{product_data['reorder_point']:,}** ชิ้น")
                    st.markdown(f"Safety Stock: **{product_data['safety_stock']:,}** ชิ้น")
                
                # แสดงกราฟแนวโน้มการขาย
                sales_trend = optimizer.analyze_sales_trend()
                product_trend = sales_trend[sales_trend['product'] == selected_product]
                
                if not product_trend.empty:
                    fig_trend = px.line(
                        product_trend,
                        x='Date',
                        y='productQuantity',
                        title=f'แนวโน้มการขาย {selected_product}'
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                
            else:  # เทรนด์การขาย
                st.header("เทรนด์การขาย")
                
                # วิเคราะห์เทรนด์การขายรายเดือน
                sales_trend = optimizer.analyze_sales_trend()
                
                # สร้างกราฟเทรนด์การขายรวม
                fig_total_trend = px.line(
                    sales_trend.groupby('Date')['productQuantity'].sum().reset_index(),
                    x='Date',
                    y='productQuantity',
                    title='เทรนด์การขายรวมทุกสินค้า'
                )
                st.plotly_chart(fig_total_trend, use_container_width=True)
                
                # สร้างกราฟเทรนด์การขายแยกตามสินค้า
                fig_product_trend = px.line(
                    sales_trend,
                    x='Date',
                    y='productQuantity',
                    color='product',
                    title='เทรนด์การขายแยกตามสินค้า'
                )
                st.plotly_chart(fig_product_trend, use_container_width=True)
            
            # แสดงรายละเอียดข้อมูล
            with st.expander("📋 รายละเอียดข้อมูล"):
                st.dataframe(
                    results.style.format({
                        'avg_monthly_sales': '{:,.0f}',
                        'stock_coverage_months': '{:.2f}',
                        'safety_stock': '{:,.0f}',
                        'reorder_point': '{:,.0f}',
                        'predicted_sales_3m': '{:,.0f}',
                        'excess_stock': '{:,.0f}',
                        'excess_stock_value': '{:,.2f}',
                        'inventory_turnover': '{:.2f}'
                    }),
                    hide_index=True,
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {str(e)}")
            st.error("กรุณาตรวจสอบรูปแบบไฟล์และข้อมูลที่นำเข้า")
    else:
        st.info("👆 กรุณาอัพโหลดไฟล์ข้อมูลทั้งสองไฟล์เพื่อดูผลการวิเคราะห์")

if __name__ == "__main__":
    main()
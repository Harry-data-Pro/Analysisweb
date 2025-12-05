import helper as hp
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Auto EDA", layout="wide")
st.title("Automatic EDA Dashboard")
data = st.file_uploader(label='select file',type = 'csv')


if data is not None:
    data = pd.read_csv(data)
    tab1, tab2, tab3 = st.tabs(["Analysis", "Outliers","QQPlot"])
    num_col = []
    cat_col = []
    for i in data.columns:
        if data[i].dtype in ['int64','float64','float32']:
            num_col.append(i)
        else:
            cat_col.append(i)
    num_df = data[num_col]
    cat_df = data[cat_col]

    with tab1:
        tab11, tab12 = st.tabs(["Univariate analysis", "Bivariate analysis"])

        #Univariate Analysis Tab
        with tab11:
            col1, col2, col3 = st.columns(3)
            with col1:
                hp.h_plot(data)
            with col2:
                hp.Count(data)
            with col3:
                hp.info(data)

        # Bivariate analysis
        with tab12 :
            col1, col2, col3 = st.columns(3)
            with col1 :
                hp.num_num_plot(num_df)

            with col2:
                hp.cat_cat_plot(cat_df)

            with col3:
                hp.num_cat_plot(data)


    with tab2:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('Before IQR')
            hp.h_plot(data,type = 'num')

        with col2:
            st.write('After IQR')
            hp.iqr(num_df)

        with col3:
            st.write('Change in Data [Old - New]')
            hp.iqr(num_df,change = True)

    with tab3:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write('Before Yeo-Johnson')
            hp.qq_plot(num_df)

        with col2:
            st.write('After Yeo-Johnson')
            hp.qq_plot(num_df,yeo=True)
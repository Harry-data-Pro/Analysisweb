import matplotlib.pyplot as plt

import helper as hp
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Auto EDA", layout="wide")
st.title("Automatic EDA Dashboard")
data = st.file_uploader(label='select file',type = 'csv')

if data is not None:
    data = pd.read_csv(data)
    data = data.dropna(axis=1, how='all')
    tab1, tab2, tab3, tab4 = st.tabs(["Analysis", 'Missing Values',"Outliers",'Normalization'])
    num_col = []
    label = []
    cat_col = []
    for i in data.columns:
        if data[i].dtype in ['int64','float64','float32']:
            num_col.append(i)
            if data[i].nunique() < 3:
                label.append(i)
        else:
            cat_col.append(i)
    num_df = data[num_col]
    cat_df = data[cat_col]
    # new_df = hp.impute_missing_values(data,full = True)

    with tab1:
        tab11, tab12 = st.tabs(["Univariate analysis", "Bivariate analysis"])

        #Univariate Analysis Tab
        with tab11:
            col1, col2 = st.columns([3,1])
            with col1:
                hp.h_plot(data)
            with col2:
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
        col1, col2 = st.columns(2)
        with col1:
            new_num_df,n_cols = hp.impute_missing_values(num_df)
            hp.h_plot(new_num_df[n_cols])
        with col2:
            new_col_df,c_cols = hp.impute_missing_values(cat_df)
            hp.h_plot(new_col_df[c_cols])

    with tab3:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('Before IQR')
            hp.h_plot(new_num_df)

        with col2:
            st.write('After IQR and Filling Missing Values')
            new_num_df = hp.iqr(new_num_df,df = True)
            hp.iqr(new_num_df)

        with col3:
            st.write('Change in Data [Old - New]')
            hp.iqr(new_num_df,table = True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            # st.write('After Scaling')
            new_num_df = hp.normalization(new_num_df.select_dtypes(include = 'number'))
            st.dataframe(new_num_df.sample(5))
        with col2:
            for i in new_num_df.columns:
                fig,ax = plt.subplots()
                num_df[i].plot(kind="kde",ax = ax,label="Original", color="blue")
                new_num_df[i].plot(kind = 'kde',ax = ax,label="Normalized", color="red")
                ax.set_xlabel(i)
                ax.legend()
                st.pyplot(fig)


    new_df = pd.concat([new_num_df, new_col_df, data[label]], axis=1)
    if 'new_df' in locals():
        csv_data = new_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Cleaned CSV",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    else:
        st.info("Complete the missing value handling in Tab 2 to enable download.")
    # with tab3:#wait
    #     col1, col2 = st.columns(2)
    #
    #     with col1:
    #         st.write('Before Yeo-Johnson')
    #         hp.qq_plot(num_df.drop(columns = label))
    #
    #     with col2:
    #         st.write('After Yeo-Johnson')
    #         hp.qq_plot(num_df.drop(columns = label),yeo=True)

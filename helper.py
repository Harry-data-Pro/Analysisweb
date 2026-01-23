import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Graphs

def limit_cat(series):
    if series.nunique() > 15:
        return series.value_counts().head(15)
    else:
        return series.value_counts()

def h_plot(data = None, typee = 'all'):
    plt.style.use('ggplot')
    # plt.style.use('default')
    if data is None:
        return 'No Input'
    names = data.columns
    for i in names:
        fig, ax = plt.subplots(figsize = (10,5),ncols = 2, nrows = 1)
        if data[i].dtype in ['int64','float64']:
            sns.histplot(x=i, data=data, kde=True,fill=True,ax = ax[0])
            ax[0].set_title(f'Distribution of {i}')
            ax[0].tick_params(rotation = 90)
            sns.violinplot(x=data[i], ax = ax[1])
            st.pyplot(fig)
            st.markdown("---")
            plt.close(fig)
            # st.write('skew = ',data[i]
        else:
            if typee == 'all':
                counts = limit_cat(data[i])
                ax[0].bar(x = counts.index.astype(str), height = counts.values)
                ax[0].set_title(f'Count plot of {i}')
                ax[0].tick_params(rotation = 90)
                # counts = data[i].value_counts()
                ax[1].pie(x=counts, labels=counts.index, autopct='%1.1f%%', pctdistance=1.45)
                st.pyplot(fig)
                st.markdown("---")
                plt.close(fig)
# space
def space(num):
    for i in range(num):
        st.write('\n')


# categorical data count
def Count(data):
    for col in data.columns:
        vc = data[col].value_counts().sort_values(ascending=False)
        if len(vc) > 5:
            top5 = vc.iloc[:5]
            others = vc.iloc[5:].sum()
            df = top5.to_frame(name='count')
            df.loc['Others'] = others
            df['Percentage(%)'] = (df['count'] / df['count'].sum() * 100).round(2)
            st.write("Unique Values")
            st.write(df.round(2))
            num = 9 - df.shape[0]
            space(num)
            st.markdown("---")

        else:
            df = vc.to_frame(name='count')
            st.write("Unique Values")
            st.write(df.round(2))
            num = 9 - df.shape[0]
            space(num)
            st.markdown("---")
    return None


# Basic descriptive statistics
def info(data):
    for i in data.columns:
        des = data[i].describe().reset_index()
        result = des.rename(columns = {'index':'Statistics',i:'Values'})
        st.write('Basic Info')
        st.dataframe(round(result,2))
        space(30 - result.shape[0] )


# Outlies --------------------------------tab-2

# missing values
# def num_missing_values(data,num_method = 'median',flag = False):
#     columns = data.columns.tolist()
#     imputer = SimpleImputer(strategy=num_method, add_indicator=flag)
#     imputed = imputer.fit_transform(data)
#     if flag:
#         flag_index = imputer.indicator_.features_
#         new_cols_name = [ columns[i] + '_missing_flag' for i in flag_index]
#         return pd.DataFrame(data = imputed,columns = columns + new_cols_name, index = data.index)
#     return pd.DataFrame(imputed, columns=columns, index=data.index)


def impute_missing_values(data):

    df = data.copy()
    cols_with_missing = [col for col in df.columns if df[col].isnull().any()]

    if not cols_with_missing:
        st.success(" No missing values found in the dataset")
        return data,[]

    st.write("### Missing Value")

    for col in cols_with_missing:

        is_numeric = is_numeric_dtype(df[col])
        col_type = "Numerical" if is_numeric else "Categorical"

        with st.container():
            st.markdown(f"**Column:** `{col}` ({col_type}) with {data[col].isnull().sum()} missing values")

            if is_numeric:
                strategies = [
                    "Do Nothing",
                    "Fill with Mean (Average)",
                    "Fill with Median (Middle Value)",
                    "Fill with Zero (0)",
                    "Drop Rows"
                ]
            else:
                strategies = [
                    "Do Nothing",
                    "Fill with Mode (Most Frequent)",
                    "Fill as 'Unknown'",
                    "Drop Rows"
                ]
            selected_method = st.selectbox(
                f"Choose a strategy for '{col}':",
                options=strategies,
                key=f"method_{col}"
            )

            if selected_method == "Fill with Mean (Average)":
                df[col] = df[col].fillna(df[col].mean())

            elif selected_method == "Fill with Median (Middle Value)":
                df[col] = df[col].fillna(df[col].median())

            elif selected_method == "Fill with Zero (0)":
                df[col] = df[col].fillna(0)

            elif selected_method == "Fill with Mode (Most Frequent)":
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])

            elif selected_method == "Fill as 'Unknown'":
                df[col] = df[col].fillna("Unknown")

            elif selected_method == "Drop Rows":
                df.dropna(subset=[col], inplace=True)

            st.markdown("---")
    return df, cols_with_missing

#tab 2,1

def iqr(data,table = False,df = False):
    # data = num_missing_values(data,flag = False)
    new = data.copy()
    for i in data.columns:
        percentile25 = data[i].quantile(0.25)
        percentile75 = data[i].quantile(0.75)
        iqr = percentile75 - percentile25
        min_value = percentile25 - (1.5 * iqr)
        max_value = percentile75 + (1.5 * iqr)
        new = new[(new[i] <= max_value) & (new[i] >= min_value)]
        if table:
            old_desc = data[i].describe()
            new_desc = new[i].describe()
            diff = old_desc.astype(float) - new_desc.astype(float)
            diff['data removed (%)'] = round(100 - (new[i].shape[0] / data[i].shape[0]) * 100, 2)
            st.dataframe(diff)
            st.markdown('---')
    if df == True:
        return new
    if table == False:
        h_plot(data = new)
        st.markdown("---")
    return new

# muti types of graphs
def multy_plot(data,kind = 'histplot',name = None):
    fig, ax = plt.subplots()
    if kind == 'histplot':
        sns.histplot(x=name, data=data, kde=True, fill=True, color='red')
        ax.set_title(f'Distribution of {name}')
        ax.legend()
        return fig

    if kind == 'Scatterplot':
        sns.scatterplot(data,x = data.columns[0],y = data.columns[1])
        return fig

    if kind == 'Line Plot':
        sns.lineplot(data, x = data.columns[0], y = data.columns[1])
        return fig

    if kind == 'Hexbin Plot':
        plt.hexbin(data.iloc[::,0],data.iloc[::,1], gridsize=20, cmap='viridis')
        plt.colorbar()
        ax.set_xlabel(data.columns[0])
        ax.set_ylabel(data.columns[1])
        return fig

# tab 11
def num_num_plot(data,kind = 'Hexbin Plot'):
    select_cols = st.multiselect(
        "### Select columns for numeric–numeric analysis:",
        options=data.columns
    )
    kind = st.selectbox(
        "Select plot type",
        ["Scatterplot", "Line Plot", "Hexbin Plot"]
    )
    if select_cols is not None:
        df = data[select_cols]

        names = df.columns
        if len(names) > 1:
            for i in range(len(names)):
                for j in range(len(names)):
                    if j!= i:
                        st.pyplot(multy_plot(df.iloc[::,[i,j]], kind=kind))

#tab12
def cat_cat_plot(data):
    select = st.multiselect(
        '### Choose 2 columns: 1st Index then Column',
        options=data.columns
    )
    if len(select) == 2:
        index, column = select
        st.write(f"### Crosstab: {index} × {column}")
        ct = pd.crosstab(index = data[index], columns = data[column], normalize=True, margins=True).mul(100).round(2)
        st.dataframe(ct)
        # st.table(
        #     pd.crosstab(index = data[index], columns = data[column], normalize=True, margins=True).mul(100).round(2)
        # )


# tab13-----------group by
def num_cat_plot(data):
    st.write('### Pivot Table with UI')
    all_cols = data.columns.tolist()

    index_cols = st.multiselect("Index (Rows)", all_cols)
    column_cols = st.multiselect("Columns", all_cols)
    value_cols = st.multiselect("Values", all_cols)

    agg_funcs = {
        "Sum": "sum",
        "Mean": "mean",
        "Count": "count",
        "Min": "min",
        "Max": "max",
        "Median": "median",
        "Std Dev": "std"
    }

    agg_choice = st.selectbox("Aggregation Function", list(agg_funcs.keys()))

    if st.button("Generate Pivot Table"):
        if len(value_cols) == 0:
            st.error("Please select at least one value column.")
        else:
            try:
                pivot = pd.pivot_table(
                    data,
                    index=index_cols if index_cols else None,
                    columns=column_cols if column_cols else None,
                    values=value_cols,
                    aggfunc=agg_funcs[agg_choice],
                )
                st.write("Pivot Table Output:")
                st.dataframe(pivot)
            except Exception as e:
                st.error(f"Error generating pivot table: {e}")

# Feature Scaling & Normalization
def normalization(data):
    for col in data.columns:
        with st.container():
            st.markdown(f"**Column Name:** `{col}`")
            strategies = [
                "Do Nothing",
                "MinMaxScaler",
                "StandardScaler"
            ]
            selected_method = st.selectbox(
                f"Choose a strategy for '{col}':",
                options=strategies,
                key=f"Scaler{col}"
            )

            if selected_method == 'MinMaxScaler':
                scaler = MinMaxScaler()
                data[[col]] = scaler.fit_transform(data[[col]])
            elif selected_method == 'StandardScaler':
                scaler = StandardScaler()
                data[[col]] = scaler.fit_transform(data[[col]])
    return data


# # skew--------------------tab-3
# def best_qq_col(data):
#     data = num_missing_values(data)
#     old_skew = data.skew()
#     pf = PowerTransformer(method = 'yeo-johnson',standardize=True)
#     new_data = pd.DataFrame(pf.fit_transform(data),columns = data.columns,index = data.index)
#     new_skew = new_data.skew()
#     improved = new_skew.abs() < old_skew.abs()
#     result = data.copy()
#     for i in data.columns:
#         if i in improved:
#             result[i] = new_data[i]
#     return result
#
# # plot QQ
# def qq_plot(data,yeo = False):
#     if yeo:
#         data = best_qq_col(data)
#     for i in data.columns:
#         fig = plt.figure()
#         stats.probplot(data[i], dist="norm", plot=plt)
#         st.write(i)
#         st.pyplot(fig)
#         st.write(data[i].skew())
#         st.markdown('---')

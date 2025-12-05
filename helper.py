import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import scipy.stats as stats
from sklearn.preprocessing import PowerTransformer

# Graphs
def h_plot(data = None, type = 'all'):
    if data is None:
        return 'No Input'
    for i in data.columns:
        fig, ax = plt.subplots()
        if data[i].dtype in ['int64','float64']:
            sns.histplot(x=i, data=data, kde=True,fill=True,color='red')
            ax.set_title(f'Distribution of {i}')
            st.pyplot(fig)
            st.markdown("---")
            plt.close(fig)
            # st.write('skew = ',data[i]
        else:
            if type == 'all':
                sns.countplot(x = i , data = data , ax = ax,color='green')
                ax.set_title(f'Count plot of {i}')
                st.pyplot(fig)
                st.markdown("---")
                plt.close(fig)
    return None
# space
def space(num):
    for i in range(num):
        st.write('\n')


# categorical data count
def Count(data):
    for i in data.columns:
        count = data[i].value_counts().sort_values(ascending = False)
        if len(count) > 5:
            others = count[5::].sum()
            count = count.head(5)
            count['Others'] = others
            df = pd.DataFrame(count)
            df['Percentage(%)'] = round((df['count'] / df['count'].sum()) * 100,2)
            st.write('Unique Values',round(df,2))
            num = 11 - df.shape[0]
            space(num)
            st.markdown("---")
        else:
            st.write('Unique Values',round(count,2))
            num = 11 - count.shape[0]
            space(num)
            st.markdown("---")
    return None


# Basic descriptive statistics
def info(data):
    for i in data.columns:
        des = data[i].describe().reset_index()
        result = des.rename(columns = {'index':'Statistics',i:'Values'})
        st.write('Basic Info',round(result,2))
        space(9-result.shape[0])
        st.markdown('---')


# Outlies --------------------------------tab-2

def iqr(data,change = False):
    for i in data.columns:
        percentile25 = data[i].quantile(0.25)
        percentile75 = data[i].quantile(0.75)
        iqr = percentile75 - percentile25
        min_value = percentile25 - (1.5 * iqr)
        max_value = percentile75 + (1.5 * iqr)
        new = data[(data[i] < max_value) & (data[i] > min_value)]
        if change == False:
            st.pyplot(multy_plot(data = new, kind = 'histplot',name = i))
            st.markdown("---")
        if change == True:
            ser = round(data[i].describe() - new[i].describe(),2)
            ser['data removed(%)'] = round(100 - (new[i].shape[0] / data[i].shape[0]) * 100,2)
            st.dataframe(ser)
            st.markdown('---')


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


# skew--------------------tab-3

def qq_plot(data,yeo = False):
    if yeo == True:
        pt = PowerTransformer(method="yeo-johnson", standardize=True)
        data = pd.DataFrame(
            pt.fit_transform(data.dropna()),
            columns=data.columns
        )
    for i in data.columns:
        fig = plt.figure()
        stats.probplot(data[i].dropna(), dist="norm", plot=plt)
        st.pyplot(fig)
        st.markdown('---')

# tab 11
def num_num_plot(data,kind = 'Hexbin Plot'):
    select_cols = st.multiselect(
        "### Select columns for numeric–numeric analysis:",
        options=data.columns
    )
    if len(select_cols) != 0 :
        df = data[select_cols]
    else:
        df = data

    kind = st.selectbox(
        "Select plot type",
        ["Scatterplot", "Line Plot", "Hexbin Plot"]
    )

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
    if len(select) == 0:
        cols = data.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col1 = cols[i]
                col2 = cols[j]
                uniq1 = data[col1].nunique()
                uniq2 = data[col2].nunique()
                if uniq1 <= uniq2:
                    df1 = data[col1]
                    df2 = data[col2]
                else:
                    df1 = data[col2]
                    df2 = data[col1]
                st.write(f"### Crosstab: {df1.name} × {df2.name}")
                st.table(
                    pd.crosstab(df1, df2, normalize=True, margins=True).mul(100).round(2)
                )
    if len(select) == 2:
        index, column = select
        st.write(f"### Crosstab: {index} × {column}")
        st.table(
            pd.crosstab(index = data[index], columns = data[column], normalize=True, margins=True).mul(100).round(2)
        )



# num * cat = Violin plot, Bar chart, Grouped summary statistics, stripplot tab 3

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

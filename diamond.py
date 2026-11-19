import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import ( AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score)
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

st.set_page_config(
    page_title="My Wide App",
    layout="wide",  # Options: "centered" (default) or "wide"
    initial_sidebar_state="expanded",
)


st.title("Diamond Price Peediction")

df = pd.read_csv("diamonds.csv")
df = df.drop(columns=["Unnamed: 0"])

num_col_eda = ["carat", "depth", "table", "x", "y", "z", "price"]
cat_col_eda = ["cut", "color", "clarity"]


# st.write(df.head())
# st.table(df.head())
# st.dataframe(df.head())

b1 = st.sidebar.checkbox("show data set")
if b1:
    st.write(df)

b2 = st.sidebar.checkbox("EDA")
col1, col2 = st.columns(2)
if b2:
    s1 = st.sidebar.selectbox(
        "type of eda", ["Univariate", "Biavariate", "Multivariate"]
    )

    if s1 == "Univariate":
        cols_name = st.sidebar.selectbox("select column", df.columns)

        if cols_name in num_col_eda:
            chart1 = px.histogram(df[cols_name])
            col1.plotly_chart(chart1)

            chart2 = px.box(df[cols_name])
            col2.plotly_chart(chart2)

    elif cols_name in cat_col_eda:
        df_cat_eda = df.groupby(cols_name)["price"].count().reset_index()

        df_cat_eda.rename(columns={"price": "Count"}, inplace=True)

        chart3 = px.bar(df_cat_eda, cols_name, "Count")
        col1.plotly_chart(chart3)
        chart4 = px.pie(df_cat_eda, cols_name, "Count")
        col2.plotly_chart(chart4)

    elif s1 == "Biavariate":
        cols_name = st.sidebar.selectbox("select column", df.columns)
        target_col = df["price"]

        if cols_name in num_col_eda:
            chart1 = px.scatter(df, cols_name, target_col)
            col1.plotly_chart(chart1)

        df_num_eda = df.groupby(cols_name)["price"].count().reset_index()
        chart2 = px.line(df_num_eda, cols_name, "price")
        col2.plotly_chart(chart2)


else:
    pass


x = df.drop(columns=["price"])
y = df["price"]

test_size = st.sidebar.slider("select test size", 0.10, 0.95, 0.20, 0.05)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=test_size, random_state=42
)

b3 = st.sidebar.checkbox("check shape")
if b3:
    st.write(f"x_train,{x_train.shape}")
    st.write(f"x test,{x_test.shape}")
    st.write(f"y train,{y_train.shape}")
    st.write(f"y test,{y_test.shape}")

cat_col = ["cut", "color", "clarity"]
num_col = ["carat", "depth", "table", "x", "y", "z"]

num_train = x_train[num_col]
num_test = x_test[num_col]

scaler = StandardScaler()
num_train_s = scaler.fit_transform(num_train)
num_train_s = pd.DataFrame(num_train_s, columns=num_train.columns, index=num_train.index)

num_test_s = scaler.transform(num_test)
num_test_s = pd.DataFrame(num_test_s, columns=num_test.columns, index=num_test.index)


Encoder = OrdinalEncoder(
    categories=[
        ["Fair", "Good", "Very Good", "Premium", "Ideal"],
        ["J", "I", "H", "G", "F", "E", "D"],
        ["I1", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2", "IF"],
    ]
)

cat_tr = x_train[cat_col]
cat_ts = x_test[cat_col]

cat_train_e = Encoder.fit_transform(cat_tr)
cat_train_e = pd.DataFrame(cat_train_e, columns=cat_col, index=cat_tr.index)
cat_test_e = Encoder.transform(cat_ts)
cat_test_e = pd.DataFrame(cat_test_e, columns=cat_col, index=cat_ts.index)

x_train_final = num_train_s.join(cat_train_e)
x_test_final = num_test_s.join(cat_test_e)


st.dataframe(x_test_final)


model_dict = {
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
    "Support Vector Regressor": SVR(),
}


model_choice = st.sidebar.selectbox(
    "Select a model for training", list(model_dict.keys())
)


selected_model = model_dict[model_choice]

b5 = st.checkbox("show result")

if b5:
    selected_model.fit(x_train_final, y_train)
    y_pred = selected_model.predict(x_test_final)

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics_dict = {
        "Metric": [
            "Mean Squared Error (MSE)",
            "Root Mean Squared Error (RMSE)",
            "Mean Absolute Error (MAE)",
            "R2 Score",
        ],
        "Value": [round(mse, 2), round(rmse, 2), round(mae, 2), round(r2, 2)],
    }

    df_metrics = pd.DataFrame(metrics_dict)

    st.subheader(f"Performance Metrics: {model_choice}")
    st.dataframe(df_metrics, hide_index=True, use_container_width=True)


models = {"Pred_model": selected_model, "Scaler": scaler, "Encoder": Encoder}
model_bytes = pickle.dumps(models)
# Download button
st.download_button(label="Download Models", data=model_bytes, file_name="models.pkl")

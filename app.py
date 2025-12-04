import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(
    page_title="🚗🚗🚗Car Price Prediction🚗🚗🚗", page_icon="🚗", layout="wide"
)

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "car_pipeline.pkl"


@st.cache_resource
def load_model():
    """Загружаем пайплайн модели через pickle"""

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_train_data():
    """Загружаем train выборку для EDA"""
    return pd.read_csv(f"{MODEL_DIR}/cars_train.csv", index_col=0)


def prepare_dataframe(df, is_pred_form=False):
    """Приводим данные к формату обучения модели."""
    df_proc = df.copy()
    if not is_pred_form:
        df_proc["mileage"] = (
            df_proc["mileage"].str.strip(" kmpl").str.strip("  km/kg")
        )
        df_proc["engine"] = df_proc["engine"].str.strip(" CC").astype(float)
        df_proc["max_power"] = df_proc["max_power"].str.strip(" bhp")
        df_proc["mileage"] = df_proc["mileage"].str.strip(" kmpl").astype(float)
        df_proc["max_power"] = (
            df_proc["max_power"].replace("", np.nan).astype(float)
        )
        df_proc["car_model"] = df_proc["name"]
        df_proc.drop("torque", axis=1, inplace=True)
    if is_pred_form:
        df_proc["name"] = df_proc["car_model"]
    df_proc["name"] = df_proc["name"].str.split(" ").str[0]
    missing_median = {
        "mileage": np.float64(19.3),
        "engine": np.float64(1248.0),
        "max_power": np.float64(82.0),
        "seats": np.float64(5.0),
    }
    for col, median in missing_median.items():
        df_proc.fillna({col: median}, inplace=True)
    df_proc["engine"] = df_proc[col].astype(int)
    df_proc["seats"] = df_proc[col].astype(int)
    return df_proc


# Загружаем модель
try:
    MODEL = load_model()
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

try:
    train_df = load_train_data()
except Exception as e:
    st.error(f"❌ Ошибка загрузки train данных: {e}")
    st.stop()

# --- Основной интерфейс ---
st.title("🚗🚗🚗🎯 Предсказание стоимости автомобилей 🚗🚗🚗")

st.header("📊 Инфографика (EDA)")
try:
    col1, col2 = st.columns(2)
    with col1:
        fuel_transmission_box = px.box(
            data_frame=train_df,
            x="fuel",
            y="selling_price",
            title="Цена в зависимости от типа топлива и коробки передач",
            labels={"selling_price": "Цена", "fuel": "Тип топлива"},
            color="transmission",
        )
        st.plotly_chart(fuel_transmission_box, width="stretch")

    with col2:
        seller_owner_box = px.box(
            data_frame=train_df,
            x="seller_type",
            y="selling_price",
            title="Цена в зависимости от продавца и владельца авто",
            color="owner",
            labels={"selling_price": "Цена", "seller_type": "Тип продавца"},
        )
        st.plotly_chart(seller_owner_box, width="stretch")
except Exception as e:
    st.error(f"❌ Ошибка вывода графиков: {e}")
    st.stop()

try:
    col1, col2 = st.columns(2)
    train_df["car_model"] = train_df["name"]
    train_df["name"] = train_df["name"].str.split(" ").str[0]

    with col1:
        price_hist = px.histogram(
            train_df,
            x="selling_price",
            nbins=30,
            title="График распределения цен",
            labels={"selling_price": "Цена", "count": "Количество автомобилей"},
        )
        st.plotly_chart(price_hist, width="content")

    with col2:
        corr_df = (
            train_df[train_df.select_dtypes(include=np.number).columns]
            .corr()
            .round(2)
        )
        corr_plot = px.imshow(
            corr_df,
            title="Корреляция числовых признаков",
            text_auto=True,
            color_continuous_scale="RdBu_r",
        )
        st.plotly_chart(corr_plot, width="content")
except Exception as e:
    st.error(f"❌ Ошибка вывода графиков: {e}")
    st.stop()

try:
    sorted_brands = (
        train_df.groupby("name")["selling_price"]
        .max()
        .sort_values(ascending=False)
        .index
    )
    price_hist = px.box(
        data_frame=train_df,
        x="name",
        y="selling_price",
        category_orders={"name": sorted_brands.tolist()},
        title="Цена в зависимости от бренда авто",
        labels={"selling_price": "Цена", "name": "Марка авто"},
    )
    st.plotly_chart(price_hist, width="stretch")
except Exception as e:
    st.error(f"❌ Ошибка вывода графиков: {e}")
    st.stop()

try:
    st.subheader("🔍 Интерактивный Pairplot")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect(
        f"Выберите признаки для pairplot (макс. {len(numeric_cols)}):",
        numeric_cols,
        default=["selling_price", "year"],
    )

    if len(selected_cols) >= 2:
        fig_pair = px.scatter_matrix(
            train_df,
            dimensions=selected_cols,
            title=f"Pairplot: {', '.join(selected_cols)}",
            height=len(selected_cols) * 400,
            color="selling_price",
        )
        st.plotly_chart(fig_pair, width="stretch")
    else:
        st.info("👈 Выберите минимум 2 числовых признака")
except Exception as e:
    st.error(f"❌ Ошибка вывода pairplot: {e}")
    st.stop()
# ========================================================

st.header("🚗🚗🚗🎯 Предсказание стоимости автомобилей")

# Загрузка CSV файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is None:
    st.info("👈 Загрузите CSV файл для начала работы")
    st.stop()


try:
    # Загружаем данные и делаем предсказания
    raw_df = pd.read_csv(uploaded_file)
    if "selling_price" in raw_df.columns:
        raw_df_init = raw_df.copy()
        raw_df = raw_df.drop("selling_price", axis=1)
    car_columns = {
        "categorical": [
            "car_model",
            "fuel",
            "seller_type",
            "transmission",
            "owner",
        ],
        "int": ["year", "km_driven", "engine", "seats"],
        "float": ["mileage", "max_power"],
    }

    df = prepare_dataframe(raw_df)
    predictions = MODEL.predict(df.drop("car_model", axis=1))

    df["prediction"] = predictions
except Exception as e:
    st.error(f"❌ Ошибка при обработке данных: {e}")
    st.stop()


# --- Метрики ---
st.subheader("📊 Результаты предсказания стоимости автомобилей")

try:
    st.dataframe(df.sort_values("prediction", ascending=False))
except Exception as e:
    st.error(f"❌ Ошибка вывода результатов предсказания: {e}")
    st.stop()
# --- Форма для предсказания ---
st.subheader("🔮 Сделать предсказание для автомобиля")

with st.form("prediction_form"):
    col_left, col_right = st.columns(2)
    input_data = {}

    with col_left:
        st.write("**Категориальные:**")
        for col in car_columns["categorical"]:
            if df[col].dtype in ("object", "bool"):
                unique_vals = sorted(df[col].astype(str).unique().tolist())
                if col == "car_model":
                    input_data[col] = st.selectbox(
                        col,
                        unique_vals,
                        key=f"cat_{col}",
                        accept_new_options=True,
                    )
                else:
                    input_data[col] = st.selectbox(
                        col, unique_vals, key=f"cat_{col}"
                    )

    with col_right:
        st.write("**Числовые:**")
        for col in car_columns["int"]:
            if df[col].dtype not in ("object", "bool"):
                val = int(df[col].median())
                input_data[col] = st.number_input(
                    col, value=val, key=f"num_{col}"
                )
        for col in car_columns["float"]:
            if df[col].dtype not in ("object", "bool"):
                val = float(df[col].median())
                input_data[col] = st.number_input(
                    col, value=val, key=f"num_{col}"
                )

    submitted = st.form_submit_button("Предсказать", width="content")

if submitted:
    try:
        input_df = pd.DataFrame([input_data])
        prepared_input = prepare_dataframe(input_df, is_pred_form=True)
        print(prepared_input)

        prediction = MODEL.predict(
            prepared_input.drop("car_model", axis=1).reset_index(drop=True)
        )[0]
        prediction_msg = (
            f"Вы можете продать автомобиль за {prediction:,.0f} у.е.".replace(
                ",", " "
            )
        )
        st.success(prediction_msg)
    except Exception as e:
        st.error(f"❌ Ошибка при предсказании: {e} {input_data}")


st.header("⚖️ Веса модели (Feature Importance)")
try:
    feature_names = MODEL.best_estimator_["preprocessor"].get_feature_names_out()
    feature_names = [x.split("__")[1] for x in feature_names]
    feature_coef = MODEL.best_estimator_["elastic"].coef_
    coef_df = (
        pd.DataFrame({"features": feature_names, "coef": feature_coef})
        .sort_values("coef", key=lambda x: np.abs(x), ascending=False)
        .round(2)
        .reset_index(drop=True)
    )
    coef_gr_df = (
        coef_df.head(20)
        .sort_values("coef", key=lambda x: np.abs(x))
        .reset_index(drop=True)
    )

    fig_weights = px.bar(
        coef_gr_df,
        x="coef",
        y="features",
        title="🔥 ТОП-20 коэффициентов модели",
        color="coef",
        color_continuous_scale=["red", "white", "blue"],
        height=600,
    )

    st.plotly_chart(fig_weights, width="stretch")
    st.dataframe(coef_df)

except Exception as e:
    st.error(f"❌ Ошибка вывода коэффициентов модели: {e}")
    st.stop()

try:
    st.header("⚙️ Гиперпараметры лучшей модели")
    best_params = MODEL.best_params_
    st.dataframe(
        pd.DataFrame(list(best_params.items()), columns=["Параметр", "Значение"])
    )
except Exception as e:
    st.error(f"❌ Ошибка вывода гиперпараметров модели: {e}")
    st.stop()
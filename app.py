import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Cargar los datos
gpa = pd.read_csv('gpa.csv')

# Variables a graficar
variables = ['promedio', 'edad', 'sexo', 'alcohol', 'noviazgo', 'fallas']

numericas = [var for var in variables if gpa[var].dtype in ['int64', 'float64', 'float32']]
categoricas = [var for var in variables if gpa[var].dtype == 'object' or gpa[var].nunique() <= 10]
target = 'promedio'

# Crear pestañas
tab1, tab2, tab3 = st.tabs(["Gráficos univariados", "Gráficos bivariados", "Modelo predictivo"])

with tab1:
    st.header("Distribución univariada de variables")

    for var in variables:
        st.subheader(f"Variable: {var}")
        fig, ax = plt.subplots(figsize=(6, 4))

        if gpa[var].dtype in ['int64', 'float64', 'float32']:
            sns.histplot(gpa[var], kde=True, bins=20, color='skyblue', ax=ax)
            ax.set_title(f'Distribución de {var}')
            ax.set_xlabel(var)
            ax.set_ylabel("Frecuencia")
        else:
            sns.countplot(x=var, data=gpa, palette='Set2', ax=ax)
            ax.set_title(f'Frecuencia de {var}')
            ax.set_xlabel(var)
            ax.set_ylabel("Conteo")

        st.pyplot(fig)
with tab2:
    st.header("Gráficos bivariados con 'promedio' como variable dependiente")

    # Numéricas: Scatterplots
    st.markdown("### Variables numéricas")
    for var in numericas:
        st.subheader(f"{target} vs {var}")
        fig, ax = plt.subplots()
        sns.regplot(x=gpa[var], y=gpa[target], scatter_kws={"alpha":0.6}, line_kws={"color":"red"}, ax=ax)
        ax.set_title(f'{target} vs {var}')
        ax.set_xlabel(var)
        ax.set_ylabel(target)
        st.pyplot(fig)

    # Categóricas: Boxplots
    st.markdown("### Variables categóricas")
    for var in categoricas:
        st.subheader(f"{target} por {var}")
        fig, ax = plt.subplots()
        sns.boxplot(x=var, y=target, data=gpa, palette='Set3', ax=ax)
        ax.set_title(f'{target} por {var}')
        ax.set_xlabel(var)
        ax.set_ylabel(target)
        st.pyplot(fig)

with tab3:
    st.header("Predicción del promedio con modelo de regresión lineal")

    # Copiar variables necesarias
    X = gpa[['edad', 'sexo', 'alcohol', 'noviazgo', 'fallas']].copy()
    y = gpa['promedio']

    # Mostrar valores únicos para diagnóstico (puedes ocultar esto luego)
    # st.write("Valores únicos:")
    # st.write("sexo:", X['sexo'].unique())
    # st.write("alcohol:", X['alcohol'].unique())
    # st.write("noviazgo:", X['noviazgo'].unique())

    # Mapeos seguros
    X['sexo'] = X['sexo'].map({'Femenino': 1, 'Masculino': 0})
    X['alcohol'] = X['alcohol'].map({'Sí': 1, 'No': 0})
    X['noviazgo'] = X['noviazgo'].map({'Sí': 1, 'No': 0})

    # Rellenar valores faltantes si los hay
    X = X.fillna(0)
    y = y.loc[X.index]

    if len(X) == 0:
        st.error("Error: El modelo no puede entrenarse porque no hay datos válidos después de transformar las variables.")
    else:
        # Entrenar el modelo
        model = LinearRegression()
        model.fit(X, y)

        # Inputs del usuario
        edad = st.slider("Edad", int(gpa['edad'].min()), int(gpa['edad'].max()))
        sexo = st.selectbox("Sexo", ["Femenino", "Masculino"])
        alcohol = st.selectbox("¿Consume alcohol?", ["Sí", "No"])
        noviazgo = st.selectbox("¿Tiene noviazgo?", ["Sí", "No"])
        fallas = st.slider("Número de fallas", int(gpa['fallas'].min()), int(gpa['fallas'].max()))

        # Codificar
        sexo_num = 1 if sexo == "Femenino" else 0
        alcohol_num = 1 if alcohol == "Sí" else 0
        noviazgo_num = 1 if noviazgo == "Sí" else 0

        st.markdown("### Valores seleccionados:")
        st.write(f"Edad: {edad}")
        st.write(f"Sexo: {sexo} (Transformado a {sexo_num})")
        st.write(f"Alcohol: {alcohol} (Transformado a {alcohol_num})")
        st.write(f"Noviazgo: {noviazgo} (Transformado a {noviazgo_num})")
        st.write(f"Fallas: {fallas}")

        if st.button("Predecir promedio"):
            input_array = np.array([[edad, sexo_num, alcohol_num, noviazgo_num, fallas]])
            pred = model.predict(input_array)
            st.success(f"Predicción del promedio: {pred[0]:.2f}")
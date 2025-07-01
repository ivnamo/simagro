import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="Simulador de Ensayos", layout="centered")

st.title("🧪 Simulador de Ensayos Agronómicos")
st.markdown("""
Esta aplicación genera datos simulados de un experimento agrícola con un control y varios tratamientos.
Podrás aplicar un incremento esperado respecto al control y añadir variabilidad aleatoria, y luego realizar un análisis ANOVA.
""")

# Entradas del usuario
control = st.number_input("🔹 Valor medio del control", min_value=0.0, value=100.0)
n_tratamientos = st.number_input("🔹 Número de tratamientos", min_value=1, step=1, value=3)
mejora_pct = st.slider("🔹 Mejora esperada (%) respecto al control", 0, 100, 10)
aleatoriedad_pct = st.slider("🔹 Variabilidad aleatoria (%)", 0, 100, 5)
replicas = st.number_input("🔹 Réplicas por tratamiento", min_value=2, step=1, value=4)

if st.button("Simular experimento"):
    np.random.seed(42)
    datos = []
    for i in range(n_tratamientos + 1):
        if i == 0:
            nombre = "Control"
            media = control
        else:
            nombre = f"Tratamiento {i}"
            media = control * (1 + mejora_pct / 100)
        
        std_dev = media * aleatoriedad_pct / 100
        valores = np.random.normal(loc=media, scale=std_dev, size=replicas)
        
        for valor in valores:
            datos.append({"Tratamiento": nombre, "Valor": valor})
    
    df = pd.DataFrame(datos)

    # Mostrar datos simulados
    st.subheader("📊 Datos Simulados")
    st.dataframe(df)

    # Gráfico
    fig = px.box(df, x="Tratamiento", y="Valor", title="Distribución de Resultados por Tratamiento")
    st.plotly_chart(fig, use_container_width=True)

    # ANOVA
    st.subheader("📈 Análisis ANOVA")
    modelo = ols('Valor ~ C(Tratamiento)', data=df).fit()
    anova_tabla = sm.stats.anova_lm(modelo, typ=2)
    st.write(anova_tabla)

    # Tukey HSD
    st.subheader("🔍 Comparaciones múltiples (Tukey HSD)")
    tukey = pairwise_tukeyhsd(endog=df['Valor'], groups=df['Tratamiento'], alpha=0.05)
    st.text(tukey.summary())

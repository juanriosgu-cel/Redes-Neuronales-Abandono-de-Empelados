import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Clasificación del abandono de empleados", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")
BASE = Path(__file__).resolve().parent

CLASSIC = pd.DataFrame([
 {"Modelo":"Random Forest","Accuracy":0.8231,"Precision":0.4468,"Recall":0.4468,"F1":0.4468,"F1 Macro":0.6708,"ROC-AUC":0.7482,"PR-AUC":0.3815,"Tipo":"Clásico · referencia"},
 {"Modelo":"Regresión Logística","Accuracy":0.8605,"Precision":0.6154,"Recall":0.3404,"F1":0.4384,"F1 Macro":0.6794,"ROC-AUC":0.8079,"PR-AUC":0.5716,"Tipo":"Clásico · referencia"},
 {"Modelo":"Árbol de Decisión","Accuracy":0.8163,"Precision":0.4146,"Recall":0.3617,"F1":0.3864,"F1 Macro":0.6392,"ROC-AUC":0.6510,"PR-AUC":0.3444,"Tipo":"Clásico · referencia"},
 {"Modelo":"KNN","Accuracy":0.8537,"Precision":0.6111,"Recall":0.2340,"F1":0.3385,"F1 Macro":0.6281,"ROC-AUC":0.6697,"PR-AUC":0.3407,"Tipo":"Clásico · referencia"},
])

@st.cache_data
def cargar_dataset():
    candidates = [
        BASE / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        BASE / "WA_Fn-UseC_-HR-Employee-Attrition(1).csv",
    ]
    csv_path = next((p for p in candidates if p.is_file()), None)
    if csv_path is None:
        matches = sorted(p for p in BASE.glob("*.csv") if "Employee-Attrition" in p.name)
        csv_path = matches[0] if matches else None
    return pd.read_csv(csv_path) if csv_path else None

@st.cache_resource
def cargar_artifacts():
    """Carga artefactos tanto desde artifacts_neural/ como desde la raíz del repositorio."""
    import tensorflow as tf

    # Preferir la carpeta de artefactos si existe; si no, usar archivos publicados en la raíz.
    artifact_dir = BASE / "artifacts_neural"
    root = artifact_dir if artifact_dir.is_dir() else BASE

    metadata_candidates = [root / "metadata.joblib", BASE / "metadata.joblib"]
    preprocessor_candidates = [root / "preprocessor.joblib", BASE / "preprocessor.joblib"]
    metrics_candidates = [root / "neural_metrics.joblib", BASE / "neural_metrics.joblib"]
    details_candidates = [root / "neural_details.joblib", BASE / "neural_details.joblib"]

    def first_existing(paths):
        return next((p for p in paths if p.is_file()), None)

    meta_path = first_existing(metadata_candidates)
    prep_path = first_existing(preprocessor_candidates)
    if meta_path is None or prep_path is None:
        raise FileNotFoundError(
            "Faltan metadata.joblib o preprocessor.joblib. "
            "Deben estar en la raíz del repositorio o en artifacts_neural/."
        )

    meta = joblib.load(meta_path)
    prep = joblib.load(prep_path)
    metrics_path = first_existing(metrics_candidates)
    details_path = first_existing(details_candidates)
    metrics = joblib.load(metrics_path) if metrics_path else []
    details = joblib.load(details_path) if details_path else {}

    models = {}
    model_files = meta.get("model_files", {})

    # Resolver rutas guardadas en metadata: admite rutas relativas de Colab y archivos en raíz.
    for model_name, saved_path in model_files.items():
        saved = Path(str(saved_path))
        candidates = [
            saved if saved.is_absolute() else BASE / saved,
            root / saved.name,
            BASE / saved.name,
        ]
        model_path = first_existing(candidates)
        if model_path:
            models[model_name] = tf.keras.models.load_model(model_path)

    # Compatibilidad con el conjunto de artefactos que se subió directamente a GitHub.
    if not models:
        model_candidates = [
            ("MLP", "mlp.keras"),
            ("Residual MLP", "residual_mlp.keras"),
            ("Ancho y profundo", "wide_deep.keras"),
        ]
        for model_name, filename in model_candidates:
            model_path = first_existing([root / filename, BASE / filename])
            if model_path:
                models[model_name] = tf.keras.models.load_model(model_path)

    if not models:
        raise FileNotFoundError(
            "No se encontraron modelos .keras. Se esperaban mlp.keras, "
            "residual_mlp.keras o wide_deep.keras en la raíz o en artifacts_neural/."
        )

    return models, prep, meta, metrics, details

st.markdown("<style>.block-container{padding-top:1.4rem;padding-bottom:2rem}h1,h2,h3{letter-spacing:-.02em}</style>",unsafe_allow_html=True)
st.sidebar.title("Navegación")
pagina=st.sidebar.radio("Seleccione una sección",["Presentación","Análisis exploratorio (EDA)","Clasificación","Redes neuronales","Comparación global","Análisis de desempeño","Predicción individual","Metodología"])
st.sidebar.divider()
st.sidebar.caption("IBM HR Analytics · Clasificación binaria de Attrition")
df=cargar_dataset()

if pagina=="Presentación":
    st.markdown("<div style='padding:2rem;border-radius:18px;background:linear-gradient(125deg,#142d4e,#176b87,#23a6a1);color:white'><div style='font-weight:700;letter-spacing:.12em'>APRENDIZAJE AUTOMÁTICO</div><h1 style='color:white'>Clasificación del abandono de empleados</h1><p>Modelos clásicos y redes neuronales para datos tabulares</p><hr><b>Dataset:</b> IBM HR Analytics — Employee Attrition<br><br><b>Proyecto académico</b><br>Especialización en Inteligencia Artificial</div>",unsafe_allow_html=True)
    a,b,c=st.columns(3); a.metric("Problema","Clasificación binaria"); b.metric("Clase positiva","Abandonó"); c.metric("Enfoques","ML clásico + redes neuronales")
    st.write("La aplicación explora el conjunto de datos, compara algoritmos y permite analizar probabilidades estimadas de abandono. Los resultados son descriptivos y académicos.")

if pagina == "Análisis exploratorio (EDA)":
    st.header("Análisis exploratorio de datos (EDA)")
    st.write(
        "Exploración descriptiva del dataset IBM HR Analytics antes de interpretar "
        "los resultados de los modelos de clasificación. Los patrones observados "
        "son descriptivos y no demuestran causalidad."
    )

    @st.cache_data
    def cargar_dataset_eda():
        carpeta_app = Path(__file__).resolve().parent
        nombres_csv = [
            "WA_Fn-UseC_-HR-Employee-Attrition.csv",
            "WA_Fn-UseC_-HR-Employee-Attrition(1).csv"
        ]
        for nombre_csv in nombres_csv:
            ruta_csv = carpeta_app / nombre_csv
            if ruta_csv.is_file():
                return pd.read_csv(ruta_csv)
        return None

    df_eda = cargar_dataset_eda()
    if df_eda is None:
        st.error(
            "No se encontró el CSV del dataset. Incluya "
            "WA_Fn-UseC_-HR-Employee-Attrition.csv en la misma carpeta que app.py."
        )
        st.stop()

    numeric_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_eda.select_dtypes(exclude=np.number).columns.tolist()
    n_total = len(df_eda)
    n_abandono = int((df_eda["Attrition"] == "Yes").sum()) if "Attrition" in df_eda else 0
    tasa_abandono = n_abandono / n_total if n_total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros", f"{n_total:,}")
    m2.metric("Variables", len(df_eda.columns))
    m3.metric("Abandonos", f"{n_abandono:,}")
    m4.metric("Tasa de abandono", f"{tasa_abandono:.1%}")

    tab_calidad, tab_distribucion, tab_relaciones, tab_atipicos, tab_hallazgos = st.tabs([
        "Calidad de datos", "Distribuciones", "Relaciones", "Valores atípicos", "Hallazgos"
    ])

    with tab_calidad:
        st.subheader("Estructura y calidad del conjunto")
        calidad = pd.DataFrame({
            "Indicador": ["Filas", "Columnas", "Celdas vacías", "Filas duplicadas", "Variables constantes"],
            "Resultado": [
                len(df_eda), len(df_eda.columns), int(df_eda.isna().sum().sum()),
                int(df_eda.duplicated().sum()),
                int(sum(df_eda[c].nunique(dropna=False) <= 1 for c in df_eda.columns))
            ]
        })
        st.dataframe(calidad, use_container_width=True, hide_index=True)
        faltantes = df_eda.isna().sum().rename("Faltantes").reset_index().rename(columns={"index":"Variable"})
        faltantes = faltantes[ faltantes["Faltantes"] > 0 ]
        if faltantes.empty:
            st.success("No se identificaron valores faltantes en el dataset.")
        else:
            st.dataframe(faltantes, use_container_width=True, hide_index=True)
        st.markdown("#### Tipos de variables")
        tipos = pd.DataFrame({"Variable": df_eda.columns, "Tipo": df_eda.dtypes.astype(str).values,
                              "Valores distintos": [df_eda[c].nunique(dropna=False) for c in df_eda.columns]})
        st.dataframe(tipos, use_container_width=True, hide_index=True)

    with tab_distribucion:
        st.subheader("Distribución de la variable objetivo")
        if "Attrition" in df_eda:
            conteo = df_eda["Attrition"].value_counts().rename_axis("Abandono").reset_index(name="Empleados")
            fig = px.bar(conteo, x="Abandono", y="Empleados", color="Abandono",
                         color_discrete_map={"Yes":"#E76F51", "No":"#2878B5"},
                         text="Empleados", title="Distribución de empleados por abandono")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        variable_num = st.selectbox("Variable numérica", numeric_cols,
                                    index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)
        fig_hist = px.histogram(df_eda, x=variable_num, color="Attrition" if "Attrition" in df_eda else None,
                                marginal="box", barmode="overlay", opacity=.72,
                                title=f"Distribución de {variable_num}")
        st.plotly_chart(fig_hist, use_container_width=True)
        if "Attrition" in df_eda:
            st.markdown("#### Tasa de abandono por variable categórica")
            opciones_cat = [c for c in categorical_cols if c != "Attrition"]
            cat = st.selectbox("Variable categórica", opciones_cat,
                               index=opciones_cat.index("OverTime") if "OverTime" in opciones_cat else 0)
            tasas = df_eda.groupby(cat, dropna=False)["Attrition"].apply(lambda s: (s == "Yes").mean()).mul(100).reset_index(name="Tasa de abandono (%)")
            tasas[cat] = tasas[cat].astype(str)
            fig_tasa = px.bar(tasas.sort_values("Tasa de abandono (%)", ascending=False),
                              x=cat, y="Tasa de abandono (%)", color="Tasa de abandono (%)",
                              color_continuous_scale=["#8ECAE6","#219EBC","#E76F51"],
                              text=tasas.sort_values("Tasa de abandono (%)", ascending=False)["Tasa de abandono (%)"].map(lambda x:f"{x:.1f}%"),
                              title=f"Tasa de abandono según {cat}")
            st.plotly_chart(fig_tasa, use_container_width=True)

    with tab_relaciones:
        st.subheader("Relaciones entre variables")
        corr_cols = [c for c in numeric_cols if df_eda[c].nunique() > 1]
        default_corr = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"] if c in corr_cols]
        corr_selected = st.multiselect("Variables para correlación", corr_cols, default=default_corr[:8])
        if len(corr_selected) >= 2:
            corr = df_eda[corr_selected].corr(numeric_only=True)
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, title="Matriz de correlación")
            st.plotly_chart(fig_corr, use_container_width=True)
        scatter_defaults = [c for c in ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears"] if c in numeric_cols]
        if len(scatter_defaults) >= 2:
            st.markdown("#### Matriz de dispersión de variables principales")
            pair_vars = st.multiselect("Variables de la matriz", numeric_cols, default=scatter_defaults)
            if len(pair_vars) >= 2:
                fig_pair = px.scatter_matrix(df_eda, dimensions=pair_vars,
                    color="Attrition" if "Attrition" in df_eda else None,
                    color_discrete_map={"Yes":"#E76F51", "No":"#2878B5"},
                    opacity=.55, title="Relaciones multivariadas")
                fig_pair.update_layout(height=750)
                st.plotly_chart(fig_pair, use_container_width=True)
        if "Department" in df_eda and "Attrition" in df_eda:
            dept = df_eda.groupby("Department")["Attrition"].apply(lambda s:(s=="Yes").mean()*100).reset_index(name="Tasa (%)")
            fig_dept = px.bar(dept, x="Department", y="Tasa (%)", color="Tasa (%)",
                              color_continuous_scale=["#8ECAE6","#219EBC","#E76F51"],
                              title="Tasa de abandono por departamento")
            st.plotly_chart(fig_dept, use_container_width=True)

    with tab_atipicos:
        st.subheader("Detección estadística mediante rango intercuartílico (IQR)")
        variable_out = st.selectbox("Variable para detectar valores atípicos", numeric_cols,
                                    index=numeric_cols.index("MonthlyIncome") if "MonthlyIncome" in numeric_cols else 0)
        q1 = df_eda[variable_out].quantile(.25)
        q3 = df_eda[variable_out].quantile(.75)
        iqr = q3 - q1
        limite_inf, limite_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask_out = (df_eda[variable_out] < limite_inf) | (df_eda[variable_out] > limite_sup)
        st.caption(f"Límite inferior: {limite_inf:,.2f} · Límite superior: {limite_sup:,.2f}")
        a, b, c = st.columns(3)
        a.metric("Valores atípicos detectados", int(mask_out.sum()))
        b.metric("Dentro de los límites", int((~mask_out).sum()))
        c.metric("Porcentaje atípico", f"{mask_out.mean():.1%}")
        df_box = df_eda[[variable_out]].copy()
        df_box["Tipo de dato"] = np.where(mask_out, "Atípico", "Dentro del rango")
        fig_box = px.box(df_box, x="Tipo de dato", y=variable_out, color="Tipo de dato",
                         color_discrete_map={"Atípico":"#E76F51","Dentro del rango":"#2878B5"},
                         points="all", title=f"Distribución y valores atípicos: {variable_out}")
        st.plotly_chart(fig_box, use_container_width=True)
        st.info("Los valores atípicos estadísticos no son necesariamente errores. Deben interpretarse en el contexto de la variable y no eliminarse automáticamente.")

    with tab_hallazgos:
        st.subheader("Hallazgos descriptivos")
        st.write(f"La tasa global de abandono en la base es **{tasa_abandono:.2%}** ({n_abandono} de {n_total} empleados).")
        if "YearsAtCompany" in df_eda and "Attrition" in df_eda:
            med = df_eda.groupby("Attrition")["YearsAtCompany"].median()
            if "Yes" in med.index and "No" in med.index:
                st.write(f"La mediana de antigüedad es **{med['Yes']:.0f} años** entre quienes abandonaron y **{med['No']:.0f} años** entre quienes permanecieron.")
        if "MonthlyIncome" in df_eda and "Attrition" in df_eda:
            med_ing = df_eda.groupby("Attrition")["MonthlyIncome"].median()
            if "Yes" in med_ing.index and "No" in med_ing.index:
                st.write(f"La mediana de ingreso mensual es **{med_ing['Yes']:,.0f}** para quienes abandonaron y **{med_ing['No']:,.0f}** para quienes permanecieron.")
        if "OverTime" in df_eda and "Attrition" in df_eda:
            overtime = df_eda.groupby("OverTime")["Attrition"].apply(lambda s:(s=="Yes").mean()*100)
            if "Yes" in overtime.index and "No" in overtime.index:
                st.write(f"La tasa de abandono entre empleados con horas extra es **{overtime['Yes']:.1f}%**, frente a **{overtime['No']:.1f}%** para quienes no reportan horas extra.")
        st.warning("Estos hallazgos describen asociaciones observadas en los datos; no prueban que una variable cause el abandono.")


elif pagina in ["Clasificación","Redes neuronales","Comparación global","Análisis de desempeño","Predicción individual","Metodología"]:
    try:
        models,preprocessor,metadata,neural_metrics,details=cargar_artifacts()
    except Exception as e:
        st.error(f"No fue posible cargar los artefactos del modelo: {e}")
        st.stop()

    # Normalize exported metric schema into the interface table
    if isinstance(neural_metrics,dict): neural_metrics=[neural_metrics]
    neural_rows=[]
    for r in neural_metrics:
        neural_rows.append({"Modelo":r.get("Modelo",r.get("name","Red neuronal")),"Accuracy":r.get("Accuracy",np.nan),"Precision":r.get("Precision",r.get("Precision (Abandonó)",np.nan)),"Recall":r.get("Recall",r.get("Recall (Abandonó)",np.nan)),"F1":r.get("F1",r.get("F1 (Abandonó)",np.nan)),"F1 Macro":r.get("F1 Macro",np.nan),"ROC-AUC":r.get("ROC-AUC",np.nan),"PR-AUC":r.get("PR-AUC",r.get("PR-AUC (Average Precision)",np.nan)),"Tipo":"Red neuronal"})
    neural_df=pd.DataFrame(neural_rows)
    if neural_df.empty:
        # Legacy deployment: use saved known evaluation values, clearly marked as MLP result
        neural_df=pd.DataFrame([{"Modelo":"MLP","Accuracy":0.8061,"Precision":0.4107,"Recall":0.4894,"F1":0.4466,"F1 Macro":0.6645,"ROC-AUC":0.7446,"PR-AUC":0.4886,"Tipo":"Red neuronal · resultado previo"}])
    global_df=pd.concat([CLASSIC,neural_df],ignore_index=True,sort=False)
    best_neural=neural_df.sort_values("F1",ascending=False).iloc[0]
    best_global=global_df.sort_values("F1",ascending=False).iloc[0]

    if pagina=="Clasificación":
        st.title("Modelos de clasificación")
        st.write("Algoritmos clásicos evaluados previamente. Sus resultados se muestran como referencia comparativa; no se cargan aquí como modelos predictivos individuales.")
        st.dataframe(CLASSIC.drop(columns="Tipo").style.format({c:"{:.4f}" for c in CLASSIC.columns if c not in ["Modelo","Tipo"]}),use_container_width=True,hide_index=True)
        fig=px.bar(CLASSIC,x="Modelo",y="F1",color="Modelo",text_auto=".4f",title="F1-score · modelos clásicos")
        fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)

    elif pagina=="Redes neuronales":
        st.title("Redes neuronales")
        st.write("Comparación de arquitecturas entrenadas con el mismo preprocesamiento y partición de prueba.")
        st.dataframe(neural_df.drop(columns="Tipo",errors="ignore").style.format({c:"{:.4f}" for c in neural_df.columns if c not in ["Modelo","Tipo"]}),use_container_width=True,hide_index=True)
        fig=px.bar(neural_df,x="Modelo",y="F1",color="Modelo",text_auto=".4f",title="F1-score de la clase positiva · redes neuronales")
        fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
        if details:
            chosen=st.selectbox("Arquitectura para revisar curvas",list(details.keys()))
            hist=details[chosen].get("history",{})
            if "loss" in hist:
                hdf=pd.DataFrame(hist); hdf.index=hdf.index+1; hdf.index.name="Época"
                cols=[c for c in ["loss","val_loss"] if c in hdf]
                if cols: st.plotly_chart(px.line(hdf[cols],title=f"Pérdida de entrenamiento y validación · {chosen}",labels={"value":"Loss","index":"Época"}),use_container_width=True)
                metric_cols=[c for c in ["accuracy","val_accuracy","auc","val_auc"] if c in hdf]
                if metric_cols: st.plotly_chart(px.line(hdf[metric_cols],title=f"Métricas por época · {chosen}",labels={"value":"Valor","index":"Época"}),use_container_width=True)

    elif pagina=="Comparación global":
        st.title("Comparación global de modelos")
        st.write("Se contrasta la referencia de modelos clásicos con las arquitecturas neuronales evaluadas. La selección se prioriza por F1 de la clase positiva «Abandonó», considerando Recall y PR-AUC.")
        st.dataframe(global_df.sort_values("F1",ascending=False).style.format({c:"{:.4f}" for c in global_df.columns if c not in ["Modelo","Tipo"]}),use_container_width=True,hide_index=True)
        metric=st.selectbox("Métrica para comparar",["F1","Recall","Precision","PR-AUC","ROC-AUC","Accuracy"],index=0)
        fig=px.bar(global_df,x="Modelo",y=metric,color="Tipo",barmode="group",text_auto=".4f",title=f"Comparación de {metric}")
        fig.update_layout(xaxis_tickangle=-25); st.plotly_chart(fig,use_container_width=True)
        a,b=st.columns(2); a.metric("Mejor red neuronal",str(best_neural["Modelo"])); b.metric("Mejor F1 observado",f"{best_global['F1']:.4f}",str(best_global["Modelo"]))
        st.caption("Los resultados clásicos corresponden a una evaluación previa. La comparación directa es descriptiva si no comparten exactamente la misma partición de prueba.")
        st.download_button("Descargar comparación CSV",global_df.to_csv(index=False).encode("utf-8-sig"),"comparacion_global_modelos.csv","text/csv")

    elif pagina=="Análisis de desempeño":
        st.title("Análisis de desempeño")
        if not details:
            st.info("Las matrices y curvas se habilitan al cargar el paquete de artefactos exportado desde el notebook de entrenamiento.")
        else:
            chosen=st.selectbox("Modelo neuronal",list(details.keys()))
            d=details[chosen]
            cm=np.array(d["confusion_matrix"])
            fig=px.imshow(cm,text_auto=True,x=["Predijo permanencia","Predijo abandono"],y=["Real permanencia","Real abandono"],color_continuous_scale="Blues",title=f"Matriz de confusión · {chosen}")
            st.plotly_chart(fig,use_container_width=True)
            if d.get("roc_fpr") and d.get("roc_tpr"):
                fig=go.Figure(); fig.add_trace(go.Scatter(x=d["roc_fpr"],y=d["roc_tpr"],mode="lines",name=f"ROC (AUC={float(neural_df.loc[neural_df.Modelo==chosen,'ROC-AUC'].iloc[0]):.3f})")); fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",line=dict(dash="dash"),name="Azar")); fig.update_layout(title=f"Curva ROC · {chosen}",xaxis_title="Tasa de falsos positivos",yaxis_title="Tasa de verdaderos positivos"); st.plotly_chart(fig,use_container_width=True)
            if d.get("pr_recall") and d.get("pr_precision"):
                fig=go.Figure(); fig.add_trace(go.Scatter(x=d["pr_recall"],y=d["pr_precision"],mode="lines",name="Precision-Recall")); fig.update_layout(title=f"Curva Precision-Recall · {chosen}",xaxis_title="Recall",yaxis_title="Precision"); st.plotly_chart(fig,use_container_width=True)

    elif pagina=="Predicción individual":
        st.title("Predicción individual")
        if df is None: st.error("No se encontró el CSV del dataset."); st.stop()
        chosen=st.selectbox("Red neuronal para la predicción",list(models.keys()),index=list(models.keys()).index(metadata.get("best_neural_model",list(models.keys())[0])) if metadata.get("best_neural_model") in models else 0)
        input_cols=metadata.get("input_columns",[c for c in df.columns if c!="Attrition"])
        num_features=metadata.get("numeric_features",[]); cat_features=metadata.get("categorical_features",[])
        threshold=float(metadata.get("threshold",0.5))
        defaults={c:(df[c].median() if c in num_features else df[c].mode(dropna=True).iloc[0]) for c in input_cols if c in df.columns}
        with st.form("individual_prediction"):
            entered={}; columns=st.columns(2)
            for i,c in enumerate(input_cols):
                if c not in df.columns: continue
                with columns[i%2]:
                    if c in cat_features or not pd.api.types.is_numeric_dtype(df[c]):
                        opts=sorted(df[c].dropna().astype(str).unique().tolist()); val=str(defaults[c])
                        entered[c]=st.selectbox(c,opts,index=opts.index(val) if val in opts else 0)
                    else:
                        mn=float(df[c].min()); mx=float(df[c].max()); val=float(defaults[c])
                        entered[c]=st.number_input(c,min_value=mn,max_value=mx,value=min(max(val,mn),mx),step=1.0)
            submit=st.form_submit_button("Estimar probabilidad",type="primary")
        if submit:
            row=pd.DataFrame([entered]).reindex(columns=input_cols)
            for c in cat_features:
                if c in row: row[c]=row[c].astype(str)
            x=preprocessor.transform(row)
            p=float(np.asarray(models[chosen].predict(x,verbose=0)).reshape(-1)[0]); pred=int(p>=threshold)
            a,b=st.columns(2); a.metric("Probabilidad de abandono",f"{p:.1%}"); b.metric("Probabilidad de permanencia",f"{1-p:.1%}")
            st.success("Clasificación estimada: posible abandono" if pred else "Clasificación estimada: permanencia")
            fig=px.bar(x=["Abandono","Permanencia"],y=[p,1-p],title="Probabilidades estimadas",labels={"x":"","y":"Probabilidad"}); fig.update_yaxes(range=[0,1],tickformat=".0%"); st.plotly_chart(fig,use_container_width=True)
            st.warning("Resultado orientativo académico; no constituye una decisión automática sobre una persona.")

    elif pagina=="Metodología":
        st.title("Metodología")
        steps=[("1. Exploración","Se examinan estructura, calidad, distribución y relaciones descriptivas del dataset."),("2. Preparación","Se separan variables predictoras y objetivo, y se divide la muestra de forma estratificada."),("3. Preprocesamiento","Las variables numéricas se imputan y escalan; las categóricas se imputan y codifican. El ajuste se realiza solo con entrenamiento."),("4. Modelos clásicos","Se utilizan como referencia comparativa los resultados de Random Forest, Regresión Logística, Árbol de Decisión y KNN."),("5. Redes neuronales","Se comparan MLP, Wide & Deep y Residual MLP sobre datos tabulares."),("6. Evaluación","Se analizan Accuracy, Precision, Recall, F1, F1 Macro, ROC-AUC y PR-AUC, con énfasis en la clase positiva Abandonó."),("7. Selección","Se prioriza F1 de Abandonó, interpretándolo junto con Recall, Precision y PR-AUC."),("8. Despliegue","Streamlit carga los artefactos entrenados y permite consultar métricas, curvas y predicciones sin reentrenar.")]
        for title,desc in steps:
            with st.container(border=True): st.subheader(title); st.write(desc)
        st.warning("Los resultados son estimaciones académicas; no prueban causalidad y no deben utilizarse para decisiones automatizadas de empleo.")

st.divider()
st.caption("Clasificación del abandono de empleados · Especialización en Inteligencia Artificial · IBM HR Analytics")

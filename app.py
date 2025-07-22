import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import numpy as np

st.set_page_config(page_title="Adaptive Salary Advisor", layout="wide")

with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>💼 Adaptive Salary Advisor</h2>", unsafe_allow_html=True)
    mode = st.radio("Select Application Mode:", ["Predict", "Explore"])

    with st.expander("🤖 AI Chat Assistant (FAQs)"):
        suggested_questions = [
            "What is Salary Prediction?",
            "Which ML algorithm performs best?",
            "What is Feature Importance and how does it help?",
            "How do you measure model accuracy?",
            "Explain the project in simple terms.",
            "What is RMSE and why is it important?"
        ]
        user_question = st.selectbox("📝 Ask a question:", [""] + suggested_questions)
        answers = {
            "What is Salary Prediction?": "💬 Salary Prediction estimates an employee's salary based on features like experience, education, and age using ML models.",
            "Which ML algorithm performs best?": "💬 XGBoost generally performs best due to its boosting capabilities, but results vary per dataset.",
            "What is Feature Importance and how does it help?": "💬 Feature Importance shows how much each feature contributes to the model's prediction.",
            "How do you measure model accuracy?": "💬 Metrics like RMSE, R² Score, and Explained Variance help evaluate prediction quality.",
            "Explain the project in simple terms.": "💬 It predicts salary, explains prediction, compares models, and provides a chat assistant.",
            "What is RMSE and why is it important?": "💬 RMSE shows how close predictions are to actual values; lower RMSE is better."
        }
        if user_question:
            st.success(answers[user_question])

st.title("💼 Adaptive Salary Advisor Dashboard")

with st.expander("📖 About This Project"):
    st.markdown("""
    This AI-based Salary Advisor predicts salaries using ML models.

    Features:
    - Dynamic Salary Prediction
    - Feature Importance Visualizations
    - Multi-Model Performance Comparison
    - Offline AI Chat Assistant
    - Prediction History Tracking
    """)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

df = pd.read_csv('data/salary_data1.csv')
X = df.drop(columns=['salary'])
y = df['salary']

if mode == "Predict":
    tabs = st.tabs(["🎯 Prediction Panel", "📜 Prediction History"])

    with tabs[0]:
        st.markdown("<h4 style='color:#007acc;'>🔵 You are in Prediction Mode</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30)
            experience = st.slider("Years of Experience", 0, 40, 5)
        with col2:
            education = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])

        if st.button("💰 Predict Salary"):
            input_df = pd.DataFrame({'Age': [age], 'YearsExperience': [experience], 'Education': [education]})
            predicted_salary = model.predict(input_df)[0]
            st.success(f"💰 Predicted Salary: ₹ {predicted_salary:,.0f}")
            st.session_state.history.append({
                'Age': age, 'Experience': experience, 'Education': education,
                'Predicted Salary': f"₹ {predicted_salary:,.0f}"
            })

    with tabs[1]:
        if st.session_state.history:
            st.dataframe(pd.DataFrame(st.session_state.history))
        else:
            st.info("No predictions made yet.")

if mode == "Explore":
    st.markdown("<h4 style='color:#e67e00;'>🟡 You are in Data Exploration Mode</h4>", unsafe_allow_html=True)

    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    preprocessor = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)], remainder='passthrough')
    rf_pipeline = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))])
    rf_pipeline.fit(X, y)

    ohe = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    ohe.fit(X[cat_cols])
    encoded_cat_cols = ohe.get_feature_names_out(cat_cols)
    all_features = list(encoded_cat_cols) + [col for col in X.columns if col not in cat_cols]

    importances = rf_pipeline.named_steps['model'].feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    st.subheader("🏋️ Top 15 Feature Importances")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(y='Feature', x='Importance', data=feat_imp_df.head(15), ax=ax, palette='flare')
    ax.set_title('Top 15 Feature Importances')
    st.pyplot(fig, use_container_width=True)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📊 Model Performance Comparison")
    models = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(), "XGBoost": XGBRegressor()}
    results = []
    for name, algo in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', algo)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        explained_var = explained_variance_score(y_test, y_pred)
        rel_acc = (1 - rmse / np.mean(y_test)) * 100
        results.append({'Model': name, 'RMSE': round(rmse, 2), 'R² Score': round(r2, 2), 'Explained Variance': round(explained_var, 2), 'Relative Accuracy (%)': round(rel_acc, 2)})

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 RMSE Comparison")
        fig1 = px.bar(results_df, x='Model', y='RMSE', color='Model', title='Model RMSE', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("📈 Relative Accuracy Comparison")
        fig2 = px.bar(results_df, x='Model', y='Relative Accuracy (%)', color='Model', title='Model Accuracy', color_discrete_sequence=px.colors.qualitative.Safe)
        st.plotly_chart(fig2, use_container_width=True)

    best_model = results_df.loc[results_df['Relative Accuracy (%)'].idxmax()]
    st.success(f"✅ Best Model: {best_model['Model']} (Accuracy: {best_model['Relative Accuracy (%)']}%)")

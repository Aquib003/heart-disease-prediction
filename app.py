from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

st.set_page_config(page_title='Heart Disease ML Demo', page_icon='🫀', layout='centered')
st.title('Heart Disease Prediction — ML Demonstration')
st.warning('Educational research demo only. Not a diagnostic tool, not a heart-attack detector, and not suitable for medical decisions. Do not enter real patient information.')

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
@st.cache_resource
def load_models():
    df = pd.read_csv(Path(__file__).with_name('heart.csv'), encoding='utf-8-sig')
    if not set(FEATURES + ['target']).issubset(df.columns):
        raise ValueError('Dataset is missing required columns')
    df = df[FEATURES + ['target']].dropna().drop_duplicates()
    X, y = df[FEATURES], df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=2)),
        'Decision Tree': DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0),
    }
    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        scores[name] = (accuracy_score(y_test, predictions), recall_score(y_test, predictions))
    return models, scores, len(df)

try:
    models, scores, sample_count = load_models()
except Exception as exc:
    st.error(f'Unable to load demonstration models: {exc}')
    st.stop()

st.caption(f'Trained from {sample_count} deduplicated records in the bundled example dataset. Test-set scores are exploratory and not clinical validation.')
model_name = st.selectbox('Model', list(models))
st.write(f'Holdout accuracy: **{scores[model_name][0]:.1%}** · Holdout recall: **{scores[model_name][1]:.1%}**')
st.subheader('Enter synthetic example values')
with st.form('prediction'):
    a,b=st.columns(2)
    with a:
        age=st.number_input('Age (years)',18,100,54)
        sex=st.selectbox('Sex coding', [0,1], format_func=lambda v:'0 — Female' if v==0 else '1 — Male')
        cp=st.selectbox('Chest pain category (dataset code)',[0,1,2,3])
        trestbps=st.number_input('Resting blood pressure (mmHg)',70,240,130)
        chol=st.number_input('Cholesterol (mg/dL)',100,650,240)
        fbs=st.selectbox('Fasting blood sugar >120 mg/dL',[0,1])
        restecg=st.selectbox('Resting ECG category (dataset code)',[0,1,2])
    with b:
        thalach=st.number_input('Maximum heart rate achieved',60,230,150)
        exang=st.selectbox('Exercise-induced angina',[0,1])
        oldpeak=st.number_input('ST depression',0.0,10.0,1.0,step=0.1)
        slope=st.selectbox('ST slope (dataset code)',[0,1,2])
        ca=st.selectbox('Number of major vessels (dataset code)',[0,1,2,3,4])
        thal=st.selectbox('Thal category (dataset code)',[0,1,2,3])
    submitted=st.form_submit_button('Run example prediction')
if submitted:
    sample=pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]],columns=FEATURES)
    result=int(models[model_name].predict(sample)[0])
    st.info(f'Model output: class {result} — '+('dataset label: heart disease present' if result==1 else 'dataset label: heart disease absent'))
    st.caption('This output is a classification of the entered example values, not a diagnosis, personalized risk estimate, or prediction of an imminent heart attack.')

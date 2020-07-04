import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write(""" # Heart Disease Prediction App
This app predicts that *the person has heart disease or not* 
""")

def user_input_features():
    
    st.write("""**1. Select Age :**""") 
    age = st.slider('', 0, 100, 25)
    st.write("""**You selected this option **""",age)
    
    st.write("""**2. Select Gender :**""")
    sex = st.selectbox("(1=Male, 0=Female)",["1","0"])
    st.write("""**You selected this option **""",sex)
    
    st.write("""**3. Select Chest Pain Type :**""")
    cp = st.selectbox("(1 = Typical Angina, 2 = Atypical Angina, 3 = Non—anginal Pain, 4 = Asymptotic) : ",["1","2","3","4"])
    st.write("""**You selected this option **""",cp)
    
    st.write("""**4. Select Resting Blood Pressure :**""")
    trestbps = st.slider('In mm/Hg unit', 0, 200, 110)
    st.write("""**You selected this option **""",trestbps)
    
    st.write("""**5. Select Serum Cholesterol :**""")
    chol = st.slider('In mg/dl unit', 0, 600, 115)
    st.write("""**You selected this option **""",chol)
    
    st.write("""**6. Maximum Heart Rate Achieved (THALACH) :**""")
    thalach = st.slider('', 0, 220, 115)
    st.write("""**You selected this option **""",thalach)
    
    st.write("""**7. Exercise Induced Angina (Pain in chest while exersice) :**""")
    exang = st.selectbox("(1=Yes, 0=No)",["1","0"])
    st.write("""**You selected this option **""",exang)
    
    st.write("""**8. Oldpeak (ST depression induced by exercise relative to rest) :**""")
    oldpeak = float(st.slider('', 0.0, 10.0, 2.0))
    st.write("""**You selected this option **""",oldpeak)
    
    st.write("""**9. Slope (The slope of the peak exercise ST segment) :**""")
    slope = st.selectbox("(Select 0, 1 or 2)",["0","1","2"])
    st.write("""**You selected this option **""",slope)
    
    st.write("""**10. CA (Number of major vessels (0-3) colored by flourosopy) :**""")
    ca = st.selectbox("(Select 0, 1, 2 or 3)",["0","1","2","3"])
    st.write("""**You selected this option **""",ca)
    
    st.write("""**11. Thal :**""")
    thal = float(st.slider('3 = normal; 6 = fixed defect; 7 = reversable defect', 0.0, 10.0, 3.0))
    st.write("""**You selected this option **""",thal)
    
    
    data = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal,}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
st.subheader('Given Inputs : ')
st.write(df)

heart = pd.read_csv("heart1.csv")
X = heart.iloc[:,0:11].values
Y = heart.iloc[:,[11]].values

model = RandomForestClassifier()
model.fit(X, Y)

prediction = model.predict(df)
st.subheader('Prediction :')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(df)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)
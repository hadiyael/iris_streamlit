import streamlit as st
import pandas as pd
import pickle

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcnEoopczYX_eLHV5kWIGUiiow7pKpxLW-bQ&s', use_container_width=True)

st.title("MSDE6: ML Course")
st.header("Iris Flower Prediction App")
st.markdown("This app predicts the **Iris flower** type")
    
   
user_input = st.selectbox("How would you like to use the prediction model?", ['Input parameters directly','Load a file of data'], index= None, placeholder="Select an input method...")

model = pickle.load(open('modeliris6.pkl','rb'))

if user_input == 'Input parameters directly':
    with st.sidebar :
        st.image('https://cdn.britannica.com/39/91239-004-44353E32/Diagram-flowering-plant.jpg?s=1500x700&q=85')
        st.header("User Input Parameters")
        sepal_length = st.slider('Sepal length',0.00,10.00)
        sepal_width = st.slider('Sepal width',0.00,6.00)
        petal_length = st.slider('Petal length',0.00,10.00)
        petal_width = st.slider('Petal width',0.00,6.00)
    
    st.header("User Input Parameters:")
    data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    st.write(data)
    
    # st.header("Class labels and their corresponding index number")
    
    st.header("Prediction")
    prediction = model.predict(data)
    prediction_proba = model.predict_proba(data)
    st.write(prediction)
    st.header("Prediction Probability")
    st.write(prediction_proba)

elif user_input == 'Load a file of data':
    file = st.file_uploader('Upload a data file')
    if file is not None:
        data = pd.read_csv(file,header=None)
        st.header("User Input Parameters:")
        st.write(data)
        
        prediction = model.predict(data)
        prediction_proba = model.predict_proba(data)
        st.header("Prediction")
        st.write(prediction)
        st.header("Prediction Probability")
        st.write(prediction_proba)

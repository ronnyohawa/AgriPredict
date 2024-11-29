import streamlit as st 
import pandas as pd
import numpy as np
import os
import pickle
import warnings

st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:left;"> Crop Recommendation  🌱 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 2])  # Updated method

    with col1:
        with st.expander(" ℹ️ Information", expanded=True):  # Updated method
            st.write("""
            Crop recommendation plays a crucial role in precision agriculture, where various factors are considered to make informed decisions. Precision agriculture aims to tailor these factors to specific locations, addressing challenges in crop selection. Although this site-specific approach has enhanced efficiency, there remains a need for ongoing monitoring of the system's results. Not all precision agriculture systems are equally effective, and in farming, it is essential that recommendations are both accurate and reliable. Errors in these recommendations can lead to substantial losses of resources and finances.
            """)
        '''
        ## How does it work ❓ 
        Complete all the parameters and the machine learning model will predict the most suitable crops to grow in a particular farm based on various parameters.
        '''

    with col2:
        st.subheader(" Find out the most suitable crop to grow in your farm 👨‍🌾")
        N = st.number_input("Nitrogen", 1, 10000)
        P = st.number_input("Phosporus", 1, 10000)
        K = st.number_input("Potassium", 1, 10000)
        temp = st.number_input("Temperature", 0.0, 100000.0)
        humidity = st.number_input("Humidity in %", 0.0, 100000.0)
        ph = st.number_input("Ph", 0.0, 100000.0)
        rainfall = st.number_input("Rainfall in mm", 0.0, 100000.0)

        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)
        
        if st.button('Predict'):
            loaded_model = load_model('model.pkl')
            prediction = loaded_model.predict(single_pred)
            col1.write(''' ## Results 🔍 ''')
            col1.success(f"{prediction.item().title()} are recommended for this planting Season.")

    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 23:04:19 2022

@author: Shekhar Rajput
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav','rb'))

def farmer_prediction(input):
    numpy_array = np.asarray(input)
    input = numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input)
    return prediction


def main():
    st.title("farmer friend")
    
    #N,P,K,temperature,humidity,ph,rainfall
    
    N = st.slider("NITROGEN",min_value=0,max_value=140,value=5,step=1)
    P = st.slider("PHASPHORAS",min_value=0,max_value=145,value=5,step=1)
    K = st.slider("POTESISUM",min_value=0,max_value=205,value=5,step=1)
    Ph = st.slider("PH VALUE",min_value=0,max_value=7,value=4,step=1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("temperature.")
        T = st.number_input("TEMPERATURE")

    with col2:
        st.header("humidity")
        H = st.number_input("humidity")
    with col3:
        st.header("Rainfall")
        Rain = st.number_input("Rainfall")
        
    output =''
    
    if st.button('test result'):
        output = str(farmer_prediction([N,P,K,T,H,Ph,Rain]))
        
    st.success(output)
    
    
if __name__ == '__main__':
    main()
        
        
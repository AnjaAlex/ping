import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Carica il modello
model_pipe = joblib.load('penguinspipe.pkl')

st.title('App di pinguines')

# Input dall'utente
island = st.selectbox("Isola", ["Biscoe", "Dream", "Torgersen"])
bill_length_mm = st.number_input("Lunghezza del becco (mm)", min_value=0.0, step=0.1)
bill_depth_mm = st.number_input("Profondità del becco (mm)", min_value=0.0, step=0.1)
flipper_length_mm = st.number_input("Lunghezza delle pinne (mm)", min_value=0.0, step=0.1)
body_mass_g = st.number_input("Massa corporea (g)", min_value=0.0, step=1.0)
sex = st.selectbox("Sesso", ["Male", "Female"])

# Prepara i dati per la predizione
data = {
    "island": [island],
    "bill_length_mm": [bill_length_mm],
    "bill_depth_mm": [bill_depth_mm],
    "flipper_length_mm": [flipper_length_mm],
    "body_mass_g": [body_mass_g],
    "sex": [sex],
}

input_df = pd.DataFrame(data)

# Effettua la predizione
if st.button('Predizione'):
    res = model_pipe.predict(input_df).astype(int)[0]

    classes = {0: 'Adelie', 1: 'Gentoo', 2: 'Chinstrap'}
    y_pred = classes[res]

    st.write(f"Il pinguino è della specie: {y_pred}")




# import joblib
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt


# st.title ('App di pinguines')
# model_pipe = joblib.load('penguinspipe.pkl')
# print('Model loaded successfully')

# data = {
# island = st.selectbox('inserire isola', ['Dream', 'Biscoe', 'Torgersen'])
# bill_length_mm = st.number_input('inserire lungezza becco', 10, 60, 50)
# bill_depth_mm = st.number_input('inserire profondità becco', 1.0, 30.0, 5.0)
# flipper_length_mm = st.number_input('inserire lungezza pinna', 100.0, 250.0, 180.0)
# body_mass_g = st.number_input('inserire massa', 5000.0, 1500.0, 3500.00)
# sex = st.number_input('inserire sex', 'female', 'male'),
# }

# # island= 'Torgersen'
# # bill_length_mm = 39.1
# # bill_depth_mm = 18.7
# # flipper_length_mm = 181
# # body_mass_g = 3750
# # sex = 'male'

# data = {
#         "island": [island],
#         "bill_length_mm": [bill_length_mm],
#         "bill_depth_mm": [bill_depth_mm],
#         "flipper_length_mm":[flipper_length_mm],
#         "body_mass_g": [body_mass_g],
#         "sex": [sex],
#         }

# input_df = pd.DataFrame(data)
# res = model_pipe.predict(input_df).astype(int)[0]
# print(res)

# classes = {0:'Adelie',
#            1:'Gentoo',
#            2:'Chinstrap',
#            }

# y_pred = classes[res]
# y_pred

# print('prediction', y_pred)
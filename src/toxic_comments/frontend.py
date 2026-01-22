import requests
import streamlit as st

# --- Configuration ---
API_URL = 'https://toxic-backend-664727552674.europe-north2.run.app/predict'

# --- UI Layout ---
st.set_page_config(page_title='Toxic Comment Classifier', page_icon='🛡️')

st.title('🛡️ Toxic Comment Classifier')
st.write("Type a sentence below to check if it's toxic or not.")

# --- Input Section ---
user_input = st.text_area('Enter text here:', height=150, placeholder='Type something...')

if st.button('Analyze Text', type='primary'):
    if user_input.strip():
        try:
            # 1. Send request to FastAPI backend
            payload = {'text': user_input}
            response = requests.post(API_URL, json=payload)

            # 2. Handle the response
            if response.status_code == 200:
                result = response.json()
                label = result['label']
                confidence = result['confidence']
                is_toxic = result['is_toxic']

                # 3. Display Results
                st.subheader('Analysis Result')

                col1, col2 = st.columns(2)
                with col1:
                    st.metric('Label', label)
                with col2:
                    st.metric('Confidence', f'{confidence:.2%}')

                # Visual Feedback
                if is_toxic:
                    st.error(f'⚠️ Prediction: **{label}**')
                else:
                    st.success(f'✅ Prediction: **{label}**')

            else:
                st.error('Something went wrong while analyzing the text.')

        except requests.exceptions.ConnectionError:
            st.error('🚨 Connection Error: Backend service is unavailable.')
    else:
        st.warning('Please enter some text first.')

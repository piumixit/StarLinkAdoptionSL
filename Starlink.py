import streamlit as st

# Set page title and icon
st.set_page_config(
    page_title="My Simple App",
    page_icon=":smiley:"
)

# Add a title
st.title("Welcome to My Simple Streamlit App!")

# Add some text
st.write("This is a basic example of a Streamlit application.")

# Create a sidebar
with st.sidebar:
    st.header("Settings")
    user_name = st.text_input("Enter your name")
    age = st.slider("Select your age", 0, 100, 25)

# Display user input
if user_name:
    st.success(f"Hello, {user_name}! You are {age} years old.")

# Add a button
if st.button("Click me for a surprise!"):
    st.balloons()

# Add a selectbox
option = st.selectbox(
    "What's your favorite color?",
    ("Red", "Green", "Blue", "Yellow")
)
st.write(f"You selected: {option}")

# Add a file uploader
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)

# Add a checkbox
if st.checkbox("Show/hide secret message"):
    st.write("The secret of life is 42!")

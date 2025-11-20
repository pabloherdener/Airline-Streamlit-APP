# Import all libraries for the app
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Set page title and icon
st.set_page_config(page_title="Airline Satisfaction Dataset Explorer", page_icon="✈️")

# Sidebar navigation (old)
# page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Extras"])

# Sidebar navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Data Overview", "Exploratory Data Analysis", "Model Training and Evaluation", "Make Predictions!"])

# Load dataset
df = pd.read_csv('data/cleaned_airline_passenger_satisfaction.csv')

# Home Page
if page == "Home":
    st.title("📊 Airline Satisfaction Dataset Explorer")
    st.subheader("Welcome to our Ariline Satisfaction dataset explorer app!")
    st.write("""
        This app provides an interactive platform to explore the referred dataset. 
        You can visualize the distribution of data, explore relationships between features, and even make predictions on new data!\n
        Use the sidebar to navigate through the sections.
    """)
    st.image('https://wallup.net/wp-content/uploads/2019/09/344761-boeing-747-airliner-aircraft-plane-airplane-boeing-747-transport-36-2.jpg', caption="B747 Airliner")

# Data Overview
elif page == "Data Overview":
    st.title("🔢 Data Overview")

    st.subheader("About the Data")
    st.write("""
        The Ariline Satisfaction dataset has a survey data collected from airline passengers. 
        The dataset contains various features related to passengers, 
        including their demographics, flight information, and survey responses.
    """)
    st.image('https://wallpapers.com/images/hd/airport-pictures-ccbwpi23wdveguhg.jpg', caption="Passengers at the Airport")

    # Dataset Display
    st.subheader("Quick Glance at the Data")
    if st.checkbox("Show DataFrame"):
        st.dataframe(df)
    

    # Shape of Dataset
    if st.checkbox("Show Shape of Data"):
        st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

# Exploratory Data Analysis (EDA)
elif page == "Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")

    # Map numeric / coded columns to readable labels
    satisfaction_map = {0: "Unsatisfied", 1: "Satisfied"}
    class_map = {1: "Eco", 2: "Eco Plus", 3: "Business"}
    customer_type_map = {1: "Loyal", 0: "Disloyal"}
    type_travel_map = {0: "Personal", 1: "Business"}
    gender_map = {1: "Male", 0: "Female"}
    
    # Friendly column labels
    col_labels = {
        'Age': 'Passenger Age',
        'Arrival Delay in Minutes': 'Arrival Delay (min)',
        'Departure Delay in Minutes': 'Departure Delay (min)',
        'Inflight wifi service': 'Inflight WiFi Service (0-5)',
        'Food and drink': 'Food & Drink Satisfaction (0-5)',
        'Online boarding': 'Online Boarding Satisfaction (0-5)',
        'Seat comfort': 'Seat Comfort (0-5)',
        'Inflight entertainment': 'Inflight Entertainment (0-5)',
        'On-board service': 'On-board Service (0-5)',
        'Leg room service': 'Leg Room Service (0-5)',
        'Baggage handling': 'Baggage Handling (0-5)',
        'Checkin service': 'Check-in Service (0-5)',
        'Inflight service': 'Inflight Service (0-5)',
        'Cleanliness': 'Cleanliness (0-5)',
        'Class': 'Ticket Class',
        'Customer Type': 'Customer Type',
        'Gender': 'Gender',
        'satisfaction': 'Satisfaction'
    }

    # Creating a Dataframe for user friendly EDA visualizations 
    df_plot = df.copy()
    df_plot['satisfaction'] = df_plot['satisfaction'].map(satisfaction_map)
    df_plot['Class'] = df_plot['Class'].map(class_map)
    df_plot['Customer Type'] = df_plot['Customer Type'].map(customer_type_map)
    df_plot['Gender'] = df_plot['Gender'].map(gender_map)

    st.subheader("Select the type of visualization you'd like to explore:")
    eda_type = st.multiselect("Visualization Options", ['Histograms', 'Box Plots', 'Scatterplots', 'Count Plots'])

    obj_cols = df_plot.select_dtypes(include='object').columns.tolist()
    num_cols = df_plot.select_dtypes(include='number').columns.tolist()

    # ---- Histograms ----
    if 'Histograms' in eda_type:
        st.subheader("Histograms - Visualizing Numerical Distributions")
        h_selected_col = st.selectbox(
            "Select a numerical column for the histogram:", 
            num_cols, 
            format_func=lambda x: col_labels.get(x, x),
            key='hist_col'
        )
        if h_selected_col:
            chart_title = f"Distribution of {col_labels.get(h_selected_col, h_selected_col)}"
            if st.checkbox("Show by Satisfaction", key='hist_satisfaction'):
                st.plotly_chart(px.histogram(df_plot, x=h_selected_col, color='satisfaction', title=chart_title, barmode='overlay'))
            else:
                st.plotly_chart(px.histogram(df_plot, x=h_selected_col, title=chart_title))

    # ---- Box Plots ----
    if 'Box Plots' in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Distributions")
        b_selected_col = st.selectbox(
            "Select a numerical column for the box plot:", 
            num_cols, 
            format_func=lambda x: col_labels.get(x, x),
            key='box_col'
        )
        if b_selected_col:
            chart_title = f"Distribution of {col_labels.get(b_selected_col, b_selected_col)} by Satisfaction"
            st.plotly_chart(px.box(df_plot, x='satisfaction', y=b_selected_col, title=chart_title, color='satisfaction'))

    # ---- Scatterplots ----
    if 'Scatterplots' in eda_type:
        st.subheader("Scatterplots - Visualizing Relationships")
        selected_col_x = st.selectbox(
            "Select x-axis variable:", 
            num_cols, 
            format_func=lambda x: col_labels.get(x, x),
            key='scatter_x'
        )
        selected_col_y = st.selectbox(
            "Select y-axis variable:", 
            num_cols, 
            format_func=lambda x: col_labels.get(x, x),
            key='scatter_y'
        )
        if selected_col_x and selected_col_y:
            chart_title = f"{col_labels.get(selected_col_x, selected_col_x)} vs. {col_labels.get(selected_col_y, selected_col_y)}"
            st.plotly_chart(px.scatter(df_plot, x=selected_col_x, y=selected_col_y, color='satisfaction', title=chart_title))

    # ---- Count Plots ----
    if 'Count Plots' in eda_type:
        st.subheader("Count Plots - Visualizing Categorical Distributions")
        selected_col = st.selectbox(
            "Select a categorical variable:", 
            obj_cols, 
            format_func=lambda x: col_labels.get(x, x),
            key='count_col'
        )
        if selected_col:
            chart_title = f'Distribution of {col_labels.get(selected_col, selected_col)}'
            st.plotly_chart(px.histogram(df_plot, x=selected_col, color='satisfaction', title=chart_title))

# Model Training and Evaluation Page
elif page == "Model Training and Evaluation":
    st.title("🛠️ Model Training and Evaluation")

    # Sidebar for model selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the data
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the selected model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the number of neighbors (k)", min_value=1, max_value=20, value=3)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the model on the scaled data
    model.fit(X_train_scaled, y_train)

    # Display training and test accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, cmap='Blues')
    st.pyplot(fig)

# Make Predictions Page
elif page == "Make Predictions!":
    st.title("✈️ Make Predictions")
    st.subheader("Adjust the values below to make predictions on the Airline Satisfaction dataset:")

    # User inputs for prediction
    gender = st.selectbox("Gender", ["Female", "Male"])
    gender_map = {"Female": 0, "Male": 1}
    gender_val = gender_map[gender]

    customer_type = st.selectbox("Customer Type", ["Disloyal", "Loyal"])
    customer_type_map = {"Disloyal": 0, "Loyal": 1}
    customer_type_val = customer_type_map[customer_type]

    age = st.slider("Age", 0, 120, 30)
    
    type_travel = st.selectbox("Type of Travel", ["Personal", "Business"])
    type_travel_map = {"Personal": 0, "Business": 1}
    type_travel_val = type_travel_map[type_travel]
    
    class_type = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    class_map = {"Eco": 1, "Eco Plus": 2, "Business": 3}
    class_val = class_map[class_type]

    flight_distance = st.slider("Flight Distance", 30, 5000, 1500)
    inflight_wifi = st.slider("Inflight WiFi Service Satisfaction Level (1-5)", 1, 5, 3)
    dep_arr_delay = st.slider("Departure/Arrival Delay Time Convenient Satisfaction Level (1-5)", 1, 5, 3)
    ease_online_booking = st.slider("Online Booking Satisfaction Level (1-5)", 1, 5, 3)
    gate_location = st.slider("Gate Location Satisfaction Level (1-5)", 1, 5, 3)
    food_and_drink = st.slider("Food and Drink Satisfaction Level (1-5)", 1, 5, 3)
    online_boarding = st.slider("Online Boarding Satisfaction Level (1-5)", 1, 5, 3)
    seat_comfort = st.slider("Seat Comfort Satisfaction Level (1-5)", 1, 5, 3)
    inflight_entertainment = st.slider("Inflight Entertainment Satisfaction Level (1-5)", 1, 5, 3)
    onboard_service = st.slider("On-board Service Satisfaction Level (1-5)", 1, 5, 3)
    leg_room_service = st.slider("Leg Room Service Satisfaction Level (1-5)", 1, 5, 3)
    baggage_handling = st.slider("Baggage Handling Satisfaction Level (1-5)", 1, 5, 3)
    checkin_service = st.slider("Check-in Service Satisfaction Level (1-5)", 1, 5, 3)
    inflight_service = st.slider("Inflight Service Satisfaction Level (0-5)", 1, 5, 3)
    cleanliness = st.slider("Cleanliness Satisfaction Level (1-5)", 1, 5, 3)

    departure_delay = st.slider(
        "Departure Delay in Minutes (0-100)",
        min_value=0,
        max_value=100,
        value=0
    )

    arrival_delay = st.slider(
        "Arrival Delay in Minutes (0-100)",
        min_value=0,
        max_value=100,
        value=0
    )

    # User input dataframe
    user_input = pd.DataFrame({
        'Gender': [gender_val],
        'Customer Type': [customer_type_val],
        'Age': [age],
        'Type of Travel': [type_travel_val],
        'Class': [class_val],
        'Flight Distance': [flight_distance],
        'Inflight wifi service': [inflight_wifi],
        'Departure/Arrival time convenient': [dep_arr_delay],
        'Ease of Online booking': [ease_online_booking],
        'Gate location': [gate_location],
        'Food and drink': [food_and_drink],
        'Online boarding': [online_boarding],
        'Seat comfort': [seat_comfort],
        'Inflight entertainment': [inflight_entertainment],
        'On-board service': [onboard_service],
        'Leg room service': [leg_room_service],
        'Baggage handling': [baggage_handling],
        'Checkin service': [checkin_service],
        'Inflight service': [inflight_service],
        'Cleanliness': [cleanliness],
        'Departure Delay in Minutes': [departure_delay],
        'Arrival Delay in Minutes': [arrival_delay]
    })

    st.write("### Your Input Values")
    st.dataframe(user_input)

    # Use KNN (k=17) as the model for predictions
    model = KNeighborsClassifier(n_neighbors=17)
    X = df.drop(columns = 'satisfaction')
    y = df['satisfaction']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Train the model on the scaled data
    model.fit(X_scaled, y)

    # Make predictions
    prediction_num = model.predict(user_input_scaled)[0]

    # Map numeric output to human-friendly label
    prediction_label = "Satisfied" if prediction_num == 1 else "Neutral / Unsatisfied"

    # Display the result
    st.write("### Prediction")
    st.write(f"The model predicts that the passenger is **{prediction_label}**")

    st.balloons()
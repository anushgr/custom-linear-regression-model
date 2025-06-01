import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def compute_cost(self, X, y, weights, bias):
        m = len(y)
        predictions = X.dot(weights) + bias
        errors = predictions - y
        cost = (1/(2*m)) * np.sum(errors**2)
        return cost

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.n_iterations):
            predictions = X.dot(self.weights) + self.bias
            dw = (1/m) * X.T.dot(predictions - y)
            db = (1/m) * np.sum(predictions - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            cost = self.compute_cost(X, y, self.weights, self.bias)
            self.cost_history.append(cost)

    def predict(self, X):
        return X.dot(self.weights) + self.bias

# Initialize session state to store points
if 'points' not in st.session_state:
    st.session_state.points = []

# Streamlit app layout
st.title("Interactive Linear Regression App")
st.markdown("Add X and Y values to create a scatter plot and fit a linear regression line. The plot updates automatically when you add or delete points.")

# Input section
st.subheader("Add a New Point")
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    x_input = st.number_input("X value", value=0.0, format="%.2f")
with col2:
    y_input = st.number_input("Y value", value=0.0, format="%.2f")
with col3:
    if st.button("Add Point"):
        st.session_state.points.append((x_input, y_input))
        st.success("Point added!")

# Display points in a table
if st.session_state.points:
    st.subheader("Entered Points")
    df = pd.DataFrame(st.session_state.points, columns=["X", "Y"])
    df['Delete'] = [f"Delete {i}" for i in range(len(st.session_state.points))]
    
    # Custom CSS for table styling
    st.markdown("""
        <style>
        .dataframe th, .dataframe td {
            text-align: center;
            padding: 10px;
        }
        .dataframe {
            border-collapse: collapse;
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.table(df)

    # Handle point deletion
    delete_col, _ = st.columns([2, 3])
    with delete_col:
        delete_index = st.selectbox("Select point to delete (optional)", 
                                   options=["None"] + [f"Point {i}: ({x:.2f}, {y:.2f})" for i, (x, y) in enumerate(st.session_state.points)],
                                   index=0)
        if delete_index != "None" and st.button("Delete Selected Point"):
            index = int(delete_index.split(":")[0].split()[1])
            st.session_state.points.pop(index)
            st.success("Point deleted!")

# Plotting with Plotly
if len(st.session_state.points) >= 2:
    # Prepare data for regression
    X = np.array([[x] for x, y in st.session_state.points])
    y = np.array([y for x, y in st.session_state.points])

    # Train the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)

    # Generate points for regression line
    x_range = np.linspace(min(X[:, 0]), max(X[:, 0]), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=y,
        mode='markers',
        name='Data Points',
        marker=dict(size=10, color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=x_range[:, 0], y=y_pred,
        mode='lines',
        name='Regression Line',
        line=dict(color='red')
    ))

    # Update layout for better visuals
    fig.update_layout(
        title="Linear Regression Fit",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        template="plotly_white",
        width=600,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    st.plotly_chart(fig)

    slope = model.weights[0]
    intercept = model.bias
    st.markdown(f"### Regression Line Equation: y = {slope:.2f}x + {intercept:.2f}")
elif st.session_state.points:
    st.info("Please add at least two points to display the regression plot.")
else:
    st.info("No points added yet. Use the input fields above to add points.")
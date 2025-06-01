# Linear Regression Streamlit App

This repository contains a Streamlit application (`app.py`) for interactive linear regression modeling and a test script (`test.py`) for training a linear regression model on synthetic data. The app allows users to input X and Y values, visualize a scatter plot with a fitted regression line using Plotly, and display the regression line equation. The test script demonstrates the model with synthetic data and Matplotlib.

## Features
- **Interactive App (`app.py`)**:
  - Input X and Y values to add points.
  - Dynamically updates a Plotly scatter plot and regression line (appears after 2+ points).
  - Displays the regression line equation (e.g., \( y = mx + b \)).
  - Allows deletion of points with automatic plot and equation updates.
  - User-friendly interface with a styled table for entered points.
- **Test Script (`test.py`)**:
  - Generates synthetic data for linear regression.
  - Trains the same `LinearRegression` class used in the app.
  - Visualizes results with Matplotlib.

## Prerequisites
- Python 3.8+
- Required packages:
  ```bash
  pip install streamlit numpy pandas plotly matplotlib
  ```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Streamlit App
1. Ensure dependencies are installed.
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser to `http://localhost:8501`.
4. Interact with the app:
   - Enter X and Y values and click "Add Point".
   - View the points in the table and the Plotly graph (appears after 2+ points).
   - See the regression line equation below the graph.
   - Delete points using the dropdown and "Delete Selected Point" button.

### Running the Test Script
1. Run the test script:
   ```bash
   python test.py
   ```
2. A Matplotlib plot will display synthetic data points and the fitted regression line.

## Files
- `app.py`: Streamlit app for interactive linear regression.
- `test.py`: Script to train and visualize linear regression on synthetic data.
- `README.md`: This file.
- `requirements.txt`: List of required Python packages.

## Example
### Streamlit App
- Add points like (1, 2), (3, 5), (2, 3.5).
- The plot shows blue scatter points and a red regression line.
- The equation (e.g., \( y = 1.67x + 0.33 \)) appears below the plot.
- Delete a point to update the plot and equation dynamically.

### Test Script
- Generates 100 synthetic points with \( y \approx 4 + 3x + \text{noise} \).
- Trains the model and plots the data with the fitted line.

## Notes
- The `LinearRegression` class uses gradient descent with a learning rate of 0.01 and 1000 iterations.
- The app requires at least two points to display the plot and equation.
- Plotly provides interactive features like zooming and hovering.

## License
MIT License

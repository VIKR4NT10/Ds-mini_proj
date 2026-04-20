
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

df = pd.read_csv("student_performance_cleaned.csv")

X = df[
    [
        "weekly_self_study_hours",
        "attendance_percentage",
        "class_participation"
    ]
]

# For score prediction
y_score = df["total_score"]

# For grade prediction
y_grade = df["grade_encoded"]


# Linear Regression for total score
score_model = LinearRegression()
score_model.fit(X, y_score)

# Logistic Regression for grade
grade_model = LogisticRegression()
grade_model.fit(X, y_grade)


grade_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "F"
}


# Streamlit UI

st.title("Student Performance Predictor")

st.write("Enter student details to predict score and grade")


# User Input
study_hours = st.slider(
    "Weekly Self Study Hours",
    0, 40, 10
)

attendance = st.slider(
    "Attendance Percentage",
    50, 100, 80
)

participation = st.slider(
    "Class Participation",
    0, 10, 5
)


if st.button("Predict Performance"):

    input_data = pd.DataFrame(
        [[study_hours, attendance, participation]],
        columns=[
            "weekly_self_study_hours",
            "attendance_percentage",
            "class_participation"
        ]
    )

    # Predict total score
    predicted_score = score_model.predict(input_data)[0]

    # Predict grade
    predicted_grade_num = grade_model.predict(input_data)[0]
    predicted_grade = grade_map[predicted_grade_num]

    # Show Result
    st.subheader("Prediction Result")

    st.write(f"Predicted Total Score: {predicted_score:.2f}")
    st.write(f"Predicted Grade: {predicted_grade}")


# ------------------------------------------------------------
# STEP 8: Conclusion
# ------------------------------------------------------------

st.write("""
This simple web app allows real-time prediction
of student performance using Data Science models.
""")
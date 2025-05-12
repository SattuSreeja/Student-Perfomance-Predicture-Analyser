
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Student Data tisko file nundi
data_file = "students.csv"
students_df = pd.read_csv(data_file)

# Recommendation of subject , aah subjects ki linklu 
recommendations = {
    "c language": {"books": ["problem solving", "logic building"], "youtube": ["https://youtu.be/irqbmMNs2Bo?si=brr6DAbZ1M_FINDK ", "https://youtu.be/aZb0iu4uGwA?si=veXtSVlMHaBVizn6"]},
    "python": {"books": ["master python", "hands on python"], "youtube": [" https://youtu.be/UrsmFxEIp5k?si=QNl30SQpXLm7RWfC", " https://youtu.be/ERCMXc8x7mc?si=ovOukiSLIhvwLQCK"]},
    "signals and systems": {"books": ["signals and systems", "signal transforms"], "youtube": ["https://youtu.be/up55tuwestg?si=Ey9_nsJKXeo_40zF", "https://youtu.be/s8rsR_TStaA?si=_u7t9iv9_sE6WuUt"]},
    "java": {"books": ["master java", "threads in java"], "youtube": ["https://youtu.be/BGTx91t8q50?si=OQT_1v6LuGcMiq5m", "https://youtu.be/yRpLlJmRo2w?si=wrZ-QsGfA6mbymhu"]},
}

# Predict Next Exam Performance
def predict_performance(marks):
    X = np.array(range(len(marks))).reshape(-1, 1)
    y = np.array(marks).reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)

    next_exam_score = model.predict([[len(marks)]])[0][0]
    return round(next_exam_score, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    student_name = request.form.get("student_name")

    if student_name not in students_df["Name"].values:
        return "Error: Student not found in database. Please check the name and try again."

    student_data = students_df[students_df["Name"] == student_name].iloc[:, 1:].to_dict(orient='list')

    predictions = {subject: predict_performance(marks) for subject, marks in student_data.items()}

    # Plot Performance Graph
    plt.figure(figsize=(8, 4)) 
    for subject, marks in student_data.items():
        plt.plot(range(len(marks)), marks, marker='o', label=subject)
    plt.xlabel("Previous Exams")
    plt.ylabel("Marks")
    plt.legend()
    plt.title(f"Performance of {student_name}")

    graph_path = "static/performance.png"
    if not os.path.exists("static"):
        os.makedirs("static")
    plt.savefig(graph_path)
    plt.close()

    return render_template('result.html', student_name=student_name, predictions=predictions, recommendations=recommendations, graph_path=graph_path)

if __name__ == "__main__":
    app.run(debug=True)
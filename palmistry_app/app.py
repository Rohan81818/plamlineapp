from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import joblib
from palm_detect import detect_palm_lines

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load trained classifier and label encoder
clf = joblib.load('palm_line_classifier.joblib')
le = joblib.load('label_encoder.joblib')

def extract_line_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    features = []

    for cnt in contours:
        length = cv2.arcLength(cnt, False)
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            orientation = ellipse[2]
        else:
            orientation = None
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        else:
            cX, cY = 0, 0

        features.append({
            'length': length,
            'orientation': orientation,
            'centroid_x': cX,
            'centroid_y': cY,
        })
    return features

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Detect palm lines and save highlighted image
            output_filename = 'lines_detected.jpg'
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            detect_palm_lines(filepath, output_path)

            # Extract features from uploaded image
            features = extract_line_features(filepath)

            # Prepare features list for prediction
            feature_list = []
            for f in features:
                orientation = f['orientation'] if f['orientation'] is not None else 0
                feature_list.append([f['length'], orientation, f['centroid_x'], f['centroid_y']])

            # Predict line types
            preds = clf.predict(feature_list)
            class_names = le.inverse_transform(preds)

            # Descriptions for lines
            line_descriptions = {
                "life_line": "Indicates general health and vitality.",
                "heart_line": "Relates to emotional life and relationships.",
                "head_line": "Represents intellect and decision making.",
                "fate_line": "Associated with career and destiny."
            }

            # Build results with predictions and descriptions
            results = []
            for i, f in enumerate(features):
                line_type = class_names[i]
                results.append({
                    "index": i+1,
                    "length": f['length'],
                    "orientation": f['orientation'],
                    "centroid_x": f['centroid_x'],
                    "centroid_y": f['centroid_y'],
                    "line_type": line_type,
                    "description": line_descriptions.get(line_type, "No description available.")
                })

            return render_template('result.html', filename=output_filename, results=results)

        else:
            return "No file selected."
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

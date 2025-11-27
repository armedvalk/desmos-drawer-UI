import io
import cv2
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

HTML_PAGE = r"""
<!doctype html>
<html>
<head>
    <title>Image → Desmos Equations</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            max-width: 900px;
            margin: 2rem auto;
            padding: 0 1rem;
            line-height: 1.5;
        }
        h1 {
            text-align: center;
        }
        form {
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
        }
        .field {
            margin-bottom: 1rem;
        }
        label {
            display: block;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        input[type="file"],
        input[type="number"] {
            width: 100%;
        }
        button {
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: 1px solid #555;
            cursor: pointer;
            background: #222;
            color: #fff;
            font-weight: 600;
        }
        button:hover {
            opacity: 0.9;
        }
        textarea {
            width: 100%;
            height: 350px;
            font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
            font-size: 0.9rem;
            padding: 0.5rem;
            border-radius: 8px;
            border: 1px solid #ddd;
            box-sizing: border-box;
        }
        .error {
            color: #b00020;
            font-weight: 600;
        }
        .hint {
            font-size: 0.9rem;
            color: #555;
        }
    </style>
</head>
<body>
    <h1>Image → Desmos Equation Generator</h1>

    <form method="POST" enctype="multipart/form-data">
        <div class="field">
            <label for="image">Upload image (PNG/JPG):</label>
            <input type="file" name="image" id="image" accept="image/png, image/jpeg" required>
        </div>

        <div class="field">
            <label for="inaccuracy">Inaccuracy value (0.001–0.05 recommended):</label>
            <input type="number" step="0.001" min="0.000" max="0.100" name="inaccuracy" id="inaccuracy"
                   value="{{ inaccuracy }}">
            <div class="hint">
                Lower = more accurate but more equations. Try 0.002–0.01 for complex images.
            </div>
        </div>

        <button type="submit">Generate Equations</button>
    </form>

    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}

    {% if equations %}
        <h2>Generated Equations</h2>
        <p class="hint">Copy these into Desmos. Each line is a separate expression.</p>
        <textarea readonly>{{ equations }}</textarea>
    {% endif %}
</body>
</html>
"""

def bezier_to_equations(start, c1, c2, end, img_height):
    """
    Convert cubic Bezier curve control points to parametric equations for Desmos.
    We flip Y (img coordinates → Cartesian coordinates).
    """
    start = (float(start[0]), float(img_height - start[1]))
    c1    = (float(c1[0]),    float(img_height - c1[1]))
    c2    = (float(c2[0]),    float(img_height - c2[1]))
    end   = (float(end[0]),   float(img_height - end[1]))

    equation_x = (
        f"((1-t)^3*{start[0]} + "
        f"3*(1-t)^2*t*{c1[0]} + "
        f"3*(1-t)*t^2*{c2[0]} + "
        f"t^3*{end[0]})"
    )
    equation_y = (
        f"(1-t)^3*{start[1]} + "
        f"3*(1-t)^2*t*{c1[1]} + "
        f"3*(1-t)*t^2*{c2[1]} + "
        f"t^3*{end[1]}"
    )
    return equation_x, equation_y

def image_to_equations(image_bgr, inaccuracy_value=0.002):
    """
    Core logic: takes a BGR OpenCV image and returns a string
    of Desmos equations, one per line.
    """
    if image_bgr is None:
        raise ValueError("Could not decode image.")

    img_height = image_bgr.shape[0]

    # Convert to grayscale for Canny
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # For complex images, a little blur can help denoise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    lines = []

    for contour in contours:
        if len(contour) < 2:
            continue

        epsilon = float(inaccuracy_value) * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 2:
            continue

        for i in range(len(approx)):
            start = approx[i][0]
            end   = approx[(i + 1) % len(approx)][0]

            x1, y1 = int(start[0]), int(start[1])
            x2, y2 = int(end[0]),   int(end[1])

            dx = x2 - x1
            dy = y2 - y1

            # Vertical line
            if dx == 0:
                y_top    = img_height - max(y1, y2)
                y_bottom = img_height - min(y1, y2)
                eq = f"x = {x1} \\left\\{{{y_top} <= y <= {y_bottom}\\right\\}}"
                lines.append(eq)

            # Horizontal line
            elif dy == 0:
                y_cart = img_height - y1
                x_left = min(x1, x2)
                x_right = max(x1, x2)
                eq = f"y = {y_cart} \\left\\{{{x_left} <= x <= {x_right}\\right\\}}"
                lines.append(eq)

            else:
                # Simple synthetic control points (still approximate, but smooth)
                c1 = ((2*x1 + x2) / 3.0, (2*y1 + y2) / 3.0)
                c2 = ((x1 + 2*x2) / 3.0, (y1 + 2*y2) / 3.0)

                eq_x, eq_y = bezier_to_equations((x1, y1), c1, c2, (x2, y2), img_height)
                # Desmos parametric form: (x(t), y(t))
                lines.append(f"({eq_x}, {eq_y})")

    if not lines:
        return "# No contours/equations were detected. Try a clearer or higher-contrast image."

    return "\n".join(lines)

@app.route("/", methods=["GET", "POST"])
def index():
    equations = ""
    error = ""
    inaccuracy_default = 0.002

    if request.method == "POST":
        # Get inaccuracy value
        inaccuracy_str = request.form.get("inaccuracy", str(inaccuracy_default))
        try:
            inaccuracy_value = float(inaccuracy_str)
            if inaccuracy_value <= 0:
                raise ValueError()
        except ValueError:
            error = "Invalid inaccuracy value. Please enter a positive number."
            return render_template_string(
                HTML_PAGE,
                equations="",
                error=error,
                inaccuracy=inaccuracy_str
            )

        # Get image file
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Please upload an image file."
            return render_template_string(
                HTML_PAGE,
                equations="",
                error=error,
                inaccuracy=inaccuracy_value
            )

        try:
            # Read file into OpenCV image
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            equations = image_to_equations(image_bgr, inaccuracy_value)
        except Exception as e:
            error = f"Error processing image: {e}"

        return render_template_string(
            HTML_PAGE,
            equations=equations,
            error=error,
            inaccuracy=inaccuracy_value
        )

    # GET
    return render_template_string(
        HTML_PAGE,
        equations="",
        error="",
        inaccuracy=inaccuracy_default
    )

if __name__ == "__main__":
    # Run the app on http://127.0.0.1:5000
    app.run(debug=True)

Project Installation Instructions
**1. Clone the project from the github repo**
https://github.com/unnit/traffic_app

**2. Make sure python is installed in your machine. You can check using**
python --version
If Python is not installed, you can download it from the official Python website(https://www.python.org/downloads/)

**3. pip is the package installer for Python, and it should be installed by default with Python. You can check if it’s installed by running:**
pip --version
If pip is not installed, you can install it using:
python -m ensurepip --upgrade

**4. Create a virtual environment (Optional)**
pip install virtualenv
python -m venv {env_name}
Activate the environment

**5. Install dependencies**
pip install -r requirements.txt

**6. Run the server**
python app.py for development server
gunicorn app:app for production server

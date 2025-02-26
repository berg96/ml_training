FROM python:3.9

RUN python -m pip install flask gunicorn scikit-learn

WORKDIR /app

ADD iris.py iris.py

EXPOSE 5003

CMD [ "gunicorn", "--bind", "0.0.0.0:5003", "iris:app" ]
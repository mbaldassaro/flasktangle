//Procfile

web: FLASK_APP=app.py python -m flask run --host=0.0.0.0 --port=$PORT
worker: python worker.py

//FLASK_APP should be set to whatever main file is
//port = $PORT b/c Heroku will assign a port to the app

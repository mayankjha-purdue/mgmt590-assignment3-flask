import time
from flask import Flask
from flask import jsonify
from flask import request
from transformers.pipelines import pipeline
import os
global modelList
global default_model
import sqlite3
from flask import abort

DATABASE_NAME = "prodscale.db"


def get_db():
    conn = sqlite3.connect(DATABASE_NAME)
    return conn


def create_tables():
    tables = [
        """CREATE TABLE IF NOT EXISTS prodscale(
                timestamp INTEGER PRIMARY KEY,
                model TEXT NOT NULL,
                answer TEXT NOT NULL,
                question TEXT NOT NULL,
				context TEXT NOT NULL)
            """
    ]
    db = get_db()
    cursor = db.cursor()
    for table in tables:
        cursor.execute(table)


modelList = [
    {
        'name': "distilled-bert",
        'tokenizer': "distilbert-base-uncased-distilled-squad",
        'model': "distilbert-base-uncased-distilled-squad"
    }
]
default_model = modelList[0]

# Create my flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# Define a handler for the / path, which
# returns "Hello World"
@app.route("/")
def hello_world():
    return 'Hello, World!'


def insert_db(timestamp, model, answer,question,context):
    db = get_db()
    cursor = db.cursor()
    statement = "INSERT OR IGNORE INTO prodscale(timestamp, model, answer,question,context) VALUES (?, ?, ?, ?, ?)"
    cursor.execute(statement, [timestamp, model, answer,question,context])
    db.commit()

def my_funct(text):
   abort(400, text)

@app.route("/answer", methods=['POST','GET'])
def answers():
    if request.method == 'POST':
        # Get the request body data
        model = request.args.get('model')
        data = request.json

        if (model == None):
            model = default_model['name']
            try:
                hg_comp = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad',
                               tokenizer='distilbert-base-uncased-distilled-squad')
            except:
                my_funct("Invalid Model Name")
            # Answer the answer
            answer = hg_comp({'question': data['question'], 'context': data['context']})['answer']

            timestamp = int(time.time())

            # Create the response body.
            out = {
                "timestamp": timestamp,
                "model": model,
                "answer": answer,
                "question": data['question'],
                "context": data['context']

            }

            insert_db(timestamp,model,answer,data['question'],data['context'])

            return jsonify(out)

        else:
            model_name = ""
            tokenizer = ""

            for i in range(len(modelList)):
                if modelList[i]['name'] == model:
                    model_name = modelList[i]['model']
                    tokenizer = modelList[i]['tokenizer']
                    break

            try:
                hg_comp = pipeline('question-answering', model=model_name,
                               tokenizer=tokenizer)
            except:
                my_funct("Invalid")
            # Answer the answer
            answer = hg_comp({'question': data['question'], 'context': data['context']})['answer']

            timestamp = int(time.time())

            # Create the response body.
            out = {
                "timestamp": timestamp,
                "model": model,
                "answer": answer,
                "question": data['question'],
                "context": data['context']

            }

            insert_db(timestamp, model, answer, data['question'], data['context'])

            return jsonify(out)
    else:

        if request.args.get('start') == None or request.args.get('end') == None:
            return "Query timestamps not provided", 400
        model = request.args.get('model')
        start = request.args.get('start')
        end = request.args.get('end')

        if (model == None):
            query = "SELECT timestamp, model, answer, question,context FROM prodscale WHERE timestamp BETWEEN ? AND ?"
            #query = "SELECT timestamp, model, answer, question,context FROM prodscale"
            db = get_db()
            cursor = db.cursor()
            cursor.execute(query,[start,end])
            result = cursor.fetchall()
            out=[]
            for index, tuple in enumerate(result):
                dict={
                "timestamp": tuple[0],
                "model": tuple[1],
                "answer": tuple[2],
                "question": tuple[3],
                "context": tuple[4]}
                out.append(dict)

            return jsonify(out)

        else:
            query = "SELECT timestamp, model, answer, question,context FROM prodscale WHERE timestamp BETWEEN ? AND ? AND model=?"
            db = get_db()
            cursor = db.cursor()
            cursor.execute(query,[start,end,model])
            result = cursor.fetchall()
            out=[]
            for index, tuple in enumerate(result):
                dict={
                "timestamp": tuple[0],
                "model": tuple[1],
                "answer": tuple[2],
                "question": tuple[3],
                "context": tuple[4]}
                out.append(dict)

            return jsonify(out)



@app.route("/models", methods=['GET','PUT','DELETE'])
def getModels(modelList=modelList):
    if request.method == 'PUT':
        data = request.json

        try:
            hg_comp = pipeline('question-answering', model=data['model'],
                               tokenizer=data['tokenizer'])
            modelList.append({
                'name': data['name'],
                'tokenizer': data['tokenizer'],
                'model': data['model']
            })

        except:
            my_funct("Invalid Model Name")

        seen = set()
        new_l = []
        for d in modelList:
            t = tuple(d.items())
            if t not in seen:
                seen.add(t)
                new_l.append(d)

        modelList = new_l
        return jsonify(modelList)

    elif request.method == 'DELETE':
        if request.args.get('model') == None:
            return "Model name not provided in query string", 400
        model = request.args.get('model')
        for i in range(len(modelList)):
            if modelList[i]['name'] == model:
                del modelList[i]
                break
        seen = set()
        new_l = []
        for d in modelList:
            t = tuple(d.items())
            if t not in seen:
                seen.add(t)
                new_l.append(d)

        modelList = new_l
        return jsonify(modelList)

    else:
        seen = set()
        new_l = []
        for d in modelList:
            t = tuple(d.items())
            if t not in seen:
                seen.add(t)
                new_l.append(d)

        modelList = new_l
        return jsonify(modelList)


# Run if running "python answer.py"
if __name__ == '__main__':
    # Run our Flask app and start listening for requests!
    create_tables()
    default_model = modelList[0]
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), threaded=True)

import flask
import numpy as np
import model

app = flask.Flask(__name__)
data = ''
def init():
    return 'Hi ' + data


def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('data'))
    return parameters

def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

@app.route('/start', methods=['GET'])
def processData():
    char_name = flask.request.args.get('name')
    parameters = getParameters()
    inputFeatures = np.asarray(parameters).reshape(1, -1)
    processed_data = inputFeatures[0][0]
   #processed_data = data
    reply = model.response(processed_data)
    return sendResponse({char_name: reply})


if __name__ == '__main__':
    print(("* Loading data and Flask starting server...\n"
            "please wait untill server has fully started"))
    data = init()
    app.run(threaded=True)
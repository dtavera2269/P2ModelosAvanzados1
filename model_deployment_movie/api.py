#!/usr/bin/python
# coding=utf-8
from flask import Flask
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
from p2model_deployment import predict_price

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Prediction API G8',
    description='Carolina Padilla Hernández 201111402\nWilson Felipe González Cantor 200924943\nJonny Eduardo Coronel Villamil 201411692\nDavid Tavera Sánchez 201016123')

ns = api.namespace('valores', 
     description='Movie Prediction genre')
   
parser = api.parser()

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='plot movie to be analyzed', 
    location='args')


resource_fields = api.model('Resource', {
    'result': fields.Integer,
})

@ns.route('/')
class PhishingApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        plot = args['plot']
        print(plot)
        
        return {
         "result": valores(plot)
        }, 200
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

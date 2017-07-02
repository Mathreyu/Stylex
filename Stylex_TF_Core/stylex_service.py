import logging
import werkzeug
from flask import Flask
from flask_restful import Resource, reqparse, Api

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
api = Api(app)


class StylexService(Resource):
    @staticmethod
    def post():
        parse = reqparse.RequestParser()
        parse.add_argument('style')
        parse.add_argument('image', type=werkzeug.FileStorage, location='files')

        args = parse.parse_args()
        logging.debug(args)

        image = args['image']
        image.save('myfile.jpg')


api.add_resource(StylexService, '/upload')

if __name__ == '__main__':
    app.run(debug=True)

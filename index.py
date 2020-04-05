import json
from modules.MaskProcessor import MaskProcessor
from modules.MaskAnalyzer import MaskAnalyzer

with open('config.json', 'r') as configFile:
    config=configFile.read()

config = json.loads(config)

maskPath = config['inputDir'] + '/' + config['inputName']

maskProcessor = MaskProcessor(maskPath)
# MaskAnalyzer(maskProcessor)
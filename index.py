import json
import sys
from modules.AppManager import AppManager

with open('config.json', 'r') as configFile:
    config=configFile.read()

config = json.loads(config)

filename = sys.argv[1]
imagePath = config['input_dir'] + '/' + filename
maskPath = config['output_dir'] + '/' + config['filename']

appManager = AppManager(imagePath, maskPath)
print(appManager.process_image())
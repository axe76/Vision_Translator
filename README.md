# Vision_Translator
A project that takes image of German text as input and outputs the English translation. The project uses EAST and Tesseract for text detection and OCR followed by an LSTM model which translates the detected text to english. The LSTM translator model and the pickled data used is available in my repo: https://github.com/axe76/Neural-translator.-. The model has been trained on a small dataset with a mediocre laptop and hence will not accurately translate complex sentences. 
In case text is not being fully detected, try changing either the psm value in the config in ocr.py or the padding value in ocr.py

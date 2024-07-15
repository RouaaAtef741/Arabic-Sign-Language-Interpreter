# Arabic-Sign-Language-Interpreter
Using python collect images, process them and train a random-forest-classifier AI model to translate signs into the Arabic language

this project has 4 python files
- Collect-imgs.py: collects the images that you will use to train the model.
- Create-dataset.py: Processes the images you collected to fit into the randomForestClassifier model.
- Train-classifier.py: Trains the model and creates two files "model.p" and "model.pkl" use whichever one you like model.p is used in the python inference classifier, while model.pkl is used in the app.py that allows you to integrate the AI model into other programs using flask api and flask server.
- Inference-classifier.py: is a python program that showcases the model's prediction, it's not a realistic implementation of the model. Use whatever you want for the GUI.
- app.py: is the implementation flask api/server. I used this to integrate the model into a flutter app. (not included in the repository).

The Noto_Naskh_Arabic folder includes afont that supports writting in arabic, this is used in the inference classifier to display the prediction in arabic.

I will not be including the dataset that the model was trained on or the model file itself. The model needs to be trained on the python version of YOUR device or it won't work correctly this is a version compatability from the skcit-learn library.

if you have any further questions about this let me know! Good Luck and Happy coding.

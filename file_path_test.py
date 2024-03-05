import os

MODEL_NAME = "DQN_Tianshou_Vector.pth"
model_path = './machine_leaning/' + MODEL_NAME

if os.path.exists(model_path):
    print('The file exists')
else:
    print('The file does not exist')

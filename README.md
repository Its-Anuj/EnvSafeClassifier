# EnvSafeClassifier
A AI program which classifies the given image into bio or non-bio degredable


## 1️⃣ Create a virtual environment

```bash
python -m venv venv
```
 # On Windows
 
```bash
venv\Scripts\activate
```
  # On linux
 
```bash
source venv/bin/activate
```

 # Install Dependencies
 
```bash
pip install -r requirements.txt
 ```

# Now run the program
```bash
python pred.py Model.pth Image_path.jpg
```

## Example
<img width="200" height="200" alt="image" src="https://github.com/user-attachments/assets/eed4b976-77ce-4e36-898d-899ce3922324" />


# Now run the program
```bash
python pred.py Model.pth Test.png
```

# The Output
<img width="792" height="680" alt="image" src="https://github.com/user-attachments/assets/d7a81c4a-2afe-4730-968f-cfbe62338360" />

```bash
2.5.1+cu121 cuda
Model path: Model.pth
Image path: archive (3)\TEST\B\TEST_BIODEG_HFL_0.jpg
EnvironmentClassifier(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (Classifier): ModuleList(
    (0): Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential(
      (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ReLU()
      (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Sequential(
      (0): Flatten(start_dim=1, end_dim=-1)
      (1): Linear(in_features=8192, out_features=512, bias=True)
      (2): ReLU()
      (3): Linear(in_features=512, out_features=1, bias=True)
    )
  )
)
True
NVIDIA GeForce MX33
```

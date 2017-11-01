# DLR implementation
## 1. Requirements
```
matplotlib==2.1.0
librosa==0.5.1
numpy==1.13.3
tensorflow==1.3.0 or tensorflow-gpu==1.3.0
```

## 1.1 Possible Error
```
No backenderror on Ubuntu
sudo apt-get install libav-tools
```

## 2. Content
### 2.1 python files
```
dlr.py # transform dlr funcs
models.py # models to build dlr
ops.py # tensorflow operations required on models.py
test.py # how to use
```
### 2.2 save
trained model to be restored

### 2.3 test_case
```
test.mp3
test.png
```

##3. Example
![mel spectogram](./asset/mel.png)
![tempogram](./asset/tempo.png)
![dlr](./asset/DLR.png)




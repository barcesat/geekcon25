@echo off
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" opencv-cuda
python c:\Projects\geekcon25\camera_greeter\test_cuda.py
pause
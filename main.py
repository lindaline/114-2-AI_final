import os
import tkinter as tk
import train
import test
import enroll_face

folder_path = 'database/train'
cls_list = []

root = tk.Tk()
root.title('臉部辨識解鎖系統')
root.geometry('300x325')        # 高度加一點，留空間放結果文字

name = tk.StringVar()        # 用來綁定輸入框的文字，方便後續使用
name.set('')

result_text = tk.StringVar()    # 用來顯示辨識結果，綁定下面的 Label
result_text.set('')

def open_enroll_face(name):
    enroll_face.main(name)
    for folder in os.listdir(folder_path):
        cls_list.append(folder)

def open_train():
    train.train(num_class=len(cls_list))

def open_test(name):
    done = test.detection(name)
    result_text.set(done)       # 更新 Label 的文字顯示辨識結果

label = tk.Label(root, text='請輸入使用者名稱(英文)', font=('清松手寫體1',16,'bold')).pack()

tk.Entry(root, font=('清松手寫體1',16,'bold'), textvariable=name).pack()

shooting = tk.Button(root,
                      text='開始拍攝臉部資料',
                      font=('清松手寫體1',16,'bold'),
                      command=lambda: open_enroll_face(name.get())
                    ).pack(pady=10)
training = tk.Button(root,
                      text='開始訓練辨識模型',
                      font=('清松手寫體1',16,'bold'),
                      command=lambda: open_train()
                    ).pack(pady=10)
testing = tk.Button(root,
                      text='開始辨識',
                      font=('清松手寫體1',16,'bold'),
                      command=lambda: open_test(name.get())
                    ).pack(pady=10)

tk.Label(root, textvariable=result_text, font=('清松手寫體1',16,'bold'), fg='green').pack(pady=10)

root.mainloop()
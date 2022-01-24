import tkinter as tk
from tkinter import filedialog, scrolledtext
from googletrans import Translator
from langdetect import detect
import pickle
from keras.preprocessing import text, sequence
from keras.models import load_model

with open('./src/tokens.pickle', 'rb') as handle:
    tokenize = pickle.load( handle)


def detect_lang(input_text):
    lang = detect(input_text)
    return lang

def load_models(file_path):
    model = load_model(file_path)
    return model

def tokenizer(x, vocabulary_size=5000, char_level=True):
    tok = []
    x_ = tokenize.texts_to_sequences(x)
    for i in x_:
        tok = tok + i

    return [tok]


def pad(x, max_document_length=300):
    x_ = sequence.pad_sequences(x, maxlen=max_document_length, padding='post', truncating='post')
    return x_
detector = Translator()

def eval_txt():

    input_txt = txt_box.get(1.0,tk.END)

    trans = detector.translate(input_txt, dest='fa')
    translate_exp_lbl.configure(text= trans.text,  foreground="green")


    lang = detector.detect(input_txt)
    language_exp_lbl.configure(text= lang.lang,  foreground="green")


    try:
        vector = pad(tokenizer(input_txt))
        model = load_models(model_path)

        sent_pred = model.predict(vector)
        print(sent_pred)
        out_pred = list(sent_pred[0]).index(max(sent_pred[0]))
        sent_exp_lbl.configure(text= out_pred,  foreground="green")
    except:
        sent_exp_lbl.configure(text="No Model Added", foreground="red")

window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry('800x600')

translate_exp_lbl = tk.Label(window, text="", font=("Arial Bold", 16), foreground="green")
translate_exp_lbl.pack()
translate_exp_lbl.place(x=200, y=400)

language_exp_lbl = tk.Label(window, text="", font=("Arial Bold", 18), foreground="green")
language_exp_lbl.pack()
language_exp_lbl.place(x=200, y=450)

sent_exp_lbl = tk.Label(window, text="", font=("Arial Bold", 18), foreground="green")
sent_exp_lbl.pack()
sent_exp_lbl.place(x=200, y=500)


input_model_lbl = tk.Label(window, text="Select your ML model to load:", font=("Arial Bold", 16))
input_model_lbl.pack()
input_model_lbl.place(x=200, y=20)

output_model_lbl = tk.Label(window, text="No model added", font=("Arial Bold", 10),foreground="red")
# output_model_lbl.grid(column=2, row=1)
output_model_lbl.pack()
output_model_lbl.place(x=300, y=53)


def load_model_clicked():
    global model_path

    model_path = filedialog.askopenfilename()
    model_name = model_path.split("/")[-1]
    showing_name = model_name + "  successfully added."

    output_model_lbl.configure(text= showing_name,  foreground="green")
    output_model_lbl.place(x=200, y=53)

add_model_btn = tk.Button(window, text="Add", command=load_model_clicked)
add_model_btn.pack()
add_model_btn.place(x=510, y=20)


input_txt_lbl = tk.Label(window, text="input your text:", font=("Arial Bold", 15))
input_txt_lbl.pack()
input_txt_lbl.place(x=50, y=85)

txt_box = scrolledtext.ScrolledText(window,width=85,height=15)
txt_box.pack()
txt_box.place(x=50, y=120)

sentiment_lbl = tk.Label(window, text="Translate --->", font=("Arial Bold", 15))
sentiment_lbl.pack()
sentiment_lbl.place(x=50, y=400)


language_lbl = tk.Label(window, text="Language --->", font=("Arial Bold", 15))
language_lbl.pack()
language_lbl.place(x=50, y=450)

sentiment_lbl = tk.Label(window, text="Sentiment --->", font=("Arial Bold", 15))
sentiment_lbl.pack()
sentiment_lbl.place(x=50, y=500)






eval_btn = tk.Button(window, text="eval", command=eval_txt,  foreground="blue", background="yellow", width=6, height=1)
eval_btn.pack()
eval_btn.place(x=680, y=380)



window.mainloop()




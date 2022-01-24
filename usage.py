import pickle
from keras.models import load_model
from keras.preprocessing import sequence
import src.pre_pro

with open('./src/tokens.pickle', 'rb') as handle:
    tokenize = pickle.load(handle)


def tokenizer(x, vocabulary_size=5000, char_level=True):
    tok = []
    x_ = tokenize.texts_to_sequences(x)
    for i in x_:
        tok = tok + i

    return [tok]


def pad(x, max_document_length=300):
    x_ = sequence.pad_sequences(x, maxlen=max_document_length, padding='post', truncating='post')
    return x_


def eval_txt(input_txt):

    clean_txt = src.pre_pro.stop_word_remover(src.pre_pro.tokenize(input_txt), is_split=True, return_split=False)

    vector = pad(tokenizer(clean_txt))
    print(vector)
    model = load_model("./src/model.h5")

    sent_pred = model.predict(vector)
    label_list = ["Positive", "negetive"]
    final_dict = {}
    for i in range(len(label_list)):
        final_dict[label_list[i]] = str(sent_pred[0][i])

    # print("before", final_dict)
    final_dict = {k: v for k, v in sorted(final_dict.items(), key=lambda item: item[1], reverse=True)}
    print(sent_pred)
    out_pred = list(sent_pred[0]).index(max(sent_pred[0]))


    return final_dict


## usage
# input_txt = "חוסר האיפוק והסובלנות של הימין הקיצוני מזכיר דע אש במידה רבה וחבל ...."
# print(eval_txt(input_txt))
import pandas as pd
import nltk
from textblob import TextBlob
from textblob.translate import NotTranslated
import random
import operator
import numpy as np
import math
import tqdm

sr = random.SystemRandom()

# Read file
file_name = 'token_test.csv'
# Read file using pandas
df = pd.read_csv(file_name, encoding= "utf-8", quoting=1, sep="~")

language = ["es", "de", "fr", "ar", "te", "hi", "ja", "fa", "sq", "bg", "nl", "gu", "ig", "kk", "mt", "ps"]


def data_augmentation(message, language, aug_range=1):
    augmented_messages = []
    if hasattr(message, "decode"):
        message = message.decode("utf-8")

    for j in range(0, aug_range):
        new_message = ""
        text = TextBlob(message)
        try:
            text = text.translate(to=sr.choice(language))  # Converting to random langauge for meaningful variation
            text = text.translate(to="en")
        except NotTranslated:
            pass
        augmented_messages.append(str(text))

    return augmented_messages


# Dictionary for intent count
# Intent is column name
intent_count = df.Intent.value_counts().to_dict()

# Get max intent count to match other minority classes through data augmentation

max_intent_count = max(intent_count.items(), key=operator.itemgetter(1))[1]

# Loop to interate all messages


newdf = pd.DataFrame()
for intent, count in intent_count.items():
    count_diff = max_intent_count - count  # Difference to fill
    multiplication_count = math.ceil(
        (count_diff) / count)  # Multiplying a minority classes for multiplication_count times
    if (multiplication_count):
        old_message_df = pd.DataFrame()
        new_message_df = pd.DataFrame()
        for message in tqdm.tqdm(df[df["Intent"] == intent]["Message"]):
            # Extracting existing minority class batch
            dummy1 = pd.DataFrame([message], columns=['Message'])
            dummy1["Intent"] = intent
            old_message_df = old_message_df.append(dummy1)

            # Creating new augmented batch from existing minority class
            new_messages = data_augmentation(message, language, multiplication_count)
            dummy2 = pd.DataFrame(new_messages, columns=['Message'])
            dummy2["Intent"] = intent
            new_message_df = new_message_df.append(dummy2)

        # Select random data points from augmented data
        new_message_df = new_message_df.take(np.random.permutation(len(new_message_df))[:count_diff])

        # Merge existing and augmented data points
        newdf = newdf.append([old_message_df, new_message_df])
    else:
        newdf = newdf.append(df[df["Intent"] == intent])


# Print count of all new data points
newdf.Intent.value_counts()
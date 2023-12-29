import json
import numpy as np
from datasketch import MinHash, MinHashLSH
import re
import time

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

def tokenize_text(text):
    tokens = text.split()
    return tokens

def create_shingles(tokens, k=1):
    shingles = set()
    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i:i + k])
        shingles.add(shingle)
    return shingles

def jaccard_similarity_minhash(minhash1, minhash2):
    return minhash1.jaccard(minhash2)

def calculate(similarities):     
    sorted_first5_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
    sum_bot = 0
    sum_human = 0
    for user, similarity in sorted_first5_similarities:
        if user in users:
            user_category = users[user]['Categoria']
            print(f"Utente: {user}, Similarità di Jaccard: {similarity}, Categoria: {user_category}")
            if user_category == "Bot":
                sum_bot = sum_bot + similarity
            elif user_category == "Umano":
                sum_human = sum_human + similarity

    if(sum_bot > sum_human):
        print("Calcolato come BOT")

        if(query_user_info['Categoria'] == "Bot"):
            print("TRUE POSITIVE", " - BOT")

        elif(query_user_info['Categoria'] == "Umano"):
            print("FALSE POSITIVE", " - BOT")

    elif(sum_human > sum_bot):
        print("Calcolato come UMANO")

        if(query_user_info['Categoria'] == "Umano"):
            print("TRUE POSITIVE", " - UMANO")

        elif(query_user_info['Categoria'] == "Bot"):
            print("FALSE POSITIVE", " - UMANO")
    else:
        print("Non sono in grado di calcolarlo")



num_hash_functions = 256 
threshold_value = 0.1

start_time = time.time()


# File JSON di ground truth
with open('ground_truth_400_record.json', 'r') as json_file:
    user_data = json.load(json_file)

users = {}
for user_entry in user_data:
    user_id = user_entry['User ID']
    tweets = user_entry["Tweets"]
    tweets_data = [(tweet["Testo"], tweet["Data"]) for tweet in tweets]
    user_info = {
        "Categoria": user_entry["Categoria"],
        "Tweets": tweets_data
    }
    users[user_id] = user_info


minhashes = {}
for user, data in users.items():
    minhash = MinHash(num_perm=num_hash_functions)
    for tweet_data in data["Tweets"]:
        data_tweet = tweet_data[1]
        tweet = normalize_text(tweet_data[0])
        tokens = tokenize_text(tweet)
        shingles = create_shingles(tokens)
        for shingle in shingles:
            minhash.update(shingle.encode('utf-8'))
        minhash.update(str(data_tweet).encode('utf-8'))
    minhashes[user] = minhash             


lsh = MinHashLSH(threshold=threshold_value, num_perm=num_hash_functions)

for user, minhash in minhashes.items():
    lsh.insert(user, minhash)


# File JSON di test
with open('test_1600.json', 'r') as query_file:
    query_data_list = json.load(query_file)

for query_data in query_data_list:
    query_user = query_data["User ID"]
    query_tweets = [(tweet["Testo"], tweet["Data"]) for tweet in query_data["Tweets"]]
    query_user_info = {
        "Categoria": query_data["Categoria"],
        "Tweets": query_tweets
    }
    
    print("\n-------------------------------------------------------------------------------------------------------------------")
    print(f"Dato di query: {query_user}")
    print(f"Categoria: {query_user_info['Categoria']}")

    query_minhash = MinHash(num_perm=num_hash_functions)
    for tweet_data in query_user_info["Tweets"]:
        data_tweet = tweet_data[1]
        tweet = normalize_text(tweet_data[0])
        tokens = tokenize_text(tweet)
        shingles = create_shingles(tokens)
        for shingle in shingles:
            query_minhash.update(shingle.encode('utf-8'))
        query_minhash.update(str(data_tweet).encode('utf-8'))

    result = lsh.query(query_minhash)
    print(f"Totale: {len(result)}")

    category_bot = {}
    category_human = {}
    for user_id in result:
        if user_id in users:
            category = users[user_id]["Categoria"]
            if category == "Bot":
                category_bot[user_id] = users[user_id]
            elif category == "Umano":
                category_human[user_id] = users[user_id]

    print(f"Bot: {len(category_bot)}")
    print(f"Umano: {len(category_human)}\n")

    if len(result) > 0:
        similarities = {}
        for user, minhash in minhashes.items():
            if user in result:
                jaccard_coefficient = jaccard_similarity_minhash(query_minhash, minhash)
                similarities[user] = jaccard_coefficient
        calculate(similarities)
    else:
        print("Non riesco a determinare risultati simili per questo record")



end_time = time.time()

execution_time = end_time - start_time

print(f"Il tempo di esecuzione è stato di {execution_time} secondi")

